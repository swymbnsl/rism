import asyncio
import logging
import time
import numpy as np
from typing import Optional, List
import aiortc
from av import AudioFrame

import fractions
from aiortc.mediastreams import MediaStreamTrack
from getstream.video.rtc import PcmData
from vision_agents.core.processors import AudioProcessorPublisher

logger = logging.getLogger(__name__)


class QueuedAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self._queue: asyncio.Queue = asyncio.Queue()
        self._timestamp: int = 0

    async def add_frame(self, frame: AudioFrame):
        await self._queue.put(frame)

    async def recv(self) -> AudioFrame:
        """Called by aiortc to get the next audio frame."""
        frame = await self._queue.get()
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, frame.sample_rate)
        self._timestamp += frame.samples
        return frame

    async def flush(self):
        """Clears the pending audio frames queue (called by the agent on turn changes)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class RISMAudioProcessor(AudioProcessorPublisher):
    """
    Audio processor that maintains a delay buffer and dynamically
    overwrites specific time spans with a 1000Hz sine wave when instructed.
    
    The delay buffer holds audio for `delay_seconds` before publishing,
    giving the STT engine time to detect blocked words and inject bleeps.
    """
    name = "streamguard_audio_processor"

    def __init__(self, delay_seconds: float = 2.0, sample_rate: int = 48000, channels: int = 1):
        self.delay_seconds = delay_seconds
        self.sample_rate = sample_rate
        self.channels = channels

        self._audio_track = QueuedAudioTrack(sample_rate=self.sample_rate, channels=self.channels)

        self._audio_buffer: List[dict] = []
        self._running = False
        self._buffer_task: Optional[asyncio.Task] = None

        # List of bad intervals (start_time, end_time) to bleep out
        self._bleep_intervals: List[tuple] = []

        # Phase-continuous sine wave generator state
        self._beep_phase: float = 0.0
        self._beep_freq: float = 1000.0

        self._frames_received = 0
        self._frames_published = 0
        self._bleeps_applied = 0

    async def start_processing(self):
        if not self._running:
            self._running = True
            self._buffer_task = asyncio.create_task(self._process_buffer())
            logger.info("🔊 Audio buffer processor started (delay=%.1fs)", self.delay_seconds)

    async def process_audio(self, audio_data: PcmData) -> None:
        """Process incoming audio, queueing it in the delay buffer."""
        if not self._running:
            await self.start_processing()

        receive_time = time.time()
        self._frames_received += 1

        # PcmData.samples is a numpy ndarray (int16), store it directly
        samples = audio_data.samples  # numpy ndarray
        sr = audio_data.sample_rate
        ch = audio_data.channels

        self._audio_buffer.append({
            'timestamp': receive_time,
            'samples': samples,
            'sample_rate': sr,
            'channels': ch,
        })

        if self._frames_received % 500 == 1:
            logger.info(
                "🔊 Audio buffer: received=%d, published=%d, bleeps=%d, buffered=%d, intervals=%d",
                self._frames_received, self._frames_published, self._bleeps_applied,
                len(self._audio_buffer), len(self._bleep_intervals),
            )

    def add_bleep_interval(self, start_time: float, end_time: float):
        """Instruct the processor to overwrite this time window with a 1000Hz sine wave."""
        logger.warning(
            "🔇 BLEEP INTERVAL ADDED: %.3f -> %.3f (duration=%.3fs, buffer_size=%d)",
            start_time, end_time, end_time - start_time, len(self._audio_buffer),
        )
        self._bleep_intervals.append((start_time, end_time))

    def _generate_mute_samples(self, num_samples: int, channels: int) -> np.ndarray:
        """Generates silence (all zeros) as int16 numpy array to mute blocked audio."""
        if channels > 1:
            return np.zeros((num_samples, channels), dtype=np.int16)
        return np.zeros(num_samples, dtype=np.int16)

    async def _process_buffer(self):
        """Continuously check buffer and dispatch audio after the configured delay."""
        while self._running:
            now = time.time()

            while self._audio_buffer and (now - self._audio_buffer[0]['timestamp']) >= self.delay_seconds:
                chunk = self._audio_buffer.pop(0)
                frame_time = chunk['timestamp']
                samples: np.ndarray = chunk['samples']
                sr: int = chunk['sample_rate']
                ch: int = chunk['channels']

                # Determine number of samples (first dimension)
                num_samples = samples.shape[0]

                # Check if this frame falls inside any bleep intervals
                should_bleep = False
                for start, end in self._bleep_intervals:
                    if start <= frame_time <= end:
                        should_bleep = True
                        break

                if should_bleep:
                    self._bleeps_applied += 1
                    if self._bleeps_applied % 20 == 1:
                        logger.warning(
                            "🔇 BLEEPING frame at t=%.3f (bleep #%d)",
                            frame_time, self._bleeps_applied,
                        )
                    out_samples = self._generate_mute_samples(num_samples, ch)
                else:
                    # Reset beep phase when not bleeping so each new bleep starts clean
                    self._beep_phase = 0.0
                    out_samples = samples

                # Build av.AudioFrame from numpy
                out_bytes = out_samples.astype(np.int16).tobytes()
                layout = 'stereo' if ch == 2 else 'mono'
                frame = AudioFrame(format='s16', layout=layout, samples=num_samples)
                frame.sample_rate = sr

                for p in frame.planes:
                    p.update(out_bytes)

                await self._audio_track.add_frame(frame)
                self._frames_published += 1

            # Clean up expired bleep intervals (older than buffer window + margin)
            cutoff = now - self.delay_seconds - 5.0
            self._bleep_intervals = [(s, e) for s, e in self._bleep_intervals if e > cutoff]

            await asyncio.sleep(0.005)

    def publish_audio_track(self) -> aiortc.AudioStreamTrack:
        """Return the audio track to publish."""
        return self._audio_track

    async def close(self) -> None:
        self._running = False
        if self._buffer_task:
            self._buffer_task.cancel()
        if self._audio_track:
            self._audio_track.stop()
        logger.info(
            "🔊 Audio processor closed. Total: received=%d, published=%d, bleeps=%d",
            self._frames_received, self._frames_published, self._bleeps_applied,
        )
