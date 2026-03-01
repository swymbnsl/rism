import asyncio
import logging
import time
import math
import numpy as np
from typing import Optional, List
import aiortc
from av import AudioFrame

import fractions
from aiortc.mediastreams import MediaStreamTrack
from getstream.video.rtc import PcmData
from vision_agents.core.processors import AudioProcessorPublisher

class QueuedAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self._queue = asyncio.Queue()
        self._timestamp: int = 0
        
    async def add_frame(self, frame: AudioFrame):
        await self._queue.put(frame)

    async def recv(self) -> AudioFrame:
        """Called by aiortc to get the next audio frame."""
        frame = await self._queue.get()
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._timestamp += frame.samples
        return frame


class StreamGuardAudioProcessor(AudioProcessorPublisher):
    """
    Audio processor that maintains a 2-second delay buffer and dynamically
    overwrites specific time spans with a 1000Hz sine wave when instructed.
    """
    name = "streamguard_audio_processor"

    def __init__(self, delay_seconds: float = 2.0, sample_rate: int = 48000, channels: int = 1):
        self.delay_seconds = delay_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        
        # aiortc usually expects output via an AudioStreamTrack
        self._audio_track = QueuedAudioTrack(sample_rate=self.sample_rate, channels=self.channels)
        
        self._audio_buffer = []  # List of dicts: {'timestamp': time, 'pcm': bytes}
        self._running = False
        self._buffer_task: Optional[asyncio.Task] = None
        
        # List of bad intervals (start_time, end_time) to bleep out
        self._bleep_intervals = []
        
        # Audio formatting configuration
        self.frame_duration_ms = 20 # Standard WebRTC audio frame duration
        self.samples_per_frame = int(self.sample_rate * (self.frame_duration_ms / 1000.0))

    async def start_processing(self):
        if not self._running:
            self._running = True
            self._buffer_task = asyncio.create_task(self._process_buffer())

    async def process_audio(self, audio_data: PcmData) -> None:
        """Process incoming audio, queueing it in the delay buffer."""
        if not self._running:
            await self.start_processing()

        receive_time = time.time()
        
        # Store raw PCM bytes and metadata in buffer
        self._audio_buffer.append({
            'timestamp': receive_time,
            'pcm': audio_data.samples,
            'sample_rate': audio_data.sample_rate,
            'channels': audio_data.channels
        })
        
        # Note: In a complete implementation, this is where we would also dispatch 
        # a duplicate stream of AudioData or bytes to a background STT service like Deepgram.
        # When Deepgram detects profanity, it calls `add_bleep_interval()`.
        # Deepgram integration usually happens as a background websocket stream.

    def add_bleep_interval(self, start_time: float, end_time: float):
        """Instruct the processor to overwrite this time window with a 1000Hz sine wave."""
        self._bleep_intervals.append((start_time, end_time))

    def _generate_beep(self, num_samples: int, sample_rate: int = 48000, freq: float = 1000.0) -> bytes:
        """Generates a 1000Hz sine wave in 16-bit PCM format."""
        t = np.arange(num_samples) / sample_rate
        # Generate sine wave scaled to 16-bit PCM amplitude
        wave = np.sin(2 * np.pi * freq * t) * 32767.0
        wave = wave.astype(np.int16)
        
        if self.channels == 2:
            # Duplicate for stereo
            wave = np.repeat(wave, 2)
            
        return wave.tobytes()

    async def _process_buffer(self):
        """Continuously check buffer and dispatch audio exactly 2 seconds after receipt."""
        while self._running:
            now = time.time()
            
            while self._audio_buffer and (now - self._audio_buffer[0]['timestamp']) >= self.delay_seconds:
                chunk = self._audio_buffer.pop(0)
                frame_time = chunk['timestamp']
                pcm_data = chunk['pcm']
                num_frames = len(pcm_data) // (2 * self.channels) # 16-bit = 2 bytes per sample
                
                # Check if this frame falls inside any bleep intervals
                should_bleep = False
                for start, end in self._bleep_intervals:
                    # Very simple overlap check: if the frame's origin time falls within the bleep window
                    if start <= frame_time <= end:
                        should_bleep = True
                        break
                
                if should_bleep:
                    # Overwrite with 1000Hz sine wave
                    processed_pcm = self._generate_beep(num_frames, chunk['sample_rate'], 1000.0)
                else:
                    processed_pcm = pcm_data
                
                # Convert PCM bytes to av.AudioFrame
                frame = AudioFrame(format='s16', layout='stereo' if self.channels == 2 else 'mono', samples=num_frames)
                frame.sample_rate = chunk['sample_rate']
                # Fill the underlying plane with byte data
                for p in frame.planes:
                    p.update(processed_pcm)
                
                # Push to the track
                await self._audio_track.add_frame(frame)
                
            # Prevent zero-delay looping
            await asyncio.sleep(0.01)

    def publish_audio_track(self) -> aiortc.AudioStreamTrack:
        """Return the audio track to publish."""
        return self._audio_track

    async def close(self) -> None:
        self._running = False
        if self._buffer_task:
            self._buffer_task.cancel()
        if self._audio_track:
            self._audio_track.stop()
