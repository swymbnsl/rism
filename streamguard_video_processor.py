import asyncio
import logging
import time
from typing import Optional, List
import av
import cv2
import aiortc
from ultralytics import YOLO

from vision_agents.core.processors import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

logger = logging.getLogger(__name__)

class StreamGuardVideoProcessor(VideoProcessorPublisher):
    """
    Video processor using YOLO to detect NSFW content, mask it with solid boxes, 
    and maintain a 2-second delay buffer to synchronize with audio processing.
    """
    name = "streamguard_video_processor"

    def __init__(
        self,
        fps: float = 30.0,
        model_path: str = "./erax_nsfw_yolo11n.pt",
        conf_threshold: float = 0.3,
        nsfw_class_ids: Optional[List[int]] = None,
        box_color: tuple = (0, 0, 0),
        delay_seconds: float = 2.0
    ):
        self.fps = fps
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nsfw_class_ids = nsfw_class_ids if nsfw_class_ids is not None else []
        self.box_color = box_color
        self.delay_seconds = delay_seconds
        
        # Load the edge-optimized model (e.g. YOLO Nano)
        logger.info(f"Loading YOLO model from {self.model_path} for Visual Moderation Pipeline")
        self.model = YOLO(self.model_path)
        
        self._forwarder: Optional[VideoForwarder] = None
        self._track = QueuedVideoTrack()
        
        # Jitter buffer for maintaining sync and the 2-second delay
        self._frame_buffer = []
        self._buffer_task: Optional[asyncio.Task] = None
        self._running = False

    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        self._forwarder = shared_forwarder
        if self._forwarder:
            self._running = True
            # Start the background task to flush the buffer based on time
            self._buffer_task = asyncio.create_task(self._process_buffer())
            
            self._forwarder.add_frame_handler(
                self._on_frame_received, fps=self.fps, name="nsfw_blur_annotator"
            )
            logger.info("StreamGuardVideoProcessor attached to video forwarder.")

    async def _on_frame_received(self, frame: av.VideoFrame):
        """Immediately process inference, then place in the delay buffer."""
        loop = asyncio.get_running_loop()
        # Run the blocking YOLO inference in a background thread to prevent starving aiortc
        annotated_frame = await loop.run_in_executor(None, self._annotate, frame)
        receive_time = time.time()
        
        # Append to buffer with receipt timestamp
        self._frame_buffer.append((receive_time, annotated_frame))

    async def _process_buffer(self):
        """Continuously check the buffer and dispatch frames once they exactly hit the delay threshold."""
        while self._running:
            now = time.time()
            
            # Enforce strictly monotonic PTS ordering (thread pools can return slightly out of order)
            self._frame_buffer.sort(key=lambda x: x[1].pts)
            
            # If the oldest frame in the buffer has been held for exactly `delay_seconds` (or longer)
            while self._frame_buffer and (now - self._frame_buffer[0][0]) >= self.delay_seconds:
                _, frame_to_publish = self._frame_buffer.pop(0)
                await self._track.add_frame(frame_to_publish)
            
            # Sleep a minimal amount to avoid blocking but keep precision tight
            await asyncio.sleep(0.01)

    def _annotate(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            # Convert to numpy array (BGR for OpenCV/Ultralytics)
            img = frame.to_ndarray(format="bgr24")
            
            # Run inference - targeting < 20ms for YOLO Nano on GPU
            results = self.model(img, conf=self.conf_threshold, verbose=False)
            
            # Draw blur
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        
                        if not self.nsfw_class_ids or cls_id in self.nsfw_class_ids:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            h, w = img.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:
                                # Draw solid box over the detected region
                                cv2.rectangle(img, (x1, y1), (x2, y2), self.box_color, -1)
            
            new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

    def publish_video_track(self) -> aiortc.VideoStreamTrack:
        return self._track

    async def stop_processing(self) -> None:
        self._running = False
        if self._buffer_task:
            self._buffer_task.cancel()
            self._buffer_task = None
            
        if self._forwarder:
            await self._forwarder.remove_frame_handler(self._on_frame_received)
            self._forwarder = None
            logger.info("StreamGuardVideoProcessor detached from video forwarder.")

    async def close(self) -> None:
        await self.stop_processing()
        self._track.stop()
