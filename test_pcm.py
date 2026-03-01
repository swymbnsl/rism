import asyncio
from getstream.video.rtc import PcmData
import numpy as np

pcm = PcmData(sample_rate=48000, format='s16', channels=2)
print("PCM created.")
