import os
import asyncio
import logging
from dotenv import load_dotenv
from getstream.video.rtc import Call

load_dotenv()
logger = logging.getLogger(__name__)

async def start_rtmp_egress(call: Call, rtmp_url: str):
    """
    Instructs the GetStream call to take the current assembled WebRTC stream
    (which contains our moderated Video and Audio) and egress it to a standard RTMP destination.
    
    Args:
        call (Call): The active GetStream Call object.
        rtmp_url (str): The specific RTMP ingest URL (e.g., rtmp://live.twitch.tv/app/STREAM_KEY).
    """
    try:
        logger.info(f"Starting RTMP broadcast to {rtmp_url}...")
        
        # Note: Depending on the specific GetStream Video Python SDK version, 
        # the broadcast method might differ. This represents the logical trigger.
        # Call.start_rtmp_broadcast or similar API call to edge.
        await call.start_broadcasting() 
        
        logger.info("RTMP Egress successfully started. The stream is now live on the destination platform!")
    except Exception as e:
        logger.error(f"Failed to start RTMP egress: {e}")

async def stop_rtmp_egress(call: Call):
    """Stops the RTMP broadcast."""
    try:
        logger.info("Stopping RTMP Egress...")
        await call.stop_broadcasting()
    except Exception as e:
        logger.error(f"Failed to stop RTMP egress: {e}")

if __name__ == "__main__":
    logger.warning("This is a utility module meant to be imported into agent.py or an external trigger script.")
