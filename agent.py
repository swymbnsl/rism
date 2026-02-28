import asyncio
import logging
import os
import time
from dotenv import load_dotenv

from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gemini, deepgram

from streamguard_video_processor import StreamGuardVideoProcessor
from streamguard_audio_processor import StreamGuardAudioProcessor

load_dotenv()
logger = logging.getLogger(__name__)

# List of severe profanities. Real-world systems use comprehensive blocklists or fast LLMs.
# Example list for POC purposes.
BLOCKLIST = ["fuck", "shit", "bitch", "asshole", "cunt"]

async def check_context_for_violations(agent: Agent, video_processor: StreamGuardVideoProcessor):
    """
    Context Pipeline: Background task that runs at 1 FPS, sampling frames
    and asking Gemini if there are Terms of Service violations.
    """
    logger.info("Context Pipeline initialized. Monitoring video stream at 1 FPS.")
    
    # We will use the agent's LLM to analyze images.
    while True:
        try:
            # Note: A real implementation would extract the raw frame cleanly.
            # Here we simulate the 1 FPS sampling loop and "Nuclear Option".
            # For demonstration, we assume we can run a prompt against the VLM.
            # Example heuristic: if a violation is detected > 80% confidence, we nuke.
            
            # Simulated check logic:
            context_safe = True 
            
            if not context_safe:
                logger.critical("🚨 NUCLEAR OPTION TRIGGERED by Context Pipeline! 🚨")
                # Trigger WebRTC connection drop or "Technical Difficulties" slate.
                # await agent.stop()
                break
                
            await asyncio.sleep(1.0)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Context Pipeline error: {e}")
            await asyncio.sleep(1.0)

async def create_agent(**kwargs) -> Agent:
    """Factory to create the Cloud Proxy StreamGuard Agent."""
    llm = gemini.LLM("gemini-2.5-flash") # VLM for Context tracking
    stt = deepgram.STT(eager_turn_detection=True) # STT for Audio Bleeping

    # 1. The Visual Pipeline
    video_processor = StreamGuardVideoProcessor(
        fps=30.0,
        model_path="erax_nsfw_yolo11n.pt", # Changed from default yolo11n.pt to your custom downloaded model
        conf_threshold=0.3,
        nsfw_class_ids=[], # empty = mask all detections for testing
        box_color=(0, 0, 0),
        delay_seconds=0 # Wait exactly 2 seconds before egress
    )

    # 2. The Audio Pipeline
    audio_processor = StreamGuardAudioProcessor(
        delay_seconds=2.0 
    )

    # Note: Vision Agents GetStream plugin allows handling WHIP via Call parameters,
    # but the internal networking is largely transparent.
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="StreamGuard Engine", id="agent"),
        instructions="You are the internal brain of the StreamGuard cloud proxy.",
        llm=llm,
        stt=stt,
        processors=[video_processor, audio_processor],
    )

    # 3. Audio Trigger logic (combining STT to the Audio Pipeline)
    # Using agent event subscription to listen for user transcription
    @agent.events.subscribe
    async def on_user_speech(event):
        # We need to detect if event is a transcription event
        # Depending on the deepgram integration, Vision Agents emits AgentTranscriptEvent or similar
        # For POC, we'll assume a generic event structure where we get text
        if hasattr(event, "text") and getattr(event, "is_final", False):
            transcript = event.text.lower()
            current_time = time.time()
            
            # Very basic keyword matching
            for bad_word in BLOCKLIST:
                if bad_word in transcript:
                    logger.warning(f"Audio Pipeline: Detected '{bad_word}' - scheduling bleep!")
                    # The STT transcription has a slight latency. If the word was spoken recently,
                    # we instruct the audio buffer to wipe the time window.
                    # As we have a 2.0s delay buffer, an STT that completes in < 1.0s gives us plenty of time
                    # to inject the bleep over the historic timestamp in the buffer.
                    # Approximate the hit location based on receipt - 0.5s duration.
                    audio_processor.add_bleep_interval(current_time - 1.0, current_time + 0.5)

    return agent

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    
    async with agent.join(call):
        agent.logger.info("StreamGuard Cloud Proxy is Active.")
        
        # Extract processors from agent to start background tasks
        video_proc = next((p for p in agent.processors if p.name == "streamguard_video_processor"), None)
        
        # Start the Context Pipeline (3)
        context_task = asyncio.create_task(check_context_for_violations(agent, video_proc))
        
        # Wait indefinitely while the agent runs
        try:
            await agent.finish()
        except asyncio.CancelledError:
            pass
        finally:
            context_task.cancel()

if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
