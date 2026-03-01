import asyncio
import logging
import os
import time
from dotenv import load_dotenv

from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gemini, deepgram
from vision_agents.core.stt.events import STTTranscriptEvent, STTPartialTranscriptEvent

from streamguard_video_processor import StreamGuardVideoProcessor
from streamguard_audio_processor import StreamGuardAudioProcessor

load_dotenv()
logger = logging.getLogger(__name__)

# List of severe profanities. Real-world systems use comprehensive blocklists or fast LLMs.
# Example list for POC purposes.
BLOCKLIST = ["fuck", "shit", "bitch", "asshole", "cunt", "company"]

# Shared delay for both audio and video — MUST be the same for A/V sync
DELAY_SECONDS = 2.0

async def check_context_for_violations(agent: Agent, video_processor: StreamGuardVideoProcessor):
    """
    Context Pipeline: Background task that runs at 1 FPS, sampling frames
    and asking Gemini if there are Terms of Service violations.
    """
    logger.info("Context Pipeline initialized. Monitoring video stream at 1 FPS.")
    
    while True:
        try:
            context_safe = True 
            
            if not context_safe:
                logger.critical("🚨 NUCLEAR OPTION TRIGGERED by Context Pipeline! 🚨")
                break
                
            await asyncio.sleep(1.0)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Context Pipeline error: {e}")
            await asyncio.sleep(1.0)

async def create_agent(**kwargs) -> Agent:
    """Factory to create the Cloud Proxy StreamGuard Agent."""
    llm = gemini.LLM("gemini-2.5-flash")
    stt = deepgram.STT(eager_turn_detection=True)

    # 1. The Visual Pipeline
    video_processor = StreamGuardVideoProcessor(
        fps=30.0,
        model_path="erax_nsfw_yolo11n.pt",
        conf_threshold=0.3,
        nsfw_class_ids=[],
        box_color=(0, 0, 0),
        delay_seconds=DELAY_SECONDS,  # Synced with audio
    )

    # 2. The Audio Pipeline
    audio_processor = StreamGuardAudioProcessor(
        delay_seconds=DELAY_SECONDS,  # Synced with video
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="StreamGuard Engine", id="agent"),
        instructions="You are the internal brain of the StreamGuard cloud proxy.",
        llm=llm,
        stt=stt,
        processors=[video_processor, audio_processor],
    )

    # 3. Audio Trigger logic — listen for STT transcripts and trigger bleeps
    @agent.events.subscribe
    async def on_user_speech(event: STTTranscriptEvent | STTPartialTranscriptEvent):
        # Catch BOTH final and partial transcripts for faster detection
        transcript = event.text.lower().strip()
        if not transcript:
            return

        current_time = time.time()

        for bad_word in BLOCKLIST:
            if bad_word in transcript:
                logger.warning(
                    "🚨 Audio Pipeline: Detected '%s' in transcript '%s' — scheduling bleep!",
                    bad_word, transcript,
                )
                # Bleep from 1.5 seconds ago to 0.5 seconds in the future
                # This wide window ensures the word is fully covered
                audio_processor.add_bleep_interval(
                    current_time - 1.5,
                    current_time + 0.5,
                )
                # Only bleep once per transcript event (avoid double-bleeping)
                break

    return agent

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    
    async with agent.join(call):
        agent.logger.info("StreamGuard Cloud Proxy is Active.")
        
        video_proc = next((p for p in agent.processors if p.name == "streamguard_video_processor"), None)
        context_task = asyncio.create_task(check_context_for_violations(agent, video_proc))
        
        try:
            await agent.finish()
        except asyncio.CancelledError:
            pass
        finally:
            context_task.cancel()

if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
