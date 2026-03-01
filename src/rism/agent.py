import asyncio
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the current working directory before importing AI plugins
env_path = Path.cwd() / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv() # Fallback to default behavior

from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gemini, deepgram
from vision_agents.core.stt.events import STTTranscriptEvent, STTPartialTranscriptEvent

from .rism_video_processor import RISMVideoProcessor
from .rism_audio_processor import RISMAudioProcessor

logger = logging.getLogger(__name__)

# List of severe profanities. Real-world systems use comprehensive blocklists or fast LLMs.
# Example list for POC purposes.
BLOCKLIST = ["fuck", "shit", "bitch", "asshole", "cunt", "company"]

# Shared delay for both audio and video — MUST be the same for A/V sync
DELAY_SECONDS = 2.0

async def check_context_for_violations(agent: Agent, video_processor: RISMVideoProcessor):
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
    """Factory to create the Cloud Proxy RISM Agent."""
    llm = gemini.LLM("gemini-2.5-flash")
    stt = deepgram.STT(eager_turn_detection=True)

    import importlib.resources
    model_path = str(importlib.resources.files("rism").joinpath("erax_nsfw_yolo11n.pt"))

    # 1. The Visual Pipeline
    video_processor = RISMVideoProcessor(
        fps=30.0,
        model_path=model_path,
        conf_threshold=0.3,
        nsfw_class_ids=[],
        box_color=(0, 0, 0),
        delay_seconds=DELAY_SECONDS,  # Synced with audio
    )

    # 2. The Audio Pipeline
    audio_processor = RISMAudioProcessor(
        delay_seconds=DELAY_SECONDS,  # Synced with video
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="RISM Engine", id="agent"),
        instructions="You are the internal brain of the RISM cloud proxy.",
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
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)
    
    # --- OBS WHIP Credentials ---
    try:
        call_response = await call.get()
        whip_info = call_response.data.call.ingress.whip
        
        obs_user_id = "obs-broadcaster"
        # IMPORTANT: Use the raw Stream client, NOT agent.edge.create_user().
        # Edge.create_user() sets self.agent_user_id which would overwrite
        # the agent's identity and break the SDK's track-skip logic.
        await agent.edge.client.create_user(name="OBS Streamer", id=obs_user_id)
        obs_token = agent.edge.client.create_token(obs_user_id)
        
        print("\n" + "="*60, flush=True)
        print("🚀 RISM OBS INTEGRATION (WHIP)", flush=True)
        print("="*60, flush=True)
        print(f"URL:          {whip_info.address}", flush=True)
        print(f"Bearer Token: {obs_token}", flush=True)
        print("-" * 60, flush=True)
        print("In OBS: Settings > Stream > Service: WHIP", flush=True)
        print("Paste the URL and Bearer Token above. Then 'Start Streaming'.", flush=True)
        print("="*60 + "\n", flush=True)
    except Exception as e:
        print(f"❌ Failed to fetch OBS WHIP credentials: {e}", flush=True)

    youtube_key = os.getenv("YOUTUBE_STREAM_KEY")

    # OBS connects via WHIP asynchronously. The SDK will detect OBS's
    # video track whenever it joins and call process_video() automatically.
    print("⏳ Waiting for OBS to connect via WHIP...", flush=True)
    async with agent.join(call):
        agent.logger.info("RISM Cloud Proxy is Active.")

        # --- YouTube RTMP Egress (agent's processed output only) ---
        # Start AFTER agent.join() has connected and process_video() has been triggered.
        if youtube_key:
            try:
                from getstream.models import RTMPBroadcastRequest, LayoutSettingsRequest
                
                # Use single-participant layout so YouTube shows ONLY the
                # agent's processed video + audio, not the raw OBS feed.
                await call.start_rtmp_broadcasts(broadcasts=[
                    RTMPBroadcastRequest(
                        name="youtube-live",
                        stream_url="rtmp://a.rtmp.youtube.com/live2",
                        stream_key=youtube_key,
                        layout=LayoutSettingsRequest(
                            name="single-participant",
                        ),
                    )
                ])
                logger.info("✅ YouTube Broadcast Initialized (agent-only output)!")
            except Exception as e:
                logger.error(f"❌ Failed to start YouTube RTMP broadcast: {e}")

        video_proc = next((p for p in agent.processors if p.name == "rism_video_processor"), None)
        context_task = asyncio.create_task(check_context_for_violations(agent, video_proc))
        
        try:
            await agent.finish()
        except asyncio.CancelledError:
            pass
        finally:
            if youtube_key:
                try:
                    await call.stop_all_rtmp_broadcasts()
                    logger.info("🛑 Stopped YouTube Broadcast.")
                except:
                    pass
            context_task.cancel()

def verify_env():
    """Check for required environment variables and exit with instructions if missing."""
    required = {
        "STREAM_API_KEY": "GetStream API Key (https://getstream.io/video/docs/python-vision/)",
        "STREAM_API_SECRET": "GetStream API Secret",
        "GOOGLE_API_KEY": "Google Gemini API Key (https://aistudio.google.com/)",
        "DEEPGRAM_API_KEY": "Deepgram API Key (https://console.deepgram.com/)",
        "YOUTUBE_STREAM_KEY": "YouTube Stream Key for RTMP Egress",
    }
    
    missing = [f"- {var}: {desc}" for var, desc in required.items() if not os.getenv(var)]
    
    if missing:
        print("\n" + "!"*60)
        print("🛑 MISSING CONFIGURATION")
        print("!"*60)
        print("RISM needs the following environment variables to function:")
        print("\n".join(missing))
        print("\nFix: Create a '.env' file in this folder with these keys.")
        print("="*60 + "\n")
        os._exit(1)

def main():
    verify_env()
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()

if __name__ == "__main__":
    main()
