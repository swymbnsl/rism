# StreamGuard

StreamGuard is a zero-latency, AI-powered cloud moderation proxy designed for live streamers. It intercepts the stream before it reaches the public, utilizing the Vision Agents framework to process audio and video in real-time. It bleeps profanity, blurs NSFW content, and intelligently monitors context, outputting a clean, brand-safe feed to platforms like Twitch or YouTube.

## Architecture: Cloud Proxy

StreamGuard uses a 3-Stage "Cloud Proxy" architecture that introduces a strict and unnoticeable 2-second processing buffer:

1. **Ingestion (The Origin)**: Stream directly to StreamGuard via the WebRTC HTTP Ingestion Protocol (WHIP).
2. **The Moderation Engine (The Proxy)**: The stream is forked into parallel Video (YOLO11), Audio (STT & 1000Hz Bleep), and Context (Gemini 2.5 Flash) pipelines. The 2-second buffer ensures processing completes and the stream stays perfectly synchronized. 
3. **Egress (The Destination)**: A headless worker muxes the cleaned tracks into a standard RTMP feed and pushes it to your actual Twitch/YouTube stream key.

## Prerequisites
1. Python 3.10+
2. **OBS Studio** (v30+ supporting native WHIP output)
3. Free [GetStream](https://getstream.io/) account to host the Vision Agents session

## Installation

1. Install dependencies:
```bash
uv init
uv add "vision-agents[getstream,gemini,ultralytics,deepgram]" av aiortc opencv-python numpy
```

2. Download your preferred YOLO NSFW detection model (e.g., `nsfw_yolo.pt`) or use the default `yolo11n.pt` (automatically downloads).

3. Create your `.env` file:
```env
STREAM_API_KEY="..."
STREAM_API_SECRET="..."
GOOGLE_API_KEY="..." # For Gemini 2.5 Flash
DEEPGRAM_API_KEY="..." # For STT Audio Moderation
```

## How to use with OBS Studio

### Part 1: Starting the Proxy Services

Start the main moderation engine:
```bash
uv run agent.py run
```
The agent establishes the GetStream Edge connection and waits for the broadcast. It initializes the Video and Audio processors, holding the 2-second synchronicity buffers.

### Part 2: OBS Ingestion (WHIP)

Configure OBS to stream to StreamGuard instead of your end destination.
1. In OBS Studio, go to **Settings > Stream**.
2. Service: **WHIP**
3. Server: `<Your GetStream WHIP Endpoint URL for the active Call>`
4. Click **Start Streaming**. OBS will ingest directly into the Vision Agent cloud network with near zero latency.

### Part 3: RTMP Egress (Broadcasting the Safe Stream)

With OBS successfully dumping frames into the Moderation Engine, start the Egress Worker to push the moderated result to your real audience.

Call the start method dynamically within your project or run the module directly if structured as a standalone script:
```python
# Assuming you pass the active 'Call' object that OBS is streaming into
from egress_worker import start_rtmp_egress

await start_rtmp_egress(call, "rtmp://live.twitch.tv/app/YOUR_TWITCH_STREAM_KEY")
```

**You are now done!** StreamGuard intercepts your raw OBS feed over WHIP, runs the 3 AI pipelines asynchronously over the 2-second jitter buffer, and ships the safe, moderated feed sequentially to Twitch/YouTube!
