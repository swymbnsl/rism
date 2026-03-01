# RISM: Realtime Intelligent Stream Moderator

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Vision Agents](https://img.shields.io/badge/Powered%20By-Vision%20Agents-purple.svg)](https://github.com/landing-ai/vision-agents)
[![CLI](https://img.shields.io/badge/CLI-uv%20tool-green.svg)](https://docs.astral.sh/uv/)

## 📋 Table of Contents
1. [Overview](#-overview)
2. [How It Works](#-how-it-works)
3. [Features](#-features)
4. [Prerequisites](#-prerequisites)
5. [Installation](#-installation)
6. [Configuration](#-configuration)
7. [Getting Started (OBS Guide)](#-getting-started-obs-guide)
8. [Uninstalling](#-uninstalling)

## 🚀 Overview
RISM (Realtime Intelligent Stream Moderator) is a zero-latency, AI-powered "Cloud Proxy" built for live streamers. It intercepts your raw video stream before it reaches the public, utilizing the Vision Agents framework to process audio and video in real-time. 

RISM acts as an automated digital bodyguard: it bleeps profanity, blurs NSFW content, and intelligently monitors context to protect your channel from ToS violations, outputting a clean, brand-safe feed to platforms like YouTube or Twitch.

## 🔄 How It Works
RISM uses a clever **3-Stage "Cloud Proxy" architecture**. The secret to its seamless performance is a strict **2-second processing buffer**. This invisible delay ensures the AI has time to analyze the content and maintain perfect audio/video synchronization.

1. **Ingestion (The Origin)**: Stream directly from your OBS to RISM via the WebRTC HTTP Ingestion Protocol (WHIP).
2. **The Moderation Engine (The Proxy)**: The stream is forked into three parallel AI pipelines:
    - **Video**: YOLO11 continuously scans for and blacks out NSFW visual content.
    - **Audio**: Deepgram STT listens for blocklisted profanities and injects a 1000Hz bleep mask dynamically.
    - **Context**: Gemini 2.5 Flash monitors the overall situation for complex Terms of Service violations.
3. **Egress (The Destination)**: The cleaned, fully moderated tracks are automatically muxed back into a standard RTMP feed and pushed to your actual YouTube/Twitch stream key.

## ✨ Features
- **Pure CLI Experience:** Installs globally as an executable tool. No heavy GUIs required.
- **Near-Zero Latency:** Built on ultra-fast WebRTC edge networks.
- **Perfect A/V Sync:** The custom jitter buffer guarantees your bleeps and blurs happen at the exact right millisecond.
- **Bring-Your-Own "Brain":** Powered by standard API keys from GetStream, Gemini, and Deepgram.
- **Pre-bundled Models:** Comes with an edge-optimized YOLO NSFW detection model ready out of the box.

## 📋 Prerequisites
### Infrastructure Requirements
- **Python 3.12+**
- **uv** (The fast Python package installer and resolver)
- Free accounts for **GetStream**, **Google AI Studio** (Gemini), and **Deepgram**.

### Application Requirements
- **OBS Studio** (v30+ supporting native WHIP output)
- Your **YouTube** (or Twitch) Stream Key

## 📥 Installation

Because RISM is packaged as a standard CLI tool, installation is a single command. Use `uv` to install it globally without messing with your local Python environments:

```bash
uv tool install rism
```

*(Alternatively, if you are developing, clone this repo and run `uv pip install -e .`)*

## ⚙️ Configuration

RISM needs your API keys to power its 3-stage brain. Navigate to the folder where you want to run your stream from, and create a `.env` file with the following keys:

```env
# GetStream API Key (https://getstream.io/video/docs/python-vision/)
STREAM_API_KEY="..."
STREAM_API_SECRET="..."

# Google Gemini API Key (https://aistudio.google.com/)
GOOGLE_API_KEY="..."

# Deepgram API Key (https://console.deepgram.com/)
DEEPGRAM_API_KEY="..."

# Your YouTube Stream Key for RTMP Egress
YOUTUBE_STREAM_KEY="..."
```

*Note: RISM has a built-in safety check. If you forget a key, it will instantly tell you which one is missing and safely exit.*

## 🎬 Getting Started (OBS Guide)

Using RISM is incredibly simple. It takes less than 30 seconds to start your proxy stream.

### 1. Start the Moderation Engine
Open your terminal in the directory containing your `.env` file and type:
```bash
rism run --no-demo
```

RISM will warm up its AI models, connect to the edge network, and present you with a beautiful CLI box containing your **WHIP URL** and a **Bearer Token**.

### 2. Connect OBS
1. Open OBS Studio.
2. Go to **Settings > Stream**.
3. Change Service to **WHIP**.
4. Paste the **URL** and **Bearer Token** exactly as outputted by RISM.
5. Click **Start Streaming**.

### 3. You are Live!
As soon as OBS connects, RISM will lock onto your signal, ingest your frames, run the moderation pipelines, and instantly route the safe stream out to the `YOUTUBE_STREAM_KEY` you provided.

## 🗑️ Uninstalling

Decided to turn off your digital bodyguard? Cleanly remove the tool and its entire isolated environment with one command:
```bash
uv tool uninstall rism
```
