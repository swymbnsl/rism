import asyncio
import websockets
async def test():
    try:
        async with websockets.connect('wss://api.deepgram.com/v1/listen') as ws:
            print("Connected successfully")
    except Exception as e:
        print(f"Failed: {e}")
asyncio.run(test())
