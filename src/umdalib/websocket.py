import asyncio
from dataclasses import dataclass
import websockets
import json
from umdalib.utils import logger

stop = asyncio.Event()


async def echo(websocket, pyfile: str) -> None:
    try:
        async for message in websocket:
            logger.info(f"Received: {message} from {pyfile}")

            if not isinstance(message, str):
                await websocket.send("Invalid message received.")

            if message.lower() == "stop":
                logger.warning("Stop command received. Shutting down server...")
                stop.set()
                await websocket.send("Server is shutting down.")
                break
            else:
                message = json.loads(message)
                # time.sleep(10)
                # await asyncio.sleep(10)
                # try:
                #     result = compute(pyfile, message)
                #     await websocket.send(json.dumps(result))
                # except Exception as e:
                #     logger.error(f"Error: {e}")
                #     message["error"] = str(e)
                await websocket.send(json.dumps(message))
                logger.info(f"Sent: {message}")
    finally:
        logger.info("Connection closed.")


async def start(wsport: int) -> None:
    # async with websockets.serve(echo, "localhost", wsport):
    #     await asyncio.Future()  # Run forever
    server = await websockets.serve(echo, "localhost", wsport)
    logger.info(f"WebSocket server started on ws://localhost:{wsport}")
    await stop.wait()
    server.close()
    await server.wait_closed()
    logger.warning("WebSocket server stopped.")


@dataclass
class Args:
    wsport: int = 8765


def main(args: Args) -> None:
    asyncio.run(start(args.wsport))


if __name__ == "__main__":
    asyncio.run(main(Args()))
