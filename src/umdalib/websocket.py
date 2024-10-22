import asyncio
from dataclasses import dataclass
import websockets
import json
from umdalib.utils import logger

stop = asyncio.Event()


async def long_running_computation(data):
    # Simulate a long-running computation
    await asyncio.sleep(10)  # Replace with actual computation
    return {"result": f"Processed {data}"}


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
                data = json.loads(message)
                await websocket.send(json.dumps(data))
                logger.info(f"Sent: {data}")
                # result = await long_running_computation(data)
                # await websocket.send(json.dumps(result))
                # logger.info(f"Sent: {result}")

    except json.JSONDecodeError:
        await websocket.send("Invalid JSON received.")
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await websocket.send("Error processing message.")
    finally:
        logger.info("Connection closed.")


async def start(wsport: int) -> None:
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
