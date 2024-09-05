import asyncio
from dataclasses import dataclass

import websockets

from umdalib.utils import logger


@dataclass
class Args:
    wsport: int
    action: str


def main(args: Args):
    logger.info(f"Starting WebSocket server on port {args.wsport}")
    uri = f"ws://localhost:{args.wsport}"

    if args.action == "stop":
        logger.info("Stopping WebSocket server")
        asyncio.get_event_loop().run_until_complete(stop_server(uri))
        return

    logger.info("Starting WebSocket server")
    asyncio.get_event_loop().run_until_complete(start_server(uri, args.wsport))


async def handle_connection(websocket, path) -> None:
    while True:
        message = await websocket.recv()
        logger.info(f"Received from JavaScript: {message}")
        await websocket.send(f"Hello from umdapy! {message}")


async def start_server(uri: str, port: int) -> None:
    async with websockets.serve(handle_connection, "localhost", int(port)):
        logger.info(f"WebSocket server started on {uri}")
        await asyncio.Future()  # run forever


async def stop_server(uri: str) -> None:
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send("STOP")
            logger.info("Stop signal sent to WebSocket server")

            # Wait for a short time to allow the server to process the stop signal
            await asyncio.sleep(1)
            logger.info(f"WebSocket connection closed: {websocket.closed}")
            # Check if the connection is closed
            if websocket.closed:
                logger.info("WebSocket connection closed successfully")
            else:
                logger.warning("WebSocket connection is still open")
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection was closed by the server")
    except Exception as e:
        logger.error(f"Error while stopping WebSocket server: {str(e)}")
