import asyncio
import websockets
from umdalib.utils import logger
from dataclasses import dataclass


async def handle_connection(websocket, path):
    while True:
        # Receive message from JavaScript client
        message = await websocket.recv()
        logger.info(f"Received from JavaScript: {message}")
        # Send message back to JavaScript client
        await websocket.send(f"Hello from umdapy! {message}")


@dataclass
class Args:
    wsport: int
    action: str


def main(args: Args):
    logger.info(f"Starting WebSocket server on port {args.wsport}")

    if args.action == "stop":
        stop_websocket_server()
        return

    start_server = websockets.serve(handle_connection, "localhost", args.wsport)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


def stop_websocket_server():

    asyncio.get_event_loop().stop()
    logger.info("WebSocket server stopped")
