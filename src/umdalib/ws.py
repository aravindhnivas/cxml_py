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


server = None


def main(args: Args):
    global server
    logger.info(f"Starting WebSocket server on port {args.wsport}")

    start_server = websockets.serve(handle_connection, "localhost", 8765)
    server = asyncio.get_event_loop().run_until_complete(start_server)

    if args.action == "stop":
        stop_websocket_server()
        return

    asyncio.get_event_loop().run_forever()


def stop_websocket_server():
    server.close()
    asyncio.get_event_loop().run_until_complete(server.wait_closed())
    logger.info("WebSocket server stopped")
