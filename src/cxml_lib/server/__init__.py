from dataclasses import dataclass

from cxml_lib.logger import logger
from cxml_lib.server.flask import app, pubsub_listener, socketio


@dataclass
class Args:
    port: int
    debug: int


def main(args: Args):
    try:
        logger.info(f"Starting server on port {args.port}")
        logger.info("Server running")
        if args.debug:
            logger.warning("Debug mode is enabled")

        # Start pubsub listener
        _background_thread = socketio.start_background_task(pubsub_listener)
        socketio.run(
            app, host="localhost", port=args.port, debug=args.debug, log_output=False
        )
    except Exception as e:
        logger.error(e)
        raise
    finally:
        logger.warning("Server stopped")
