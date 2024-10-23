from dataclasses import dataclass
from umdalib.server.flask import app, socketio, pubsub_listener
from umdalib.utils import logger


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
