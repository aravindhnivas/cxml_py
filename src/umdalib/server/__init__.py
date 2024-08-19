import waitress
from dataclasses import dataclass
from umdalib.utils import logger
from umdalib.server.flask import app


@dataclass
class Args:
    port: int
    debug: int


def main(args: Args):
    logger.info(f"Starting server on port {args.port}")
    if args.debug:
        app.run(port=args.port, debug=True)
        logger.warning("Server running in debug mode")
        return

    logger.info("Server running in production mode")
    waitress.serve(app, port=args.port, url_scheme="http")
