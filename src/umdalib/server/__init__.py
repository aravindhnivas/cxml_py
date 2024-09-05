from dataclasses import dataclass

import waitress

from umdalib.server.flask import app
from umdalib.utils import logger


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
