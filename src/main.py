import json
import multiprocessing
import sys
from umdalib.utils import logger, compute


if __name__ == "__main__":
    multiprocessing.freeze_support()

    logger.info("Starting main.py")
    pyfile = sys.argv[1]
    args = None
    if len(sys.argv) > 2:
        try:
            if not sys.argv[2].strip():
                raise ValueError("Input JSON string is empty")
            args = json.loads(sys.argv[2])
            logger.success("Successfully loaded JSON string")
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e}")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"ValueError: {e}")
            sys.exit(1)

    compute(pyfile, args)
