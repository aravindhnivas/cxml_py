import sys
import json
import warnings
from importlib import import_module
from umdalib.utils import logger, Paths

log_dir = Paths().app_log_dir


class MyClass(object):
    @logger.catch
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


if __name__ == "__main__":
    logger.info("Starting main.py")
    logger.info(f"{sys.argv=}")
    pyfile = sys.argv[1]
    args = None
    if len(sys.argv) > 2:
        try:
            if not sys.argv[2].strip():
                raise ValueError("Input JSON string is empty")
            args = json.loads(sys.argv[2])
            logger.info("Successfully loaded JSON string")
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: {e}")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"ValueError: {e}")
            sys.exit(1)
        # args = json.loads(sys.argv[2])

    logger.info("\n[Received arguments]\n" + json.dumps(args, indent=4))
    logger.info(f"{pyfile=}\n")
    args = MyClass(**args)

    with warnings.catch_warnings(record=True) as warn:
        pyfunction = import_module(f"umdalib.{pyfile}")
        if args:
            result = pyfunction.main(args)
            if result:
                logger.success(f"{result=}")
                with open(log_dir / f"{pyfile}.json", "w") as f:
                    json.dump(result, f, indent=4)
                    logger.success(f"Result saved to {log_dir / f'{pyfile}.json'}")
        else:
            pyfunction.main()
