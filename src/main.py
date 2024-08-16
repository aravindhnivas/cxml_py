import sys
import json
from typing import Dict
import warnings
from importlib import import_module
import multiprocessing
import numpy as np
from umdalib.utils import logger, Paths
from time import perf_counter

log_dir = Paths().app_log_dir


class MyClass(object):
    @logger.catch
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


if __name__ == "__main__":
    multiprocessing.freeze_support()

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

    result_file = log_dir / f"{pyfile}.json"
    if result_file.exists():
        logger.info(f"Removing existing file: {result_file}")
        result_file.unlink()

    with warnings.catch_warnings(record=True) as warn:
        pyfunction = import_module(f"umdalib.{pyfile}")

        if args:
            start_time = perf_counter()
            result: Dict = pyfunction.main(args)
            computed_time = f"{(perf_counter() - start_time):.2f} s"

            if isinstance(result, Dict):
                for k, v in result.items():
                    if isinstance(v, np.ndarray):
                        result[k] = v.tolist()

            if not result:
                result = {"info": "No result returned from main() function"}

            result["done"] = True
            result["error"] = False
            result["computed_time"] = computed_time

            if result:
                logger.success(f"{result=}")
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=4)
                    logger.success(f"Result saved to {result_file}")
        else:
            pyfunction.main()
    logger.info("Finished main.py")
