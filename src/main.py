import json
import multiprocessing
import sys
import warnings
from importlib import import_module
from time import perf_counter
from pathlib import Path as pt
import numpy as np

from umdalib.utils import Paths, logger, safe_json_dump

log_dir = Paths().app_log_dir


class MyClass(object):
    @logger.catch
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    logger.info("Starting main.py")
    # logger.info(f"sys.argv = {json.dumps(sys.argv, indent=4)}")
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
        # args = json.loads(sys.argv[2])

    args_file = log_dir / f"{pyfile}.args.json"
    safe_json_dump(args, args_file)
    # with open(args_file, "w") as f:
    #     json.dump(args, f, indent=4)
    #     logger.success(f"Result saved to {args_file}")

    logger.info(f"{pyfile=}")
    logger.info(f"\n[Received arguments]\n{json.dumps(args, indent=4)}")
    logger.info(f"{pyfile=}\n")
    args = MyClass(**args)

    result_file = log_dir / f"{pyfile}.json"
    if result_file.exists():
        logger.warning(f"Removing existing file: {result_file}")
        result_file.unlink()

    with warnings.catch_warnings(record=True) as warn:
        pyfunction = import_module(f"umdalib.{pyfile}")

        start_time = perf_counter()
        result: dict = {}

        if args:
            result = pyfunction.main(args)
        else:
            result = pyfunction.main()

        computed_time = f"{(perf_counter() - start_time):.2f} s"

        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    result[k] = v.tolist()
                if isinstance(v, pt):
                    result[k] = str(v)

        if not result:
            result = {"info": "No result returned from main() function"}

        result["done"] = True
        result["error"] = False
        result["computed_time"] = computed_time
        logger.success(f"Computation completed successfully in {computed_time}")
        logger.success(f"{result=}")
        safe_json_dump(result, result_file)
        # with open(result_file, "w") as f:
        #     json.dump(result, f, indent=4)
        #     logger.success(f"Result saved to {result_file}")

    logger.info("Finished main.py")
