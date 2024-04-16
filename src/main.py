import sys
import json
import warnings
from importlib import import_module
from umdapy.utils import logger


class MyClass(object):
    @logger.catch
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


if __name__ == "__main__":
    pyfile = sys.argv[1]
    args = None
    if len(sys.argv) > 2:
        args = json.loads(sys.argv[2])

    args = MyClass(**args)
    logger.info(f"{pyfile=}\n{args=}")

    logger.info(f"{pyfile=}\n")
    # if "verbose" in args and args.verbose:
    #     logger.info(f"{args=}")

    with warnings.catch_warnings(record=True) as warn:
        pyfunction = import_module(f"umdapy.{pyfile}")
        if args:
            pyfunction.main(args)
        else:
            pyfunction.main()
