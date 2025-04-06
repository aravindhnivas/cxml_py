import json
import traceback
import warnings
from importlib import import_module, reload
from pathlib import Path as pt
from time import perf_counter

import joblib
from gensim.models import word2vec

from umdalib.logger import Paths, logger
from umdalib.utils.json import convert_to_json_compatible, safe_json_dump


def load_model(filepath: str, use_joblib: bool = False):
    logger.info(f"Loading model from {filepath}")
    if not pt(filepath).exists():
        logger.error(f"Model file not found: {filepath}")
        raise FileNotFoundError(f"Model file not found: {filepath}")
    logger.info(f"Model loaded from {filepath}")

    if use_joblib:
        return joblib.load(filepath)
    return word2vec.Word2Vec.load(str(filepath))


class MyClass(object):
    @logger.catch
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def compute(pyfile: str, args: dict | str):
    try:
        logger.info(f"{pyfile=}")

        log_dir = Paths().app_log_dir
        args_file = log_dir / f"{pyfile}.args.json"
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            args = {}

        safe_json_dump(args, args_file)
        logger.info(f"\n[Received arguments]\n{json.dumps(args, indent=4)}")

        args = MyClass(**args)

        result_file = log_dir / f"{pyfile}.json"
        # if result_file.exists():
        #     logger.warning(f"Removing existing file: {result_file}")
        #     result_file.unlink()

        with warnings.catch_warnings(record=True) as warnings_list:
            pyfunction = import_module(f"umdalib.{pyfile}")
            pyfunction = reload(pyfunction)

            start_time = perf_counter()
            result: dict = {}

            if args:
                result = pyfunction.main(args)
            else:
                result = pyfunction.main()

            computed_time = f"{(perf_counter() - start_time):.2f} s"

            if not result:
                result = {"info": "No result returned from main() function"}

            result["done"] = True
            result["error"] = False
            result["computed_time"] = computed_time
            result["warnings"] = [str(warning.message) for warning in warnings_list]

            result = convert_to_json_compatible(result)
            logger.success(f"Computation completed successfully in {computed_time}")
            logger.success(f"result = {json.dumps(result, indent=4)}")
            safe_json_dump(result, result_file)

        logger.info(f"Finished main.py execution for {pyfile} in {computed_time}")
        return result
    except Exception:
        error = traceback.format_exc(5)
        logger.error(error)
        raise Exception(error)
