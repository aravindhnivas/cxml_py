import sys
import threading
import traceback
import warnings
from importlib import import_module, reload
from time import perf_counter

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from umdalib.utils import logger

app = Flask(__name__)
CORS(app)


@app.errorhandler(Exception)
def handle_exception(e):
    # Get the full traceback
    tb = traceback.format_exception(*sys.exc_info())

    # Create a detailed error response
    error_response = {"error": str(e), "traceback": tb}

    # You can choose to keep 500 as the status code for all server errors
    return jsonify(error_response), 500


@app.route("/umdapy")
def home():
    return "Server running: umdapy"


# Module cache
module_cache = {}


class MyClass(object):
    @logger.catch
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def preload_modules():
    """Preload frequently used modules."""
    frequently_used_modules = [
        # Add your frequently used module names here
        "training.read_data",
        "training.check_duplicates_on_x_column",
        "training.embedd_data",
        # "training.ml_model",
    ]
    for module_name in frequently_used_modules:
        try:
            module = import_module(f"umdalib.{module_name}")
            module_cache[module_name] = module
            logger.info(f"Preloaded module: {module_name}")
        except ImportError as e:
            logger.error(f"Failed to preload module {module_name}: {e}")


def warm_up():
    """Perform warm-up tasks."""
    logger.info("Starting warm-up phase...")
    preload_modules()
    # Add any other initialization tasks here
    logger.info("Warm-up phase completed.")


# Start warm-up in a separate thread
threading.Thread(target=warm_up, daemon=True).start()


@app.route("/", methods=["POST"])
def compute():
    logger.info("fetching request")

    try:
        startTime = perf_counter()
        data = request.get_json()
        pyfile = data["pyfile"]

        args = MyClass(**data["args"])

        logger.info(f"{pyfile=}\n{args=}")

        with warnings.catch_warnings(record=True) as warnings_list:
            # Use the module cache
            if pyfile in module_cache:
                pyfunction = module_cache[pyfile]
            else:
                pyfunction = import_module(f"umdalib.{pyfile}")
                module_cache[pyfile] = pyfunction

            # pyfunction = import_module(f"umdalib.{pyfile}")
            # Always reload the module to ensure we have the latest version
            pyfunction = reload(pyfunction)
            output = pyfunction.main(args)

            if warnings_list:
                logger.warning(f"Warnings: {warnings_list}")
                output["warnings"] = [str(warning.message) for warning in warnings_list]
        computed_time = perf_counter() - startTime
        if not output:
            output = {"info": "No result returned from main() function"}

        output["computed_time"] = f"{computed_time:.2f} s"
        output["done"] = True
        output["error"] = False
        logger.info(f"function execution done in {computed_time:.2f} s")

        if isinstance(output, dict):
            logger.success("Computation done!!")

            for k, v in output.items():
                if isinstance(v, np.ndarray):
                    output[k] = v.tolist()
            logger.info(f"Returning received to client\n{output=}")
            return jsonify(output)

    except Exception:
        error = traceback.format_exc(5)
        logger.error(error)
        raise
