import warnings
import traceback
from time import perf_counter
from importlib import import_module, reload
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from umdalib.utils import logger
import sys
# from umdalib.utils import Paths
# import json
# from pathlib import Path as pt

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


class MyClass(object):
    @logger.catch
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


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
            pyfunction = import_module(f"umdalib.{pyfile}")
            pyfunction = reload(pyfunction)
            output = pyfunction.main(args)

            if warnings_list:
                logger.warning(f"Warnings: {warnings_list}")
                output["warnings"] = [str(warning.message) for warning in warnings_list]
        computed_time = perf_counter() - startTime
        if not output:
            output = {"info": "No result returned from main() function"}

        output["computed_time"] = f"{computed_time:.2f} s"
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
