import json
import warnings
import traceback
from time import perf_counter
from importlib import import_module, reload
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from pathlib import Path as pt

import numpy as np
from umdalib.utils import Paths
from umdalib.utils import logger

app = Flask(__name__)
CORS(app)


@app.route("/umdapy")
def home():
    return "Server running: umdapy"


@app.errorhandler(400)
def pyError(error):
    return jsonify(error=str(error)), 400


class MyClass(object):
    @logger.catch
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def log_output(logfilename: pt):
    logger.info(f"Reading log file: {logfilename=}")
    if not logfilename.exists():
        raise Exception(
            "Computed file is neither returned from main function or saved to temperary location"
        )

    with open(logfilename, "r") as f:
        data = json.load(f)
    return data


@app.route("/", methods=["POST"])
def compute():
    logger.info("fetching request")

    try:
        startTime = perf_counter()
        data = request.get_json()
        pyfile = data["pyfile"]

        calling_file = pyfile.split(".")[-1]
        logfilename = Paths().temp_dir / f"{calling_file}_data.json"
        args = MyClass(**data["args"])

        logger.info(f"{pyfile=}\n{data['args']=}")

        with warnings.catch_warnings(record=True) as warnings_list:
            pyfunction = import_module(f"umdalib.{pyfile}")
            pyfunction = reload(pyfunction)
            output = pyfunction.main(args)

            if warnings_list:
                logger.warning(f"Warnings: {warnings_list}")
                output["warnings"] = [str(warning.message) for warning in warnings_list]
        computed_time = perf_counter() - startTime
        output["computed_time"] = f"{computed_time:.2f} s"

        logger.info(f"function execution done in {computed_time:.2f} s")

        if isinstance(output, dict):
            logger.success(f"Computation done!!")

            for k, v in output.items():
                if isinstance(v, np.ndarray):
                    output[k] = v.tolist()

            logger.info(f"Returning received to client\n{output=}")
            return jsonify(output)

        data = log_output(logfilename)

        for k, v in output.items():
            if isinstance(v, np.ndarray):
                output[k] = v.tolist()
        return jsonify(data)

    except Exception:
        error = traceback.format_exc(5)
        logger.error(error)
        abort(400, description=error)
