import sys
import traceback

import rq_dashboard
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from redis import Redis
from rq import Queue
from rq.job import Job

from umdalib.utils import Paths, compute, logger

# flask app
log_dir = Paths().app_log_dir
app = Flask(__name__)
CORS(app)

# Redis configuration
app.config["REDIS_URL"] = "redis://localhost:6379/0"
redis_conn = Redis.from_url(app.config["REDIS_URL"])
queue = Queue(connection=redis_conn)

# Configure and initialize RQ Dashboard
app.config.from_object(rq_dashboard.default_settings)
app.config["RQ_DASHBOARD_REDIS_URL"] = app.config["REDIS_URL"]
rq_dashboard.web.setup_rq_connection(app)
app.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")


def long_computation(pyfile: str, args: dict | str):
    result = compute(pyfile, args)
    return result


@app.route("/enqueue_job", methods=["POST"])
def enqueue_job():
    logger.info("fetching request")
    data = request.get_json()
    job = queue.enqueue(long_computation, data["pyfile"], data["args"])
    return jsonify({"job_id": job.id}), 202
    # output = compute(data["pyfile"], data["args"])
    # return jsonify(output), 200


@app.route("/job_status/<job_id>", methods=["GET"])
def get_job_status(job_id):
    job = Job.fetch(job_id, connection=redis_conn)
    return jsonify({"status": job.get_status()}), 200


@app.route("/job_result/<job_id>", methods=["GET"])
def get_job_result(job_id):
    job = Job.fetch(job_id, connection=redis_conn)
    if job.is_finished:
        return jsonify({"result": job.result}), 200
    else:
        return jsonify({"message": "Job not completed yet"}), 202


@app.route("/cancel_job/<job_id>", methods=["POST"])
def cancel_job(job_id):
    job = Job.fetch(job_id, connection=redis_conn)
    job.cancel()
    return jsonify({"message": "Job cancelled"}), 200


@app.errorhandler(Exception)
def handle_exception(e):
    # Get the full traceback
    tb = traceback.format_exception(*sys.exc_info())

    # Create a detailed error response
    error_response = {"error": str(e), "traceback": tb}

    # You can choose to keep 500 as the status code for all server errors
    return jsonify(error_response), 500


@app.route("/umdapy")
def umdapy():
    return "Server running: umdapy"


@app.route("/")
def home():
    # return "Python server running for UMDA_UI"
    return render_template("index.html")


# class MyClass(object):
#     @logger.catch
#     def __init__(self, **kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)

# Module cache
# module_cache = {}


# def preload_modules():
#     """Preload frequently used modules."""
#     frequently_used_modules = [
#         # Add your frequently used module names here
#         "training.read_data",
#         "training.check_duplicates_on_x_column",
#         "training.embedd_data",
#         "training.ml_prediction",
#     ]
#     for module_name in frequently_used_modules:
#         try:
#             module = import_module(f"umdalib.{module_name}")
#             module_cache[module_name] = module
#             logger.info(f"Preloaded module: {module_name}")
#         except ImportError as e:
#             logger.error(f"Failed to preload module {module_name}: {e}")


# def warm_up():
#     """Perform warm-up tasks."""
#     logger.info("Starting warm-up phase...")
#     preload_modules()
#     # Add any other initialization tasks here
#     logger.info("Warm-up phase completed.")


# # Start warm-up in a separate thread
# threading.Thread(target=warm_up, daemon=True).start()
