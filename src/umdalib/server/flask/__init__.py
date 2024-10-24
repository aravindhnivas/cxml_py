import eventlet

eventlet.monkey_patch()  # This needs to happen before other imports

import sys  # noqa: E402
import traceback  # noqa: E402
import rq_dashboard  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402
from flask_cors import CORS  # noqa: E402
from redis import Redis  # noqa: E402
from rq import Queue  # noqa: E402
from rq.job import Job  # noqa: E402
from flask_socketio import SocketIO  # noqa: E402
from umdalib.utils import Paths, compute, logger  # noqa: E402
import uuid  # noqa: E402
import json  # noqa: E402


# flask app
log_dir = Paths().app_log_dir
app = Flask(__name__)

# CORS(app, resources={r"/*": {"origins": "http://localhost:1420"}})
# socketio = SocketIO(app, cors_allowed_origins="http://localhost:1420")
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
)

# Redis configuration
app.config["REDIS_URL"] = "redis://localhost:6379/0"
redis_conn = Redis.from_url(app.config["REDIS_URL"])
queue = Queue(connection=redis_conn)

# Configure and initialize RQ Dashboard
app.config.from_object(rq_dashboard.default_settings)
app.config["RQ_DASHBOARD_REDIS_URL"] = app.config["REDIS_URL"]
rq_dashboard.web.setup_rq_connection(app)
app.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")

# Track connected clients
connected_clients = set()


@socketio.on("connect")
def handle_connect():
    client_id = request.sid
    connected_clients.add(client_id)
    logger.info(f"Client connected: {client_id}")
    # Emit directly to the connected client
    socketio.emit(
        "connection_response",
        {"status": "connected", "client_id": client_id},
        room=client_id,
    )


@socketio.on("disconnect")
def handle_disconnect():
    client_id = request.sid
    if client_id in connected_clients:
        connected_clients.remove(client_id)
    logger.info(f"Client disconnected: {client_id}")


def emit_to_all_clients(event_type, data):
    """Helper function to emit to all connected clients"""
    logger.info(f"Emitting {event_type} to {len(connected_clients)} clients")
    for client_id in connected_clients:
        try:
            socketio.emit(event_type, data, room=client_id)
            logger.info(f"Emitted {event_type} to client {client_id}")
        except Exception as e:
            logger.error(f"Error emitting to client {client_id}: {str(e)}")


def handle_worker_message(message):
    """Handle messages from workers through Redis pubsub"""
    try:
        if message and message.get("type") == "message":
            data = json.loads(message["data"])
            event_type = data.get("event")
            if event_type:
                logger.info(f"Broadcasting event: {event_type}")
                # Use the helper function to emit to all clients
                emit_to_all_clients(event_type, data["payload"])
    except Exception as e:
        logger.error(f"Error handling worker message: {str(e)}")
        logger.error(traceback.format_exc())


def pubsub_listener():
    """Listen for messages from workers"""
    pubsub = redis_conn.pubsub()
    pubsub.subscribe("job_channel")
    logger.info("Pubsub listener started")

    while True:
        try:
            message = pubsub.get_message(timeout=1.0)
            if message:
                logger.info(f"Received message from Redis: {message}")
                socketio.start_background_task(handle_worker_message, message)
            eventlet.sleep(0.1)  # Use eventlet sleep
        except Exception as e:
            logger.error(f"Error in pubsub listener: {str(e)}")
            eventlet.sleep(1)


@app.route("/enqueue_job", methods=["POST"])
def enqueue_job():
    try:
        data = request.get_json()
        job_id = f"job_{uuid.uuid4().hex}"

        # Enqueue the job
        job = queue.enqueue(
            "umdalib.worker.long_computation",
            job_id,
            data["pyfile"],
            data["args"],
            job_id=job_id,
        )

        # Emit job queued event to all clients
        emit_to_all_clients("job_queued", {"job_id": job.id, "status": "queued"})

        return jsonify({"job_id": job.id}), 202
    except Exception as e:
        logger.error(f"Error enqueueing job: {str(e)}")
        return jsonify({"error": str(e)}), 500


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


@app.route("/compute", methods=["POST"])
def run_compute():
    try:
        data = request.get_json()
        result = compute(data["pyfile"], data["args"])
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error enqueueing job: {str(e)}")
        return jsonify({"error": str(e)}), 500


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
