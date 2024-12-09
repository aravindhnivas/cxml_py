import json
from redis import Redis
from umdalib.utils.computation import compute
from umdalib.logger import logger
import traceback
import multiprocessing

redis_conn = Redis.from_url("redis://localhost:6379/0")


def publish_event(event_type, payload):
    """Publish event to Redis channel"""
    try:
        message = {"event": event_type, "payload": payload}
        redis_conn.publish("job_channel", json.dumps(message))
    except Exception as e:
        logger.error(f"Error publishing event: {str(e)}")


def run_computation_in_process(result_queue, pyfile, args):
    """Run the computation in a separate process"""
    try:
        result = compute(pyfile, args)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)


def long_computation(job_id: str, pyfile: str, args: dict | str):
    """Worker function that performs computation and publishes events"""
    try:
        # Publish job started event
        publish_event("job_started", {"job_id": job_id, "status": "started"})

        # Create a process for the computation
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=run_computation_in_process, args=(result_queue, pyfile, args)
        )
        process.start()

        # Monitor for cancellation
        while process.is_alive():
            if redis_conn.get(f"job_cancelled_{job_id}"):
                # More graceful termination
                process.terminate()
                grace_period = 5  # seconds
                process.join(timeout=grace_period)
                if process.is_alive():
                    logger.warning(
                        f"Process for job {job_id} did not terminate gracefully, forcing kill"
                    )
                    process.kill()  # Force kill if still running after grace period
                    process.join()

                publish_event(
                    "job_cancelled",
                    {
                        "job_id": job_id,
                        "status": "cancelled",
                        "message": "Job was forcefully cancelled",
                    },
                )
                return None
            process.join(timeout=0.5)  # Check every 0.5 seconds

        # Get result if process completed normally
        if not redis_conn.get(f"job_cancelled_{job_id}"):
            result = result_queue.get()
            if isinstance(result, Exception):
                raise result
            publish_event(
                "job_result",
                {"job_id": job_id, "result": result, "status": "completed"},
            )
            return result

    except Exception as e:
        error_msg = str(e)
        error = traceback.format_exc(5)
        logger.error(f"Error in computation for job {job_id}: {error_msg}")

        # Publish error event
        publish_event(
            "job_error",
            {
                "job_id": job_id,
                "error": error,
                "error_msg": error_msg,
                "status": "error",
            },
        )

        raise
    finally:
        # Clean up
        redis_conn.delete(f"job_cancelled_{job_id}")
