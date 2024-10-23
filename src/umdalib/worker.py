import json
from redis import Redis
from umdalib.utils import compute, logger

redis_conn = Redis.from_url("redis://localhost:6379/0")


def publish_event(event_type, payload):
    """Publish event to Redis channel"""
    try:
        message = {"event": event_type, "payload": payload}
        redis_conn.publish("job_channel", json.dumps(message))
    except Exception as e:
        logger.error(f"Error publishing event: {str(e)}")


def long_computation(job_id: str, pyfile: str, args: dict | str):
    """Worker function that performs computation and publishes events"""
    try:
        # Publish job started event
        publish_event("job_started", {"job_id": job_id, "status": "started"})

        # Perform computation
        result = compute(pyfile, args)

        # Publish result
        publish_event(
            "job_result", {"job_id": job_id, "result": result, "status": "completed"}
        )

        return result
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in computation for job {job_id}: {error_msg}")

        # Publish error event
        publish_event(
            "job_error", {"job_id": job_id, "error": error_msg, "status": "error"}
        )

        raise
