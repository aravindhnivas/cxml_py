import multiprocessing
import os
import subprocess
import sys
import time

from redis import Redis
from rq import Connection, Queue, Worker

from umdalib.logger import logger

# Check if the environment variable is set
if os.getenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY") != "YES":
    # Set the environment variable and restart the script
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run([sys.executable] + sys.argv)
    sys.exit()


# class CustomWorker(Worker):
#     def execute_job(self, job, queue):
#         try:
#             return super().execute_job(job, queue)
#         except Exception as e:
#             logger.error(f"Error executing job {job.id}: {e}")
#             raise

#     def handle_job_failure(self, job, exc_info):
#         logger.error(f"Job {job.id} failed: {exc_info}")
#         super().handle_job_failure(job, exc_info)

#     def main_work_horse(self, *args, **kwargs):
#         try:
#             super().main_work_horse(*args, **kwargs)
#         except Exception as e:
#             logger.error(f"Work-horse error: {e}")
#             raise


# def create_worker(redis_url: str, listen: list[str] = ["default"]):
#     try:
#         conn = Redis.from_url(redis_url, socket_timeout=30, retry_on_timeout=True)

#         with Connection(conn):
#             worker = CustomWorker(
#                 queues=map(Queue, listen), connection=conn, job_monitoring_interval=30
#             )
#             worker.work(with_scheduler=True, burst=False)
#     except Exception as e:
#         logger.error(f"Error in worker: {e}")
#         sys.exit(1)


def create_worker(redis_url: str, listen: list[str] = ["default"]):
    conn = Redis.from_url(redis_url)
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work(with_scheduler=True)


class Args:
    redis_port: int = 6379
    listen: list[str] = ["default"]


def main(args: Args):
    listen = args.listen
    redis_url = f"redis://localhost:{args.redis_port}"
    logger.info(f"Connecting to Redis at {redis_url} and listening to {listen}")

    # ncpus = multiprocessing.cpu_count()
    # num_workers = max(2, ncpus // 2)
    num_workers = 1
    processes: list[multiprocessing.Process] = []

    try:
        for i in range(num_workers):
            process = multiprocessing.Process(
                target=create_worker, args=(redis_url, listen), name=f"Worker-{i}"
            )
            process.start()
            processes.append(process)
            logger.info(f"Started worker {i+1}")

        # Monitor and restart workers if they die
        while True:
            for i, process in enumerate(processes):
                if not process.is_alive():
                    logger.warning(f"Worker {i} died, restarting...")
                    processes[i] = multiprocessing.Process(
                        target=create_worker,
                        args=(redis_url, listen),
                        name=f"Worker-{i}",
                    )
                    processes[i].start()

            time.sleep(5)  # Check every 5 seconds

    except KeyboardInterrupt:
        logger.warning("Shutting down workers...")
        for process in processes:
            process.terminate()
            process.join(timeout=1)


if __name__ == "__main__":
    main(Args())
