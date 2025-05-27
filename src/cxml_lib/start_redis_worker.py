import multiprocessing
import os
import subprocess
import sys
import time

from redis import Redis
from rq import Queue, Worker

from cxml_lib.logger import logger

# Check if the environment variable is set
if os.getenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY") != "YES":
    # Set the environment variable and restart the script
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run([sys.executable] + sys.argv)
    sys.exit()


def create_worker(
    redis_url: str, listen: list[str] = ["default"], worker_name: str = "worker"
):
    conn = Redis.from_url(redis_url)
    queues = [Queue(name, connection=conn) for name in listen]
    worker = Worker(queues, connection=conn, name=worker_name)
    logger.info(f"Starting Redis worker: {worker_name}")
    worker.work(with_scheduler=True)


class Args:
    port: int = 6379
    listen: list[str] = ["default"]


def main(args: Args):
    listen = args.listen
    redis_url = f"redis://localhost:{args.port}"
    logger.info(f"Connecting to Redis at {redis_url} and listening to {listen}")

    # Run the worker in the main process
    logger.info("Starting Redis worker in main process")
    try:
        logger.info("Redis worker running")
        create_worker(redis_url, listen)
        # run_worker_in_subprocess(redis_url, listen)
    except KeyboardInterrupt:
        logger.warning("Redis worker interrupted. Shutting down...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        logger.info("Redis worker stopped")
        sys.exit(0)


def run_worker_in_subprocess(redis_url: str, listen: list[str]):
    ncpus = multiprocessing.cpu_count()
    num_workers = max(2, ncpus // 2)
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

        logger.info("Redis worker running")
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
                break

            time.sleep(5)  # Check every 5 seconds

    except KeyboardInterrupt:
        logger.warning("Shutting down workers...")
        for process in processes:
            process.terminate()
            process.join(timeout=1)

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        logger.info("All workers shut down.")
        logger.info("Redis worker stopped")
        sys.exit(0)


if __name__ == "__main__":
    main(Args())
