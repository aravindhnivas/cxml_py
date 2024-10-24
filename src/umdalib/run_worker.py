import multiprocessing
import os
import subprocess
import sys

from redis import Redis
from rq import Connection, Queue, Worker

from umdalib.logger import logger

# Check if the environment variable is set
if os.getenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY") != "YES":
    # Set the environment variable and restart the script
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run([sys.executable] + sys.argv)
    sys.exit()


def create_worker(redis_url: str, listen: list[str] = ["default"]):
    conn = Redis.from_url(redis_url)
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work(with_scheduler=True)


def main():
    listen = ["default"]
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    logger.info("Starting worker")
    # Get the number of CPU cores
    ncpus = multiprocessing.cpu_count()
    num_workers = max(2, ncpus // 2)

    # Create a pool of workers
    processes: list[multiprocessing.Process] = []

    try:
        # Start multiple worker processes
        for _ in range(num_workers):
            process = multiprocessing.Process(
                target=create_worker, args=(redis_url, listen)
            )
            process.start()
            processes.append(process)

        # Wait for all processes to complete
        for process in processes:
            process.join()

    except KeyboardInterrupt:
        # Handle graceful shutdown
        logger.warning("Shutting down workers...")
        for process in processes:
            process.terminate()
            process.join()


# conn = Redis.from_url(redis_url)
# if __name__ == "__main__":
#     with Connection(conn):
#         worker = Worker(map(Queue, listen))
#         worker.work()

if __name__ == "__main__":
    main()
