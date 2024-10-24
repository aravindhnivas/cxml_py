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


class Args:
    redis_port: int = 6379
    listen: list[str] = ["default"]


def main(args: Args):
    listen = args.listen
    redis_url = f"redis://localhost:{args.redis_port}"
    logger.info(f"Connecting to Redis at {redis_url} and listening to {listen}")

    # Get the number of CPU cores
    # ncpus = multiprocessing.cpu_count()
    # num_workers = max(2, ncpus // 2)
    num_workers = 1

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


if __name__ == "__main__":
    main(Args())
