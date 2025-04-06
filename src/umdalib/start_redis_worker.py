import multiprocessing
import os
import subprocess
import sys
import time

from redis import Redis
from rq import Queue, Worker

from umdalib.logger import logger

# Check if the environment variable is set
if os.getenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY") != "YES":
    # Set the environment variable and restart the script
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run([sys.executable] + sys.argv)
    sys.exit()


def create_worker(redis_url: str, listen: list[str] = ["default"]):
    conn = Redis.from_url(redis_url)
    queues = [Queue(name, connection=conn) for name in listen]
    worker = Worker(queues, connection=conn)
    worker.work(with_scheduler=True)
    # with connections.Connection(conn):
    #     worker = Worker(map(Queue, listen), connection=conn)
    #     worker.work(with_scheduler=True)


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
