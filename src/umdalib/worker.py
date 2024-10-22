import os
import sys
import subprocess
from redis import Redis
from rq import Worker, Queue, Connection
import gevent.monkey

gevent.monkey.patch_all()
listen = ["default"]

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

# Disable fork safety
# os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# Check if the environment variable is set
if os.getenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY") != "YES":
    # Set the environment variable and restart the script
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    subprocess.run([sys.executable] + sys.argv)
    sys.exit()

conn = Redis.from_url(redis_url)

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()
