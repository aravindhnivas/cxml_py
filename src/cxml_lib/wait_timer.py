from dataclasses import dataclass
from cxml_lib.logger import logger
from time import sleep


@dataclass
class Args:
    wait_time: int


def main(args: Args):
    wait_time = int(args.wait_time)

    logger.info(f"Starting to sleep for {wait_time}s!")
    # raise Exception("Intentional error!")
    sleep(wait_time)
    logger.info("Finished sleeping!")

    return {
        "status": "completed",
        "message": f"Slept for {wait_time}s!",
    }
