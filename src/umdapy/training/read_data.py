import pandas as pd
import dask.dataframe as dd
import time


def main(args):
    print("Reading data...")
    time.sleep(10)
    print(args)
    return {"data": args}
