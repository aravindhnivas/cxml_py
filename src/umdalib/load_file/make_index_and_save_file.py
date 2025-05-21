from dataclasses import dataclass
from umdalib.logger import logger
from umdalib.load_file.read_data import read_as_ddf


@dataclass
class Args:
    filetype: str
    filename: str
    key: str = None
    use_dask: bool = False
    index_column_name: str = None


def main(args: Args):
    logger.info(f"Making index and saving file: {args.filename}")
    training_df = read_as_ddf(
        args.filetype,
        args.filename,
        args.key,
        use_dask=args.use_dask,
        computed=True,
    )
    # return {}
    if args.index_column_name in training_df.columns:
        training_df.set_index(args.index_column_name, inplace=True)
        training_df.to_csv(args.filename, index=True)
    else:
        training_df.to_csv(
            args.filename, index=True, index_label=args.index_column_name
        )
    logger.info(f"Saved {args.filename}")
    return {
        "index_column_name": args.index_column_name,
        "filename": args.filename,
    }
