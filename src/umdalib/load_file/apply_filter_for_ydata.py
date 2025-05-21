from dataclasses import dataclass
from pathlib import Path as pt
from umdalib.load_file.read_data import read_as_ddf


@dataclass
class Args:
    filename: str
    filetype: str
    key: str
    use_dask: bool
    min_yvalue: float
    max_yvalue: float
    property_column: str


def main(args: Args):
    df = read_as_ddf(
        args.filetype,
        args.filename,
        args.key,
        use_dask=args.use_dask,
        computed=True,
    )

    filename = pt(args.filename)
    save_loc = filename.parent

    # make sure the property_column is in the df and dtype is float
    if args.property_column not in df.columns:
        raise ValueError(f"{args.property_column} not in the DataFrame")

    df[args.property_column] = df[args.property_column].astype(float)

    min_yvalue = None
    max_yvalue = None

    if args.min_yvalue:
        min_yvalue = float(args.min_yvalue)
    if args.max_yvalue:
        max_yvalue = float(args.max_yvalue)

    # filter the y values based on the min and max values of property_column and make a new df

    if min_yvalue is None:
        min_yvalue = df[args.property_column].min()
    if max_yvalue is None:
        max_yvalue = df[args.property_column].max()

    new_df = df[
        (df[args.property_column] >= min_yvalue)
        & (df[args.property_column] <= max_yvalue)
    ]

    savefile = save_loc / f"{filename.stem}_filtered_ydata.csv"
    new_df.to_csv(savefile, index=False)

    return {
        "savefile": savefile,
    }
