import argparse
from pathlib import Path
import plot_raw_data as prd
import numpy as np
import pandas as pd


def reformat_csv(file_path, temp_path, out_path):
    df = pd.read_csv(file_path, index_col=0)  # rows: snapshot_i, columns: temp_i
    temps = prd.read_temperatures(temp_path)

    if len(temps) != df.shape[1]:
        raise ValueError(
            f"{len(temps)} temperatures in {temp_path}, but {df.shape[1]} columns in {file_path}"
        )

    temp_row = pd.DataFrame([temps], index=["temp"], columns=df.columns)
    out = pd.concat([temp_row, df])
    out.to_csv(out_path, header=False)  # keep row names, drop column names


def parse_args():
    parser = argparse.ArgumentParser(description="Replace column header with a 'temps' row.")
    parser.add_argument("--file-path", type=Path, help="Input CSV (snapshot rows, temp columns)")
    parser.add_argument("--temp-path", type=Path, help="File with the temperatures")
    parser.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help="Output CSV (default: <input>_reformatted.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_path = args.out_path or args.file_path.with_name(args.file_path.stem + "_reformatted.csv")
    reformat_csv(args.file_path, args.temp_path, out_path)