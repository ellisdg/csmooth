import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file",
                        help="input csv file with results from multiple iterations of the grid "
                             "resolution effect simulations.",
                        required=True)
    parser.add_argument("--out-file",
                        help="output pdf file with the plot.",
                        required=True)
    parser.add_argument("--column", nargs="+", required=True,)
    parser.add_argument("--column-label", nargs="+",)
    return parser.parse_args()


def plot_results(in_file, out_file, column, column_names=None):
    df = pd.read_csv(in_file)
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    n_cols = len(column)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4),
                             constrained_layout=True)
    try:
        axes[0]
    except TypeError:
        axes = [axes]

    for col, ax, label in zip(column, axes, column_names):
        col_dfs = list()
        for method in ["raw", "gaussian", "constrained"]:
            col_name = f"{method}_{col}"
            tmp_df = df[["label", "fwhm_mm", col_name]].copy()
            tmp_df["method"] = method
            tmp_df[col] = tmp_df[col_name]
            if method == "raw":
                tmp_df["fwhm_mm"] = 0.0
                # remove duplicate rows
                tmp_df = tmp_df.drop_duplicates(subset=["label", "fwhm_mm", "method"])
            tmp_df.set_index(["label", "fwhm_mm", "method"])
            col_dfs.append(tmp_df)


        col_df = pd.concat(col_dfs, axis=0)
        sns.boxplot(data=col_df, x="fwhm_mm", y=col, hue="method", ax=ax,
                    legend=ax == axes[-1])
        ax.set_xlabel("FWHM (mm)")
        ax.set_ylabel(label)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left",
                      bbox_to_anchor=(1.05, 1.02),
                      )
    fig.savefig(out_file, dpi=300, bbox_inches="tight")


def main():
    args = parse_args()
    column = "-".join(args.column)
    out_file = args.out_file.format(column)
    if args.column_label is not None:
        assert len(args.column_label) == len(args.column), "--column-label must have the same number of inputs as --column"
        column_names = args.column_label
    else:
        column_names = args.column
    plot_results(args.in_file, out_file, column=args.column, column_names=column_names)


if __name__ == "__main__":
    main()
