import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file",
                        help="input csv file with results from multiple iterations of the grid "
                             "resolution effect simulations.")
    parser.add_argument("--out-file",
                        help="output pdf file with the plot.")
    return parser.parse_args()


def plot_results(in_file, out_file):
    df = pd.read_csv(in_file)
    ax, fig = plt.subplots()
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    sns.barplot(df, x="voxel_size_mm", y="")


def main():
    args = parse_args()
    plot_results(args.in_file, args.out_file)


if __name__ == "__main__":
    main()
