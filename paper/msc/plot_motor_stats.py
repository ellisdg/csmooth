import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import seaborn as sns

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # or another sans-serif font like Arial or Helvetica

def main():
    motor_stats_filename = "/media/conda2/public/MSC/derivatives/motor_stats/motor_stats.csv"
    stats_df = pd.read_csv(motor_stats_filename)
    # Ensure "gaussian" is plotted first (on the left)
    stats_df["method"] = pd.Categorical(stats_df["method"], categories=["no_smoothing", "gaussian", "constrained"], ordered=True)
    stats_df = stats_df.sort_values(["fwhm", "method"])

    seaborn.set_palette("muted")
    sns.set_style("whitegrid")

    # Prepare data for plotting
    fwhm_vals = stats_df["fwhm"].unique()
    methods = ["no_smoothing", "gaussian", "constrained"]
    colors = {"no_smoothing": "gray", "gaussian": "#1f77b4", "constrained": "#2ca02c"}  # matplotlib C0 (blue), C2 (green)

    metrics = [
        ("mse", "Mean Squared Error (MSE)"),
        ("mae", "Mean Absolute Error (MAE)"),
        ("dice", "Dice Coefficient"),
    ]

    x = np.arange(len(fwhm_vals))
    width = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        # Compute means and standard errors for each group and metric
        means = []
        ses = []
        for fwhm in fwhm_vals:
            group = stats_df[stats_df["fwhm"] == fwhm]
            means.append([group[group["method"] == m][metric].mean() for m in methods])
            ses.append([group[group["method"] == m][metric].sem() for m in methods])
        means = np.array(means)
        ses = np.array(ses)

        for i, method in enumerate(methods):
            ax.bar(
                x + (i - 1) * width,
                means[:, i],
                width=width,
                label=method,
                color=colors[method],
                edgecolor=None,
                linewidth=0,
                yerr=ses[:, i],
                capsize=8,
                error_kw=dict(lw=1, capthick=1, ecolor="black")
            )

        # Remove all spines (borders)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks(x)
        ax.set_xticklabels([str(f) for f in fwhm_vals], fontsize=12)
        ax.set_xlabel("FWHM (mm)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(ylabel, fontsize=15)

    # Only add the legend to the first subplot
    legend = axes[0].legend(
        title="Smoothing Method",
        fontsize=12,
        title_fontsize=14,
        alignment="left",
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        frameon=False
    )

    plt.tight_layout()
    plt.savefig("motor_stats_subplots.pdf")
    plt.close()


if __name__ == "__main__":
    main()