#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(RColorBrewer)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript plot_fpr_boxplots.R <fpr_csv> [output_dir]")
}

fpr_csv <- args[[1]]
output_dir <- if (length(args) >= 2) args[[2]] else dirname(fpr_csv)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Read data
fpr <- read_csv(fpr_csv, show_col_types = FALSE)

# Ensure factors are ordered for nicer plots
if (!"fwhm" %in% names(fpr)) stop("CSV must contain column 'fwhm'")
fpr <- fpr %>%
  mutate(
    fwhm_num = suppressWarnings(as.numeric(fwhm)),
    fwhm = ifelse(is.na(fwhm_num), fwhm, as.character(fwhm_num)),
    fwhm = factor(fwhm, levels = unique(fwhm[order(fwhm_num, fwhm)])),
    method = tolower(as.character(method)),
    method = recode(method,
                    "csmooth" = "constrained",
                    "constrained" = "constrained",
                    "gaussian" = "gaussian",
                    .default = method),
    method = factor(method,
                    levels = c("gaussian", "constrained"),
                    labels = c("Gaussian", "Constrained"))
  )

# Shared colors/position for boxplots
gg_colors <- c("Gaussian" = brewer.pal(3, "Set1")[2],
               "Constrained" = brewer.pal(3, "Set1")[3])
gg_pd <- position_dodge(width = 0.7)
# Shared y-limits so combined plots align
fpr_y_limits <- range(c(fpr$voxel_fp_pct, fpr$cluster_fp_pct), na.rm = TRUE)

# Summary stats for barplots (per metric/method/FWHM)
fpr_summary <- fpr %>%
  select(fwhm, method, voxel_fp_pct, cluster_fp_pct) %>%
  pivot_longer(cols = c(voxel_fp_pct, cluster_fp_pct), names_to = "metric", values_to = "value") %>%
  group_by(fwhm, method, metric) %>%
  summarize(
    n = sum(!is.na(value)),
    mean = mean(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    se = sd / sqrt(pmax(n, 1)),
    ci = qt(0.975, pmax(n - 1, 1)) * se,
    .groups = "drop"
  )
bar_y_limits <- range(c(fpr_summary$mean - fpr_summary$ci, fpr_summary$mean + fpr_summary$ci), na.rm = TRUE)
bar_pad <- ifelse(is.finite(diff(bar_y_limits)) && diff(bar_y_limits) > 0, 0.05 * diff(bar_y_limits), 0.05)
bar_y_limits[2] <- bar_y_limits[2] + bar_pad

compute_paired_tests <- function(df, metrics) {
  paired <- list()
  for (metric in metrics) {
    wide <- df %>%
      select(fwhm, subject, dir, run, method, !!sym(metric)) %>%
      pivot_wider(names_from = method, values_from = !!sym(metric)) %>%
      drop_na(Gaussian, Constrained)
    for (f in levels(df$fwhm)) {
      sub <- wide %>% filter(fwhm == f)
      if (nrow(sub) < 2) next
      tt <- t.test(sub$Constrained, sub$Gaussian, paired = TRUE)
      paired[[length(paired) + 1]] <- tibble(
        metric = metric,
        fwhm = f,
        n = nrow(sub),
        mean_gaussian = mean(sub$Gaussian, na.rm = TRUE),
        mean_constrained = mean(sub$Constrained, na.rm = TRUE),
        estimate_mean_diff = unname(tt$estimate),
        conf_low = tt$conf.int[1],
        conf_high = tt$conf.int[2],
        p_value = tt$p.value
      )
    }
  }
  bind_rows(paired)
}

paired_df <- compute_paired_tests(fpr, c("cluster_fp_pct", "voxel_fp_pct"))

sig_positions <- fpr_summary %>%
  mutate(upper = mean + ci) %>%
  group_by(metric, fwhm) %>%
  summarize(y_base = max(upper, na.rm = TRUE), .groups = "drop")

sig_labels <- paired_df %>%
  mutate(
    metric = as.character(metric),
    label = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01  ~ "**",
      p_value < 0.05  ~ "*",
      TRUE ~ "n.s."
    )
  ) %>%
  left_join(sig_positions, by = c("metric", "fwhm")) %>%
  mutate(
    bump = ifelse(is.finite(bar_pad), bar_pad * 0.4, 0.02),
    y = y_base + bump
  )

# Boxplot helper (returns a plot for reuse)
make_boxplot <- function(df, value_col, ylab, title, drop_y_title = FALSE) {
  p <- ggplot(df, aes(x = fwhm, y = .data[[value_col]], fill = method)) +
    geom_boxplot(position = gg_pd, width = 0.65, outlier.size = 0.8) +
    scale_fill_manual(values = gg_colors) +
    labs(x = "FWHM (mm)", y = if (drop_y_title) NULL else ylab, fill = "Smoothing Method", title = title) +
    coord_cartesian(ylim = fpr_y_limits) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "right",
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      panel.grid.minor = element_blank()
    )
  p
}

# Barplot helper (returns a plot for reuse)
make_barplot <- function(summary_df, sig_df, metric_name, ylab, title, drop_y_title = FALSE) {
  plot_df <- summary_df %>% filter(metric == metric_name)
  sig_plot <- sig_df %>% filter(metric == metric_name)
  p <- ggplot(plot_df, aes(x = fwhm, y = mean, fill = method)) +
    geom_col(position = gg_pd, width = 0.65, color = "gray30", alpha = 0.9) +
    geom_errorbar(aes(ymin = mean - ci, ymax = mean + ci), position = gg_pd, width = 0.2, linewidth = 0.6) +
    scale_fill_manual(values = gg_colors) +
    labs(x = "FWHM (mm)", y = if (drop_y_title) NULL else ylab, fill = "Smoothing Method", title = title) +
    coord_cartesian(ylim = bar_y_limits) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "right",
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      panel.grid.minor = element_blank()
    )
  if (nrow(sig_plot) > 0) {
    p <- p + geom_text(data = sig_plot, aes(x = fwhm, y = y, label = label),
                       vjust = -0.3, size = 3.2, color = "black", inherit.aes = FALSE)
  }
  p
}

# Save helper
save_boxplot <- function(plot_obj, filename) {
  ggsave(filename, plot = plot_obj, width = max(7, 5 + 0.6 * nlevels(fpr$fwhm)), height = 4, dpi = 300)
}

p_cluster <- make_boxplot(fpr, "cluster_fp_pct", "FPR (%)", "With cluster significance threshold", drop_y_title = TRUE)
p_voxel   <- make_boxplot(fpr, "voxel_fp_pct",   "FPR (%)",   "Without cluster significance threshold", drop_y_title = FALSE)

p_bar_cluster <- make_barplot(fpr_summary, sig_labels, "cluster_fp_pct", "FPR (%)", "With cluster significance threshold", drop_y_title = TRUE)
p_bar_voxel   <- make_barplot(fpr_summary, sig_labels, "voxel_fp_pct",   "FPR (%)", "Without cluster significance threshold", drop_y_title = FALSE)

save_boxplot(p_cluster, file.path(output_dir, "fpr_cluster_boxplots.pdf"))
save_boxplot(p_voxel,   file.path(output_dir, "fpr_voxel_boxplots.pdf"))

save_boxplot(p_bar_cluster, file.path(output_dir, "fpr_cluster_barplots.pdf"))
save_boxplot(p_bar_voxel,   file.path(output_dir, "fpr_voxel_barplots.pdf"))

# Combined plot: voxelwise left, clusterwise right, shared y-axis/legend
combined <- p_voxel + p_cluster + plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "right")

# Combined barplot: voxelwise left, clusterwise right
combined_bar <- p_bar_voxel + p_bar_cluster + plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "right")

ggsave(file.path(output_dir, "fpr_combined_boxplots.pdf"), plot = combined,
       width = max(10, 7 + 1.0 * nlevels(fpr$fwhm)), height = 4, dpi = 300)

ggsave(file.path(output_dir, "fpr_combined_barplots.pdf"), plot = combined_bar,
       width = max(10, 7 + 1.0 * nlevels(fpr$fwhm)), height = 4, dpi = 300)

# Paired t-tests (Constrained vs Gaussian within each FWHM)
if (nrow(paired_df) > 0) {
  write_csv(paired_df, file.path(output_dir, "fpr_paired_ttests.csv"))
  message("Wrote paired t-tests to ", file.path(output_dir, "fpr_paired_ttests.csv"))
} else {
  message("No paired t-tests computed (insufficient matched rows)")
}

message("Saved plots to ", output_dir)
