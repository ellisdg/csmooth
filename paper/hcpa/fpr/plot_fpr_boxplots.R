#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
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
    method = factor(method, levels = c("csmooth", "gaussian"))
  )

# Boxplot helper
save_boxplot <- function(df, value_col, filename, ylab) {
  p <- ggplot(df, aes(x = method, y = .data[[value_col]], fill = method)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.8) +
    facet_wrap(~fwhm, scales = "free_x") +
    ylab(ylab) +
    xlab("Method") +
    theme_bw() +
    theme(legend.position = "none")
  ggsave(filename, plot = p, width = 8, height = 4 + 0.5 * length(levels(df$fwhm)), dpi = 300)
}

save_boxplot(fpr, "cluster_fp_pct", file.path(output_dir, "fpr_cluster_boxplots.png"), "Clusterwise FPR (%)")
save_boxplot(fpr, "voxel_fp_pct",   file.path(output_dir, "fpr_voxel_boxplots.png"),   "Voxelwise FPR (%)")

# Paired t-tests (csmooth vs gaussian within each fwhm)
paired_tests <- list()
metrics <- c("cluster_fp_pct", "voxel_fp_pct")
for (metric in metrics) {
  wide <- fpr %>%
    select(fwhm, subject, dir, run, method, !!sym(metric)) %>%
    pivot_wider(names_from = method, values_from = !!sym(metric)) %>%
    drop_na(csmooth, gaussian)
  for (f in levels(fpr$fwhm)) {
    sub <- wide %>% filter(fwhm == f)
    if (nrow(sub) < 2) next
    tt <- t.test(sub$csmooth, sub$gaussian, paired = TRUE)
    paired_tests[[length(paired_tests) + 1]] <- tibble(
      metric = metric,
      fwhm = f,
      n = nrow(sub),
      estimate_mean_diff = unname(tt$estimate),
      conf_low = tt$conf.int[1],
      conf_high = tt$conf.int[2],
      p_value = tt$p.value
    )
  }
}

paired_df <- bind_rows(paired_tests)
if (nrow(paired_df) > 0) {
  write_csv(paired_df, file.path(output_dir, "fpr_paired_ttests.csv"))
  message("Wrote paired t-tests to ", file.path(output_dir, "fpr_paired_ttests.csv"))
} else {
  message("No paired t-tests computed (insufficient matched rows)")
}

message("Saved plots to ", output_dir)
