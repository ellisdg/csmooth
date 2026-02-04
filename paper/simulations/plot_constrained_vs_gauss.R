#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(broom)
  library(RColorBrewer)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript plot_constrained_vs_gauss.R <input_csv> <out_png> <out_pdf>", call. = FALSE)
}

in_csv <- args[1]
out_png <- args[2]
out_pdf <- args[3]

metrics_map <- c(
  dice = "Dice Coefficient",
  pearson_r = "Correlation",
  active_gm = "GM Active Voxels",
  active_wm = "WM Active Voxels"
)
method_levels <- c("no smoothing", "gaussian", "constrained")

# load and reshape
raw_df <- read.csv(in_csv, header = TRUE, stringsAsFactors = FALSE)
cat("Input CSV columns:\n")
print(colnames(raw_df))

# Robustly detect columns like raw_dice, gaussian_pearson.r, constrained_active_gm, etc.
methods <- c("raw", "gaussian", "constrained")
metric_keys <- names(metrics_map)
long_pieces <- list()
for (m in methods) {
  pref <- paste0(m, "_")
  cols_m <- grep(paste0("^", pref), colnames(raw_df), value = TRUE)
  for (col in cols_m) {
    suffix <- sub(paste0("^", pref), "", col)
    norm <- tolower(gsub("[^a-z0-9]+", "_", suffix))
    metric_key <- NA_character_
    if (grepl("dice", norm)) {
      metric_key <- "dice"
    } else if (grepl("pearson|corr|correlation|\br\b|rho", norm)) {
      metric_key <- "pearson_r"
    } else if (grepl("gm|grey|gray|gmm|gm_active|gm_active_voxels", norm)) {
      metric_key <- "active_gm"
    } else if (grepl("wm|white|wm_active|wm_active_voxels", norm)) {
      metric_key <- "active_wm"
    } else {
      # unknown metric suffix - skip
      next
    }

    tmp <- raw_df %>%
      select(label, fwhm_mm, all_of(col)) %>%
      rename(value = all_of(col)) %>%
      mutate(method = m, metric_key = metric_key)
    if (m == "raw") tmp <- tmp %>% mutate(fwhm_mm = 0)
    long_pieces[[length(long_pieces) + 1]] <- tmp
  }
}

if (length(long_pieces) == 0) stop("No method-prefixed metric columns found in input CSV (expected prefixes raw_, gaussian_, constrained_)")

long_df <- bind_rows(long_pieces) %>%
  filter(metric_key %in% metric_keys) %>%
  mutate(
    metric = recode(metric_key, !!!as.list(metrics_map)),
    metric = factor(metric, levels = unname(metrics_map)),
    metric_key = factor(metric_key, levels = metric_keys),
    method = if_else(method == "raw", "no smoothing", method),
    method = factor(method, levels = method_levels)
  ) %>%
  distinct(label, fwhm_mm, method, metric, .keep_all = TRUE)

# Diagnostic summary to help debug missing metrics
metric_counts <- long_df %>% group_by(metric_key) %>% summarize(n = n(), fwhm_vals = paste(sort(unique(fwhm_mm)), collapse = ","), .groups = "drop")
cat("Detected metric keys and sample counts:\n")
print(metric_counts)

if (nlevels(drop_na(long_df)$metric) < length(metrics_map)) {
  missing <- setdiff(unname(metrics_map), levels(drop_na(long_df)$metric))
  warning(paste("Some expected metrics are missing from input:", paste(missing, collapse = ", ")))
}

perform_pairwise_tests <- function(data_long) {
  data_long <- data_long %>%
    mutate(metric_key = recode(metric,
                               "Dice Coefficient" = "dice",
                               "Correlation" = "pearson_r",
                               "GM Active Voxels" = "active_gm",
                               "WM Active Voxels" = "active_wm"))

  comps <- list()
  fwhm_levels <- sort(unique(data_long$fwhm_mm))

  for (m in unique(data_long$metric_key)) {
    for (f in fwhm_levels) {
      subset_data <- data_long %>% filter(metric_key == m, fwhm_mm == f)

      g_vals <- subset_data %>% filter(method == "gaussian") %>% pull(value)
      c_vals <- subset_data %>% filter(method == "constrained") %>% pull(value)
      if (length(g_vals) > 1 && length(c_vals) > 1) {
        t_res <- t.test(g_vals, c_vals, paired = TRUE)
        comps[[length(comps) + 1]] <- tibble(
          metric = m, fwhm = f,
          group1 = "gaussian", group2 = "constrained",
          t = unname(t_res$statistic), p = t_res$p.value
        )
      }

      baseline <- data_long %>%
        filter(metric_key == m, method == "no smoothing", fwhm_mm == 0) %>%
        pull(value)
      for (method_name in c("gaussian", "constrained")) {
        m_vals <- subset_data %>% filter(method == method_name) %>% pull(value)
        if (length(baseline) > 1 && length(m_vals) > 1) {
          t_res <- t.test(m_vals, baseline, paired = TRUE)
          comps[[length(comps) + 1]] <- tibble(
            metric = m, fwhm = f,
            group1 = method_name, group2 = "no smoothing",
            t = unname(t_res$statistic), p = t_res$p.value
          )
        }
      }
    }
  }

  bind_rows(comps) %>%
    group_by(metric) %>%
    mutate(
      p.adj = p.adjust(p, method = "bonferroni"),
      p.signif = case_when(
        p.adj < 0.001 ~ "***",
        p.adj < 0.01  ~ "**",
        p.adj < 0.05  ~ "*",
        TRUE          ~ "ns"
      )
    ) %>%
    ungroup()
}

test_results <- perform_pairwise_tests(long_df)
cat("Pairwise t-tests (Bonferroni-adjusted):\n")
print(test_results)

# Prepare plotting data: show only FWHM 3,6,9,12 on the x-axis, but include 'no smoothing' as 0
plot_fwhm_values <- c(3, 6, 9, 12)
plot_fwhm_values_chr <- as.character(plot_fwhm_values)

plot_df <- long_df %>%
  mutate(
    # represent no smoothing as '0' category, keep others as their numeric strings
    plot_x = if_else(method == "no smoothing", "0", as.character(fwhm_mm))
  ) %>%
  # keep rows that are either no smoothing (now '0') or one of the requested fwhm values
  filter(plot_x == "0" | plot_x %in% plot_fwhm_values_chr)

# If no data for the requested fwhm values (apart from no smoothing), warn and fall back to full data
if (nrow(plot_df) == 0) {
  warning("No rows found for FWHM values 0,3,6,9,12; falling back to full data for plotting.")
  plot_df <- long_df %>% mutate(plot_x = if_else(method == "no smoothing", "0", as.character(fwhm_mm)))
}

# Ensure x-order places '0' first then the numeric FWHM levels
x_levels <- c("0", plot_fwhm_values_chr)

p <- ggplot(plot_df, aes(x = factor(plot_x, levels = x_levels), y = value, fill = method)) +
  geom_boxplot(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ metric, scales = "free_y", ncol = 2, drop = FALSE) +
  scale_x_discrete(drop = FALSE, limits = x_levels) +
  scale_fill_manual(values = c(
    "no smoothing" = "gray80",
    "gaussian"     = brewer.pal(3, "Set1")[2],
    "constrained"  = brewer.pal(3, "Set1")[3]
  )) +
  labs(x = "FWHM (mm)", y = "", fill = "Smoothing Method") +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    axis.title.y = element_blank()
  )

ggsave(out_png, p, width = 8, height = 6, dpi = 300)
ggsave(out_pdf, p, width = 8, height = 6)

stats_out <- sub("\\.[^.]+$", "_stats.csv", out_pdf)
write.csv(test_results, stats_out, row.names = FALSE)
