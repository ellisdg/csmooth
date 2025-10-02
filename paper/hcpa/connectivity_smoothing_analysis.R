# Load required library
library(lme4)
library(broom.mixed)
library(ggplot2)
library(tidyr)
library(readr)
library(dplyr)
library(tibble)

# Read command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Ensure two arguments are provided
if (length(args) != 2) {
  stop("Please provide exactly two file paths as arguments.")
}

# Assign arguments to variables
metrics_file <- args[1]
output_dir <- args[2]

# Print the file paths
cat("Graphy Connectivity Metrics Filename:", metrics_file, "\n")
cat("Output Directory:", output_dir, "\n")

# read in the csv file
metrics_data = read_csv(metrics_file)

# --- Boxplots for each metric across smoothing conditions ---

# Ensure FWHM=0 (no smoothing) is included
metrics_data$FWHM <- as.factor(metrics_data$FWHM)
metrics_data$Method <- factor(metrics_data$Method, levels = c("gaussian", "constrained"))

metrics_levels <- c("0", "3", "6", "9", "12")
metrics_data$FWHM <- factor(metrics_data$FWHM, levels = metrics_levels)

metrics = c("LocalEfficiency", "GlobalEfficiency", "ClusteringCoefficient", "Modularity")

# Gather data for plotting
plot_data <- metrics_data %>%
  select(Subject, Method, FWHM, all_of(metrics)) %>%
  tidyr::pivot_longer(
    cols = all_of(metrics),
    names_to = "Metric",
    values_to = "Value"
  )

# Create boxplots for each metric in one figure
boxplot_figure <- ggplot(plot_data, aes(x = FWHM, y = Value, fill = Method)) +
  geom_boxplot(position = position_dodge(width = 0.8), outlier.shape = NA) +
  facet_wrap(~ Metric, scales = "free_y") +
  theme_minimal() +
  labs(
    title = "Metric Values by Smoothing Condition",
    x = "Smoothing FWHM",
    y = "Metric Value",
    fill = "Method"
  ) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

ggsave(file.path(output_dir, "metrics_boxplots.pdf"), boxplot_figure, width = 12, height = 8)

# --- T-tests comparing smoothing conditions ---

ttest_results <- tibble(
  Metric = character(),
  FWHM = character(),
  Comparison = character(),
  t_statistic = numeric(),
  p_value = numeric()
)

for (metric in metrics) {
  for (fwhm in metrics_levels) {
    # Compare methods at each FWHM
    vals_gaussian <- metrics_data %>% filter(FWHM == fwhm, Method == "gaussian") %>% pull(metric)
    vals_constrained <- metrics_data %>% filter(FWHM == fwhm, Method == "constrained") %>% pull(metric)
    if (length(vals_gaussian) > 1 && length(vals_constrained) > 1) {
      ttest <- t.test(vals_gaussian, vals_constrained)
      ttest_results <- add_row(ttest_results,
        Metric = metric,
        FWHM = fwhm,
        Comparison = "gaussian_vs_constrained",
        t_statistic = ttest$statistic,
        p_value = ttest$p.value
      )
    }
    # Compare each method at FWHM vs no smoothing (FWHM=0)
    if (fwhm != "0") {
      for (method in c("gaussian", "constrained")) {
        vals_fwhm <- metrics_data %>% filter(FWHM == fwhm, Method == method) %>% pull(metric)
        vals_nosmooth <- metrics_data %>% filter(FWHM == "0", Method == method) %>% pull(metric)
        if (length(vals_fwhm) > 1 && length(vals_nosmooth) > 1) {
          ttest <- t.test(vals_fwhm, vals_nosmooth)
          ttest_results <- add_row(ttest_results,
            Metric = metric,
            FWHM = fwhm,
            Comparison = paste0(method, "_vs_nosmoothing"),
            t_statistic = ttest$statistic,
            p_value = ttest$p.value
          )
        }
      }
    }
  }
}

# Save t-test results table
ttest_file <- file.path(output_dir, "smoothing_ttest_results.csv")
write.csv(ttest_results, ttest_file, row.names = FALSE)
print(ttest_results)
