# Load required library
library(lme4)
library(broom.mixed)
library(ggplot2)
library(tidyr)
library(readr)
library(dplyr)
library(tibble)
library(RColorBrewer)

# # Read command-line arguments
# args <- commandArgs(trailingOnly = TRUE)
# 
# # Ensure two arguments are provided
# if (length(args) != 2) {
#   stop("Please provide exactly two file paths as arguments.")
# }
# 
# # Assign arguments to variables
# metrics_file <- args[1]
# output_dir <- args[2]
# 
# # Print the file paths
# cat("Graphy Connectivity Metrics Filename:", metrics_file, "\n")
# cat("Output Directory:", output_dir, "\n")

metrics_file <- "~/Box Sync/Aizenberg_Documents/Papers/csmooth/results/conn_smoothing_graph_metrics.csv"
output_dir <- "~/Box Sync/Aizenberg_Documents/Papers/csmooth/figures/hcpa_conn/"

# read in the csv file
metrics_data = read_csv(metrics_file)

# --- Boxplots for each metric across smoothing conditions ---

# In the Method column there is gaussian and constrained
# However, when FWHM=0, there is no smoothing, so we will treat that as a separate condition
# So, set FWHM=0 rows to Method="No Smoothing"
# and ensure the Method factor levels are ordered as: No Smoothing, Gaussian, Constrained
# Also ensure FWHM is treated as a factor for plotting purposes
# Also, also remove duplicate rows, as some No Smoothing rows are duplicated
metrics_data <- metrics_data %>%
  mutate(
    Method = if_else(FWHM == 0, "no smoothing", Method)
  )

# If there are duplicate rows (same Subject, Method, FWHM), keep only one
metrics_data <- metrics_data %>%
  distinct(Subject, Method, FWHM, .keep_all = TRUE)


# Ensure FWHM=0 (no smoothing) is included
metrics_data$FWHM <- as.factor(metrics_data$FWHM)
metrics_data$Method <- factor(metrics_data$Method, levels = c("no smoothing", "gaussian", "constrained"))

metrics_levels <- c("0", "3", "6", "9", "12")
metrics_data$FWHM <- factor(metrics_data$FWHM, levels = metrics_levels)

# Change column names to have a spaece for better plotting labels
colnames(metrics_data) <- gsub("LocalEfficiency", "Local Efficiency", colnames(metrics_data))
colnames(metrics_data) <- gsub("GlobalEfficiency", "Global Efficiency", colnames(metrics_data))
colnames(metrics_data) <- gsub("ClusteringCoefficient", "Clustering Coefficient", colnames(metrics_data))
metrics = c("Local Efficiency", "Global Efficiency", "Clustering Coefficient", "Modularity")

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
  geom_boxplot(position = position_dodge(width = 0.8)) +
  facet_wrap(~ Metric, scales = "free_y", nrow=1) +
  theme_minimal() +
  labs(
    x = "FWHM (mm)",
    y = "Metric Value",
    fill = "Method:"
  ) +
  # force "no smoothing" = gray, + two colors from Set1
  scale_fill_manual(values = c(
    "no smoothing" = "gray80",
    "gaussian"     = brewer.pal(3, "Set1")[2],
    "constrained"  = brewer.pal(3, "Set1")[3]
  )) +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    legend.position = "bottom" # or "bottom"
  )
print(boxplot_figure)

ggsave(file.path(output_dir, "metrics_boxplots.pdf"), boxplot_figure, width = 8, height = 4)

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
        vals_nosmooth <- metrics_data %>% filter(FWHM == "0", Method == "no smoothing") %>% pull(metric)
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
