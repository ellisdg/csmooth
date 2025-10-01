# Load required library
library(lme4)
library(broom.mixed)
library(ggplot2)
library(tidyr)
library(readr)

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

# Fit a linear mixed-effects model to assess the effect of smoothing (FWHM) on connectivity metrics
# a different beta should be fit for Gaussian and constrained smoothing
# metric ~ FWHM * Method + (1 | Subject)
metrics_data$Method = factor(metrics_data$Method, levels = c("gaussian", "constrained"))
metrics_data$Subject = factor(metrics_data$Subject)



# Initialize results dataframe
results_table <- data.frame(
  Metric = character(),
  BetaGaussian = numeric(),
  BetaGaussianCILow = numeric(),
  BetaGaussianCIHigh = numeric(),
  PvalueGaussian = numeric(),
  BetaConstrained = numeric(),
  BetaConstrainedCILow = numeric(),
  BetaConstrainedCIHigh = numeric(),
  PvalueConstrained = numeric(),
  stringsAsFactors = FALSE
)

# loop over all the metrics
metrics = c("LocalEfficiency", "GlobalEfficiency", "ClusteringCoefficient", "Modularity")
for (metric in metrics) {
  cat("Fitting model for metric:", metric, "\n")

  formula = as.formula(paste(metric, "~ FWHM * Method + (1 | Subject)"))
  model <- lmer(formula, data = metrics_data)

  # Print the summary of the model
  print(summary(model))

  # Save the summary to a text file
  model_summary_file <- file.path(output_dir, paste0(metric, "_connectivity_metrics_model_summary.txt"))
  capture.output(summary(model), file = model_summary_file)

  # Extract coefficients and confidence intervals
  coef_summary <- summary(model)$coefficients
  confint_result <- confint(model, method = "Wald")

  # Extract FWHM effects for each method
  # FWHM coefficient is the effect for gaussian (reference level)
  beta_gaussian <- coef_summary["FWHM", "Estimate"]
  pvalue_gaussian <- coef_summary["FWHM", "Pr(>|t|)"]
  ci_gaussian <- confint_result["FWHM", ]

  # FWHM:Methodconstrained is the additional effect for constrained
  # Total effect for constrained = FWHM + FWHM:Methodconstrained
  beta_interaction <- coef_summary["FWHM:Methodconstrained", "Estimate"]
  beta_constrained <- beta_gaussian + beta_interaction
  pvalue_constrained <- coef_summary["FWHM:Methodconstrained", "Pr(>|t|)"]
  ci_interaction <- confint_result["FWHM:Methodconstrained", ]
  ci_constrained <- ci_gaussian + ci_interaction

  # Add to results table
  results_table <- rbind(results_table, data.frame(
    Metric = metric,
    BetaGaussian = beta_gaussian,
    BetaGaussianCILow = ci_gaussian[1],
    BetaGaussianCIHigh = ci_gaussian[2],
    PvalueGaussian = pvalue_gaussian,
    BetaConstrained = beta_constrained,
    BetaConstrainedCILow = ci_constrained[1],
    BetaConstrainedCIHigh = ci_constrained[2],
    PvalueConstrained = pvalue_constrained
  ))
}

# Save results table
results_file <- file.path(output_dir, "connectivity_metrics_results_table.csv")
write.csv(results_table, results_file, row.names = FALSE)
print(results_table)

# Load required library
library(ggplot2)

# Reshape the data for visualization
results_long <- results_table %>%
  tidyr::pivot_longer(
    cols = c(starts_with("BetaGaussian"), starts_with("BetaConstrained")),
    names_to = c("Method", ".value"),
    names_pattern = "Beta(Gaussian|Constrained)(.*)"
  ) %>%
  rename(Estimate = "", CILow = "CILow", CIHigh = "CIHigh")

# Create the plot
plot <- ggplot(results_long, aes(x = Metric, y = Estimate, color = Method)) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_errorbar(aes(ymin = CILow, ymax = CIHigh), width = 0.2, position = position_dodge(width = 0.5)) +
  theme_minimal() +
  labs(
    title = "Beta Estimates with Confidence Intervals",
    x = "Metric",
    y = "Beta Estimate",
    color = "Method"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Save the plot
ggsave(file.path(output_dir, "results_visualization.pdf"), plot, width = 8, height = 5)
