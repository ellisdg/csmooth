# load libraries
library(ggplot2)
library(mgcv)       # for GAM modeling
library(dplyr)      # for data manipulation
library(arrow)     # for reading/writing parquet files
library(tidyr)     # for pivot_wider
# Load the patchwork library for combining plots
library(patchwork)

# Read command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Ensure two arguments are provided
if (length(args) != 2) {
  stop("Please provide exactly two file paths as arguments.")
}

# Assign arguments to variables
data_file <- args[1]
output_dir <- args[2]

# Print the file paths
cat("Connectivity Distance Data Filename:", data_file, "\n")
cat("Output Directory:", output_dir, "\n")

# read in parquet file
data = read_parquet(data_file)

# Diagnostic checks
print("Data dimensions:")
print(dim(data))
print("Column names:")
print(names(data))
print("Data structure:")
print(str(data))

# Check the range and distribution of key variables
print("Distance_mm summary:")
print(summary(data$Distance_mm))
print("FWHM summary:")
print(summary(data$FWHM))
print("FisherZ_Connectivity summary:")
print(summary(data$FisherZ_Connectivity))

# Define Method based on FWHM: when FWHM == 0 call it "no_smoothing".
# Remove any duplicate rows so there are no repeated rows in the dataset.
# Ensure FWHM is numeric and Method/Subject are factors with the intended levels.
data <- data %>%
  mutate(
    FWHM = as.numeric(FWHM),
    Method = as.character(Method),
    Method = ifelse(FWHM == 0, "no_smoothing", Method),
    # if Method is NA for non-zero FWHM (e.g. original data didn't include Method),
    # assume gaussian smoothing for those rows
    Method = ifelse(is.na(Method) & FWHM != 0, "gaussian", Method)
  ) %>%
  distinct()

# Make Method a factor with three levels: no_smoothing, gaussian, constrained
data$Method <- factor(data$Method, levels = c("no_smoothing", "gaussian", "constrained"))
# Subject as a factor
data$Subject <- factor(data$Subject)

# Check factor levels
print("Method levels:")
print(table(data$Method))
print("FWHM levels:")
print(table(data$FWHM))
print("Number of subjects:")
print(length(unique(data$Subject)))

# Try a simpler model first - separate smooths by Method and FWHM
print("Trying simpler interaction model...")
model_interaction <- bam(FisherZ_Connectivity ~ 
                          s(Distance_mm, by = interaction(Method, FWHM)) +
                          s(Subject, bs="re"),
                        data = data,
                        discrete = TRUE)
summary(model_interaction)
model_interaction_summary_file <- file.path(output_dir, "connectivity_distance_by_method_and_fwhm_model_summary.txt")
capture.output(summary(model_interaction), file = model_interaction_summary_file)

# Create prediction data - much more efficient approach
print("Creating prediction grid...")
# compute the 99 percentile range of Distance_mm
# (use the same bounds as before)
dist_range <- quantile(data$Distance_mm, probs = c(0.005, 0.995))
print(paste("Distance_mm 2.5%:", dist_range[1], "97.5%:", dist_range[2]))
# sequence of distances to predict over
dist_seq <- seq(dist_range[1], dist_range[2], length.out = 50)

# Build the valid Method x FWHM combinations from the data (this avoids invalid pairs)
combos <- data %>% distinct(Method, FWHM)
# Cartesian product of distances with valid combos
newdat <- merge(data.frame(Distance_mm = dist_seq), combos, by = NULL, stringsAsFactors = FALSE)

# Add Subject column - required even when excluding the random effect
newdat$Subject <- factor("sub-HCA6018857", levels = levels(data$Subject))

# Ensure FWHM is numeric here (will convert to factor for plotting later)
newdat$FWHM <- as.numeric(newdat$FWHM)

print(paste("Prediction grid size:", nrow(newdat), "rows"))

# Predict from the model - exclude random effects for group average
print("Making predictions...")
newdat$fit <- predict(model_interaction, newdata = newdat, type = "response", 
                      exclude = "s(Subject)")
print("Predictions complete.")

# Make FWHM a factor for plotting
newdat$FWHM <- factor(newdat$FWHM, levels = sort(unique(data$FWHM)))

# Create the plot with proper grouping
pred_plot <- ggplot() +
  geom_line(data = newdat, aes(x = Distance_mm, y = fit, linetype = Method, color = FWHM),
            linewidth = 0.7, alpha = 0.8) +
  # change the x ticks
  scale_x_continuous(breaks = seq(0, 150, by = 20)) +
  theme_minimal() +
  labs(
    y = "Connectivity",
    x = "Distance (mm)",
    linetype = "Smoothing Method",
    color = "FWHM (mm)",
  ) +
  theme(strip.text = element_text(size = 10))

# Save the plot
plot_filename <- file.path(output_dir, "connectivity_distance_by_method_and_fwhm.pdf")
ggsave(plot_filename, pred_plot, width = 5, height = 5)

# Get predictions with standard errors
newdat$fit_se <- predict(model_interaction, newdata = newdat, type = "response", 
                         exclude = "s(Subject)", se.fit = TRUE)$se.fit

# Compute differences with propagated uncertainty
present_methods <- unique(as.character(data$Method))
required_methods <- c("gaussian", "constrained")

if (all(required_methods %in% present_methods)) {
  newdat_with_ci <- newdat %>%
    pivot_wider(names_from = Method, values_from = c(fit, fit_se)) %>%
    mutate(
      Difference = fit_constrained - fit_gaussian,
      Diff_SE = sqrt(fit_se_gaussian^2 + fit_se_constrained^2),
      Lower_CI = Difference - 1.96 * Diff_SE,
      Upper_CI = Difference + 1.96 * Diff_SE,
      Significant = !(Lower_CI <= 0 & Upper_CI >= 0)
    )

  # Plot with confidence intervals
  ci_plot <- ggplot(data = subset(newdat_with_ci, FWHM != 0),
                    aes(x = Distance_mm, y = Difference, color = FWHM)) +
    geom_ribbon(aes(ymin = Lower_CI, ymax = Upper_CI, fill = FWHM),
                alpha = 0.4, color = NA) +
    geom_line(linewidth = 0.7) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_color_discrete(drop = FALSE) +
    scale_fill_discrete(guide = "none", drop = FALSE) +
    scale_x_continuous(breaks = seq(0, 150, by = 20)) +
    theme_minimal() +
    labs(
      y = "Connectivity Difference\n(Constrained - Gaussian)",
      x = "Distance (mm)",
      color = "FWHM (mm)"
    )

  ci_plot_filename <- file.path(output_dir, "connectivity_difference_with_ci.pdf")
  ggsave(ci_plot_filename, ci_plot, width = 5, height = 5)
} else {
  missing_methods <- setdiff(required_methods, present_methods)
  warning_msg <- paste("Skipping difference/CI plot because missing methods:", paste(missing_methods, collapse = ", "))
  message(warning_msg)
  # write a small log file so it's clear in the output directory
  writeLines(warning_msg, con = file.path(output_dir, "connectivity_difference_skipped.txt"))
  # create a placeholder plot so downstream code that composes plots doesn't break
  ci_plot <- ggplot() +
    geom_blank() +
    ggtitle(paste("CI plot skipped - missing methods:", paste(missing_methods, collapse = ", ")))
}

# Create combined plot with labels
combined_plot <- pred_plot + ci_plot +
  plot_layout(ncol = 2) +
  plot_annotation(tag_levels = 'A')

# Save the combined plot
combined_plot_filename <- file.path(output_dir, "connectivity_distance_combined_plots.pdf")
ggsave(combined_plot_filename, combined_plot, width = 10, height = 5)
