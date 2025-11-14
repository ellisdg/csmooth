#!/usr/bin/env Rscript

# Sensory analysis with NB-GLMM and visualizations
# This script:
#  - reads a CSV of per-ROI FSL FEAT summary stats (including n_active per ROI)
#  - computes descriptive summaries (mean, sd, skew) by region
#  - plots per-region histograms (each panel has its own x-axis range)
#  - fits negative-binomial GLMMs for counts per ROI with random intercepts for subject and run
#    and fixed effects for FWHM (continuous), method (Gaussian vs constrained) and their interaction
#  - outputs slope and intercept estimates (with CIs and p-values) for Gaussian and constrained
#  - produces a forest plot comparing smoothing slopes by ROI

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(glmmTMB)
  library(e1071)   # for skewness
  library(purrr)
  library(tibble)
  library(RColorBrewer)  # added for brewer.pal used in forest plots
})

#---------------------------------------------------------------------
# I/O (adjust these paths if your data/figure folders differ)
#---------------------------------------------------------------------
input_csv <- "/Users/david.ellis/Box Sync/Aizenberg_Documents/Papers/csmooth/results/fsl_stats_task-lefthand.csv"
out_dir   <- "/Users/david.ellis/Box Sync/Aizenberg_Documents/Papers/csmooth/figures"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

#---------------------------------------------------------------------
# Load and prepare data
# - ensure consistent column names
# - coerce types and drop rows with missing critical values
#---------------------------------------------------------------------
df <- readr::read_csv(input_csv, show_col_types = FALSE)

# Normalize column names that might vary across exports
if (!"fwhm" %in% names(df) && "smoothing_parameter" %in% names(df)) {
  df <- df %>% rename(fwhm = smoothing_parameter)
}
if (!"method" %in% names(df) && "smoothing_method" %in% names(df)) {
  df <- df %>% rename(method = smoothing_method)
}

required_cols <- c("region", "method", "fwhm", "subject", "run", "n_active")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

# Standardize columns and types
# - method lower-cased and converted to factor with Gaussian as reference
# - region as character; subject/run as factor
# - fwhm and n_active coerced to numeric
# - drop rows with missing fwhm or n_active

df <- df %>%
  mutate(
    method = tolower(as.character(method)),
    method = recode(method,
                    "gaussian" = "gaussian",
                    "constrained" = "constrained",
                    .default = method),
    method = factor(method, levels = c("gaussian", "constrained")),
    region = as.character(region),
    subject = factor(subject),
    run = factor(run),
    fwhm = suppressWarnings(as.numeric(fwhm)),
    n_active = suppressWarnings(as.numeric(n_active))
  ) %>%
  filter(!is.na(n_active), !is.na(fwhm))

rois <- intersect(c("gm", "wm", "rh_precentral", "rh_postcentral"),
                  unique(df$region))

# filter to only these ROIs
df <- df %>% filter(region %in% rois)

# Map ROI codes to display names once and update 'rois' to use the display names
# This centralizes region renaming so we don't repeat it before every plot.
display_map <- c(
  "rh_postcentral" = "RH Postcentral",
  "rh_precentral" = "RH Precentral",
  "gm" = "GM",
  "wm" = "WM"
)

# Update the 'rois' vector to the mapped (display) names for downstream use
rois <- unname(display_map[rois])

# Replace region codes in the main dataframe with display names and set a consistent factor ordering
df <- df %>%
  mutate(region = as.character(display_map[region])) %>%
  # Set ordering so RH Postcentral appears on top in plots by placing it last
  mutate(region = factor(region, levels = c("WM", "GM", "RH Precentral", "RH Postcentral")))

#---------------------------------------------------------------------
# Histograms: number of active voxels by region
# - use scales = 'free_x' so each facet uses an x-axis range appropriate for that region
# - reduce bins for readability (30)
# - save as PDF
#---------------------------------------------------------------------
p_hist <- df %>%
  ggplot(aes(x = n_active)) +
  geom_histogram(bins = 30, color = "white", fill = "#2c7fb8") +
  facet_wrap(~ region, scales = "free_x", ncol = 4) +
  labs(x = "Number of active voxels", y = "Count") +
  theme_minimal(base_size = 12)

ggsave(filename = file.path(out_dir, "hist_by_region.pdf"),
       plot = p_hist, width = 8, height = 4)

#---------------------------------------------------------------------
# Summary stats: mean, sd, skew by region
# - skewness uses e1071::skewness with type=2 (consistent with common definitions)
#---------------------------------------------------------------------
summary_tbl <- df %>%
  group_by(region) %>%
  summarize(
    mean_n_active = mean(n_active, na.rm = TRUE),
    sd_n_active   = sd(n_active, na.rm = TRUE),
    skew_n_active = e1071::skewness(n_active, na.rm = TRUE, type = 2),
    .groups = "drop"
  ) %>%
  arrange(region)

write.csv(summary_tbl,
          file = file.path(out_dir, "n_active_summary_by_region.csv"),
          row.names = FALSE)

# Print to console for quick inspection
print(summary_tbl)

#---------------------------------------------------------------------
# Negative-binomial GLMM per ROI (counts only)
# - We'll analyze a set of ROIs of interest, fit a glmmTMB NB model per ROI
# - Model: n_active ~ fwhm * method + (1 | subject) + (1 | run)
# - We will extract both slope estimates (effect of fwhm) and intercepts for
#   Gaussian (reference) and Constrained (reference + method effect)
#---------------------------------------------------------------------


if (length(rois) == 0) {
  stop("No expected ROIs found among regions. Regions present: ",
       paste(sort(unique(df$region)), collapse = ", "))
}

# Helper to compute linear combinations of fixed effects (estimate, se, CI, p)
lincom <- function(beta, V, coefs, level = 0.95) {
  # beta: named vector of fixed-effect estimates
  # V: variance-covariance matrix for fixed effects
  # coefs: named numeric vector of weights for the linear combination
  terms <- names(beta)
  w <- rep(0, length(beta))
  names(w) <- names(beta)
  for (nm in names(coefs)) {
    if (nm %in% names(w)) w[nm] <- w[nm] + coefs[[nm]]
  }
  est <- as.numeric(t(w) %*% beta)
  se  <- sqrt(as.numeric(t(w) %*% V %*% w))
  z   <- est / se
  alpha <- 1 - level
  crit <- qnorm(1 - alpha / 2)
  lwr <- est - crit * se
  upr <- est + crit * se
  p   <- 2 * pnorm(-abs(z))
  list(est = est, se = se, lwr = lwr, upr = upr, p = p)
}

# Fit one ROI and return a tidy summary including intercepts and slopes
fit_one_roi <- function(dat_roi) {
  # Fit NB-GLMM with random intercepts for subject and run
  m <- glmmTMB(
    n_active ~ fwhm * method + (1 | subject) + (1 | run),
    family = nbinom2(),
    data = dat_roi
  )
  
  # Extract fixed effects and covariance
  beta <- fixef(m)$cond
  V    <- as.matrix(vcov(m)$cond)

  # Compute Gaussian (reference) slope: coef on 'fwhm'
  g_slope <- lincom(beta, V, c("fwhm" = 1.0))
  # Compute Constrained slope: fwhm + fwhm:methodconstrained (if present)
  if ("fwhm:methodconstrained" %in% names(beta)) {
    c_slope <- lincom(beta, V, c("fwhm" = 1.0, "fwhm:methodconstrained" = 1.0))
  } else {
    c_slope <- g_slope
  }

  # Compute intercepts: Gaussian intercept is '(Intercept)'
  g_intercept <- lincom(beta, V, c("(Intercept)" = 1.0))
  # Constrained intercept = intercept + methodconstrained (if present)
  if ("methodconstrained" %in% names(beta)) {
    c_intercept <- lincom(beta, V, c("(Intercept)" = 1.0, "methodconstrained" = 1.0))
  } else {
    c_intercept <- g_intercept
  }

  tibble::tibble(
    region = dat_roi$region[1],
    gaussian_intercept = g_intercept$est,
    gaussian_intercept_lwr = g_intercept$lwr,
    gaussian_intercept_upr = g_intercept$upr,
    gaussian_intercept_p   = g_intercept$p,
    constrained_intercept = c_intercept$est,
    constrained_intercept_lwr = c_intercept$lwr,
    constrained_intercept_upr = c_intercept$upr,
    constrained_intercept_p   = c_intercept$p,

    gaussian_slope = g_slope$est,
    gaussian_lwr = g_slope$lwr,
    gaussian_upr = g_slope$upr,
    gaussian_p   = g_slope$p,
    constrained_slope = c_slope$est,
    constrained_lwr = c_slope$lwr,
    constrained_upr = c_slope$upr,
    constrained_p   = c_slope$p
  )
}

# Run model for each ROI and collect results
glmm_results <- map_dfr(rois, function(r) {
  dat_roi <- df %>% filter(region == r)
  fit_one_roi(dat_roi)
})

# Save results table with intercepts and slopes
write.csv(glmm_results,
          file = file.path(out_dir, "sensory_nb_glmm_slopes_by_roi.csv"),
          row.names = FALSE)

print(glmm_results)

#---------------------------------------------------------------------
# Forest plot: Gaussian vs Constrained smoothing slopes (per ROI)
# - pivot so we have columns: region, method, slope, lwr, upr
# - ensure names match the pattern '(Gaussian|Constrained)_(slope|lwr|upr)'
# - minimal change: data already contains display names; only ensure ordering and transform to response scale
#---------------------------------------------------------------------
plot_df <- glmm_results %>%
  select(region,
         Gaussian_slope = gaussian_slope, Gaussian_lwr = gaussian_lwr, Gaussian_upr = gaussian_upr,
         Constrained_slope = constrained_slope, Constrained_lwr = constrained_lwr, Constrained_upr = constrained_upr) %>%
  pivot_longer(cols = -region,
               names_to = c("method", ".value"),
               names_pattern = "(Gaussian|Constrained)_(.*)") %>%
  # Keep only the display-name regions (rois variable already contains display names)
  filter(region %in% rois) %>%
  # Standardize method labels and ensure ordering
  mutate(method = factor(method, levels = c("Gaussian", "Constrained"))) %>%
  arrange(region, method) %>%
  # response-scale transformations for slopes
  mutate(slope_rr = exp(slope), lwr_rr = exp(lwr), upr_rr = exp(upr)) %>%
  # enforce the same region factor ordering as df/glmm_results
  mutate(region = factor(as.character(region), levels = c("WM", "GM", "RH Precentral", "RH Postcentral")))

# Create slope forest plot with specified colors (dodge side-by-side) on response scale
p_forest <- ggplot(plot_df, aes(x = slope_rr, y = region, color = method, shape = method)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +  # null rate ratio at 1
  geom_point(position = position_dodge(width = 0.5), size = 2) +
  geom_errorbar(aes(xmin = lwr_rr, xmax = upr_rr),
                position = position_dodge(width = 0.5), width = 0.2) +
  scale_color_manual(values = c("Gaussian" = brewer.pal(3, "Set1")[2], "Constrained" = brewer.pal(3, "Set1")[3]),
                     breaks = c("Gaussian", "Constrained")) +
  scale_shape_manual(values = c("Gaussian" = 16, "Constrained" = 17),
                     breaks = c("Gaussian", "Constrained")) +
  # Ensure a tick at 1 by adding it to pretty breaks
  scale_x_continuous(breaks = function(x) {
    rng <- range(x, na.rm = TRUE)
    base <- pretty(rng)
    sort(unique(c(base, 1)))
  }) +
  labs(x = "Incidence Rate Ratio per 1 mm FWHM",
       y = NULL, color = "Method", shape = "Method") +
  theme_minimal(base_size = 12)

#---------------------------------------------------------------------
# Forest plot for intercepts (constant terms)
# - similar to slopes but plotting the intercept +/- CI
# - use the same color scheme and region ordering
#---------------------------------------------------------------------
intercept_df <- glmm_results %>%
  select(region,
         Gaussian_intercept = gaussian_intercept, Gaussian_lwr = gaussian_intercept_lwr, Gaussian_upr = gaussian_intercept_upr,
         Constrained_intercept = constrained_intercept, Constrained_lwr = constrained_intercept_lwr, Constrained_upr = constrained_intercept_upr) %>%
  pivot_longer(cols = -region,
               names_to = c("method", ".value"),
               names_pattern = "(Gaussian|Constrained)_(.*)") %>%
  # Keep only the display-name regions
  filter(region %in% rois) %>%
  # rename the estimate column to 'intercept' for clarity (pivot_longer created 'intercept')
  rename(intercept = intercept) %>%
  mutate(method = factor(method, levels = c("Gaussian", "Constrained"))) %>%
  # response-scale transformations for intercepts
  mutate(intercept_resp = exp(intercept), lwr_resp = exp(lwr), upr_resp = exp(upr)) %>%
  # enforce display-name region ordering
  mutate(region = factor(as.character(region), levels = c("WM", "GM", "RH Precentral", "RH Postcentral")))

# Plot intercepts (dodge side-by-side) on response scale (remove null line at 0)
p_forest_intercept <- ggplot(intercept_df, aes(x = intercept_resp, y = region, color = method, shape = method)) +
  geom_point(position = position_dodge(width = 0.5), size = 2) +
  geom_errorbar(aes(xmin = lwr_resp, xmax = upr_resp),
                position = position_dodge(width = 0.5), width = 0.2) +
  scale_color_manual(values = c("Gaussian" = brewer.pal(3, "Set1")[2], "Constrained" = brewer.pal(3, "Set1")[3]),
                     breaks = c("Gaussian", "Constrained")) +
  scale_shape_manual(values = c("Gaussian" = 16, "Constrained" = 17),
                     breaks = c("Gaussian", "Constrained")) +
  # Ensure x-axis lower limit is 0 (counts cannot be negative)
  scale_x_continuous(limits = c(0, NA)) +
  labs(x = "Expected Count at Baseline",
       y = NULL, color = "Method", shape = "Method") +
  theme_minimal(base_size = 12)


# Save PDF versions of both forest plots
ggsave(filename = file.path(out_dir, "sensory_smoothing_glmm_forest_plot.pdf"),
       plot = p_forest, width = 6, height = 4)
ggsave(filename = file.path(out_dir, "sensory_intercept_glmm_forest_plot.pdf"),
       plot = p_forest_intercept, width = 6, height = 4)
