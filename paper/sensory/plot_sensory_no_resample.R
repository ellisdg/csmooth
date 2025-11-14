#!/usr/bin/env Rscript

# Sensory analysis with NB-GLMM and visualizations
# This script:
#  - reads a CSV of per-ROI FSL FEAT summary stats (including n_active per ROI)
#  - computes descriptive summaries (mean, sd, skew) by region
#  - plots per-region histograms (each panel has its own x-axis range)
#  - fits negative-binomial GLMMs for counts per ROI with random intercepts for subject and run
#    and fixed effects for FWHM (continuous), method (Gaussian vs constrained vs constrained-no-resample) and their interaction
#  - outputs slope and intercept estimates (with CIs and p-values) for Gaussian, Constrained, and Constrained-NoResample
#  - produces a forest plot comparing smoothing slopes by ROI
#  - additionally produces forest plots of differences vs Gaussian (interaction-based) for slopes and intercepts

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
  library(patchwork)
})

#---------------------------------------------------------------------
# I/O (adjust these paths if your data/figure folders differ)
#---------------------------------------------------------------------
input_csv <- "/Users/david.ellis/Box Sync/Aizenberg_Documents/Papers/csmooth/results/fsl_stats_task-lefthand.csv"
out_dir   <- "/Users/david.ellis/Box Sync/Aizenberg_Documents/Papers/csmooth/figures/no_resample"
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
                    "constrained_nr" = "constrained_nr",
                    .default = method),
    method = factor(method, levels = c("gaussian", "constrained", "constrained_nr")),
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
  # Compute Constrained-NoResample slope: fwhm + fwhm:methodconstrained_nr (if present)
  if ("fwhm:methodconstrained_nr" %in% names(beta)) {
    cnr_slope <- lincom(beta, V, c("fwhm" = 1.0, "fwhm:methodconstrained_nr" = 1.0))
  } else {
    cnr_slope <- g_slope
  }

  # Compute intercepts: Gaussian intercept is '(Intercept)'
  g_intercept <- lincom(beta, V, c("(Intercept)" = 1.0))
  # Constrained intercept = intercept + methodconstrained (if present)
  if ("methodconstrained" %in% names(beta)) {
    c_intercept <- lincom(beta, V, c("(Intercept)" = 1.0, "methodconstrained" = 1.0))
  } else {
    c_intercept <- g_intercept
  }
  # Constrained-NoResample intercept = intercept + methodconstrained_nr (if present)
  if ("methodconstrained_nr" %in% names(beta)) {
    cnr_intercept <- lincom(beta, V, c("(Intercept)" = 1.0, "methodconstrained_nr" = 1.0))
  } else {
    cnr_intercept <- g_intercept
  }

  # Differences vs Gaussian (interaction or method coefficients alone)
  c_slope_diff <- if ("fwhm:methodconstrained" %in% names(beta)) lincom(beta, V, c("fwhm:methodconstrained" = 1)) else list(est = NA, lwr = NA, upr = NA, p = NA)
  cnr_slope_diff <- if ("fwhm:methodconstrained_nr" %in% names(beta)) lincom(beta, V, c("fwhm:methodconstrained_nr" = 1)) else list(est = NA, lwr = NA, upr = NA, p = NA)
  c_intercept_diff <- if ("methodconstrained" %in% names(beta)) lincom(beta, V, c("methodconstrained" = 1)) else list(est = NA, lwr = NA, upr = NA, p = NA)
  cnr_intercept_diff <- if ("methodconstrained_nr" %in% names(beta)) lincom(beta, V, c("methodconstrained_nr" = 1)) else list(est = NA, lwr = NA, upr = NA, p = NA)

  tibble(
    region = dat_roi$region[1],
    gaussian_intercept = g_intercept$est,
    gaussian_intercept_lwr = g_intercept$lwr,
    gaussian_intercept_upr = g_intercept$upr,
    gaussian_intercept_p   = g_intercept$p,
    constrained_intercept = c_intercept$est,
    constrained_intercept_lwr = c_intercept$lwr,
    constrained_intercept_upr = c_intercept$upr,
    constrained_intercept_p   = c_intercept$p,
    constrained_nr_intercept = cnr_intercept$est,
    constrained_nr_intercept_lwr = cnr_intercept$lwr,
    constrained_nr_intercept_upr = cnr_intercept$upr,
    constrained_nr_intercept_p   = cnr_intercept$p,

    gaussian_slope = g_slope$est,
    gaussian_lwr = g_slope$lwr,
    gaussian_upr = g_slope$upr,
    gaussian_p   = g_slope$p,
    constrained_slope = c_slope$est,
    constrained_lwr = c_slope$lwr,
    constrained_upr = c_slope$upr,
    constrained_p   = c_slope$p,
    constrained_nr_slope = cnr_slope$est,
    constrained_nr_lwr = cnr_slope$lwr,
    constrained_nr_upr = cnr_slope$upr,
    constrained_nr_p   = cnr_slope$p,

    # Differences (log scale); exponentiate later for ratios
    slope_diff_constrained = c_slope_diff$est,
    slope_diff_constrained_lwr = c_slope_diff$lwr,
    slope_diff_constrained_upr = c_slope_diff$upr,
    slope_diff_constrained_p   = c_slope_diff$p,
    slope_diff_constrained_nr = cnr_slope_diff$est,
    slope_diff_constrained_nr_lwr = cnr_slope_diff$lwr,
    slope_diff_constrained_nr_upr = cnr_slope_diff$upr,
    slope_diff_constrained_nr_p   = cnr_slope_diff$p,
    intercept_diff_constrained = c_intercept_diff$est,
    intercept_diff_constrained_lwr = c_intercept_diff$lwr,
    intercept_diff_constrained_upr = c_intercept_diff$upr,
    intercept_diff_constrained_p   = c_intercept_diff$p,
    intercept_diff_constrained_nr = cnr_intercept_diff$est,
    intercept_diff_constrained_nr_lwr = cnr_intercept_diff$lwr,
    intercept_diff_constrained_nr_upr = cnr_intercept_diff$upr,
    intercept_diff_constrained_nr_p   = cnr_intercept_diff$p
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

# Save wide differences vs Gaussian (log-scale coefficients)
write.csv(glmm_results %>% select(region,
                                  starts_with("slope_diff_"),
                                  starts_with("intercept_diff_")),
          file = file.path(out_dir, "sensory_nb_glmm_differences_vs_gaussian_by_roi.csv"),
          row.names = FALSE)

print(glmm_results)

# NEW: Tidy differences dataset with exponentiated ratios (response scale)
diffs_tidy <- bind_rows(
  glmm_results %>%
    transmute(region,
              method = "Constrained",
              slope_log = slope_diff_constrained,
              slope_log_lwr = slope_diff_constrained_lwr,
              slope_log_upr = slope_diff_constrained_upr,
              intercept_log = intercept_diff_constrained,
              intercept_log_lwr = intercept_diff_constrained_lwr,
              intercept_log_upr = intercept_diff_constrained_upr),
  glmm_results %>%
    transmute(region,
              method = "Constrained-NoResample",
              slope_log = slope_diff_constrained_nr,
              slope_log_lwr = slope_diff_constrained_nr_lwr,
              slope_log_upr = slope_diff_constrained_nr_upr,
              intercept_log = intercept_diff_constrained_nr,
              intercept_log_lwr = intercept_diff_constrained_nr_lwr,
              intercept_log_upr = intercept_diff_constrained_nr_upr)
) %>%
  mutate(
    slope_ratio = exp(slope_log),
    slope_ratio_lwr = exp(slope_log_lwr),
    slope_ratio_upr = exp(slope_log_upr),
    intercept_ratio = exp(intercept_log),
    intercept_ratio_lwr = exp(intercept_log_lwr),
    intercept_ratio_upr = exp(intercept_log_upr)
  )

write.csv(diffs_tidy,
          file = file.path(out_dir, "sensory_nb_glmm_differences_vs_gaussian_tidy.csv"),
          row.names = FALSE)

#---------------------------------------------------------------------
# Forest plot: Gaussian vs Constrained smoothing slopes (per ROI)
#---------------------------------------------------------------------
plot_df <- glmm_results %>%
  select(region,
         Gaussian_slope = gaussian_slope, Gaussian_lwr = gaussian_lwr, Gaussian_upr = gaussian_upr,
         Constrained_slope = constrained_slope, Constrained_lwr = constrained_lwr, Constrained_upr = constrained_upr,
         ConstrainedNR_slope = constrained_nr_slope, ConstrainedNR_lwr = constrained_nr_lwr, ConstrainedNR_upr = constrained_nr_upr) %>%
  pivot_longer(cols = -region,
               names_to = c("method", ".value"),
               names_pattern = "(Gaussian|Constrained|ConstrainedNR)_(.*)") %>%
  filter(region %in% rois) %>%
  mutate(method = recode(method, ConstrainedNR = "Constrained-NoResample")) %>%
  mutate(method = factor(method, levels = c("Gaussian", "Constrained", "Constrained-NoResample"))) %>%
  arrange(region, method) %>%
  mutate(slope_rr = exp(slope), lwr_rr = exp(lwr), upr_rr = exp(upr)) %>%
  mutate(region = factor(as.character(region), levels = c("WM", "GM", "RH Precentral", "RH Postcentral")))

p_forest <- ggplot(plot_df, aes(x = slope_rr, y = region, color = method, shape = method)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_point(position = position_dodge(width = 0.5), size = 2) +
  geom_errorbar(aes(xmin = lwr_rr, xmax = upr_rr),
                position = position_dodge(width = 0.5), width = 0.2) +
  scale_color_manual(values = c("Gaussian" = brewer.pal(3, "Set1")[2],
                                "Constrained" = brewer.pal(3, "Set1")[3],
                                "Constrained-NoResample" = brewer.pal(3, "Set1")[1]),
                     breaks = c("Gaussian", "Constrained", "Constrained-NoResample")) +
  scale_shape_manual(values = c("Gaussian" = 16, "Constrained" = 17, "Constrained-NoResample" = 15),
                     breaks = c("Gaussian", "Constrained", "Constrained-NoResample")) +
  scale_x_continuous(breaks = function(x) { rng <- range(x, na.rm = TRUE); base <- pretty(rng); sort(unique(c(base, 1))) }) +
  labs(x = "Incidence Rate Ratio per 1 mm FWHM",
       y = NULL, color = "Method", shape = "Method") +
  theme_minimal(base_size = 12)

#---------------------------------------------------------------------
# Forest plot for intercepts (constant terms)
#---------------------------------------------------------------------
intercept_df <- glmm_results %>%
  select(region,
         Gaussian_intercept = gaussian_intercept, Gaussian_lwr = gaussian_intercept_lwr, Gaussian_upr = gaussian_intercept_upr,
         Constrained_intercept = constrained_intercept, Constrained_lwr = constrained_intercept_lwr, Constrained_upr = constrained_intercept_upr,
         ConstrainedNR_intercept = constrained_nr_intercept, ConstrainedNR_lwr = constrained_nr_intercept_lwr, ConstrainedNR_upr = constrained_nr_intercept_upr) %>%
  pivot_longer(cols = -region,
               names_to = c("method", ".value"),
               names_pattern = "(Gaussian|Constrained|ConstrainedNR)_(.*)") %>%
  filter(region %in% rois) %>%
  rename(intercept = intercept) %>%
  mutate(method = recode(method, ConstrainedNR = "Constrained-NoResample")) %>%
  mutate(method = factor(method, levels = c("Gaussian", "Constrained", "Constrained-NoResample"))) %>%
  mutate(intercept_resp = exp(intercept), lwr_resp = exp(lwr), upr_resp = exp(upr)) %>%
  mutate(region = factor(as.character(region), levels = c("WM", "GM", "RH Precentral", "RH Postcentral")))

p_forest_intercept <- ggplot(intercept_df, aes(x = intercept_resp, y = region, color = method, shape = method)) +
  geom_point(position = position_dodge(width = 0.5), size = 2) +
  geom_errorbar(aes(xmin = lwr_resp, xmax = upr_resp),
                position = position_dodge(width = 0.5), width = 0.2) +
  scale_color_manual(values = c("Gaussian" = brewer.pal(3, "Set1")[2],
                                "Constrained" = brewer.pal(3, "Set1")[3],
                                "Constrained-NoResample" = brewer.pal(3, "Set1")[1]),
                     breaks = c("Gaussian", "Constrained", "Constrained-NoResample")) +
  scale_shape_manual(values = c("Gaussian" = 16, "Constrained" = 17, "Constrained-NoResample" = 15),
                     breaks = c("Gaussian", "Constrained", "Constrained-NoResample")) +
  scale_x_continuous(limits = c(0, NA)) +
  labs(x = "Expected Count at Baseline",
       y = NULL, color = "Method", shape = "Method") +
  theme_minimal(base_size = 12)


#---------------------------------------------------------------------
# NEW: Forest plots for differences vs Gaussian (ratio-of-rate-ratios)
#---------------------------------------------------------------------
# Slope differences (interaction terms) on response scale
slope_diff_df <- glmm_results %>%
  select(region,
         Constrained_slope_diff_est = slope_diff_constrained,
         Constrained_slope_diff_lwr = slope_diff_constrained_lwr,
         Constrained_slope_diff_upr = slope_diff_constrained_upr,
         ConstrainedNR_slope_diff_est = slope_diff_constrained_nr,
         ConstrainedNR_slope_diff_lwr = slope_diff_constrained_nr_lwr,
         ConstrainedNR_slope_diff_upr = slope_diff_constrained_nr_upr) %>%
  pivot_longer(cols = -region,
               names_to = c("method", ".value"),
               names_pattern = "(Constrained|ConstrainedNR)_slope_diff_(.*)") %>%
  filter(region %in% rois) %>%
  mutate(method = recode(method, ConstrainedNR = "Constrained-NoResample", Constrained = "Constrained")) %>%
  mutate(method = factor(method, levels = c("Constrained", "Constrained-NoResample"))) %>%
  mutate(ratio_rr = exp(est), lwr_rr = exp(lwr), upr_rr = exp(upr)) %>%
  mutate(region = factor(as.character(region), levels = c("WM", "GM", "RH Precentral", "RH Postcentral")))

p_forest_slope_diff <- ggplot(slope_diff_df, aes(x = ratio_rr, y = region, color = method, shape = method)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_point(position = position_dodge(width = 0.5), size = 2) +
  geom_errorbar(aes(xmin = lwr_rr, xmax = upr_rr), position = position_dodge(width = 0.5), width = 0.2) +
  scale_color_manual(values = c("Constrained" = brewer.pal(3, "Set1")[3],
                                "Constrained-NoResample" = brewer.pal(3, "Set1")[1])) +
  scale_shape_manual(values = c("Constrained" = 17, "Constrained-NoResample" = 15)) +
  scale_x_continuous(breaks = function(x) { rng <- range(x, na.rm = TRUE); base <- pretty(rng); sort(unique(c(base, 1))) }) +
  labs(x = "Ratio of Slope Rate Ratios vs Gaussian (per 1 mm FWHM)", y = NULL, color = "Method", shape = "Method") +
  theme_minimal(base_size = 12)

# Intercept differences on response scale
intercept_diff_df <- glmm_results %>%
  select(region,
         Constrained_intercept_diff_est = intercept_diff_constrained,
         Constrained_intercept_diff_lwr = intercept_diff_constrained_lwr,
         Constrained_intercept_diff_upr = intercept_diff_constrained_upr,
         ConstrainedNR_intercept_diff_est = intercept_diff_constrained_nr,
         ConstrainedNR_intercept_diff_lwr = intercept_diff_constrained_nr_lwr,
         ConstrainedNR_intercept_diff_upr = intercept_diff_constrained_nr_upr) %>%
  pivot_longer(cols = -region,
               names_to = c("method", ".value"),
               names_pattern = "(Constrained|ConstrainedNR)_intercept_diff_(.*)") %>%
  filter(region %in% rois) %>%
  mutate(method = recode(method, ConstrainedNR = "Constrained-NoResample", Constrained = "Constrained")) %>%
  mutate(method = factor(method, levels = c("Constrained", "Constrained-NoResample"))) %>%
  mutate(ratio_resp = exp(est), lwr_resp = exp(lwr), upr_resp = exp(upr)) %>%
  mutate(region = factor(as.character(region), levels = c("WM", "GM", "RH Precentral", "RH Postcentral")))

p_forest_intercept_diff <- ggplot(intercept_diff_df, aes(x = ratio_resp, y = region, color = method, shape = method)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_point(position = position_dodge(width = 0.5), size = 2) +
  geom_errorbar(aes(xmin = lwr_resp, xmax = upr_resp), position = position_dodge(width = 0.5), width = 0.2) +
  scale_color_manual(values = c("Constrained" = brewer.pal(3, "Set1")[3],
                                "Constrained-NoResample" = brewer.pal(3, "Set1")[1])) +
  scale_shape_manual(values = c("Constrained" = 17, "Constrained-NoResample" = 15)) +
  scale_x_continuous(breaks = function(x) { rng <- range(x, na.rm = TRUE); base <- pretty(rng); sort(unique(c(base, 1))) }) +
  labs(x = "Ratio of Baseline Expected Counts vs Gaussian", y = NULL, color = "Method", shape = "Method") +
  theme_minimal(base_size = 12)

#---------------------------------------------------------------------
# Save all plots
#---------------------------------------------------------------------
ggsave(filename = file.path(out_dir, "sensory_smoothing_glmm_forest_plot.pdf"),
       plot = p_forest, width = 6, height = 4)

ggsave(filename = file.path(out_dir, "sensory_intercept_glmm_forest_plot.pdf"),
       plot = p_forest_intercept, width = 6, height = 4)

ggsave(filename = file.path(out_dir, "sensory_combined_forest_plots.pdf"),
       plot = p_forest / p_forest_intercept + plot_layout(ncol = 2, guides = "collect"),
       width = 12, height = 4)

ggsave(filename = file.path(out_dir, "sensory_smoothing_glmm_slope_diff_forest_plot.pdf"),
       plot = p_forest_slope_diff, width = 6, height = 4)

ggsave(filename = file.path(out_dir, "sensory_intercept_glmm_diff_forest_plot.pdf"),
       plot = p_forest_intercept_diff, width = 6, height = 4)
