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
input_csv <- "/Users/david.ellis/Library/CloudStorage/Box-Box/Aizenberg_Documents/Papers/csmooth/results/fsl_stats_task-lefthand.csv"
out_dir   <- "/Users/david.ellis/Library/CloudStorage/Box-Box/Aizenberg_Documents/Papers/csmooth/figures/no_resample"
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

# Add ROI size column if available in input, otherwise estimate per subject/run/region
size_candidates <- c("roi_size","n_voxels","n_vox","n_mask","mask_count","mask_voxels","nvox","nvoxels","n_roi","roi_voxels","n_voxels_roi","n_total_voxels")
found_size <- intersect(names(df), size_candidates)
if (length(found_size) > 0) {
  fname <- found_size[1]
  message("Using input ROI size column: ", fname)
  # Safely coerce to numeric; create a consistent column name 'roi_size'
  df$roi_size <- suppressWarnings(as.numeric(df[[fname]]))
} else {
  warning("No explicit ROI-size column found in input; estimating roi_size as max observed n_active per subject/run/region.")
  # Estimate ROI size as the maximum observed n_active per subject/run/region (fallback)
  df_est <- df %>% group_by(subject, run, region) %>% summarize(roi_size = max(n_active, na.rm = TRUE), .groups = "drop")
  df <- dplyr::left_join(df, df_est, by = c("subject", "run", "region"))
  # If still NA (e.g., single row), fallback to region-level max
  df <- df %>% group_by(region) %>% mutate(roi_size = ifelse(is.na(roi_size), max(roi_size, na.rm = TRUE), roi_size)) %>% ungroup()
}

# Compute proportion of active voxels within the ROI (may be an estimate if roi_size was not provided)
df <- df %>% mutate(prop_active = n_active / roi_size)

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

ggsave(filename = file.path(out_dir, "hist_by_region_no_resample.pdf"),
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
          file = file.path(out_dir, "n_active_summary_by_region_no_resample.csv"),
          row.names = FALSE)

# Print to console for quick inspection
print(summary_tbl)

#---------------------------------------------------------------------
# Active voxels by region, method, and FWHM (boxplots)
# - includes optional no-smoothing baseline (method coded as none/raw/nosmooth)
# - keeps WM/GM/RH Precentral/RH Postcentral and FWHM 3/6/9/12 (baseline shown as 0)
#---------------------------------------------------------------------
plot_fwhm_values <- c(3, 6, 9, 12)
method_display_map <- c(
  "gaussian" = "Gaussian",
  "constrained" = "Constrained",
  # rename the no-resample variant back to the simpler label
  "constrained_nr" = "Constrained",
  "none" = "No Smoothing",
  "no_smoothing" = "No Smoothing",
  "nosmooth" = "No Smoothing",
  "raw" = "No Smoothing"
)

active_plot_df <- df %>%
  mutate(
    method_chr = tolower(as.character(method)),
    method_display = recode(method_chr, !!!method_display_map, .default = method_chr),
    # use the simpler 'Constrained' label for the no-resample variant; drop the resampling 'constrained' method by checking the raw code
    method_display = factor(method_display, levels = c("No Smoothing", "Gaussian", "Constrained")),
    fwhm_plot = if_else(method_display == "No Smoothing", 0, fwhm),
    # override legend label to No Smoothing whenever FWHM is 0, regardless of method
    method_display_plot = if_else(fwhm_plot == 0, "No Smoothing", as.character(method_display)),
    method_display_plot = factor(method_display_plot, levels = c("No Smoothing", "Gaussian", "Constrained"))
  ) %>%
  filter(region %in% rois, fwhm_plot %in% c(0, plot_fwhm_values)) %>%
  # explicitly remove rows corresponding to the regular (resampling) constrained method by inspecting the original method code
  filter(method_chr != "constrained") %>%
  drop_na(method_display_plot, fwhm_plot, n_active)

if (nrow(active_plot_df) == 0) {
  warning("No data available for active-voxel plot after filtering for expected methods/FWHM.")
}

p_active <- ggplot(active_plot_df, aes(x = factor(fwhm_plot), y = n_active, fill = method_display_plot)) +
  geom_boxplot(position = position_dodge(width = 0.8), width = 0.7, outlier.size = 0.8) +
  facet_wrap(~ region, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = c(
    "No Smoothing" = "gray80",
    "Gaussian" = brewer.pal(3, "Set1")[2],
    # use the original green for Constrained
    "Constrained" = brewer.pal(3, "Set1")[3]
  ), drop = FALSE, name = "Method") +
  scale_x_discrete(limits = c("0", "3", "6", "9", "12")) +
  labs(x = "FWHM (mm)", y = "Number of active voxels") +
  theme_minimal(base_size = 12)

ggsave(filename = file.path(out_dir, "sensory_active_voxels_by_method_fwhm.png"),
       plot = p_active, width = 8, height = 6, dpi = 300)

ggsave(filename = file.path(out_dir, "sensory_active_voxels_by_method_fwhm_no_resample.pdf"),
       plot = p_active, width = 8, height = 6)

# ---------------------------------------------------------------------
# Boxplots of proportions derived from per-scan counts:
#  - Percentage of active voxels in GM = 100 * GM / (GM + WM)
#  - Percentage of active voxels in postcentral = 100 * Postcentral / (GM + WM)
# Compute these per subject/run/fwhm/method and then make boxplots
# ---------------------------------------------------------------------

# Pivot per-scan region counts to wide so we can compute cross-region percentages
scan_wide <- active_plot_df %>%
  select(subject, run, fwhm_plot, method_display_plot, region, n_active) %>%
  pivot_wider(names_from = region, values_from = n_active)

# Ensure numeric and compute percentages with safe division
scan_wide <- scan_wide %>%
  mutate(
    GM = as.numeric(.data$GM),
    WM = as.numeric(.data$WM),
    Postcentral = as.numeric(`RH Postcentral`),
    Precentral = as.numeric(`RH Precentral`),
    denom_GM_WM = GM + WM,
    # percent of GM within GM+WM
    percent_GM_of_total = ifelse(is.na(denom_GM_WM) | denom_GM_WM == 0, NA, 100 * GM / denom_GM_WM),
    # percent of postcentral compared to postcentral + precentral (postcentral vs precentral)
    denom_post_pre = Postcentral + Precentral,
    percent_Postcentral_of_pair = ifelse(is.na(denom_post_pre) | denom_post_pre == 0, NA, 100 * Postcentral / denom_post_pre)
  )

# Prepare long dataframe for plotting two measures side-by-side
prop_plot_df <- scan_wide %>%
  select(subject, run, fwhm_plot, method_display_plot, percent_GM_of_total, percent_Postcentral_of_pair) %>%
  pivot_longer(cols = c(percent_GM_of_total, percent_Postcentral_of_pair),
               names_to = "measure", values_to = "percent") %>%
  mutate(measure = recode(measure,
                          percent_GM_of_total = "% active voxels in GM",
                          percent_Postcentral_of_pair = "% active voxels in postcentral vs precentral"),
         measure = factor(measure, levels = c("% active voxels in GM", "% active voxels in postcentral vs precentral"))) %>%
  drop_na(percent, method_display_plot, fwhm_plot)

if (nrow(prop_plot_df) == 0) {
  warning("No data available to plot the derived percent measures after filtering.")
}

p_prop <- ggplot(prop_plot_df, aes(x = factor(fwhm_plot), y = percent, fill = method_display_plot)) +
  geom_boxplot(position = position_dodge(width = 0.8), width = 0.7, outlier.size = 0.8) +
  facet_wrap(~ measure, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = c(
    "No Smoothing" = "gray80",
    "Gaussian" = brewer.pal(3, "Set1")[2],
    "Constrained" = brewer.pal(3, "Set1")[3]
  ), drop = FALSE, name = "Method") +
  scale_x_discrete(limits = c("0", "3", "6", "9", "12")) +
  labs(x = "FWHM (mm)", y = "Percent") +
  theme_minimal(base_size = 12)

# Save derived-percent plots
if (nrow(prop_plot_df) > 0) {
  ggsave(filename = file.path(out_dir, "sensory_percent_derived_gm_postcentral_by_method_fwhm_no_resample.png"),
         plot = p_prop, width = 8, height = 4, dpi = 300)
  ggsave(filename = file.path(out_dir, "sensory_percent_derived_gm_postcentral_by_method_fwhm_no_resample.pdf"),
         plot = p_prop, width = 8, height = 4)
}


# ---------------------------------------------------------------------
# Paired t-tests (Gaussian vs Constrained) per measure and FWHM
# - pairs on subject + run within each fwhm_plot
# ---------------------------------------------------------------------
paired_ttest_by_fwhm <- prop_plot_df %>%
  filter(method_display_plot %in% c("Gaussian", "Constrained")) %>%
  group_by(measure, fwhm_plot) %>%
  group_modify(~{
    wide <- .x %>%
      select(subject, run, method_display_plot, percent) %>%
      pivot_wider(names_from = method_display_plot, values_from = percent) %>%
      drop_na(Gaussian, Constrained)
    if (nrow(wide) < 2) return(tibble())
    tt <- try(stats::t.test(wide$Gaussian, wide$Constrained, paired = TRUE), silent = TRUE)
    if (inherits(tt, "try-error")) return(tibble())
    tibble(
      n_pairs = nrow(wide),
      mean_gaussian = mean(wide$Gaussian, na.rm = TRUE),
      mean_constrained = mean(wide$Constrained, na.rm = TRUE),
      diff_mean = mean(wide$Gaussian - wide$Constrained, na.rm = TRUE),
      t_stat = unname(tt$statistic),
      df = unname(tt$parameter),
      p_value = tt$p.value
    )
  }) %>%
  ungroup() %>%
  mutate(fwhm_plot = as.character(fwhm_plot))

if (nrow(paired_ttest_by_fwhm) == 0) {
  warning("No paired t-test results computed (insufficient paired data for Gaussian vs Constrained comparisons).")
} else {
  write.csv(paired_ttest_by_fwhm,
            file = file.path(out_dir, "sensory_percent_paired_ttests_gaussian_vs_constrained_no_resample.csv"),
            row.names = FALSE)
}

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
          file = file.path(out_dir, "sensory_nb_glmm_slopes_by_roi_no_resample.csv"),
          row.names = FALSE)

# Save wide differences vs Gaussian (log-scale coefficients)
write.csv(glmm_results %>% select(region,
                                  starts_with("slope_diff_"),
                                  starts_with("intercept_diff_")),
          file = file.path(out_dir, "sensory_nb_glmm_differences_vs_gaussian_by_roi_no_resample.csv"),
          row.names = FALSE)

print(glmm_results)

# Tidy differences dataset with exponentiated ratios (response scale)
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
          file = file.path(out_dir, "sensory_nb_glmm_differences_vs_gaussian_tidy_no_resample.csv"),
          row.names = FALSE)

#---------------------------------------------------------------------
# Forest plot: Gaussian vs Constrained smoothing slopes (per ROI)
#---------------------------------------------------------------------
pd <- position_dodge(width = 0.6)
plot_df <- glmm_results %>%
  select(region,
         Gaussian_slope = gaussian_slope, Gaussian_lwr = gaussian_lwr, Gaussian_upr = gaussian_upr,
         Constrained_slope = constrained_slope, Constrained_lwr = constrained_lwr, Constrained_upr = constrained_upr,
         ConstrainedNR_slope = constrained_nr_slope, ConstrainedNR_lwr = constrained_nr_lwr, ConstrainedNR_upr = constrained_nr_upr) %>%
  pivot_longer(cols = -region,
               names_to = c("method", ".value"),
               names_pattern = "(Gaussian|Constrained|ConstrainedNR)_(.*)") %>%
  filter(region %in% rois) %>%
  # drop the regular Constrained rows and rename the NR variant to the simple 'Constrained'
  filter(method != "Constrained") %>%
  mutate(method = recode(method, ConstrainedNR = "Constrained")) %>%
  mutate(method = factor(method, levels = c("Gaussian", "Constrained"))) %>%
  arrange(region, method) %>%
  mutate(slope_rr = exp(slope), lwr_rr = exp(lwr), upr_rr = exp(upr)) %>%
  mutate(region = factor(as.character(region), levels = c("WM", "GM", "RH Precentral", "RH Postcentral")))

p_forest <- ggplot(plot_df, aes(x = slope_rr, y = region, color = method, shape = method, group = method)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_point(position = pd, size = 2) +
  geom_errorbar(aes(xmin = lwr_rr, xmax = upr_rr),
                position = pd, width = 0.2) +
  scale_color_manual(values = c("Gaussian" = brewer.pal(3, "Set1")[2],
                                "Constrained" = brewer.pal(3, "Set1")[3]),
                     breaks = c("Gaussian", "Constrained")) +
  scale_shape_manual(values = c("Gaussian" = 16, "Constrained" = 17),
                     breaks = c("Gaussian", "Constrained")) +
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
  # drop the regular Constrained rows and rename the NR variant to the simple 'Constrained'
  filter(method != "Constrained") %>%
  rename(intercept = intercept) %>%
  mutate(method = recode(method, ConstrainedNR = "Constrained")) %>%
  mutate(method = factor(method, levels = c("Gaussian", "Constrained"))) %>%
  mutate(intercept_resp = exp(intercept), lwr_resp = exp(lwr), upr_resp = exp(upr)) %>%
  mutate(region = factor(as.character(region), levels = c("WM", "GM", "RH Precentral", "RH Postcentral")))

p_forest_intercept <- ggplot(intercept_df, aes(x = intercept_resp, y = region, color = method, shape = method, group = method)) +
  geom_point(position = pd, size = 2) +
  geom_errorbar(aes(xmin = lwr_resp, xmax = upr_resp),
                position = pd, width = 0.2) +
  scale_color_manual(values = c("Gaussian" = brewer.pal(3, "Set1")[2],
                                "Constrained" = brewer.pal(3, "Set1")[3]),
                     breaks = c("Gaussian", "Constrained")) +
  scale_shape_manual(values = c("Gaussian" = 16, "Constrained" = 17),
                     breaks = c("Gaussian", "Constrained")) +
  scale_x_continuous(limits = c(0, NA)) +
  labs(x = "Expected Count at Baseline",
       y = NULL, color = "Method", shape = "Method") +
  theme_minimal(base_size = 12)


#---------------------------------------------------------------------
# Forest plots for differences vs Gaussian (ratio-of-rate-ratios)
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
  # keep only the NR (no-resample) differences and rename
  filter(method != "Constrained") %>%
  mutate(method = recode(method, ConstrainedNR = "Constrained")) %>%
  mutate(method = factor(method, levels = c("Constrained"))) %>%
  mutate(ratio_rr = exp(est), lwr_rr = exp(lwr), upr_rr = exp(upr)) %>%
  mutate(region = factor(as.character(region), levels = c("WM", "GM", "RH Precentral", "RH Postcentral")))

p_forest_slope_diff <- ggplot(slope_diff_df, aes(x = ratio_rr, y = region, color = method, shape = method, group = method)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_point(position = pd, size = 2) +
  geom_errorbar(aes(xmin = lwr_rr, xmax = upr_rr), position = pd, width = 0.2) +
  scale_color_manual(values = c("Constrained" = brewer.pal(3, "Set1")[3])) +
  scale_shape_manual(values = c("Constrained" = 17)) +
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
  # keep only the NR (no-resample) differences and rename
  filter(method != "Constrained") %>%
  mutate(method = recode(method, ConstrainedNR = "Constrained")) %>%
  mutate(method = factor(method, levels = c("Constrained"))) %>%
  mutate(ratio_resp = exp(est), lwr_resp = exp(lwr), upr_resp = exp(upr)) %>%
  mutate(region = factor(as.character(region), levels = c("WM", "GM", "RH Precentral", "RH Postcentral")))

p_forest_intercept_diff <- ggplot(intercept_diff_df, aes(x = ratio_resp, y = region, color = method, shape = method, group = method)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_point(position = pd, size = 2) +
  geom_errorbar(aes(xmin = lwr_resp, xmax = upr_resp), position = pd, width = 0.2) +
  scale_color_manual(values = c("Constrained" = brewer.pal(3, "Set1")[3])) +
  scale_shape_manual(values = c("Constrained" = 17)) +
  scale_x_continuous(breaks = function(x) { rng <- range(x, na.rm = TRUE); base <- pretty(rng); sort(unique(c(base, 1))) }) +
  labs(x = "Ratio of Baseline Expected Counts vs Gaussian", y = NULL, color = "Method", shape = "Method") +
  theme_minimal(base_size = 12)


#---------------------------------------------------------------------
# Save all plots
#---------------------------------------------------------------------
ggsave(filename = file.path(out_dir, "sensory_smoothing_glmm_forest_plot_no_resample.pdf"),
       plot = p_forest, width = 6, height = 4)

ggsave(filename = file.path(out_dir, "sensory_intercept_glmm_forest_plot_no_resample.pdf"),
       plot = p_forest_intercept, width = 6, height = 4)

ggsave(filename = file.path(out_dir, "sensory_combined_forest_plots_no_resample.pdf"),
       plot = p_forest / p_forest_intercept + plot_layout(ncol = 2, guides = "collect"),
       width = 12, height = 4)

ggsave(filename = file.path(out_dir, "sensory_smoothing_glmm_slope_diff_forest_plot_no_resample.pdf"),
       plot = p_forest_slope_diff, width = 6, height = 4)

ggsave(filename = file.path(out_dir, "sensory_intercept_glmm_diff_forest_plot_no_resample.pdf"),
       plot = p_forest_intercept_diff, width = 6, height = 4)
