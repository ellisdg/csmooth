library(dplyr)
library(ggplot2)
library(rstatix)    # for pairwise_t_test()
library(ggpubr)     # for stat_pvalue_manual()
library(RColorBrewer)
library(glmmTMB)
library(emmeans)

fn <- "/Users/david.ellis/Library/CloudStorage/Box-Box/Aizenberg_Documents/Papers/csmooth/results/dice_scores_task-lefthand.csv"
dice <- read.csv(fn, header = TRUE, stringsAsFactors = FALSE)
# Rename legacy constrained label to the resampling variant and set factor order with NR first
DiceMethod <- dplyr::recode(as.character(dice$SmoothingMethod),
                            "constrained" = "constrained_rs",
                            .default = as.character(dice$SmoothingMethod))
dice$SmoothingMethod <- factor(DiceMethod,
                               levels = c("gaussian", "constrained_nr", "constrained_rs"))
dice$SmoothingMethod <- relevel(dice$SmoothingMethod, ref = "gaussian")
# model uses this factor directly
DiceMethod <- NULL
dice$method <- dice$SmoothingMethod

# 1) PREPARE DATA FOR BOXPLOT ----------------------------------------------------------

boxplot_data <- dice %>%
  filter(FWHM %in% c(0,3,6,9,12)) %>%
  # recode the 0 mm rows to "no smoothing" and relabel methods for display
  mutate(
    SmoothingMethod = if_else(FWHM == 0, "no smoothing", as.character(SmoothingMethod)),
    SmoothingMethod = dplyr::recode(SmoothingMethod,
                                    "gaussian" = "Gaussian",
                                    "constrained_rs" = "Constrained-RS",
                                    "constrained_nr" = "Constrained-NR",
                                    .default = SmoothingMethod),
    SmoothingMethod = factor(SmoothingMethod,
                             levels = c("no smoothing", "Gaussian", "Constrained-NR", "Constrained-RS"))
  )
# 2) STATISTICAL TESTS ------------------------------------------------------
# Use emmeans/emtrends to get intercepts (at FWHM=0) and slopes and contrasts vs gaussian
fit_model <- function(data) {
  # Fit GLMM with random intercepts for Subject
  m <- glmmTMB(
    Dice ~ FWHM * method + (1 | Subject),
    data = data
  )

  # Estimated intercepts (FWHM = 0) and slopes by method
  intercept_emm <- emmeans(m, ~ method, at = list(FWHM = 0))
  slope_emm     <- emtrends(m, ~ method, var = "FWHM")

  # Summaries (estimate, SE, CIs, p-values)
  i_df <- as.data.frame(summary(intercept_emm, infer = c(TRUE, TRUE)))
  s_df <- as.data.frame(summary(slope_emm, infer = c(TRUE, TRUE)))

  # Contrasts vs Gaussian (treatment vs control where gaussian is control)
  i_contr <- as.data.frame(contrast(intercept_emm, method = "trt.vs.ctrl", ref = "gaussian", infer = c(TRUE, TRUE)))
  s_contr <- as.data.frame(contrast(slope_emm,     method = "trt.vs.ctrl", ref = "gaussian", infer = c(TRUE, TRUE)))

  # Helper to pick rows for a level
  pick_stat <- function(df, lvl) {
    row <- df[df$method == lvl, ]
    if (nrow(row) == 0) return(list(est = NA, lwr = NA, upr = NA, p = NA))
    val_col   <- dplyr::case_when("emmean" %in% names(df) ~ "emmean",
                                  "emtrend" %in% names(df) ~ "emtrend",
                                  "estimate" %in% names(df) ~ "estimate",
                                  "FWHM.trend" %in% names(df) ~ "FWHM.trend",
                                  TRUE ~ NA_character_)
    lower_col <- if ("lower.CL" %in% names(df)) "lower.CL" else if ("asymp.LCL" %in% names(df)) "asymp.LCL" else NA_character_
    upper_col <- if ("upper.CL" %in% names(df)) "upper.CL" else if ("asymp.UCL" %in% names(df)) "asymp.UCL" else NA_character_
    p_col     <- if ("p.value" %in% names(df)) "p.value" else if ("Pr(>|t|)" %in% names(df)) "Pr(>|t|)" else NA_character_
    list(
      est = if (!is.na(val_col)) row[[val_col]] else NA,
      lwr = if (!is.na(lower_col)) row[[lower_col]] else NA,
      upr = if (!is.na(upper_col)) row[[upper_col]] else NA,
      p   = if (!is.na(p_col)) row[[p_col]] else NA
    )
  }
  pick_trt_contrast <- function(df, lvl) {
    # contrast labels are like "constrained_nr - gaussian"
    lab <- paste0(lvl, " - gaussian")
    row <- df[df$contrast == lab | df$contrast == paste0(lvl, " - ", "gaussian"), ]
    if (nrow(row) == 0) return(list(est = NA, lwr = NA, upr = NA, p = NA))
    list(est = row$estimate, lwr = row$lower.CL, upr = row$upper.CL, p = row$p.value)
  }

  # pull stats for each method (levels should include gaussian, constrained_nr, constrained_rs)
  g_int  <- pick_stat(i_df, "gaussian")
  nr_int <- pick_stat(i_df, "constrained_nr")
  rs_int <- pick_stat(i_df, "constrained_rs")

  g_slope  <- pick_stat(s_df, "gaussian")
  nr_slope <- pick_stat(s_df, "constrained_nr")
  rs_slope <- pick_stat(s_df, "constrained_rs")

  nr_int_diff <- pick_trt_contrast(i_contr, "constrained_nr")
  rs_int_diff <- pick_trt_contrast(i_contr, "constrained_rs")
  nr_slope_diff <- pick_trt_contrast(s_contr, "constrained_nr")
  rs_slope_diff <- pick_trt_contrast(s_contr, "constrained_rs")

  tibble(
    gaussian_intercept = g_int$est,
    gaussian_intercept_lwr = g_int$lwr,
    gaussian_intercept_upr = g_int$upr,
    gaussian_intercept_p   = g_int$p,

    constrained_nr_intercept = nr_int$est,
    constrained_nr_intercept_lwr = nr_int$lwr,
    constrained_nr_intercept_upr = nr_int$upr,
    constrained_nr_intercept_p   = nr_int$p,

    constrained_rs_intercept = rs_int$est,
    constrained_rs_intercept_lwr = rs_int$lwr,
    constrained_rs_intercept_upr = rs_int$upr,
    constrained_rs_intercept_p   = rs_int$p,

    gaussian_slope = g_slope$est,
    gaussian_lwr = g_slope$lwr,
    gaussian_upr = g_slope$upr,
    gaussian_p   = g_slope$p,

    constrained_nr_slope = nr_slope$est,
    constrained_nr_lwr = nr_slope$lwr,
    constrained_nr_upr = nr_slope$upr,
    constrained_nr_p   = nr_slope$p,

    constrained_rs_slope = rs_slope$est,
    constrained_rs_lwr = rs_slope$lwr,
    constrained_rs_upr = rs_slope$upr,
    constrained_rs_p   = rs_slope$p,

    # Differences vs Gaussian (contrasts from emmeans)
    slope_diff_constrained_nr = nr_slope_diff$est,
    slope_diff_constrained_nr_lwr = nr_slope_diff$lwr,
    slope_diff_constrained_nr_upr = nr_slope_diff$upr,
    slope_diff_constrained_nr_p   = nr_slope_diff$p,
    intercept_diff_constrained_nr = nr_int_diff$est,
    intercept_diff_constrained_nr_lwr = nr_int_diff$lwr,
    intercept_diff_constrained_nr_upr = nr_int_diff$upr,
    intercept_diff_constrained_nr_p   = nr_int_diff$p,

    slope_diff_constrained_rs = rs_slope_diff$est,
    slope_diff_constrained_rs_lwr = rs_slope_diff$lwr,
    slope_diff_constrained_rs_upr = rs_slope_diff$upr,
    slope_diff_constrained_rs_p   = rs_slope_diff$p,
    intercept_diff_constrained_rs = rs_int_diff$est,
    intercept_diff_constrained_rs_lwr = rs_int_diff$lwr,
    intercept_diff_constrained_rs_upr = rs_int_diff$upr,
    intercept_diff_constrained_rs_p   = rs_int_diff$p
  )
}

glmm_results <- fit_model(dice)
print(glmm_results)
# Save results to CSV
write.csv(glmm_results, "/Users/david.ellis/Library/CloudStorage/Box-Box/csmooth_frontiers/figures/ch2/sensory_reliability/sensory_task_activation_reliability_by_smoothing_method_glmm_results.csv",
          row.names = FALSE)

# --- Small parameter plot: intercepts and slopes with 95% CI ------------------
# Prepare a small data frame with intercept and slope estimates (+ 95% CI)
plot_df <- tibble(
  parameter = rep(c("Intercept", "Slope"), each = 3),
  method = factor(rep(c("Gaussian", "Constrained-NR", "Constrained-RS"), times = 2),
                  levels = c("Gaussian", "Constrained-NR", "Constrained-RS")),
  est = c(
    glmm_results$gaussian_intercept,
    glmm_results$constrained_nr_intercept,
    glmm_results$constrained_rs_intercept,
    glmm_results$gaussian_slope,
    glmm_results$constrained_nr_slope,
    glmm_results$constrained_rs_slope
  ),
  lwr = c(
    glmm_results$gaussian_intercept_lwr,
    glmm_results$constrained_nr_intercept_lwr,
    glmm_results$constrained_rs_intercept_lwr,
    glmm_results$gaussian_lwr,
    glmm_results$constrained_nr_lwr,
    glmm_results$constrained_rs_lwr
  ),
  upr = c(
    glmm_results$gaussian_intercept_upr,
    glmm_results$constrained_nr_intercept_upr,
    glmm_results$constrained_rs_intercept_upr,
    glmm_results$gaussian_upr,
    glmm_results$constrained_nr_upr,
    glmm_results$constrained_rs_upr
  )
)

# Define colors consistent with the boxplot
param_colors <- c("Gaussian" = brewer.pal(3, "Set1")[2],
                  "Constrained-NR" = brewer.pal(3, "Set1")[3],
                  "Constrained-RS" = brewer.pal(3, "Set1")[1])
param_shapes <- c("Gaussian" = 16, "Constrained-NR" = 17, "Constrained-RS" = 15)
param_pd <- position_dodge(width = 0.5)

p_params <- ggplot(plot_df, aes(x = method, y = est, color = method, shape = method)) +
  geom_errorbar(aes(ymin = lwr, ymax = upr),
                position = param_pd,
                width = 0.15, size = 0.8) +
  geom_point(position = param_pd, size = 3) +
  facet_wrap(~ parameter, scales = "free_y") +
  scale_color_manual(values = param_colors) +
  scale_shape_manual(values = param_shapes) +
  labs(x = NULL, y = "Estimate (95% CI)") +
  theme_minimal(base_size = 12) +
  theme(strip.text = element_text(face = "bold"),
        axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank())

print(p_params)

# Save the small figure next to the other outputs
ggsave("/Users/david.ellis/Library/CloudStorage/Box-Box/csmooth_frontiers/figures/ch2/sensory_reliability/combined/sensory_task_activation_reliability_by_smoothing_method_parameters.pdf",
       plot = p_params, width = 6, height = 3.5)


# 4) THE FINAL PLOT ------------------------------------------------------

ggplot(boxplot_data,
       aes(x = factor(FWHM, levels = c(0,3,6,9,12)),
           y = Dice,
           fill = SmoothingMethod)) +
  
  # boxplots
  geom_boxplot(position = position_dodge(width = 0.8),
               width = 0.7) +
  
  # force "no smoothing" = gray, + three colors from Set1 for the smoothing methods
  scale_fill_manual(values = c(
    "no smoothing"            = "gray80",
    "Gaussian"                = brewer.pal(3, "Set1")[2],
    "Constrained-NR"          = brewer.pal(3, "Set1")[3],
    "Constrained-RS"          = brewer.pal(3, "Set1")[1]
  )) +
  
  labs(
    x = "FWHM (mm)",
    y = "Dice Coefficient",
    fill = "Smoothing Method"
  ) +
  theme_minimal()

# save combined plot with both constrained variants
combined_out <- "/Users/david.ellis/Library/CloudStorage/Box-Box/csmooth_frontiers/figures/ch2/sensory_reliability/combined/sensory_task_activation_reliability_by_smoothing_method_boxplot.pdf"
ggsave(combined_out, width = 6, height = 4)

# Create a no-resample-only view: drop RS and relabel NR as Constrained
nr_only_data <- boxplot_data %>%
  filter(SmoothingMethod != "Constrained-RS") %>%
  mutate(SmoothingMethod = if_else(SmoothingMethod == "Constrained-NR", "Constrained", SmoothingMethod),
         SmoothingMethod = factor(SmoothingMethod, levels = c("no smoothing", "Gaussian", "Constrained")))

ggplot(nr_only_data,
       aes(x = factor(FWHM, levels = c(0,3,6,9,12)),
           y = Dice,
           fill = SmoothingMethod)) +
  geom_boxplot(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = c(
    "no smoothing" = "gray80",
    "Gaussian"     = brewer.pal(3, "Set1")[2],
    "Constrained"  = brewer.pal(3, "Set1")[3]
  )) +
  labs(x = "FWHM (mm)", y = "Dice Coefficient", fill = "Smoothing Method") +
  theme_minimal()

nr_only_out <- "/Users/david.ellis/Library/CloudStorage/Box-Box/csmooth_frontiers/figures/ch2/sensory_reliability/no_resample_only/sensory_task_activation_reliability_by_smoothing_method_boxplot_no_resample_only.pdf"
ggsave(nr_only_out, width = 6, height = 4)
