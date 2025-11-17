library(dplyr)
library(ggplot2)
library(rstatix)    # for pairwise_t_test()
library(ggpubr)     # for stat_pvalue_manual()
library(RColorBrewer)
library(glmmTMB)

fn <- "~/Box Sync/Aizenberg_Documents/Papers/csmooth/results/dice.csv"
dice <- read.csv(fn, header = TRUE, stringsAsFactors = FALSE)
# Contains methods "gaussian" and "constrained"
dice$SmoothingMethod <- factor(dice$SmoothingMethod)
dice$SmoothingMethod <- relevel(dice$SmoothingMethod, ref = "gaussian")

# Copy "SmoothingMethod" to "method" for modeling
dice$method <- dice$SmoothingMethod


# 2) STATISTICAL TESTS ------------------------------------------------------
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

# Fit and return a tidy summary including intercepts and slopes
fit_model <- function(data) {
  # Fit GLMM with random intercepts for Subject
  m <- glmmTMB(
    Dice ~ FWHM * method + (1 | Subject),
    data = data
  )
  
  # Extract fixed effects and covariance
  beta <- fixef(m)$cond
  V    <- as.matrix(vcov(m)$cond)

  # Compute Gaussian (reference) slope: coef on 'FWHM'
  g_slope <- lincom(beta, V, c("FWHM" = 1.0))
  # Compute Constrained slope: FWHM + FWHM:methodconstrained (if present)
  if ("FWHM:methodconstrained" %in% names(beta)) {
    c_slope <- lincom(beta, V, c("FWHM" = 1.0, "FWHM:methodconstrained" = 1.0))
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

  # Differences vs Gaussian (interaction or method coefficients alone)
  c_slope_diff <- if ("FWHM:methodconstrained" %in% names(beta)) lincom(beta, V, c("FWHM:methodconstrained" = 1)) else list(est = NA, lwr = NA, upr = NA, p = NA)
  c_intercept_diff <- if ("methodconstrained" %in% names(beta)) lincom(beta, V, c("methodconstrained" = 1)) else list(est = NA, lwr = NA, upr = NA, p = NA)

  tibble(
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
    constrained_p   = c_slope$p,

    # Differences vs Gaussian
    slope_diff_constrained = c_slope_diff$est,
    slope_diff_constrained_lwr = c_slope_diff$lwr,
    slope_diff_constrained_upr = c_slope_diff$upr,
    slope_diff_constrained_p   = c_slope_diff$p,
    intercept_diff_constrained = c_intercept_diff$est,
    intercept_diff_constrained_lwr = c_intercept_diff$lwr,
    intercept_diff_constrained_upr = c_intercept_diff$upr,
    intercept_diff_constrained_p   = c_intercept_diff$p,
  )
}

glmm_results <- fit_model(dice)
print(glmm_results)
# Save results to CSV
write.csv(glmm_results, "~/Box Sync/Aizenberg_Documents/Papers/csmooth/figures/sensory_task_activation_reliability_by_smoothing_method_glmm_results.csv",
row.names = FALSE)

# 3) PREPARE DATA FOR BOXPLOT ----------------------------------------------------------

boxplot_data <- dice %>%
  filter(FWHM %in% c(0,3,6,9,12)) %>%
  # recode the 0 mm rows to "no smoothing"
  mutate(
    SmoothingMethod = if_else(FWHM == 0, "no smoothing", as.character(SmoothingMethod)),
    SmoothingMethod = factor(SmoothingMethod,
                             levels = c("no smoothing", "gaussian", "constrained"))
  )

# 4) THE FINAL PLOT ------------------------------------------------------

ggplot(boxplot_data,
       aes(x = factor(FWHM),
           y = Dice,
           fill = SmoothingMethod)) +
  
  # boxplots
  geom_boxplot(position = position_dodge(width = 0.8),
               width = 0.7) +
  
  # force "no smoothing" = gray, + two colors from Set1
  scale_fill_manual(values = c(
    "no smoothing" = "gray80",
    "gaussian"     = brewer.pal(3, "Set1")[2],
    "constrained"  = brewer.pal(3, "Set1")[3]
  )) +
  
  labs(
    x = "FWHM (mm)",
    y = "Dice Similarity Coefficient",
    fill = "Smoothing Method"
  ) +
  theme_minimal()

# save to file
ggsave("~/Box Sync/Aizenberg_Documents/Papers/csmooth/figures/sensory_task_activation_reliability_by_smoothing_method_boxplot.pdf", width = 6, height = 4)