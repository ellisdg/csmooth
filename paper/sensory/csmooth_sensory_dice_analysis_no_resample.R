library(dplyr)
library(ggplot2)
library(rstatix)    # for pairwise_t_test()
library(ggpubr)     # for stat_pvalue_manual()
library(RColorBrewer)

fn <- "~/Box Sync/Aizenberg_Documents/Papers/csmooth/results/dice_scores_task-lefthand.csv"
dice <- read.csv(fn, header = TRUE, stringsAsFactors = FALSE)
dice$SmoothingMethod <- factor(dice$SmoothingMethod)
dice$SmoothingMethod <- relevel(dice$SmoothingMethod, ref = "gaussian")
dice$SmoothingMethod2 <- with(dice,
                              ifelse(FWHM == 0, "none", as.character(SmoothingMethod))
)
dice$SmoothingMethod2 <- factor(dice$SmoothingMethod2,
                                levels = c("none", "gaussian", "constrained", "constrained_nr")
)

# 1) PREPARE DATA FOR BOXPLOT ----------------------------------------------------------

boxplot_data <- dice %>%
  filter(FWHM %in% c(0,3,6,9,12)) %>%
  # recode the 0 mm rows to "no smoothing" and relabel methods for display
  mutate(
    SmoothingMethod = if_else(FWHM == 0, "no smoothing", as.character(SmoothingMethod)),
    SmoothingMethod = dplyr::recode(SmoothingMethod,
                                    "gaussian" = "Gaussian",
                                    "constrained" = "Constrained",
                                    "constrained_nr" = "Constrained-NoResample",
                                    .default = SmoothingMethod),
    SmoothingMethod = factor(SmoothingMethod,
                             levels = c("no smoothing", "Gaussian", "Constrained", "Constrained-NoResample"))
  )

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
    "Constrained"             = brewer.pal(3, "Set1")[3],
    "Constrained-NoResample"  = brewer.pal(3, "Set1")[1]
  )) +
  
  labs(
    x = "FWHM (mm)",
    y = "Dice Coefficient",
    fill = "Smoothing Method"
  ) +
  theme_minimal()

# save to file
ggsave("~/Box Sync/Aizenberg_Documents/Papers/csmooth/figures/no_resample/sensory_task_activation_reliability_by_smoothing_method_boxplot.pdf", width = 6, height = 4)