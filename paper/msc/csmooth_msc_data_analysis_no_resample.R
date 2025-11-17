library(dplyr)
library(ggplot2)
library(ggpubr)     # for stat_pvalue_manual()
library(RColorBrewer)
library(tidyr)
library(broom)      # for tidy()

# TODO: ignore constrained_nr and remove from plots and stats
# Load both datasets
fn1 <- "~/Box Sync/Aizenberg_Documents/Papers/csmooth/results/motor_stats.csv"
motor_stats_df <- read.csv(fn1, header = TRUE, stringsAsFactors = FALSE)

fn2 <- "~/Box Sync/Aizenberg_Documents/Papers/csmooth/results/motor_stats_single_run.csv"
motor_stats_single_run_df <- read.csv(fn2, header = TRUE, stringsAsFactors = FALSE)

# Function to perform paired t-tests
perform_pairwise_tests <- function(data) {
  
  # Filter data for analysis
  test_data <- data %>%
    filter(fwhm %in% c(0,3,6,9,12)) %>%
    mutate(
      method = if_else(fwhm == 0, "no smoothing", as.character(method)),
      method = factor(method, levels = c("no smoothing", "gaussian", "constrained", "constrained_nr"))
    )
  
  # Define the comparisons we want to make
  comparisons <- list()
  metrics <- c("dice", "pearson_r", "mae")
  fwhm_levels <- c(0, 3, 6, 9, 12)
  
  for (metric in metrics) {
    for (fwhm_val in fwhm_levels) {
      
      # Get data for this FWHM and metric
      subset_data <- test_data %>% filter(fwhm == fwhm_val)
      
      if (fwhm_val == 0) {
        # For FWHM = 0, we only have "no smoothing", so skip comparisons
        next
      } else {
        # Compare gaussian vs constrained at this FWHM
        gaussian_data <- subset_data %>% filter(method == "gaussian") %>% pull(!!sym(metric))
        constrained_data <- subset_data %>% filter(method == "constrained") %>% pull(!!sym(metric))
        
        if (length(gaussian_data) > 0 && length(constrained_data) > 0) {
          test_result <- t.test(gaussian_data, constrained_data, paired = TRUE)
          comparisons <- append(comparisons, list(data.frame(
            metric = metric,
            fwhm = fwhm_val,
            group1 = "gaussian",
            group2 = "constrained",
            t = test_result$statistic,
            p = test_result$p.value,
            p.adj = NA,  # Will adjust later
            p.signif = case_when(
              test_result$p.value < 0.001 ~ "***",
              test_result$p.value < 0.01 ~ "**",
              test_result$p.value < 0.05 ~ "*",
              TRUE ~ "ns"
            )
          )))
        }
        
        # Compare each method vs no smoothing
        no_smooth_data <- test_data %>% filter(fwhm == 0, method == "no smoothing") %>% pull(!!sym(metric))
        
        for (method_name in c("gaussian", "constrained")) {
          method_data <- subset_data %>% filter(method == method_name) %>% pull(!!sym(metric))
          
          if (length(method_data) > 0 && length(no_smooth_data) > 0) {
            test_result <- t.test(method_data, no_smooth_data, paired = TRUE)
            comparisons <- append(comparisons, list(data.frame(
              metric = metric,
              fwhm = fwhm_val,
              group1 = method_name,
              group2 = "no smoothing",
              t = test_result$statistic,
              p = test_result$p.value,
              p.adj = NA,  # Will adjust later
              p.signif = case_when(
                test_result$p.value < 0.001 ~ "***",
                test_result$p.value < 0.01 ~ "**",
                test_result$p.value < 0.05 ~ "*",
                TRUE ~ "ns"
              )
            )))
          }
        }
      }
    }
  }
  
  # Combine all comparisons
  all_comparisons <- do.call(rbind, comparisons)
  
  # Adjust p-values within each metric using Bonferroni correction
  all_comparisons <- all_comparisons %>%
    group_by(metric) %>%
    mutate(
      p.adj = p.adjust(p, method = "bonferroni"),
      p.adj.signif = case_when(
        p.adj < 0.001 ~ "***",
        p.adj < 0.01 ~ "**",
        p.adj < 0.05 ~ "*",
        TRUE ~ "ns"
      )
    ) %>%
    ungroup()
  
  return(all_comparisons)
}

# Function to create motor stats plot with significance
create_motor_plot_with_stats <- function(data) {
  
  # Perform statistical tests
  stat_results <- perform_pairwise_tests(data)
  
  # 1) Filter and prepare data for plotting -------------------------------
  
  plot_data <- data %>%
    filter(fwhm %in% c(0,3,6,9,12)) %>%
    # recode the 0 mm rows to "no smoothing"
    mutate(
      method = if_else(fwhm == 0, "no smoothing", as.character(method)),
      method = factor(method,
                      levels = c("no smoothing", "gaussian", "constrained", "constrained_nr"))
    ) %>%
    # Reshape data for faceting
    pivot_longer(
      cols = c(dice, mae, pearson_r),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric = factor(metric, 
                      levels = c("dice",  "pearson_r", "mae"),
                      labels = c("Dice Coefficient", "Correlation", "Mean Absolute Error"))
    )
  
  # Prepare stat results for plotting
  stat_plot_data <- stat_results %>%
    filter(p.adj.signif != "ns") %>%  # Only show significant results
    mutate(
      metric_label = case_when(
        metric == "dice" ~ "Dice Coefficient",
        metric == "pearson_r" ~ "Correlation", 
        metric == "mae" ~ "Mean Absolute Error"
      ),
      metric_label = factor(metric_label, 
                           levels = c("Dice Coefficient", "Correlation", "Mean Absolute Error"))
    )
  
  # Calculate y positions for significance markers
  y_positions <- plot_data %>%
    group_by(metric) %>%
    summarise(
      max_val = max(value, na.rm = TRUE),
      min_val = min(value, na.rm = TRUE),
      range = max_val - min_val,
      .groups = 'drop'
    ) %>%
    mutate(
      y_pos_base = max_val + 0.05 * range
    )
  
  # Add y positions to stat data
  stat_plot_data <- stat_plot_data %>%
    left_join(y_positions, by = c("metric_label" = "metric")) %>%
    group_by(metric_label, fwhm) %>%
    mutate(
      y_pos = y_pos_base + (row_number() - 1) * 0.05 * range
    ) %>%
    ungroup()
  
  # 2) THE FINAL PLOT ------------------------------------------------------
  
  p <- ggplot(plot_data,
         aes(x = factor(fwhm),
             y = value,
             fill = method)) +
    
    # box plots
    geom_boxplot(position = position_dodge(width = 0.8),
                 width = 0.7) +
    
    # Create subplots
    facet_wrap(~ metric, scales = "free_y", ncol = 3) +
    
    # force "no smoothing" = gray, + two colors from Set1
    scale_fill_manual(values = c(
      "no smoothing" = "gray80",
      "gaussian"     = brewer.pal(3, "Set1")[2],
      "constrained"  = brewer.pal(3, "Set1")[3],
      "constrained_nr" = brewer.pal(3, "Set1")[1]
    )) +
    
    labs(
      x = "FWHM (mm)",
      y = "",
      fill = "Smoothing Method"
    ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 12, face = "bold"),
      axis.text.x = element_text(angle = 0, hjust = 0.5)
    )
  
  # Add significance annotations
#  if (nrow(stat_plot_data) > 0) {
#    p <- p + 
#      geom_text(data = stat_plot_data,
#                aes(x = factor(fwhm), y = y_pos, label = p.adj.signif),
#                inherit.aes = FALSE,
#                size = 3, vjust = 0.5)
#  }
  
  return(p)
}

plot_motor <- create_motor_plot_with_stats(motor_stats_df)
print(plot_motor)

# save to file
ggsave("~/Box Sync/Aizenberg_Documents/Papers/csmooth/figures/msc_motor_activation_accuracy_by_smoothing_method_no_resample.pdf", width = 10, height = 4)

#plot_motor_single_run <- create_motor_plot_with_stats(motor_stats_single_run_df)
#print(plot_motor_single_run)
# save to file
#ggsave("~/Box Sync/Aizenberg_Documents/Papers/csmooth/figures/msc_motor_activation_accuracy_by_smoothing_method_single_run.png", width = 10, height = 4, dpi = 300)
#ggsave("~/Box Sync/Aizenberg_Documents/Papers/csmooth/figures/msc_motor_activation_accuracy_by_smoothing_method_single_run.pdf", width = 10, height = 4)

# Print statistical results
cat("Statistical Test Results:\n")
stat_results_main <- perform_pairwise_tests(motor_stats_df)
print(stat_results_main)
# save to csv
write.csv(stat_results_main, "~/Box Sync/Aizenberg_Documents/Papers/csmooth/results/msc_motor_stats_pairwise_tests.csv", row.names = FALSE)

#cat("\nSingle Run Statistical Test Results:\n")
#stat_results_single <- perform_pairwise_tests(motor_stats_single_run_df)
#print(stat_results_single)