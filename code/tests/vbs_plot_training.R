# Required Libraries
library(ggplot2)
library(dplyr)


import <- function(text) {
  
  # Split the text into lines
  lines <- strsplit(text, "\n")[[1]]
  
  # Remove empty lines
  lines <- lines[lines != ""]
  
  # Filter out Training lines
  filtered_lines <- lines[!grepl("^Training", lines)]
  
  # Parse validation data
  data <- data.frame(
    Epoch = numeric(),
    Loss = numeric(),
    Accuracy = numeric()
  )
  
  for (i in seq_along(filtered_lines)) {
    line <- filtered_lines[i]
    if (grepl("^Epoch", line)) {
      epoch <- as.numeric(gsub("Epoch (\\d+)", "\\1", line))
    } else if (grepl("^Validation", line)) {
      loss <- as.numeric(gsub(".*Loss: ([0-9.]+),.*", "\\1", line))
      accuracy <- as.numeric(gsub(".*Accuracy: ([0-9.]+)", "\\1", line))
      data <- rbind(data, data.frame(Epoch = epoch, Loss = loss, Accuracy = accuracy))
    }
  }
  # Print the organized table
  print(data)
}


import_csv <- function(file_path) {
  # Read the CSV file
  data <- read.csv(file_path)
  
  # Select the columns of interest: Epoch, Val_Loss, Val_Accuracy
  result <- data.frame(
    Epoch = data$Epoch,
    Loss = data$Val_Loss,
    Accuracy = data$Val_Accuracy
  )
  
  # Print the organized table
  print(result)
  
  # Optionally return the table if further processing is needed
  return(result)
}


plot <- function(data, title) {
  
  savename <- paste0(title, ".png")

  
  # Create the plot
  ggplot(data, aes(x = Epoch)) +
    # Plot Loss on the primary y-axis
    geom_line(aes(y = Loss, color = "Loss"), size = 1) +
    # Plot Accuracy on the secondary y-axis (scaled)
    geom_line(aes(y = Accuracy * max(data$Loss) / max(data$Accuracy), color = "Accuracy"), size = 1) +
    # Customize scales and labels
    scale_y_continuous(
      name = "Loss",
      sec.axis = sec_axis(~ . * max(data$Accuracy) / max(data$Loss), name = "Accuracy")
    ) +
    scale_x_continuous(
      breaks = 1:10,  # Set x-axis ticks to 1 to 10
      name = "Epoch"
    ) +
    labs(
      title = title,
      x = "Epoch",
      color = "Metric"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position = "bottom"
    )
  
}

totalplot <- function(statsIn) {
  
  custom_colors <- c("BR" = "#FF9E4A", "LN" = "#69B5A2")
  
  # Plot
  ggplot(statsIn, aes(x = Epoch, y = Mean_Accuracy, color = Script, fill = Script)) +
    geom_line(size = 3) +  # Line for mean accuracy
    geom_ribbon(aes(ymin = CI_Low, ymax = CI_High), alpha = 0.2, color = NA) +  # Shaded confidence interval
    scale_x_continuous(breaks = 1:10) +  # Set x-axis ticks from 1 to 10
    scale_color_manual(values = custom_colors) +  # Custom colors for lines
    scale_fill_manual(values = custom_colors) +  # Custom colors for confidence intervals
    labs(x = "Epoch", y = "Accuracy") +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 14),       # Increase x and y tick size
      axis.title = element_text(size = 16),      # Increase x and y label size
      legend.position = "none"                   # Remove legend
    )
  
  ggsave("../../outputs/figures/subs-5_script-both_plot-accuracy.png", width = 8, height = 5, dpi = 300)
   
}

### Plot all the pilots

br1 <- import_csv('../../outputs/logs/training-BR_2025-01-21_13-24-59.csv')
br2 <- import_csv('../../outputs/logs/training-BR_2025-01-25_17-50-05.csv')
br3 <- import_csv('../../outputs/logs/training-BR_2025-01-28_21-03-23.csv')
br4 <- import_csv('../../outputs/logs/training-BR_2025-01-31_08-10-18.csv')
br5 <- import_csv('../../outputs/logs/training-BR_2025-02-02_17-55-44.csv')

ln1 <- import_csv('../../outputs/logs/training-LN_2025-01-22_20-47-02.csv')
ln2 <- import_csv('../../outputs/logs/training-LN_2025-01-24_13-31-10.csv')
ln3 <- import_csv('../../outputs/logs/training-LN_2025-01-27_14-37-52.csv')
ln4 <- import_csv('../../outputs/logs/training-LN_2025-01-30_02-12-52.csv')
ln5 <- import_csv('../../outputs/logs/training-LN_2025-02-01_12-03-44.csv')

br <- rbind(br1, br2, br3, br4, br5)
br$Script <- "BR"

ln <- rbind(ln1, ln2, ln3, ln4, ln5)
ln$Script <- "LN"

learning <- rbind(br,ln)

learning_stats <- learning %>% group_by(Epoch, Script) %>%
                               summarise(Mean_Accuracy = mean(Accuracy),
                                         SD = sd(Accuracy),
                                         N = n(),  # Number of observations per group
                                         SE = SD / sqrt(N),  # Standard error
                                         CI_Low = Mean_Accuracy - 1.96 * SE,  # 95% CI lower bound
                                         CI_High = Mean_Accuracy + 1.96 * SE  # 95% CI upper bound
                                         )

allplot(learning_stats)




