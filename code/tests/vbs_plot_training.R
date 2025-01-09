# Required Libraries
library(ggplot2)

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

plot <- function(data, title, filename) {
  
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

### Plot all the pilots

data <- import("LR 1e-4\tMOM .9\tBATCH 128\tTR .7
Epoch 1
Validation - Loss: 6.6293, Accuracy: 0.0045
Epoch 2
Validation - Loss: 4.1339, Accuracy: 0.1208
Epoch 3
Validation - Loss: 0.6515, Accuracy: 0.8467
Epoch 4
Validation - Loss: 0.1157, Accuracy: 0.9703
Epoch 5
Validation - Loss: 0.0438, Accuracy: 0.9893
Epoch 6
Validation - Loss: 0.0249, Accuracy: 0.9936
Epoch 7
Validation - Loss: 0.0129, Accuracy: 0.9971
Epoch 8
Validation - Loss: 0.0080, Accuracy: 0.9982
Epoch 9
Validation - Loss: 0.0067, Accuracy: 0.9984
Epoch 10
Validation - Loss: 0.0045, Accuracy: 0.9991
")
plot(data, "LR 1e-4 - MOM .9 - BATCH 128 - TR .7")


data <- import("LR 1e-4\tMOM .5\tBATCH 64\tTR .7
Epoch 1
Validation - Loss: 6.8817, Accuracy: 0.0014
Epoch 2
Validation - Loss: 6.7080, Accuracy: 0.0026
Epoch 3
Validation - Loss: 6.1792, Accuracy: 0.0110
Epoch 4
Validation - Loss: 4.8790, Accuracy: 0.0689
Epoch 5
Validation - Loss: 2.8545, Accuracy: 0.3365
Epoch 6
Validation - Loss: 1.1346, Accuracy: 0.7419
Epoch 7
Validation - Loss: 0.4109, Accuracy: 0.9097
Epoch 8
Validation - Loss: 0.1699, Accuracy: 0.9633
Epoch 9
Validation - Loss: 0.0912, Accuracy: 0.9814
Epoch 10
Validation - Loss: 0.0624, Accuracy: 0.9868")
plot(data, "LR 1e-4 MOM .5 BATCH 64 TR .7")



data <- import("LR 1e-4\tMOM .9\tBATCH 64\tTR .7
Epoch 1
Validation - Loss: 5.1268, Accuracy: 0.0405
Epoch 2
Validation - Loss: 0.2161, Accuracy: 0.9452
Epoch 3
Validation - Loss: 0.0260, Accuracy: 0.9937
Epoch 4
Validation - Loss: 0.0139, Accuracy: 0.9961
Epoch 5
Validation - Loss: 0.0079, Accuracy: 0.9979
Epoch 6
Validation - Loss: 0.0056, Accuracy: 0.9984
Epoch 7
Validation - Loss: 0.0031, Accuracy: 0.9992
Epoch 8
Validation - Loss: 0.0032, Accuracy: 0.9991
Epoch 9
Validation - Loss: 0.0016, Accuracy: 0.9997
Epoch 10
Validation - Loss: 0.0017, Accuracy: 0.9995")
plot(data, "LR 1e-4	MOM .9	BATCH 64	TR .7")



data <- import("LR 1e-3\tMOM .9\tBATCH 64\tTR .7
Epoch 1
Validation - Loss: 6.9087, Accuracy: 0.0010
Epoch 2
Validation - Loss: 6.9088, Accuracy: 0.0009
Epoch 3
Validation - Loss: 6.9087, Accuracy: 0.0009
Epoch 4
Validation - Loss: 6.9086, Accuracy: 0.0009
Epoch 5
Validation - Loss: 6.6751, Accuracy: 0.0020
Epoch 6
Validation - Loss: 0.0314, Accuracy: 0.9900
Epoch 7
Validation - Loss: 0.0031, Accuracy: 0.9990
Epoch 8
Validation - Loss: 0.0015, Accuracy: 0.9995
Epoch 9
Validation - Loss: 0.0005, Accuracy: 0.9999
Epoch 10
Validation - Loss: 0.0003, Accuracy: 0.9999
")
plot(data, "LR 1e-3	MOM .9	BATCH 64	TR .7")



data <- import("LR 1e-4\tMOM .4\tBATCH 64\tTR .7
Epoch 1
Validation - Loss: 6.8919, Accuracy: 0.0010
Epoch 2
Validation - Loss: 6.7895, Accuracy: 0.0019
Epoch 3
Validation - Loss: 6.4849, Accuracy: 0.0064
Epoch 4
Validation - Loss: 5.8097, Accuracy: 0.0227
Epoch 5
Validation - Loss: 4.3495, Accuracy: 0.1145
Epoch 6
Validation - Loss: 2.6652, Accuracy: 0.3831
Epoch 7
Validation - Loss: 1.2227, Accuracy: 0.7189
Epoch 8
Validation - Loss: 0.4778, Accuracy: 0.8958
Epoch 9
Validation - Loss: 0.2129, Accuracy: 0.9548
Epoch 10
Validation - Loss: 0.1240, Accuracy: 0.9739
")
plot(data, "LR 1e-4	MOM .4	BATCH 64	TR .7")




