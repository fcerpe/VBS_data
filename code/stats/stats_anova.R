
# Add all necessary libraries
library("readxl")
library("ez")

### Support functions 

## Compute anova
compute_anova <- function(infile, outfile, measure) {
  ##
  # Compute a rmANOVA for expertise, script, layer based on clustering or dissimilarity
  # 
  # Parameters
  # ----------
  # infile (str): name of the input file, containing the table to analyse
  # 
  # outfile (str): name of the output file, how to save the anova
  # 
  # measure (str): indication of the measure contained in the table, to set the dv
  # 
  # Outputs
  # -------
  # None, the stat will be saved in outputs/results/[corresponding folder]
  ##
  
  # Based on measure, set folder
  ifelse(measure == 'dissimilarity', folder <- 'distances',
                                     folder <- 'clustering')
  
  # Import data
  net <- read.csv(paste('../../outputs/results', folder, infile, sep = "/"))
  
  # Compute ANOVA
  ifelse(measure == 'dissimilarity',  anova <- ezANOVA(data = net, 
                                                       dv = dissimilarity, 
                                                       wid = sub, 
                                                       within = .(layer, script), 
                                                       between = expertise, 
                                                       type = 3),
                                      anova <- ezANOVA(data = net, 
                                                       dv = clustering, 
                                                       wid = sub, 
                                                       within = .(layer, script), 
                                                       between = expertise,
                                                       type = 3))
  
  # extract the relevant stats
  anova <- anova$ANOVA
  
  # Change to a more inclusive name
  names(anova)[which(names(anova) == "p<.05")] <- "significance"
  
  # Assign asterisks:
  # p < 0.05,  *
  # p < 0.01,  **
  # p < 0.001, ***
  anova$significance <- ifelse(anova$p < 0.001, "***", 
                               ifelse(anova$p < 0.01, "**",  
                                      ifelse(anova$p < 0.05, "*", "ns")))
  
  # Save p-values as character, to ease csv file
  anova$p <- as.character(anova$p)
  
  # Save as csv
  write.csv(anova, 
            paste('../../outputs/results', folder, outfile, sep = "/"), 
            row.names = FALSE)
}


### ----------------------------------------------------------------------------
### ANOVAs

## Clustering

# AlexNet
compute_anova('model-alexnet_training-all_test-VBE_data-clustering-matrices.csv',
              'model-alexnet_training-all_test-VBE_data-clustering_analysis-anova-clusters.csv',
              'clustering')
  
# CORnet 
compute_anova('model-cornet_training-all_test-VBE_data-clustering-matrices.csv',
              'model-cornet_training-all_test-VBE_data-clustering_analysis-anova-clustering.csv',
              'clustering')


## Dissimilarity

# TBD

# # AlexNet  
# compute_anova('model-alexnet_training-all_test-VBE_data-clustering-matrices.csv',
#               'model-alexnet_training-all_test-VBE_data-clustering_analysis-anova-dissimilarity_method-euclidean.csv',
#               'dissimilarity')
# 
# # CORnet 
# compute_anova('model-cornet_training-all_test-VBE_data-clustering-matrices.csv',
#               'model-cornet_training-all_test-VBE_data-clustering_analysis-anova-dissimilarity_method-euclidean.csv',
#               'dissimilarity')









