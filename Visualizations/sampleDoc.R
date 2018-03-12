library(dplyr)
library(readr)

# Read in the data
dataDoc <- read_csv("docTSNE.csv")

# Stratify the data
stratifiedDocs <- dataDoc %>% 
  group_by(group) %>% 
  sample_frac(size = 0.25)

# Count the docs
stratifiedDocs %>% 
  group_by(group) %>% 
  summarise(n = n())

# Write the stratified data
write_csv(stratifiedDocs, "stratifiedDocsTSNE.csv")
