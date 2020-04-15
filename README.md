# Metchalizer
The Metchalizer package can be used for normalization of metabolomics data. Different normalization methods can be found such as *PQN*, *Best correlated internal standard*, *Anchor* and *Metchalizer*. This package furthermore contains a regression model which can be used to calculate age and sex corrected Z-scores for features/metabolites.

# Progenesis
We developed functions/classes to merge batches/exports to a single dataset. Each single batch/dataset should contain pre-processed MS-data where peak picking, peak alignment, peak integration etc. was already performed. The datasets in this repository were processed using Progenesis QI and exported to .csv. The classes/functions work only with these exports. However, one can tranform their own datasets to the same format (see Data/Pos/ or Data/neg/ for the format).

#Data 
The Data directory contains eight batches processed as described by Bonte et al. 2019 (https://doi.org/10.3390/metabo9120289) for both ion modi. Every batch contains control samples and QC samples. Originally, these batches also contained patient samples but were removed for privacy reasons. 



