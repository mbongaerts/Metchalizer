This repository contains code and datasets used for the article ***Using Out-of-Batch Reference Populations to Improve Untargeted Metabolomics for Screening Inborn Errors of Metabolism*** published in MDPI Metabolites:

Bongaerts, M.; Bonte, R.; Demirdas, S.; Jacobs, E. H.; Oussoren, E.; van der Ploeg, A. T.; Wagenmakers, M. A. E. M.; Hofstra, R. M. W.; Blom, H. J.; Reinders, M. J. T. & Ruijter, G. J. G.
Using Out-of-Batch Reference Populations to Improve Untargeted Metabolomics for Screening Inborn Errors of Metabolism 
Metabolites, MDPI AG, 2020, 11, 8
https://doi.org/10.3390/metabo11010008

# Metchalizer

The Metchalizer package can be used for normalization of metabolomics data. Different normalization methods can be found such as *PQN*, *Best correlated internal standard*, *Anchor* and *Metchalizer*. This package furthermore contains a regression model which can be used to calculate age and sex corrected Z-scores for features/metabolites. See for details 
https://doi.org/10.3390/metabo11010008


# Progenesis
We developed functions/classes to merge batches/exports to a single dataset. Each single batch/dataset should contain pre-processed MS-data where peak picking, peak alignment, peak integration etc. was already performed. The datasets in this repository were processed using Progenesis QI and exported to .csv. The classes/functions work only with these exports. However, one can tranform their own datasets to the same format (see Data/Pos/ or Data/neg/ for the format).

# Data 
The Data directory contains eight batches processed as described by Bonte et al. 2019 (https://doi.org/10.3390/metabo9120289) for both ion modi. Every batch contains control samples and QC samples. Originally, these batches also contained patient samples but were removed for privacy reasons. Metadata for all samples are provided in Data/Sample_metadata.csv .



