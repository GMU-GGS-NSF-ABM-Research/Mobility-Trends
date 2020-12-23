# Mobility-Stats

A collection of python scripts that uses Safe-Graphs social distancing data to calculate the decrease in median-non-home-swell-time from 2019 to 2020 in response to COVID-19.

## Usage

- `Setup.py`

This should be the first script that is run, it checks that the user has downloaded the Safe-Graph data and has it in the current working space. This script also calls `aggregate_stay_at_home_data.py` to aggregate the Safe-Graph data from Census Block Groups into county level for easier analysis. The script also calls `difference.py` to calculate the difference in non-home-dwell-time of 2020 compared to the average of 2019.  

- `PCA.py`

This script will perform a principal component analysis on the time-series outputs from the aggregated county level data and write the outputs to an output shapefile for mapping. Initially performs a PCA, then removes any counties that are outliers and performs a second PCA on the data without outliers. From there K-means and Hierarchical clustering are done on the resulting principal components. These clusters are also stored in the output shapefile.  

- `3 value time series.py`

This script is a naive approach to what `PCA.py` is attempting to do by describing the reduction in non-home-dwell-time by three simple values. A "pre-lockdown" period, "initial-lockdown" period, and a "post-lockdown" period are described for each county in the dataset and can be sampled and plotted or can be built upon. 


- `utils.py, aggregate_stay_at_home_data.py, and difference.py`

These are all essentially utilities that help with the analysis carried out above. 
