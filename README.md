# Mobility-Stats

A collection of python scripts that uses Safe-Graphs social distancing data to calculate the decrease in median-non-home-swell-time from 2019 to 2020 in response to COVID-19.

All of the Python packages that were used for this analysis are listed in `requirements.txt`. It is reccomened to use anaconda as a programming environment. The following command will create a new conda env from that file.

`conda create --name <Mobility_Stats> --file <requirements.txt>`

## Required Data
Safe-Graph data is used for this analysis and can be acquired [here](https://www.safegraph.com/covid-19-data-consortium). 

All data must be stored in `../safegraph-data/` and you will need to download the following two datasets.

* Social Distancing Metrics v2.1 (formerly Physical Distancing Metrics)
    - Saved as `../safegraph_social_distancing_metrics/`
* Open Census Data
    - Saved as `../safegraph_open_census_data/`

The script `update_safegraph_data.py` is provided to help keep up to date with any new release of data from Safe-Graph. Running this file will use the AWS CLI profile set up during initial data download to see if any new data has been released and it will both parse and add those new days to the dataset.

A quick aside, the Open-Census-Data provided by Safe-Graph does not include a FIPS code for Oglala Lakota County, SD so you must edit the cbg_fips_codes.csv in the metadata for the census data. The entry should be as follows: state = SD, state_fips = 46, county_fips = 102, county = Oglala Lakota County, class_code = H1

Two additional datasets are required for mapping results, one county level shapefile that contains a FIPS column with county and state FIPS along with a state level shapefile that contains a FIPS column with only the state FIPS. 
* Must be stored in `../Data/Base Shape Files/`

## Usage

- `setup.py`

This should be the first script that is run, it checks that the user has downloaded the Safe-Graph data and has it in the current working space. This script also calls two functions from `utils.py`, *aggregate_stay_at_home_data* and *calculate_mobility_difference*. The first of which aggregates the Safe-Graph data from Census Block Groups into county level for easier analysis. The second calculates the difference in non-home-dwell-time of 2020 compared to the average of 2019.  

- `main_analysis.py`

This script will perform a principal component analysis on the time-series outputs from the aggregated county level data and write the outputs to an output shapefile for mapping. Initially performs a PCA, then removes any counties that are outliers and performs a second PCA on the data without outliers. From there K-means and Hierarchical clustering are done on the resulting principal components. These clusters are also stored in the output shapefile.  

- `3 value time series.py`

This script is a naive approach to what `PCA.py` is attempting to do by describing the reduction in non-home-dwell-time by three simple values. A "pre-lockdown" period, "initial-lockdown" period, and a "post-lockdown" period are described for each county in the dataset and can be sampled and plotted or can be built upon. 


- `utils.py`

These are all function that help to manipulate the datasets, wither through pre-processing or reading in the data for analysis. 



