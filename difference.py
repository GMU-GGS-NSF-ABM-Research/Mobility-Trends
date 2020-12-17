import pandas as pd
import os

# I used this just to see when the first epiweek in 2020 was, we only have data up until 2020-10-10 which is the end
# of EpiWeek 41

# from epiweeks import Week, Year
# from datetime import date
# print(Week(2020, 1, 'cdc').startdate(), Week.fromdate(date(2020, 10, 10)))


cwd = os.path.dirname(os.path.abspath(__file__))

save_path = os.path.join(cwd, 'Data', 'mobility')
data_path = os.path.join(cwd, 'safegraph-data', 'aggregated-patterns')
pop = os.path.join(data_path, '2020', '10', '10', '2020-10-10-countylevel.csv')


def calc_complete_home(df):
    # Util function to calculate the percentage of people that never left home each day
    df = df.copy()
    df['no_mob'] = df['completely_home_device_count'] / df['device_count'] * 100
    return df

def get_year_data(path, year, target_col='med_away'):
    # file names are saved as Year-Month-Day-countylevel.csv from the aggregation step 
    fname = '{}-{}-{}-countylevel.csv'

    # dictionary to create a dataframe 
    days ={}
    
    if year == '2020':
        # adds the days that start epiweek #1 in 2019 to the ouput (hardcoded but doesn't change so it doesn't matter)
        fpath = os.path.join(path, '2019', '12')
        for day in range(29, 32):
            days['{}-{}'.format('12',day)] = pd.read_csv(os.path.join(fpath, str(day), fname.format(2019, 12, day)) ,converters={0: lambda x:str(int(x)).zfill(5)}).rename(columns={'Unnamed: 0' :'FIPS'}).set_index('FIPS')[target_col]
    

    # Data from aggregation is stored in ...Year/Month/Day/file.csv 
    fpath = os.path.join(path, year)

    for month in os.listdir(fpath):     
        for day in os.listdir(os.path.join(fpath, month)):
            # read in the csv with, set the index as fips and get a series with only the target column
            days['{}-{}'.format(month,day)] = pd.read_csv(os.path.join(fpath, month, day, fname.format(year, month, day)),converters={0: lambda x:str(int(x)).zfill(5)}).rename(columns={'Unnamed: 0' :'FIPS'}).set_index('FIPS')[target_col]

    # return a df with all of the days in the year as columns, rows are counties
    return pd.DataFrame(days)


_2019 = get_year_data(data_path, '2019')
_2020 = get_year_data(data_path, '2020')

# adds a state column for when we look at state wide data, helps for creating graphs
fips = pd.read_csv(os.path.join(cwd, 'safegraph-data', 'safegraph_open_census_data', 'metadata', 'cbg_fips_codes.csv'), converters={'state_fips':lambda x :str(x).zfill(2), 'county_fips':lambda x:str(x).zfill(3)})
fips['FIPS'] = fips['state_fips'] + fips['county_fips']
pop_data = pd.read_csv(pop, usecols=['Unnamed: 0', 'B01003e1'], converters={'Unnamed: 0':lambda x: str(x).zfill(5)}).rename(columns={'Unnamed: 0' :'FIPS', 'B01003e1':'pop'}).set_index('FIPS') 


# Calculate the simple difference between the previous year
out = _2020.subtract(_2019.mean(axis=1), axis=0)
out = pd.merge(out, fips, left_index=True, right_on='FIPS').drop(['state_fips', 'county_fips', 'class_code', 'county'], axis=1).set_index('FIPS')
out = pd.merge(out, pop_data, left_index=True, right_index=True) 
out.to_csv(save_path + ' difference.csv') # save


# calculate the percent increase/decrease
out = _2020.subtract(_2019.mean(axis=1), axis=0).div(_2019.mean(axis=1), axis=0) * 100
out = pd.merge(out, fips, left_index=True, right_on='FIPS').drop(['state_fips', 'county_fips', 'class_code', 'county'], axis=1).set_index('FIPS')
out = pd.merge(out, pop_data, left_index=True, right_index=True) 
out.to_csv(save_path + ' percent difference.csv') # save

print('Done')
