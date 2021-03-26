import time
import pandas as pd
import os
import gzip

def create_standard_axes(figsize=(16, 9)):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import Divider, Size
    fig = plt.figure(figsize=figsize)

    # The first & third items are for padding and the second items are for the
    # axes. Sizes are in inches.
    h = [Size.Fixed(1.3), Size.Scaled(1.), Size.Fixed(.2)]
    v = [Size.Fixed(1.6), Size.Scaled(1.), Size.Fixed(.2)]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))
    return ax

def read_data(fpath:str, states_to_remove:list=['AK', 'PR', 'HI'], pop_level:int=None, state_level:bool=False, use_fips=None): 
    """
    Function reads in the csv designated by the file path (fpath) and returns a dataframe with the 
    all counties and a series that contains the states for each county.

    The returned dataframe contains a column for each day and a row for each county.

    Counties that are in the list states_to_remove will be removed from the returned dataframe.

    """
    data = pd.read_csv(fpath, converters={'FIPS':lambda x:str(x).zfill(5)}).set_index('FIPS')
    data = data[~data['state'].isin(states_to_remove)]

    if pop_level:
        data = data[data['pop'] >= pop_level]
        
    data = data.drop(['state', 'pop'], axis=1)
    
    if state_level:
        data = data.groupby(data.index.str.slice(0,2)).mean()    

    if use_fips:
        data = data.loc[use_fips]

    data.columns = pd.to_datetime(data.columns)
    return data


def reindex_with_placename(df, just_state=False):
    '''
    Function that takes a single dataframe with FIPS as its index and reindexes it with the corresponding 
    county and state-name. 

    Returns the same dataframe with a new index of place-names.
    '''
    cwd = os.path.dirname(os.path.abspath(__file__))

    county_info = pd.read_csv(os.path.join(cwd, 'safegraph-data', 'safegraph_open_census_data', 'metadata', 'cbg_fips_codes.csv'), converters={'state_fips':lambda x :str(x).zfill(2), 'county_fips':lambda x:str(x).zfill(3)})
    county_info['FIPS']= county_info['state_fips'] + county_info['county_fips']
    if just_state:
        county_info['Name'] = county_info['state']
    else:
        county_info['Name'] = county_info['county'] + ', ' + county_info['state']
    county_info = county_info.set_index('FIPS').drop(['state_fips', 'county_fips', 'class_code', 'county', 'state'], axis=1)
    out = pd.merge(df, county_info, left_index=True, right_index=True).set_index('Name')
    try:
        out.columns = pd.to_datetime(out.columns)
    except:
        pass
    return out
    

def aggregate_stay_at_home_data():
    '''
    This function aggregates the safegraph data up to a county level, only keeping the columns that are of interest.
    It takes quite a while to run, but it prints to the screen what percentage has been completed. It will create an
    identical file structure to the original safegraph data and it will be stored in .../safegraph-data/aggregated-patterns
    '''
    print('Starting to parse files, about 10 minutes or so.\n')
    start_time = time.time()
    cwd = os.path.dirname(os.path.abspath(__file__))

    out_path = os.path.join(cwd, 'safegraph-data', 'aggregated-patterns')
    if not os.path.exists(out_path):
        os.mkdir(out_path)


    pop_estimate='B01003e1' # population estimate from the census 
    data_path = os.path.join(cwd, 'safegraph-data', 'safegraph_social_distancing_metrics')

    pop_data = pd.read_csv(os.path.join(cwd,'safegraph-data', 'safegraph_open_census_data', 'data', 'cbg_b01.csv'), 
                                    error_bad_lines=False, usecols=['census_block_group', pop_estimate],  dtype={pop_estimate:'uint16'},
                                    converters={'census_block_group': lambda x:str(int(x)).zfill(12) })

    pop_data = pop_data.groupby(pop_data['census_block_group'].str.slice(0,5))[pop_estimate].sum().astype('uint32')

    counter = 0
    total_files = sum([len(files) for r, d, files in os.walk(data_path)])
    previous = -1
    for year in os.listdir(data_path):

        if not os.path.exists(os.path.join(out_path, year)):
            os.mkdir(os.path.join(out_path, year))

        for month in os.listdir(os.path.join(data_path, year)):
            # if the corresponding month doesn't exist make it
            if not os.path.exists(os.path.join(out_path, year, month)):
                os.mkdir(os.path.join(out_path, year, month))
            
            
            for day in os.listdir(os.path.join(data_path, year, month)):
                if not os.path.exists(os.path.join(out_path, year, month, day)):
                    os.mkdir(os.path.join(out_path, year, month, day))
                else:
                    # if the file exists skip iteration
                    if os.path.exists(os.path.join(out_path, year, month, day, '{}-{}-{}-countylevel.csv'.format(year, month, day))):
                        # print('exists')
                        
                        counter += 1
                        percent_done = int((counter / total_files) * 100)
                        if not percent_done % 10 and percent_done != previous:
                            previous = percent_done
                            print('{}% Completed'.format(percent_done))

                        # this could also be used to add data to any previously created datasets
                        continue
                

                current = os.path.join(data_path, year, month, day)
                current = os.path.join(current, os.listdir(current)[0])
                # read each day for each month
                # This reads in only the columns that are needed using the smallest datatype possible for each to minimize memory use
                data = pd.read_csv(gzip.open(current, 'rb'), error_bad_lines=False, 
                                    usecols=['origin_census_block_group', 'device_count', 'completely_home_device_count','median_non_home_dwell_time'], 
                                    engine='c', dtype={'device_count':'uint8', 'completely_home_device_count':'uint8','median_non_home_dwell_time':'uint8'},
                                    converters={'origin_census_block_group': lambda x:str(int(x)).zfill(12) })

                # aggregate by county level fips 
                data = data.groupby(data['origin_census_block_group'].str.slice(0,5))

                # aggregate some target columns
                dev_count = data['device_count'].sum()
                comp_home = data['completely_home_device_count'].sum()
                med_time = data['median_non_home_dwell_time'].median().rename('med_away')
                max_time = data['median_non_home_dwell_time'].max().rename('max_away')
                min_time = data['median_non_home_dwell_time'].min().rename('min_away')
                avg_time = data['median_non_home_dwell_time'].mean().rename('avg_non_home_dwell')

                # Create a df of all the aggregated data
                data = pd.concat([dev_count.astype('uint32') ,comp_home.astype('uint32') ,med_time.astype('uint32'), avg_time.astype('uint32'), max_time.astype('uint32'), min_time.astype('uint32')], axis=1) #join the columns
                
                # Only reason I did this is because there are some extra FIPS that have no data so I used the "ground truth" of census data to validate the counties
                data = pd.merge(data, pop_data, left_index=True, right_index=True)

                data.to_csv(os.path.join(out_path, year, month, day, '{}-{}-{}-countylevel.csv'.format(year, month, day)))
                
                # printing percentage completed
                counter += 1
                percent_done = int((counter / total_files) * 100)
                if not percent_done % 10 and percent_done != previous:
                    previous = percent_done
                    print('{}% Completed'.format(percent_done))


    print('Total Time : {:.2f}s\n'.format(time.time()-start_time))

def calculate_mobility_difference():
    '''
    This function uses the paresed safegraph data to calculate the differnce is median-non-home-dwell time when
    compared to the average of 2019. 2020 uses epiweeks so it includes a few days in 2019 when it is read in. 

    Saves two CSV's, one of the simple difference from the 2019 year average 
    '''
    print('Starting to calculate county mobility decrease from a baseline.')
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
        
        # if year == '2020':
        #     # adds the days that start epiweek #1 in 2019 to the ouput (hardcoded but doesn't change so it doesn't matter)
        #     fpath = os.path.join(path, '2019', '12')
        #     for day in range(29, 32):
        #         days['{}-{}-19'.format('12',day)] = pd.read_csv(os.path.join(fpath, str(day), fname.format(2019, 12, day)) ,converters={0: lambda x:str(int(x)).zfill(5)}).rename(columns={'Unnamed: 0' :'FIPS'}).set_index('FIPS')[target_col]
        

        # Data from aggregation is stored in ...Year/Month/Day/file.csv 
        fpath = os.path.join(path, year)

        for month in os.listdir(fpath):     
            for day in os.listdir(os.path.join(fpath, month)):
                # read in the csv with, set the index as fips and get a series with only the target column
                days['{}-{}-{}'.format(month,day,year[2:])] = pd.read_csv(os.path.join(fpath, month, day, fname.format(year, month, day)),converters={0: lambda x:str(int(x)).zfill(5)}).rename(columns={'Unnamed: 0' :'FIPS'}).set_index('FIPS')[target_col]

        # return a df with all of the days in the year as columns, rows are counties
        return pd.DataFrame(days).dropna()

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    def create_diff_with_window(baseline_year, y2020, window_size):
        baseline_year = baseline_year.rolling(window_size, center=True, axis=1).mean(skipna=True)
        y2020 = y2020.drop('02-27-20', axis=1).rolling(window_size, center=True, axis=1).mean(skipna=True)
        baseline_year.columns = y2020.columns

        # This section is for creating a figure for the paper, just shows the raw data after its been smoothed, commented out because we only need one but I left it in just in case
        ax = create_standard_axes()
        baseline_year.mean(axis=0).T.plot(label='2019')
        y2020.mean(axis=0).T.plot(label='2020')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b".ljust(8)))
        plt.legend(prop={'size': 18})
        plt.xticks(fontsize=18, rotation=40,ha='center')
        plt.ylabel('MoPE',labelpad=22, fontsize=18)
        plt.yticks(fontsize=18) 
        plt.xlabel('Date', labelpad=22, fontsize=18)
        plt.savefig(os.path.join(cwd, 'Outputs', 'Figures', 'Raw_Time_Series_Smoothed.png'))

        diff = (y2020 - baseline_year[baseline_year.index.isin(y2020.index)])
        percent_diff = diff / baseline_year[baseline_year.index.isin(diff.index)] * 100
        
        import numpy as np #remove any counties that have an "infinite" value
        percent_diff = percent_diff[~percent_diff.index.isin(percent_diff.index[np.isinf(percent_diff).any(1)].values)]
        return diff, percent_diff

    _2019 = get_year_data(data_path, '2019')
    _2020 = get_year_data(data_path, '2020')

    diff, percent_diff = create_diff_with_window(_2019, _2020, 7)

    # adds a state column for when we look at state wide data, helps for creating graphs
    fips = pd.read_csv(os.path.join(cwd, 'safegraph-data', 'safegraph_open_census_data', 'metadata', 'cbg_fips_codes.csv'), converters={'state_fips':lambda x :str(x).zfill(2), 'county_fips':lambda x:str(x).zfill(3)})
    fips['FIPS'] = fips['state_fips'] + fips['county_fips']
    pop_data = pd.read_csv(pop, usecols=['Unnamed: 0', 'B01003e1'], converters={'Unnamed: 0':lambda x: str(x).zfill(5)}).rename(columns={'Unnamed: 0' :'FIPS', 'B01003e1':'pop'}).set_index('FIPS') 
    

    out = pd.merge(diff, fips, left_index=True, right_on='FIPS').drop(['state_fips', 'county_fips', 'class_code', 'county'], axis=1).set_index('FIPS')
    out = pd.merge(out, pop_data, left_index=True, right_index=True) 
    out.to_csv(save_path + ' window baseline difference.csv') # save


    out = pd.merge(percent_diff, fips, left_index=True, right_on='FIPS').drop(['state_fips', 'county_fips', 'class_code', 'county'], axis=1).set_index('FIPS')
    out = pd.merge(out, pop_data, left_index=True, right_index=True) 
    out.to_csv(save_path + ' window baseline percent difference.csv') # save


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

def update_files():
    aggregate_stay_at_home_data()
    calculate_mobility_difference()

