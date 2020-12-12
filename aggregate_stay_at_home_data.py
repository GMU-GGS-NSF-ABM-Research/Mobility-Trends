import time
import pandas as pd
import os
import gzip

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
                    print('exists')
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

        print("Finished Month #{}".format(month))
    print('Finished Year #{}'.format(year))

print('Total Time : {:.2f}'.format(time.time()-start_time))


