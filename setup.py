'''
The purpose of this file is to create any necessary folders for outputs or data. 

It will also loosely check if there are any extra requirements that you will need 
for the rest of the analysis to run. 

'''

import os
import difference
import aggregate_stay_at_home_data
reqs = []

cwd = os.path.dirname(os.path.abspath(__file__))

def check_folder(folder_name):
    if not os.path.exists(os.path.join(cwd, folder_name)):
        os.mkdir(os.path.join(cwd, folder_name))
        return False
    else:
        print('...{}/ exists'.format(folder_name))
        return True

# create a Data folder for parsed data
check_folder('Data')

if not os.path.exists(os.path.join(cwd, 'Data', 'Base Shape Files')):
    reqs.append(False)
    print('''\n    Please download county level shapefiles and state level shapefiles. 
    Each shapefile needs to have a columns called FIPS, county level should have 
    State Fips + County Fips and state fips should just have State FIPS.\n''')

# Create a folder for outputs
check_folder('Outputs')


# make sure that safegraph data is stored in correct folder
if not os.path.exists(os.path.join(cwd, 'safegraph-data')):
    reqs.append(False)
    print('''\n    Please download safegraph_social_distancing_metrics along with 
    safegraph_open_census_data and store them in .../safegraph-data/
    
    Any analysis cannot be completed without those datasets...\n''')
else:
    # if the data exisits then create aggregated data file structure and calculate the percent increase from baseline
    if not check_folder('safegraph-data/aggregated-patterns'):
        aggregate_stay_at_home_data.main()
        difference.main()


if all(reqs):
    print('\nAll requirements are met, the analysis can be started.')
else:
    print('\nCheck outputs before continuing to main.')

