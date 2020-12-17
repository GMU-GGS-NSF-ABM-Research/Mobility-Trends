import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from utils import read_data

cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(os.getcwd(), 'Outputs')):
    os.mkdir(os.path.join(cwd, 'Outputs'))


def write_shape_file(outName, df, state_level=False):
    print('Starting write.')
    shp = gpd.read_file(os.path.join(cwd, 'extras', 'tl_2017_us_county', 'tl_2017_us_county.shp'))
    
    if not state_level:
        shp['FIPS'] = shp['STATEFP'] + shp['COUNTYFP']
    else:
        shp['FIPS'] = shp['STATEFP']
        shp = shp.dissolve(by='FIPS') #temporary, just merges the counties to state level. I can remove this after I save the state file

    df = df.copy()

    pcs = len(df.columns)
    # normalize all of the values from 0-1
    df[['PC_{}_Normalized'.format(i) for i in range(1,pcs+1)]] = (df - df.min()) / (df.max() - df.min())

    # only use the first 3 principal components as the rgb colors 
    df[['R','B','G']] = (df[['PC_{}_Normalized'.format(i) for i in range(1,4)]] * 255).astype(int)
    shp = shp.merge(df, left_on='FIPS', right_index=True, how='left')
    print(df)
    shp.to_file(os.path.join(cwd, 'Outputs', outName))
    print('Done writing {}.'.format(outName))


data_path = os.path.join(cwd, 'Outputs', 'percent increase.csv')
save_path = os.path.join(cwd, 'Outputs')


data, states = read_data(data_path, weekends=False, state_level=False, pop_level=250000)
print(data.shape)
data = data.dropna() # make sure there are no null values when training


# The title for the output shapefile
title = 'Counties 250k or more pop with weekends'

# you can change the number of components to however many you want
components = 3
clf = PCA(n_components=components)
clf.fit(data)
X = clf.transform(data)

# create a dataframe from the model outputs 
output = pd.DataFrame(X, index=data.index, columns=['PCA_{}_Raw'.format(i) for i in range(1, components+1)])


# write the output to a shapefile
# write_shape_file('{}.shp'.format(title), output, state_level=True)


# FOR PLOTTING THE EXPLAINED VARIANCE WITH THE NUMBER OF FEATURES 
var=np.cumsum(np.round(clf.explained_variance_ratio_, decimals=3)*100)
plt.plot([i for i in range(1,components+1)],var)
plt.grid()
plt.ylabel('% variance explained')
plt.xticks([i for i in range(1,components+1)])
plt.xlabel('Number of features')
plt.title(title)
plt.show()



# ineficient but works, isolates each principal component based on its max value in PCA space and the min for the rest 
inverse = []
for col in output.columns:
    temp = [] # holding list for each principal component maxes and mins
    for _col in output.columns:
        if col == _col:
            temp.append(output[_col].max())
        else:
            temp.append(output[_col].min())
    inverse.append(temp)

# Create a dataframe from the inverse transformation that is back into the time series space and plot each of the principal components
inverse = pd.DataFrame(clf.inverse_transform(inverse), columns=data.columns, index=['PC_{}'.format(i) for i in range(1,components+1)]).T
inverse.plot()
plt.title(title)
plt.show()