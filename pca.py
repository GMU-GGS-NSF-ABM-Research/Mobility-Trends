import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

# hardcoded fips codes, I was too lazy to come up with a fancy solution
states = [ i.zfill(2) for i in ['11','2', '1', '4', '5', '6', '8', '9', '10', '12', '13', '15', '16', '17', '18', '19', '20', '21', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56', '22']]
lower_48 = [i.zfill(2) for i in ['11', '1', '4', '5', '6', '8', '9', '10', '12', '13', '16', '17', '18', '19', '20', '21', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56', '22']]

cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(os.getcwd(), 'Outputs')):
    os.mkdir(os.path.join(cwd, 'Outputs'))


def write_shape_file(outName, df):
    print('Starting write.')
    shp = gpd.read_file(os.path.join(cwd, 'extras', 'tl_2017_us_county', 'tl_2017_us_county.shp'))
    shp['FIPS'] = shp['STATEFP'] + shp['COUNTYFP']
    df = df.copy()

    pcs = len(df.columns)
    # normalize all of the values from 0-1
    df[['PC_{}_Normalized'.format(i) for i in range(1,pcs+1)]] = (df - df.min()) / (df.max() - df.min())

    # only use the first 3 principal components as the rgb colors 
    df[['R','B','G']] = (df[['PC_{}_Normalized'.format(i) for i in range(1,4)]] * 255).astype(int)

    shp = shp.merge(df, left_on='FIPS', right_index=True, how='left')

    shp.to_file(os.path.join(cwd, 'Outputs', outName))
    print('Done writing {}.'.format(outName))



data_path = os.path.join(cwd, 'Outputs', 'percent increase.csv')
save_path = os.path.join(cwd, 'Outputs')
data = pd.read_csv(data_path, converters={'FIPS':lambda x:str(int(x)).zfill(5)}) 
data = data.dropna() #remove any cells that have N/a becuase PCA focuses on 0's so we can't just fill it


# removing extra counties, comment if you don't want it. Also change to lower_48 if you only want to 
data = data[(data.FIPS.str.slice(0,2).isin(states))]


# Removing weekends, comment out these two lines if you don't want it
data = data.set_index('FIPS').T
data = pd.concat([data[i:i+7][1:6] for i in range(0,len(data.index),7)] ).T.reset_index()

# The title for the output shapefile
title = 'All Data No Weekends'

# you can change the number of components to however many you want
components = 3
clf = PCA(n_components=components)
clf.fit(data.drop('FIPS', axis=1))
X = clf.transform(data.drop('FIPS', axis=1))

# create a dataframe from the model outputs 
output = pd.DataFrame(X, index=data['FIPS'], columns=['PCA_{}_Raw'.format(i) for i in range(1, components+1)])

# write the output to a shapefile
write_shape_file('{}.shp'.format(title), output)

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
inverse = pd.DataFrame(clf.inverse_transform(inverse), columns=data.drop('FIPS', axis=1).columns, index=['PC_{}'.format(i) for i in range(1,components+1)]).T
inverse.plot()
plt.title(title)
plt.show()