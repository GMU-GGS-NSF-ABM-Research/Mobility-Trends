import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
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


data_path = os.path.join(cwd, 'Data', 'mobility percent difference.csv')
save_path = os.path.join(cwd, 'Outputs')


title = 'Lower 48 No Weekends'
data, states = read_data(data_path, weekends=False, state_level=False, pop_level=None)
print('Analysis Type: {}'.format(title))
print('Counties: {}, Days: {}\n'.format(*data.shape))
data = data.dropna() # make sure there are no null values when training


# The title for the output shapefile

# train the original model 
components = 3
clf = PCA(n_components=components)
clf.fit(data)
X = clf.transform(data)

# create a dataframe from the model outputs 
output = pd.DataFrame(X, index=data.index, columns=['PCA_{}_Raw'.format(i) for i in range(1, components+1)])

# remove outliers that are above 3 std away from mean and transform back into original space
outliers_rm = output[(np.abs(stats.zscore(output)) < 3).all(axis=1)]
outliers_rm = pd.DataFrame(clf.inverse_transform(outliers_rm), columns=data.columns, index=outliers_rm.index )


# fix, ax = plt.subplots(3)
# for col, _ax in zip(output.columns, ax.ravel()):
#     output[col].hist(ax=_ax)
#     _ax.set_title(col)
# plt.show()

# train the model with outliers removed
clf = PCA(n_components=components)
clf.fit(outliers_rm)
X = clf.transform(outliers_rm)
output = pd.DataFrame(X, index=outliers_rm.index, columns=['PCA_{}_Raw'.format(i) for i in range(1, components+1)])


# output = (output - output.min())  / (output.max() - output.min())


# PROBABLY GOING TO RUN THIS Hierarchical CLUSTERING METHOD IN THE FUTURE
# clf = AgglomerativeClustering(n_clusters=3, affinity='manhattan')


# 3D PLOTTING SECTION
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(xs=output['PCA_1_Raw'], ys=output['PCA_2_Raw'], zs=output['PCA_3_Raw'])
# plt.show()

# TRIED DBSCAN, ONLY EVER FOUND ONE CLUSTER
# for i in range(1,11):
#     db = DBSCAN(eps=i/10, min_samples=10).fit(output)
#     labels = db.labels_
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     print(n_clusters_)

# results = {'Silhouettes':[], 'Distortion':[]}
# for i in range(2, 16):
#     clf = KMeans(n_clusters=i)
#     clf.fit(output)
#     clusters = clf.predict(output)
#     results['Silhouettes'].append(silhouette_score(output, clusters))
#     results['Distortion'].append(clf.inertia_)
    
# test = pd.DataFrame(results, index=range(2,16))
# print(test)
# test.plot(y='Silhouettes')
# plt.show()
# test.plot(y='Distortion')
# plt.show()
exit()

corr_data = pd.read_csv(os.path.join(cwd, 'Data', 'corr_data.csv'), index_col=0, converters={'FIPS':lambda x:str(x).zfill(5)}).set_index('FIPS', drop=True)
corr_data = pd.merge(output, corr_data, right_index=True, left_index=True )

focus_cols = output.columns
# print the pearson's R value for each of the principal components
print(corr_data.corr().filter(focus_cols).drop(focus_cols))

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