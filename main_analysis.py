import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from utils import read_data

color_lookup = {'PC #1':'r', 'PC #2':'g', 'PC #3':'b'}

cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(os.getcwd(), 'Outputs')):
    os.mkdir(os.path.join(cwd, 'Outputs'))


def read_shape_file(df, state_level=False, save_name=False):
    print('Reading shapefile.')
    
    if not state_level:
        # read the county level shape file
        shp = gpd.read_file(os.path.join(cwd, 'Data', 'Base Shape Files', 'counties.shp'))
    else:
        # read the state level shape file
        shp = gpd.read_file(os.path.join(cwd, 'Data', 'Base Shape Files', 'states.shp'))
    
    df = df.copy()

    shp = shp.merge(df, left_on='FIPS', right_index=True, how='left')

    # make a column with hex codes from the three principal components, N/a is filled with black 
    shp['color'] = shp.apply(lambda x : colors.to_hex(tuple(x[['PC_1_Norm', 'PC_2_Norm', 'PC_3_Norm']].fillna(0))) , axis=1)

    
    if save_name:
        shp.to_file(os.path.join(cwd, 'Outputs', save_name+'.shp'))
    return shp

def inverse_principal_components(df, model):
    '''
    Pass in the PCA model used and the output principal components. 

    This will return a dataframe with the principal components back in the original data space.
    '''
    # ineficient but works, isolates each principal component based on its max value in PCA space and the min for the rest 
    inverse = []
    for col in df.columns:
        temp = [] # holding list for each principal component maxes and mins
        for _col in df.columns:
            if col == _col:
                temp.append(df[_col].max())
            else:
                temp.append(df[_col].min())
        inverse.append(temp)

    # Create a dataframe from the inverse transformation that is back into the time series space and plot each of the principal components
    return pd.DataFrame(model.inverse_transform(inverse), columns=data.columns, index=['PC #{}'.format(i) for i in range(1,components+1)])

def plot_histograms(df, title):
    '''Plots the histograms of the input dataframe, used for showing what the counties look like in PC-space'''
    fig, ax = plt.subplots(len(df.columns))
    for col, _ax in zip(df.columns, ax.ravel()):
        output[col].hist(ax=_ax)
        _ax.set_title(col)
    fig.suptitle(title)

def plot_model_accuracy(model, title, components):
    ''' For plotting the explained model variance as the number of components increases. '''
    var=np.cumsum(np.round(model.explained_variance_ratio_, decimals=3)*100)
    fig, ax = plt.subplots()
    ax.plot([i for i in range(1,components+1)],var)
    ax.grid()
    ax.set_ylabel('% variance explained')
    ax.set_xticks([i for i in range(1,components+1)])
    ax.set_xlabel('Number of features')
    ax.set_title(title)

# file = 'mobility percent difference.csv'
# file = 'mobility window baseline percent difference.csv' 

# file = 'mobility difference.csv'
file = 'mobility window baseline difference.csv'


data_path = os.path.join(cwd, 'Data', file)
save_path = os.path.join(cwd, 'Outputs')

# The title for the output shapefile
show_plots = True
state_level = False

data = read_data(data_path, pop_level=None)
# data = read_data(r'E:\Justin\Documents\GitHub\Mobility-Stats\Data\test.csv')
print('Analysis Type: {}'.format(file))
print('Counties: {}, Days: {}'.format(*data.shape))
# TESTING ONLY
# data = data.fillna(data.mean(axis=0))
data = data.dropna(axis=1) # make sure there are no null values when training


# print(data.idxmin(axis=1).value_counts())
# exit()
# train the original model 
components = 3
clf = PCA(n_components=components)
clf.fit(data)
X = clf.transform(data)
print('Explained Var:',sum(clf.explained_variance_ratio_))

# create a dataframe from the model outputs, don't normalize at this step becuase we have to transform back into
# the original data space.
output = pd.DataFrame(X, index=data.index, columns=['PC_{}_Norm'.format(i) for i in range(1, components+1)])


# get the index of all counties that are above and below 
outliers = output[~(np.abs(stats.zscore(output)) < 4).all(axis=1)].index.values

# Select all data where the z-score is less than three from original data
no_outliers = data[data.index.isin(output[(np.abs(stats.zscore(output)) < 4).all(axis=1)].index)] 


print('Counties Lost From Removing Outliers: {}\n'.format(len(outliers)))

# agg to state level if true after removing outliers
if state_level:
    no_outliers = no_outliers.groupby(no_outliers.index.str.slice(0,2)).mean()


# train the model with outliers removed
clf = PCA(n_components=components)
clf.fit(no_outliers)
X = clf.transform(no_outliers)
print('Explained Var:',sum(clf.explained_variance_ratio_))

output = pd.DataFrame(X, index=no_outliers.index, columns=['PC_{}_Norm'.format(i) for i in range(1, components+1)])

# Show what the principal components look like after removing outliers and retraining
if show_plots:
    fig, ax = plt.subplots()
    inv = inverse_principal_components(output, clf)
    for row in inv.index:
        inv.loc[row].T.plot(ax=ax, c=color_lookup[row], label=row)
    ax.set_title('Outliers Removed')
    plt.legend()
    # plot_histograms(output, 'Outliers Removed')
    # plot_model_accuracy(clf, 'Outliers Removed', components)
    plt.show()


# After testing, K-means was a good enough method to cluster the counties
output['k_clusters'] = KMeans(n_clusters=3).fit_predict(output)



# Isolating the time series for each cluster by taking the avg of all counties in clusters
inv_clusters = []
for cluster, df_group in output.groupby('k_clusters'):
    inv_clusters.append(list(df_group.mean(axis=0)[['PC_{}_Norm'.format(i+1) for i in range(3)]]))

inv_clusters = pd.DataFrame(clf.inverse_transform(inv_clusters), columns=data.columns, index=['Cluster #{}'.format(i) for i in range(1,4)])
print(inv_clusters)
inv_clusters.T.plot()
plt.show()


# Normalize everything except k_clusters
clusters = output['k_clusters']
output.drop('k_clusters', axis=1, inplace=True)
output = (output - output.min())  / (output.max() - output.min())
output['k_clusters'] = clusters + 1


# Average the colors for each pc
temp = []
output[['clust_c{}'.format(i) for i in range(3)]] = 0
for cluster, df_group in output.groupby('k_clusters'):
    df_group[['clust_c{}'.format(i) for i in range(3)]] = df_group.mean(axis=0)[['PC_{}_Norm'.format(i+1) for i in range(3)]]
    temp.append(df_group)

output = pd.concat(temp)

# plotting the shapefile using the color created from the 3 pricipal components
shp = read_shape_file(output, state_level=state_level, save_name='Output')


# for plotting the k-means outputs
# plot the outliers with grey hashes, only show them if plotting at a county level
if not state_level:
    # base = shp.plot(column='k_clusters',cmap='copper', missing_kwds={'color': 'black'})
    base = shp.plot(color=shp['color'], missing_kwds={'color': 'black'})
    shp[shp['FIPS'].isin(outliers)].plot(ax=base, color='lightgrey', linewidth=0.1, edgecolor='black',zorder=2, hatch='///')
else:
    # shp.plot(column='k_clusters',cmap='copper')
    shp.plot(color=shp['color'])
plt.show()


# FOR TESTING HOW MANY CLUSTERS WE SHOULD USE WITH K-MEANS
results = {'Silhouettes':[], 'Distortion':[]}
for i in range(2, 16):
    clf = KMeans(n_clusters=i)
    clusters = clf.fit_predict(output)
    results['Silhouettes'].append(silhouette_score(output, clusters))
    results['Distortion'].append(clf.inertia_)

# PLOT THE ACCURACY FROM THE CLUSTERING 
test = pd.DataFrame(results, index=range(2,16))
print(test)
test.plot(y='Silhouettes')
plt.show()
test.plot(y='Distortion')
plt.show()

# 3D PLOTTING SECTION
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs=output['PC_1_Norm'], ys=output['PC_2_Norm'], zs=output['PC_3_Norm'], c=output['k_clusters'])
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.show()


if not state_level:
    corr_data = pd.read_csv(os.path.join(cwd, 'Data', 'corr_data.csv'), index_col=0, converters={'FIPS':lambda x:str(x).zfill(5)}).set_index('FIPS', drop=True)
    corr_data = pd.merge(output, corr_data, right_index=True, left_index=True )

    focus_cols = output.columns
    # print the pearson's R value for each of the principal components
    print(corr_data.corr().filter(focus_cols).drop(focus_cols))

