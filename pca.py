import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from utils import read_data

cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(os.getcwd(), 'Outputs')):
    os.mkdir(os.path.join(cwd, 'Outputs'))


def read_shape_file(df, state_level=False):
    print('Starting write.')
    
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
    return shp
    # shp.to_file(os.path.join(cwd, 'Outputs', outName))


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
    return pd.DataFrame(model.inverse_transform(inverse), columns=data.columns, index=['PC_{}'.format(i) for i in range(1,components+1)])

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



data_path = os.path.join(cwd, 'Data', 'mobility percent difference.csv')
save_path = os.path.join(cwd, 'Outputs')

# The title for the output shapefile
title = 'Lower 48 No Weekends State Level'
show_plots = False

data = read_data(data_path, weekends=False, state_level=False, pop_level=None)
print('Analysis Type: {}'.format(title))
print('Counties: {}, Days: {}'.format(*data.shape))
data = data.dropna() # make sure there are no null values when training



# train the original model 
components = 3
clf = PCA(n_components=components)
clf.fit(data)
X = clf.transform(data)

# create a dataframe from the model outputs, don't normalize at this step becuase we have to transform back into
# the original data space.
output = pd.DataFrame(X, index=data.index, columns=['PCA_{}_Raw'.format(i) for i in range(1, components+1)])


# Show what the original principal components look like before removing outliers, as well as plotting the histograms for the data
if show_plots:
    inv = inverse_principal_components(output, clf)
    ax1 = inv.T.plot()
    ax1.set_title('Including Outliers')
    plot_histograms(output, 'Including Outliers')
    plot_model_accuracy(clf, 'Including Outliers', components)


# remove outliers that are above 3 std away from mean 
outliers_rm = output[(np.abs(stats.zscore(output)) < 3).all(axis=1)]
# create a dataframe from transforming back into original space
outliers_rm = pd.DataFrame(clf.inverse_transform(outliers_rm), columns=data.columns, index=outliers_rm.index )

print('Counties Lost From Removing Outliers: {}\n'.format(len(data.index) - len(outliers_rm.index)))

# train the model with outliers removed
clf = PCA(n_components=components)
clf.fit(outliers_rm)
X = clf.transform(outliers_rm)
output = pd.DataFrame(X, index=outliers_rm.index, columns=['PC_{}_Norm'.format(i) for i in range(1, components+1)])

# Show what the principal components look like after removing outliers and retraining
if show_plots:
    inv = inverse_principal_components(output, clf)
    ax2 = inv.T.plot()
    ax2.set_title('Outliers Removed')
    plot_histograms(output, 'Outliers Removed')
    plot_model_accuracy(clf, 'Outliers Removed', components)
    plt.show()

# Normalize data
output = (output - output.min())  / (output.max() - output.min())

# After testing, K-means was a good enough method to cluster the counties
clf = KMeans(n_clusters=3)
clf.fit(output)
output['k_clusters'] = clf.predict(output)

# plotting the shapefile using the color created from the 3 pricipal components
shp = read_shape_file(output)
shp.plot(color=shp['color'])
plt.show()

# FOR TESTING HOW MANY CLUSTERS WE SHOULD USE WITH K-MEANS
# results = {'Silhouettes':[], 'Distortion':[]}
# for i in range(2, 16):
# results['Silhouettes'].append(silhouette_score(output, clusters))
# results['Distortion'].append(clf.inertia_)

# PLOT THE ACCURACY FROM THE CLUSTERING 
# test = pd.DataFrame(results, index=range(2,16))
# print(test)
# test.plot(y='Silhouettes')
# plt.show()
# test.plot(y='Distortion')
# plt.show()

# 3D PLOTTING SECTION
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(xs=output['PCA_1_Norm'], ys=output['PCA_2_Norm'], zs=output['PCA_3_Norm'], c=output['clusters'])
# ax.set_xlabel('PC 1')
# ax.set_ylabel('PC 2')
# ax.set_zlabel('PC 3')
# plt.show()

corr_data = pd.read_csv(os.path.join(cwd, 'Data', 'corr_data.csv'), index_col=0, converters={'FIPS':lambda x:str(x).zfill(5)}).set_index('FIPS', drop=True)
corr_data = pd.merge(output, corr_data, right_index=True, left_index=True )

focus_cols = output.columns
# print the pearson's R value for each of the principal components
print(corr_data.corr().filter(focus_cols).drop(focus_cols))

