import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import geopandas as gpd
import pandas as pd
import numpy as np
from pandas.core.reshape.merge import merge
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score, r2_score
from utils import read_data, reindex_with_placename, create_standard_axes

color_lookup = {'PC1':'#bd0d0d', 'PC2':'#15a340', 'PC3':'#4266f5'}

cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(os.getcwd(), 'Outputs')):
    os.mkdir(os.path.join(cwd, 'Outputs'))


def read_shape_file(df, state_level=False, save_name=False):
    print('Reading shapefile.')
    
    if not state_level:
        # read the county level shape file
        shp = gpd.read_file(os.path.join(cwd, 'Data', 'Base Shape Files', 'counties.shp'))
        pop = pd.read_csv(os.path.join(cwd, 'Data', 'mobility window baseline difference.csv'), usecols=['FIPS', 'pop'], dtype={'FIPS':str})
        shp = shp.merge(pop, left_on='FIPS', right_on='FIPS', how='left')     
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
    return pd.DataFrame(model.inverse_transform(inverse), columns=data.columns, index=['PC{}'.format(i) for i in range(1,components+1)])

def plot_histograms(df, title):
    '''Plots the histograms of the input dataframe, used for showing what the counties look like in PC-space'''
    fig, ax = plt.subplots(len(df.columns))
    for col, _ax in zip(df.columns, ax.ravel()):
        output[col].hist(ax=_ax)
        _ax.set_title(col)
    fig.suptitle(title)

def plot_model_accuracy(model, components):
    ''' For plotting the explained model variance as the number of components increases. '''
    var=np.cumsum(model.explained_variance_ratio_ * 100)
    print(var)
    ax = create_standard_axes(figsize=(9,9))
    ax.plot([i for i in range(1,components+1)], var)
    ax.grid()
    ax.set_xticks([i for i in range(1,components+1)])
    
    plt.xlabel('Number of Features', labelpad=10, fontsize=18)
    plt.ylabel('% Variance Explained', labelpad=8, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18) 
    



file = 'mobility window baseline difference.csv'

data_path = os.path.join(cwd, 'Data', file)
save_path = os.path.join(cwd, 'Outputs')

# The title for the output shapefile
show_plots = True
state_level = False

data = read_data(data_path, pop_level=None)
print('Analysis Type: {}'.format(file))
print('Counties: {}, Days: {}'.format(*data.shape))
data = data.dropna(axis=1) # make sure there are no null values when training
print('Counties: {}, Days: {}'.format(*data.shape))


# train the original model 
components = 3
clf = PCA(n_components=components)
clf.fit(data)
X = clf.transform(data)
print('Explained Var:',sum(clf.explained_variance_ratio_)) # total explained var of the model
print(clf.explained_variance_ratio_) # show the eiganvalues for the PCs

# create a dataframe from the model outputs, don't normalize at this step becuase we have to transform back into the original data space.
output = pd.DataFrame(X, index=data.index, columns=['PC_{}_Norm'.format(i) for i in range(1, components+1)])


# get the index of all counties that are above and below 
outliers = output[~(np.abs(stats.zscore(output)) < 4).all(axis=1)].index.values

# this prints the names of any outliers that are found, kept it in just in case we wanted to play around with the code later
# for place in reindex_with_placename(read_data(data_path, use_fips=list(outliers))).index:
#     print(place, end=', ')


# Select all indexes of the data where the z-score is less than three from original data
no_outliers = data[data.index.isin(output[(np.abs(stats.zscore(output)) < 4).all(axis=1)].index)] 


print('Counties Lost From Removing Outliers: {}\n'.format(len(outliers)))
print('Counties: {}, Days: {}'.format(*no_outliers.shape))

# agg to state level if true after removing outliers
if state_level:
    no_outliers = no_outliers.groupby(no_outliers.index.str.slice(0,2)).mean()


# train the model with outliers removed
clf = PCA(n_components=components)
clf.fit(no_outliers)
X = clf.transform(no_outliers)
print('Explained Var:',sum(clf.explained_variance_ratio_))
print(clf.explained_variance_ratio_)



output = pd.DataFrame(X, index=no_outliers.index, columns=['PC_{}_Norm'.format(i) for i in range(1, components+1)]) #create a df from the second PCA

data_outlier_removed = data[data.index.isin(output.index)] #get all the data that isn't dropped
inv = pd.DataFrame(clf.inverse_transform(output), columns=data_outlier_removed.columns, index=data_outlier_removed.index)

R2 = pd.Series(r2_score(data_outlier_removed.T, inv.T, multioutput='raw_values'), index=data_outlier_removed.index).rename('R2')


# Show what the principal components look like after removing outliers and re-training
if show_plots:

    ax = create_standard_axes()
    inv = inverse_principal_components(output, clf)
    print(inv)
    for row in inv.index:
        inv.loc[row].T.plot(ax=ax, c=color_lookup[row], label=row)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d '%y"))
    plt.legend(prop={'size': 18})
    plt.xlabel('Date', labelpad=10, fontsize=18)
    plt.ylabel(r'$\Delta$ MoPE', labelpad=8, fontsize=18)
    plt.xticks(fontsize=18, rotation=40)
    plt.yticks(fontsize=18) 
    plt.savefig(os.path.join(cwd, 'Outputs', 'Figures', 'PCA_Time_Series.png'))
    # plot_model_accuracy(clf, 'Accuracy', components)
    # plt.show()

# After testing, K-means was a good enough method to cluster the counties
output['k_clusters'] = KMeans(n_clusters=3).fit_predict(output)

# Isolating the time series for each cluster by taking the avg of all counties in clusters
# inv_clusters = []
# for cluster, df_group in output.groupby('k_clusters'):
#     inv_clusters.append(list(df_group.mean(axis=0)[['PC_{}_Norm'.format(i+1) for i in range(3)]]))

# inv_clusters = pd.DataFrame(clf.inverse_transform(inv_clusters), columns=data.columns, index=['Cluster #{}'.format(i) for i in range(1,4)])
# print(inv_clusters)
# inv_clusters.T.plot()
# plt.show()


# Normalize everything except k_clusters
clusters = output['k_clusters']
output.drop('k_clusters', axis=1, inplace=True)
raw = output.copy()
raw.columns = [f'PC_{i}_Raw' for i in range(1,len(raw.columns)+1)]
output = (output - output.min())  / (output.max() - output.min()) #normalize PC values

output['k_clusters'] = clusters + 1
output['R2'] = R2 
output = pd.concat([output, raw], axis=1)
# targets = [ '06107','51013', '42021']
# print(reindex_with_placename(output.loc[targets]))
# exit()
# print(output.mean(axis=0))
# print(reindex_with_placename(output.T[[ '06107', '42021', '51013']].T))

# Average the colors for each pc
temp = []
color_lookup2 = {}
output[['clust_c{}'.format(i) for i in range(3)]] = 0 
for cluster, df_group in output.groupby('k_clusters'):
    avg = df_group.mean(axis=0)[['PC_{}_Norm'.format(i+1) for i in range(3)]]
    df_group[['clust_c{}'.format(i) for i in range(3)]] = avg
    color_lookup2[cluster] = tuple(avg.values)
    temp.append(df_group)

output = pd.concat(temp)



# # plotting the shapefile using the color created from the 3 pricipal components
# shp = read_shape_file(output, state_level=state_level, save_name='final')

# # for plotting the k-means outputs
# # plot the outliers with grey hashes, only show them if plotting at a county level
# if not state_level:
#     base = shp.plot(column='k_clusters',cmap='copper', missing_kwds={'color': 'black'})
#     # base = shp.plot(color=shp['color'], missing_kwds={'color': 'black'})
#     # shp[shp['FIPS'].isin(outliers)].plot(ax=base, color='lightgrey', linewidth=0.1, edgecolor='black',zorder=2, hatch='///')
# else:
#     # shp.plot(column='k_clusters',cmap='copper')
#     shp.plot(color=shp['color'])
# plt.show()


# FOR TESTING HOW MANY CLUSTERS WE SHOULD USE WITH K-MEANS
# results = {'Silhouettes':[], 'Distortion':[]}
# for i in range(2, 16):
#     clf = KMeans(n_clusters=i)
#     clusters = clf.fit_predict(no_outliers)
#     results['Silhouettes'].append(silhouette_score(no_outliers, clusters))
#     results['Distortion'].append(clf.inertia_)

# # # PLOT THE ACCURACY FROM THE CLUSTERING 
# test = pd.DataFrame(results, index=range(2,16))
# print(test)
# test.plot(y='Silhouettes')
# plt.show()
# test.plot(y='Distortion')
# plt.show()

# 3D PLOTTING SECTION
# ANGLE #1
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(projection='3d')
for cluster, df in output.groupby('k_clusters'):
    ax.scatter(xs=df['PC_1_Norm'], ys=df['PC_2_Norm'], zs=df['PC_3_Norm'], c=colors.to_hex(color_lookup2[cluster]))
ax.set_xlabel('PC1', labelpad=18)
ax.set_ylabel('PC2',labelpad=18)
ax.set_zlabel('PC3',labelpad=18)
ax.view_init(20.6382978723405, 65)
plt.tight_layout()
plt.savefig(os.path.join(cwd, 'Outputs', 'Figures', '3-D Angle 1.png'))

# ANGLE #2
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(projection='3d')
for cluster, df in output.groupby('k_clusters'):
    ax.scatter(xs=df['PC_1_Norm'], ys=df['PC_2_Norm'], zs=df['PC_3_Norm'], c=colors.to_hex(color_lookup2[cluster]))
ax.set_xlabel('PC1', labelpad=18)
ax.set_ylabel('PC2',labelpad=18)
ax.set_zlabel('PC3',labelpad=18)
ax.view_init(20.6382978723405, -115)
plt.tight_layout()
plt.savefig(os.path.join(cwd, 'Outputs', 'Figures', '3-D Angle 2.png'))
# plt.show()
exit()
if not state_level:
    # this is a hacky way that I created a latex table, I know pandas has a method but I needed a specific format and this was easy enough
    corr_data = pd.read_csv(os.path.join(cwd, 'Data', 'parsed_json.csv'),  converters={'FIPS':lambda x:str(x).zfill(5)}).set_index('FIPS', drop=True) # read the data I want to correlate
    corr_data = pd.merge(output, corr_data, right_index=True, left_index=True ).dropna()

    focus_cols = [f'PC_{i}_Norm' for i in range(1, components+1)] # PC columns that I want to test correlation on
    test_cols = sorted(list(set(corr_data.columns) - set(output.columns))) #

    out = {} # dictionary for building a df
    first = True # check if we need to add a first column
    out['_'] = [] # first column of &'s
    
    for i,col in enumerate(focus_cols):
        # initialize the lists for the current columns
        out[col+ '_r'] = []
        out[str(i)] = []
        out[col + '_p'] = []
        out[str(i)+'_'] = []

        for _col in test_cols: # loop through the target correlation columns
            r, p = stats.pearsonr(corr_data[col] , corr_data[_col]) # calculate pearson's r and it's p-value
            
            # formatting the output columns, use scientific notation under threshold
            if r < .001:
                out[col+ '_r'].append('{:0.2e}'.format(r))
            else:
                out[col+ '_r'].append('{:0.2}'.format(r))
            
            if p < .001:
                out[col + '_p'].append('{:0.2e}'.format(p))
            else:
                out[col + '_p'].append('{:0.2}'.format(p))
            
            out[str(i)].append('&')
            if i == len(focus_cols)-1: # if its the last columns add the end line
                out[str(i)+'_'].append('\\\\')
            else:
                out[str(i)+'_'].append('&') # otherwise add a column line
            if first:
                out['_'].append('&')
        first = False
    corr = pd.DataFrame.from_dict(out)
    corr['l'] = '\\hline' # add the column bottom line
    corr.index = [' '.join(col.split('_')) for col in test_cols] # formatting the index since latex doesn't accept _
    corr['sort'] = corr['PC_1_Norm_p'].astype(float) # sort by the p-values of pc1
    print(corr.sort_values('sort', ascending=True).drop('sort', axis=1)) # print the table 
