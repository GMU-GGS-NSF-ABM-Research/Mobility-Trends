import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date
from utils import read_data, reindex_with_placename

cwd = os.path.dirname(os.path.abspath(__file__))

# file = 'mobility percent difference.csv'
# file = 'mobility window baseline percent difference.csv'

# file = 'mobility difference.csv'
file = 'mobility window baseline difference.csv'

data_path = os.path.join(cwd, 'Data', file)
save_path = os.path.join(cwd, 'Outputs', 'Figures')
data = read_data(data_path, state_level=False, pop_level=None)

# These are the selected date ranges 
date_ranges = list(map(lambda x: x.split(' - '), ['01-01-20 - 03-16-20', '03-16-20 - 06-01-20', '06-01-20 - ']))
        
def create_avg_df(df, start_date, end_date):
    '''
    Function slices the original time series data from the start data to the end date and returns a  
    new time series like dataframe with the average from that time period
    '''
    if end_date:
        target =  df.loc[:,start_date:end_date]
    else:
        target = df.loc[:,start_date:]
    return pd.DataFrame().reindex_like(target).fillna(0).add(target.mean(axis=1), axis=0)



# Create a dataframe with 3 values describing each county over the entire time period. The index will
# be replaced with the County, State_Abr for plotting.
avgs = pd.concat([create_avg_df(data, start, stop) for start, stop in date_ranges] ,axis=1)


# Take the average of all the available days and sample 2 counties from each quantile 
single_average = data.mean(axis=1)
def sample_at_mean_quantile(df, n, q1, q2, ret_avg=False):
    if not ret_avg:
        return list(df[(df > df.quantile(q1)) &  (df < df.quantile(q2))].sample(n).index.values)
    else:
        return df[(df > df.quantile(q1)) &  (df < df.quantile(q2))].mean(axis=0)

targets = []
for i in range(1,4):
    targets += sample_at_mean_quantile(single_average, 1 ,(i-1)/3, i/3)


avg_targets = reindex_with_placename(avgs[avgs.index.isin(targets)])
raw_targets = reindex_with_placename(data[data.index.isin(targets)])
# raw_targets = pd.concat([sample_at_mean_quantile(data,1 ,(i-1)/4, i/4, True) for i in range(1,5)], axis=1).rename({i:'Quant #{}'.format(i+1) for i in range(4)}, axis=1).T

from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
rcParams["legend.loc"] = 'lower right'

# Plotting section
# The US average will be plotted as a black line on the bottom for each plot. 
fig, ax = plt.subplots()
avgs.mean(axis=0).T.plot(ax=ax, style='--', c='k', label='US Avg')
avg_targets.T.plot(ax=ax)
plt.legend(prop={'size': 14})
plt.xticks(fontsize=14, rotation=40)
plt.yticks(fontsize=14) 
plt.savefig(os.path.join(save_path, '3 Value Time Series.pdf'))

fig2, ax2 = plt.subplots() 
data.mean(axis=0).T.plot(ax=ax2, style='--', c='k', label='US Avg')
raw_targets.T.plot(ax=ax2)
plt.legend(prop={'size': 14})
plt.xticks(fontsize=14, rotation=40)
plt.yticks(fontsize=14) 
plt.savefig(os.path.join(save_path, 'Raw Time Series.pdf'))
plt.show()