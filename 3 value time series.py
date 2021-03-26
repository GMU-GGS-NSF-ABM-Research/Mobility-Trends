import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date
import matplotlib.dates as mdates
from utils import read_data, reindex_with_placename, create_standard_axes

cwd = os.path.dirname(os.path.abspath(__file__))


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


targets = [ '06107','51013', '42021']
col = {'Arlington County, VA':'#bd0d0d', 'Tulare County, CA':'#15a340', 'Cambria County, PA':'#4266f5'}
avg_targets = reindex_with_placename(avgs[avgs.index.isin(targets)])
raw_targets = reindex_with_placename(data[data.index.isin(targets)])


ax = create_standard_axes()
data.mean(axis=0).T.plot(ax=ax, style='--', c='k', label='US Avg')
for key, val in col.items():
    raw_targets.T[key].plot(ax=ax, c=val)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d '%y"))
plt.legend(prop={'size': 18})
plt.xticks(fontsize=18, rotation=40)
plt.yticks(fontsize=18) 
plt.xlabel('Date', labelpad=10, fontsize=18)
plt.ylabel(r'$\Delta$ MoPE', labelpad=8, fontsize=18)
plt.savefig(os.path.join(save_path, 'Raw Time Series.png'))



# Plotting section for 3 averaged date ranges. We removed this from the paper, but I kept it in just in case. 
# The US average will be plotted as a black line on the bottom for each plot. 
# ax= create_standard_axes()
# avgs.mean(axis=0).T.plot(ax=ax, style='--', c='k', label='US Avg')
# for key, val in col.items():
#     avg_targets.T[key].plot(ax=ax, c=val)
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d '%y"))
# plt.legend(prop={'size': 18})
# plt.xticks(fontsize=18, rotation=40)
# plt.yticks(fontsize=18) 
# plt.xlabel('Date', labelpad=10, fontsize=18)
# plt.ylabel('Change in Minutes Away From Home', labelpad=8, fontsize=18)

# plt.savefig(os.path.join(save_path, '3 Value Time Series.pdf'))




plt.show()