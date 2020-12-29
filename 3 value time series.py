import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date
from utils import read_data, reindex_with_placename

cwd = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(cwd, 'Data', 'mobility percent difference.csv')

# Set this variable true if you want weekends otherwise false
weekends = False
data = read_data(data_path, weekends=weekends, state_level=False, pop_level=None)

# These are the selected date ranges 
date_ranges = list(map(lambda x: x.split(' - '), ['12-30-19 - 03-16-20', '03-16-20 - 06-01-20', '06-01-20 - ']))

def is_weekend(in_dates):
    '''
    Function just checks if a date is a weekend or not and returns a list of dates that are weekends.
    '''
    weekends = []
    for _date in in_dates:

        try:
            month, day, year = _date.split('-')
            if date(int('20'+year), int(month), int(day)).weekday()> 4 :
                weekends.append(_date)
        except:
            return weekends
    return weekends

if not weekends:
    # check if any of the dates are weekends, if there are print them and exit
    weekends = []
    for d in date_ranges:
        bad = is_weekend(d)
        if bad:
            weekends += bad
    if weekends:
        print('The following dates are weekends, please change them.')
        print(weekends)
        exit()
        
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
avgs = reindex_with_placename(pd.concat([create_avg_df(data, start, stop) for start, stop in date_ranges] ,axis=1), just_state=False)


# Plotting section
# The US average will be plotted as a black line on the bottom for each plot. 
ax = avgs.mean(axis=0).T.plot(style='--', c='k', label='US Avg')
avgs.sample(5).T.plot(ax=ax)

# 'Loudoun County, VA', 'Fairfax County, VA', 'Fairfax city, VA', 'Winchester city, VA', 'Jefferson County, WV'
# avgs.loc[['Rockingham County, VA','Cambria County, PA', 'Somerset County, PA']].T.plot(ax=ax)
plt.legend()
plt.show()