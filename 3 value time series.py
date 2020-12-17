import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date
from utils import read_data

cwd = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(cwd, 'Data', 'mobility percent difference.csv')

# Set this variable true if you want weekends otherwise false
weekends = False
data, states = read_data(data_path, weekends=weekends)

# Make sure that if you don't include weekend when reading in the data that you don't include a weekend in the ranges 
date_ranges = list(map(lambda x: x.split(', '), ['12-30, 03-16', '03-16, 06-01', '06-01, 10-09']))


def is_weekend(in_dates):
    weekends = []
    for _date in in_dates:    
        month, day = _date.split('-')
        if _date == '12-29':
            weekends.append(_date)
        if date(2020, int(month), int(day)).weekday()> 4 :
            weekends.append(_date)
         
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
    target =  df.loc[:,start_date:end_date]
    return pd.DataFrame().reindex_like(target).fillna(0).add(target.mean(axis=1), axis=0)

avgs = pd.concat([create_avg_df(data, start, stop) for start, stop in date_ranges] ,axis=1)

# avgs.mean(axis=0).T.plot()
avgs.sample(5).T.plot()
plt.show()