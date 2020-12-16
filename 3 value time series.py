import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from epiweeks import Week, Year

data = pd.read_csv(r'E:\Justin\Documents\GitHub\Mobility-Stats\Outputs\percent increase.csv', converters={'FIPS':lambda x:str(x).zfill(5)}).set_index('FIPS')

# TO REMOVE WEEKENDS
# data = data.T
# data = pd.concat([data[i:i+7][1:6] for i in range(0,len(data.index),7)] ).T

date_ranges = map(lambda x: x.split(', '), ['12-30, 03-16', '03-16, 06-01', '06-01, 10-10'])

def create_avg_df(df, start_date, end_date):
    target =  df.loc[:,start_date:end_date]
    return pd.DataFrame().reindex_like(target).fillna(0).add(target.mean(axis=1), axis=0)

avgs = pd.concat([create_avg_df(data, start, stop) for start, stop in date_ranges] ,axis=1)
print(avgs)

avgs.sample(15).T.plot()
plt.show()