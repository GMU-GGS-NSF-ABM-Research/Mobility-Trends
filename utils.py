import pandas as pd
def read_data(fpath:str, states_to_remove:list=['AK', 'PR', 'HI'], pop_level:int=None, weekends:bool=True, state_level:bool=False): 
    """
    Function reads in the csv designated by the file path (fpath) and returns a dataframe with the 
    all counties and a series that contains the states for each county.

    The returned dataframe contains a column for each day and a row for each county.

    Counties that are in the list states_to_remove will be removed from the returned dataframe.

    If weekends is set to False, it will remove the weekends from the returned data. 

    """
    data = pd.read_csv(fpath, converters={'FIPS':lambda x:str(x).zfill(5)}).set_index('FIPS')
    data = data[~data['state'].isin(states_to_remove)]
    states = data['state']

    if pop_level:
        data = data[data['pop'] >= pop_level]
        
    data = data.drop(['state', 'pop'], axis=1)
    

    if not weekends:
        data = data.T
        data = pd.concat([data[i:i+7][1:6] for i in range(0,len(data.index),7)] ).T

    if state_level:
        data = data.groupby(data.index.str.slice(0,2)).mean()
        states = states.groupby(states.index.str.slice(0,2)).max()
    
    return data, states