import os
import subprocess
import datetime

cwd = os.path.dirname(os.path.abspath(__file__))

def get_most_current_data_downloaded():
    '''
    This function parses through the data that is currently downloaded on the system and returns a datetime object of that date.
    '''

    cur = os.path.join(cwd, 'safegraph-data', 'safegraph_social_distancing_metrics')
    date = ''
    while True:
        _dir = os.listdir(cur)
        if _dir[-1].endswith('.gz'):
            year, month, day = map(int, date.split())
            return datetime.date(year, month, day)
        else:
            date += _dir[-1]+' '
            cur = os.path.join(cur, _dir[-1])

def make_date_dir(date):
    '''
    This function check to see if the current date has the corresponding year, month, and day folder and creates them if needed. 
    
    Returns a string of the path to the newest folder.
    '''
    path = os.path.join(cwd, 'safegraph-data', 'safegraph_social_distancing_metrics')
    for folder in date.strftime('%Y %m %d').split():
        path = os.path.join(path, folder)
        if not os.path.exists(path):
            os.mkdir(path)
    return path+'\\'

def list_aws_dir(path):
    '''
    This function uses the safegraphws profile and aws to list the directory of the path given. The path input must be "year/month/day/"
    
    Returns a list of all of the contents in that folder on the AWS server. If there is nothing, an empty list is returned instead. 
    '''
    call = r'aws s3 ls s3://sg-c19-response/social-distancing/v2/{} --profile safegraphws --endpoint https://s3.wasabisys.com'
    _subprocess = subprocess.Popen(call.format(path), shell=True, stdout=subprocess.PIPE)
    subprocess_return = _subprocess.stdout.read().decode().split()
    subprocess_return = [i for i in subprocess_return if i != 'PRE']
    return subprocess_return


def download_data(date, path):
    '''
    This function uses AWS CLI to download a given date to the specified path. 
    '''
    download_call = "aws s3 sync s3://sg-c19-response/social-distancing/v2/{} {} --profile safegraphws --endpoint https://s3.wasabisys.com"
    os.system(download_call.format(date.strftime('%Y/%m/%d/'), path))


def main():
    # get the most recent date
    date = get_most_current_data_downloaded()
    downloaded = False
    while True:
        # add a day to the current date 
        date += datetime.timedelta(days=1)
        # check if there are any files to be downloaded, i.e. the returned list from listing the aws directory contains files
        if len(list_aws_dir(date.strftime('%Y/%m/%d/'))):
            # download them if true
            download_data(date, make_date_dir(date))
            downloaded = True
        else:
            break
    if not downloaded:
        print('Up to date...')
    else:
        print('All files are done downloading...')
main()
