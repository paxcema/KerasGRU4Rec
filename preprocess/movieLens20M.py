import datetime as dt
import pandas as pd
import numpy as np
import keras
import time

def preprocess():
    ratings_df = pd.read_csv('./data/ratings.csv')

    # ALL TRAIN SET
    # Get only ratings between January 2008 to March 2013
    all_train_start = "09/01/1995"
    all_train_end = "01/03/2013"
    all_train_start_ts = time.mktime(dt.datetime.strptime(all_train_start, "%d/%m/%Y").timetuple())
    all_train_end_ts = time.mktime(dt.datetime.strptime(all_train_end, "%d/%m/%Y").timetuple())
    all_train_data = ratings_df.drop(['rating'], axis=1)

    # in date range
    all_train_data = all_train_data.loc[(all_train_data['timestamp'] >= all_train_start_ts) & (all_train_data['timestamp'] <= all_train_end_ts)]
    
    # only users 5 < rated_movies < 101
    all_train_data = all_train_data.groupby("userId").filter(lambda x: len(x) > 5 and len(x) < 101)

    # RECENT TRAIN SET
    # Get only ratings between January 2008 to March 2013
    train_start = "01/01/2008"
    train_end = "01/03/2013"
    train_start_ts = time.mktime(dt.datetime.strptime(train_start, "%d/%m/%Y").timetuple())
    train_end_ts = time.mktime(dt.datetime.strptime(train_end, "%d/%m/%Y").timetuple())
    train_data = ratings_df.drop(['rating'], axis=1)

    # in date range
    train_data = train_data.loc[(train_data['timestamp'] >= train_start_ts) & (train_data['timestamp'] <= train_end_ts)]
    # only users 5 < rated_movies < 101
    train_data = train_data.groupby("userId").filter(lambda x: len(x) > 5 and len(x) < 101)

    # DEV SET
    # Get only ratings between April 2014 to April 2015
    dev_start = "01/04/2013"
    dev_end = "01/04/2014"
    dev_start_ts = time.mktime(dt.datetime.strptime(dev_start, "%d/%m/%Y").timetuple())
    dev_end_ts = time.mktime(dt.datetime.strptime(dev_end, "%d/%m/%Y").timetuple())
    dev_data = ratings_df.drop(['rating'], axis=1)

    # in date range
    dev_data = dev_data.loc[(dev_data['timestamp'] >= dev_start_ts) & (dev_data['timestamp'] <= dev_end_ts)]
    # only users 5 < rated_movies < 101
    dev_data = dev_data.groupby("userId").filter(lambda x: len(x) > 5 and len(x) < 101)

    # TEST SET
    # Get only ratings between April 2015 to April 2016
    test_start = "02/04/2014"
    test_end = "01/04/2015"
    test_start_ts = time.mktime(dt.datetime.strptime(test_start, "%d/%m/%Y").timetuple())
    test_end_ts = time.mktime(dt.datetime.strptime(test_end, "%d/%m/%Y").timetuple())
    test_data = ratings_df.drop(['rating'], axis=1)

    # in date range
    test_data = test_data.loc[(test_data['timestamp'] >= test_start_ts) & (test_data['timestamp'] <= test_end_ts)]
    # only users 5 < rated_movies < 101
    test_data = test_data.groupby("userId").filter(lambda x: len(x) > 5 and len(x) < 101)

    all_train_data.columns = ['SessionId', 'ItemId', 'Time']
    train_data.columns = ['SessionId', 'ItemId', 'Time']
    dev_data.columns = ['SessionId', 'ItemId', 'Time']
    test_data.columns = ['SessionId', 'ItemId', 'Time']

    all_train_data.to_csv('./data/all_train.csv', sep='\t', index=False)
    train_data.to_csv('./data/train.csv', sep='\t', index=False)
    dev_data.to_csv('./data/dev.csv', sep='\t', index=False)
    test_data.to_csv('./data/test.csv', sep='\t', index=False)

if __name__ == '__main__':
    preprocess()