import numpy as np
import pandas as pd
import datetime as dt

fraction = 64

PATH_TO_PROCESSED_DATA = '../../data/'

data = pd.read_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_tr.txt', sep='\t', dtype={'ItemId':np.int64})
train = data
length = len(data['ItemId'])

print('Full Training Set:\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(data), data.SessionId.nunique(), data.ItemId.nunique()))

print("\nGetting most recent 1/{} fraction of training test...\n".format(fraction))
first_session = train.iloc[length-length//fraction].SessionId
train = train.loc[train['SessionId'] >= first_session]

itemids = train['ItemId'].unique()
n_items = len(itemids)

print('Fractioned train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_fraction_1_{}.txt'.format(fraction), sep='\t', index=False)