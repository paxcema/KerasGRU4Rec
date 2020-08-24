from matplotlib import pyplot as plt
import argparse
import numpy as np
import pandas as pd


def preprocess_df(df):    
    n_items = len(train_data['ItemId'].unique())
    aux = list(train_data['ItemId'].unique())
    itemids = np.array(aux)
    itemidmap = pd.Series(data=np.arange(n_items), index=itemids)  # (id_item => (0, n_items))
    
    item_key = 'ItemId'
    session_key = 'SessionId'
    time_key = 'Time'
    
    data = pd.merge(df, pd.DataFrame({item_key:itemids, 'ItemIdx':itemidmap[itemids].values}), on=item_key, how='inner')
    data.sort_values([session_key, time_key], inplace=True)

    length = len(data['ItemId'])
        
    return data


def compute_dwell_time(df):
    times_t = np.roll(df['Time'], -1)  # Take time row
    times_dt  = df['Time']             # Copy, then displace by one
    
    diffs = np.subtract(times_t, times_dt)  # Take the pairwise difference
    
    length = len(df['ItemId'])
    
    # cummulative offset start for each session
    offset_sessions = np.zeros(df['SessionId'].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = df.groupby('SessionId').size().cumsum() 
    
    offset_sessions = offset_sessions - 1
    offset_sessions = np.roll(offset_sessions, -1)
    
    # session transition implies zero-dwell-time
    # note: paper statistics do not consider null entries, 
    # though they are still checked when augmenting
    np.put(diffs.values, offset_sessions, np.zeros((offset_sessions.shape)), mode='raise')
    return diffs


def get_statistics(dts):
    filtered = np.array(list(filter(lambda x: int(x) != 0, dts)))
    pd_dts = pd.DataFrame(filtered)
    pd_dts.boxplot(vert=False, showfliers=False) # no outliers in boxplot
    plt.show()
    pd_dts.describe()


def join_dwell_reps(df, dt, threshold=2000):
    # Calculate d_ti/threshold + 1, add column to dataFrame
    dt //= threshold
    dt += 1   
    df['DwellReps'] = pd.Series(dt.astype(np.int64), index=dt.index)


def augment(df):    
    col_names = list(df.columns.values)[:3]
    print(col_names)
    augmented = np.repeat(df.values, df['DwellReps'], axis=0) 
    print(augmented[0][:3])  
    augmented = pd.DataFrame(data=augmented[:,:3],
                             columns=col_names)
    dtype = {'SessionId': np.int64, 
             'ItemId': np.int64, 
             'Time': np.float32}
    
    for k, v in dtype.items():
        augmented[k] = augmented[k].astype(v)

    return augmented


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DwellTime extractor')
    parser.add_argument('--train-path', type=str, default='../processedData/rsc15_train_tr.txt')
    parser.add_argument('--output-path', type=str, default='../processedData/augmented_train.csv')
    args = parser.parse_args()

    # load RSC15 preprocessed train dataframe
    train_data = pd.read_csv(args.train_path, sep='\t', dtype={'ItemId':np.int64})

    new_df = preprocess_df(train_data)
    dts = compute_dwell_time(new_df)

    # get_statistics(dts)

    join_dwell_reps(new_df, dts, threshold=200000)

    # Now, we augment the sessions copying each entry an additional (dwellReps[i]-1) times
    df_aug = augment(new_df)
    df_aug.to_csv(args.output_path, index=False, sep='\t')