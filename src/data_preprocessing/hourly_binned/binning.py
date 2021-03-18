import os
import sys
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
sys.path.append(head)
from src.data_preprocessing.features_preprocessing.stepII_split_sets_n_normalise import MakeSetsAndNormalise

def main(labels):
    if labels is None or labels=='rosnati':
        start_path = os.path.join(head, 'data', 'processed', 'full_labvitals_horizon_0_last.csv')
        end_path = os.path.join(head, 'data', 'processed', 'full_labvitals_binned.csv')
        stat_path = os.path.join(head, 'data', 'processed', 'full_static.csv')
    elif labels == 'moor':
        start_path = os.path.join(head, 'data', 'moor', 'processed', 'full_labvitals_horizon_0_last.csv')
        end_path = os.path.join(head, 'data', 'moor', 'processed', 'full_labvitals_binned.csv')
        stat_path = os.path.join(head, 'data', 'moor', 'processed', 'full_static.csv')
    else:
        raise NameError
    # subselect variables

    df = pd.read_csv(start_path)

    # calculate mean per hour
    df['chart_time_bucket'] = df['chart_time'].apply(lambda x: np.round(x).astype(int))
    df = df.groupby(['icustay_id', 'chart_time_bucket'], as_index=False).mean().drop_duplicates()

    # fill
    IDs = df.icustay_id.unique().tolist()
    for ID in tqdm(IDs):
        df.loc[df.icustay_id == ID] = df.loc[df.icustay_id == ID].fillna(method='ffill').fillna(method='bfill')

    df = df.drop(columns=['chart_time']).rename(columns={'chart_time_bucket' : 'chart_time'})
    df.to_csv(end_path)

    # normalise data
    moor_path = os.path.join(head, 'data', 'moor',)
    sets_n_norm = MakeSetsAndNormalise(end_path, stat_path)
    sets_n_norm.load_data()
    sets_n_norm.load_split(path=moor_path if labels == 'moor' else None)
    sets_n_norm.normalise(file_name='hello')
    sets_n_norm.save(file_name='binned', path=moor_path if labels == 'moor' else None)




