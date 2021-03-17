import os
import sys
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import pickle
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
sys.path.append(head)

def extract_horizon(h, variables, static_variables):
    dfs = {key: pd.read_csv(os.path.join(head, 'data', key, 'full_labvitals_binned.csv'))
           for key in ['train', 'val','test']}
    dfs_stat = {key: pd.read_csv(os.path.join(head, 'data', key, 'full_static_binned.csv'))
           for key in ['train', 'val','test']}
    out = {key: {} for key in dfs}
    for key in dfs:
        df = dfs[key]
        df_static = dfs_stat[key]
        # first subselect data of 5h prior to onset
        df = df.sort_values(by=['icustay_id', 'chart_time'], ascending=[True, False])
        df_static = df_static.sort_values(by='icustay_id')
        df['time_to_onset'] = df.groupby(['icustay_id']).cumcount()
        # filter for patients wgo have 6h of data, and only keep 6h of data
        df = df.loc[(df.time_to_onset < 6 + h) & (df.time_to_onset >= h)]
        df = df.dropna()
        IDs = df.icustay_id.unique().tolist()
        # then extract InSight features
        data = np.empty((len(IDs), 6, len(variables)))
        for i, ID in tqdm(enumerate(IDs)):
            n = df.loc[df.icustay_id == ID, 'time_to_onset'].max() + 1
            if n < 6:
                data[i, -n:] = df.loc[df.icustay_id == ID, variables]
            else:
                data[i] = df.loc[df.icustay_id == ID, variables]


        M = data.mean(1)
        D = data[:, -1] - data[:, 0]
        D_hat = np.zeros((len(IDs), 0))
        for f, feature in tqdm(enumerate(variables)):
            mini_data = D[:, f]
            mini_data_non_null = D[D != 0]
            q = np.quantile(mini_data_non_null, (1 / 3, 2 / 3))
            mini_data[mini_data < q[0]] = -1
            mini_data[(mini_data >= q[0]) & (mini_data < q[1])] = 0
            mini_data[mini_data >= q[1]] = 1
            D_hat = np.concatenate((D_hat, mini_data[:, np.newaxis]), -1)

        # correlations
        means = D.mean(0)
        stds = D.std(0)
        D2_hat = np.zeros((len(IDs), 0))
        for els in tqdm(itertools.combinations(np.arange(len(variables)), 2)):
            f1, f2 = els
            m_d1 = D[:, f1]
            m_d2 = D[:, f2]
            temp = (m_d1 - means[f1]) * (m_d2 - means[f2]) / (stds[f1] * stds[f2])
            q = np.quantile(temp[temp != 0], (1 / 3, 2 / 3))
            idx_neg = temp < q[0]
            idx_zero = (temp >= q[0]) & (temp < q[1])
            idx_pos = temp >= q[0]
            temp[idx_neg] = -1
            temp[idx_zero] = 0
            temp[idx_pos] = 1
            D2_hat = np.concatenate((D2_hat, temp[:, np.newaxis]), -1)


        D3_hat = np.zeros((len(IDs), 0))
        for els in tqdm(itertools.combinations(np.arange(len(variables)), 3)):
            f1, f2, f3 = els
            m_d1 = D[:, f1]
            m_d2 = D[:, f2]
            m_d3 = D[:, f3]
            temp = (m_d1 - means[f1]) * (m_d2 - means[f2]) * (m_d3 - means[f3]) / (stds[f1] * stds[f2] * stds[f3])
            q = np.quantile(temp[temp != 0], (1 / 3, 2 / 3))
            idx_neg = temp < q[0]
            idx_zero = (temp >= q[0]) & (temp < q[1])
            idx_pos = temp >= q[0]
            temp[idx_neg] = -1
            temp[idx_zero] = 0
            temp[idx_pos] = 1
            D3_hat = np.concatenate((D3_hat, temp[:, np.newaxis]), -1)


        static = df_static.loc[df_static.icustay_id.isin(IDs), static_variables]
        out[key]['X'] = np.concatenate((static, M, D_hat, D2_hat, D3_hat), -1)
        out[key]['y'] = df[['icustay_id', 'label']].drop_duplicates().label.to_numpy()
        out[key]['IDs'] = IDs
        with open(os.path.join(head, key, 'InSight_hz_{}.pkl'.format(h)), 'wb') as f:
            pickle.dump(out[key], f)
    return out
