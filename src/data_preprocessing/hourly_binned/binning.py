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

start_path = os.path.join(head, 'data', 'processed', 'full_labvitals_horizon_0_last.csv')
end_path = os.path.join(head, 'data', 'processed', 'full_labvitals_binned.csv')
stat_path = os.path.join(head, 'data', 'processed', 'full_static.csv')
# subselect variables

variables = ['sysbp', 'diabp', 'meanbp', 'resprate', 'heartrate', 'spo2_pulsoxy',
                         'tempc', 'bicarbonate', 'creatinine', 'chloride', 'glucose',
                         'hematocrit', 'hemoglobin', 'lactate', 'platelet', 'potassium', 'ptt',
                         'inr', 'pt', 'sodium', 'bun', 'wbc', 'magnesium', 'ph_bloodgas']

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
sets_n_norm = MakeSetsAndNormalise(end_path, stat_path)
sets_n_norm.load_data()
sets_n_norm.load_split()
sets_n_norm.normalise(file_name='hello')
sets_n_norm.save(file_name='binned')

# now let's calculate
df = sets_n_norm.var_data
df = df.sort_values(by=['icustay_id', 'chart_time'], ascending=[True, False])
df['time_to_onset'] = df.groupby(['icustay_id']).cumcount()
# filter for patients wgo have 6h of data, and only keep 6h of data
df = df.loc[(df.time_to_onset < 6)]
df = df.dropna()


IDs = df.icustay_id.unique().tolist()
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
    temp = (m_d1 - means[f1]) * (m_d2 - means[f2]) / (stds[f1]*stds[f2])
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

# todo: put together all this data, separate and run RidgeLogReg (should be quick?) (24 + 24 + 276 + 2024)
# todo: run RidgeLogReg without the Insight craze
# todo: run Insight craze on Insight features (see iPad for this!)
for key in sets_n_norm.sets:
    print(key, len(set(sets_n_norm.sets[key])),
          len(set(sets_n_norm.sets[key]) & set(IDs)))

ISfeatures = np.concatenate(())



