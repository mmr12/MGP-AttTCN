import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pickle
# head = os.getcwd()
# from src.data_preprocessing.hourly_binned.InSight_features import train_model

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
sys.path.append(head)


def load_horizon(labels, h, variables, static_variables, savename, force_extract, verbose):
    load = True
    out = {}
    if labels is None or labels == 'rosnati':
        starting_path = os.path.join(head, 'data')
    elif labels == 'moor':
        starting_path = os.path.join(head, 'data', 'moor')
    else:
        raise NameError
    for key in ['train', 'val','test']:
        save_path = os.path.join(starting_path, key, 'LogReg_{}_hz_{}.pkl'.format(savename, h))
        if not os.path.isfile(save_path):
            load = False
        else:
            with open(save_path, 'rb') as f:
                out[key] = pickle.load(f)
    if not load or force_extract:
        out = extract_horizon(labels, h, variables, static_variables, savename, verbose)
    return out

def extract_horizon(labels, h, variables, static_variables, savename, verbose):
    if labels is None or labels == 'rosnati':
        starting_path = os.path.join(head, 'data')
    elif labels == 'moor':
        starting_path = os.path.join(head, 'data', 'moor')
    else:
        raise NameError
    #
    dfs = {key: pd.read_csv(os.path.join(starting_path, key, 'full_labvitals_binned.csv'))
           for key in ['train', 'val','test']}
    dfs_stat = {key: pd.read_csv(os.path.join(starting_path, key, 'full_static_binned.csv'))
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
        data = np.zeros((len(IDs), 6, len(variables)))
        for i, ID in tqdm(enumerate(IDs)):
            n = df.loc[df.icustay_id == ID, 'time_to_onset'].max() \
                - df.loc[df.icustay_id == ID, 'time_to_onset'].min() + 1
            if n < 6:
                data[i, -n:] = df.loc[df.icustay_id == ID, variables]
            else:
                data[i] = df.loc[df.icustay_id == ID, variables]
        static = df_static.loc[df_static.icustay_id.isin(IDs), static_variables]
        out[key]['X'] = np.concatenate((static, data.reshape(len(IDs), -1)), -1)
        out[key]['y'] = df[['icustay_id', 'label']].drop_duplicates().label.to_numpy()
        out[key]['IDs'] = IDs
        with open(os.path.join(starting_path, key, 'LogReg_{}_hz_{}.pkl'.format(savename, h)), 'wb') as f:
            pickle.dump(out[key], f)
    return out


def large_main(labels, force_extract, verbose):
    variables = ['sysbp', 'diabp', 'meanbp', 'resprate', 'heartrate', 'spo2_pulsoxy',
                 'tempc', 'bicarbonate', 'creatinine', 'chloride', 'glucose',
                 'hematocrit', 'hemoglobin', 'lactate', 'platelet', 'potassium', 'ptt',
                 'inr', 'pt', 'sodium', 'bun', 'wbc', 'magnesium', 'ph_bloodgas']
    static_variables = ['admission_age', 'gender_M',
       'first_careunit_CCU', 'first_careunit_CSRU', 'first_careunit_MICU',
       'first_careunit_SICU', 'first_careunit_TSICU']
    Data = {}
    for hz in range(7):
        if verbose:
            print(hz)
        Data[hz] = load_horizon(labels, hz, variables, static_variables, 'extended', force_extract, verbose)
    return Data


def small_main(labels, force_extract, verbose):
    variables = ['sysbp', 'ptt', 'heartrate', 'tempc','resprate', 'wbc',  'ph_bloodgas', 'spo2_pulsoxy',   ]
    static_variables = ['admission_age',]
    Data = {}
    for hz in range(7):
        if verbose:
            print(hz)
        Data[hz] = load_horizon(labels, hz, variables, static_variables, 'original', force_extract, verbose)
    return Data


def train_model(data, labels, model_args, force_extract=False, verbose=False):
    if data == 'small':
        Data = small_main(labels, force_extract, verbose)
    elif data == 'large':
        Data = large_main(labels, force_extract, verbose)
    else:
        raise NameError
    X = np.concatenate([Data[h]['train']['X'] for h in Data], axis=0)
    y = np.concatenate([Data[h]['train']['y'] for h in Data], axis=0)

    model = LR(**model_args)
    model.fit(X, y)
    # overall
    AUROCs = {'train':[], 'val':[], 'test':[]}
    PR_AUCs = {'train':[], 'val':[], 'test':[]}
    for _set in ['train', 'val', 'test']:
        for h in Data:
            y = Data[h][_set]['y']
            y_hat = model.predict_proba(Data[h][_set]['X'])
            fpr, tpr, _ = roc_curve(y_true=y, y_score=y_hat[:, 1])
            AUROCs[_set].append(auc(fpr, tpr))
            #
            pre, rec, _ = precision_recall_curve(y_true=y, probas_pred=y_hat[:, 1])
            recall = rec[np.argsort(rec)]
            precision = pre[np.argsort(rec)]
            PR_AUCs[_set].append(auc(recall, precision))
            #
            print(h, _set, AUROCs[_set][-1], PR_AUCs[_set][-1])

    return AUROCs, PR_AUCs


