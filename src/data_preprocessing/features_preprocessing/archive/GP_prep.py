import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

# appending head path
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
sys.path.append(head)
from src.utils.debug import *


class CompactTransform:
    def __init__(self, data, onset_h, outpath):
        self.data = data
        self.onset_h = onset_h
        self.outpath = outpath

    def calculation(self, features=None):
        """
        ids                  - ids //
        values               - array of list of observations
        ind_lvs              - array of list of variable observed
        times                - array of list of observed times
        num_rnn_grid_times   - num of grid times
        rnn_grid_times       - hourly range to celing of end_time (grid times)
        labels               - label
        ind_times            - copy of times (legacy)
        num_obs_times        - # of times with observations (diff from # obs vals as multiple vals can be obs at same t)
        num_obs_values       - duh
        onset_hour           - duh
        """
        # reformatting self.data so that all numeric values are in one column and a new column indicates the var_id
        if features is None:
            variables = ['sysbp', 'diabp', 'meanbp', 'resprate', 'heartrate',
                         'spo2_pulsoxy', 'tempc', 'cardiacoutput', 'tvset', 'tvobserved',
                         'tvspontaneous', 'peakinsppressure', 'totalpeeplevel', 'o2flow', 'fio2',
                         'albumin', 'bands', 'bicarbonate', 'bilirubin', 'creatinine',
                         'chloride', 'glucose', 'hematocrit', 'hemoglobin', 'lactate',
                         'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium', 'bun', 'wbc',
                         'creatinekinase', 'ck_mb', 'fibrinogen', 'ldh', 'magnesium',
                         'calcium_free', 'po2_bloodgas', 'ph_bloodgas', 'pco2_bloodgas',
                         'so2_bloodgas', 'troponin_t']

        elif features == 'mr_features':
            variables = ['sysbp', 'diabp', 'meanbp', 'resprate', 'heartrate', 'spo2_pulsoxy',
                         'tempc', 'bicarbonate', 'creatinine', 'chloride', 'glucose',
                         'hematocrit', 'hemoglobin', 'lactate', 'platelet', 'potassium', 'ptt',
                         'inr', 'pt', 'sodium', 'bun', 'wbc', 'magnesium', 'ph_bloodgas']
        else:
            return 1

        # initialise
        var_id = len(variables)
        var = variables[-1]
        self.data["var_id"] = var_id - 1
        moddata = self.data.loc[~self.data[var].isna(), ["icustay_id", var, "chart_time", "var_id", "label"]]
        moddata.rename(columns={var: "value"}, inplace=True)
        # loop
        for var_id, var in enumerate(variables[:-1]):
            self.data["var_id"] = var_id
            temp = self.data.loc[~self.data[var].isna(), ["icustay_id", var, "chart_time", "var_id", "label"]]
            temp.rename(columns={var: "value"}, inplace=True)
            moddata = moddata.append(temp, sort=False)
        self.data = moddata.sort_values(["icustay_id", "chart_time"], inplace=False)

        temp = self.data.groupby("icustay_id", as_index=False).max()
        # ids                  - ids
        ids = temp.icustay_id.to_numpy()
        end_time = temp.chart_time.to_numpy()
        # labels               - label
        labels = temp.label.to_numpy()
        # num_rnn_grid_times   - num of grid times
        num_rnn_grid_times = np.round(end_time + 1).astype(int)

        # values               - array of list of observations
        values = []
        # times                - array of list of observed times
        times = []
        # ind_lvs              - array of list of variable observed
        ind_lvs = []
        # rnn_grid_times       - hourly range to ceiling of end_time (grid times)
        rnn_grid_times = []
        for i, x in enumerate(ids):
            if i % 300 == 0: t_print("id iteration {}".format(i))
            values.append(self.data.loc[self.data.icustay_id == x, "value"].tolist())
            times.append(self.data.loc[self.data.icustay_id == x, "chart_time"].tolist())
            ind_lvs.append(self.data.loc[self.data.icustay_id == x, "var_id"].tolist())
            rnn_grid_times.append(np.arange(num_rnn_grid_times[i]))
        values = np.array(values)
        times = np.array(times)
        ind_lvs = np.array(ind_lvs)
        rnn_grid_times = np.array(rnn_grid_times)

        # ind_times            - copy of times (legacy)
        ind_times = times
        # num_obs_values       - duh
        num_obs_values = self.data.groupby("icustay_id").value.count().to_numpy()
        # num_obs_times        - # of times with observations
        #                        (diff from # obs vals as multiple vals can be obs at same t)
        num_obs_times = self.data[["icustay_id", "chart_time"]].groupby("icustay_id").chart_time.count().to_numpy()
        onset_hour = self.onset_h.loc[self.onset_h.icustay_id.isin(ids)]. \
            sort_values(by="icustay_id").onset_hour.to_numpy()
        self.result = [values, times, ind_lvs, ind_times,
                       labels, num_rnn_grid_times, rnn_grid_times,
                       num_obs_times, num_obs_values, onset_hour, ids]

    def save(self, features=None):
        if features is None:
            path = os.path.join(head, 'data', self.outpath, 'GP_prep.pkl')
        else:
            path = os.path.join(head, 'data', self.outpath, "/GP_prep_{}.pkl".format(features))
        with open(path, "wb") as f:
            pickle.dump(self.result, f)


def main(args):
    outpath = args.out_path
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(head, 'data')
    file = path + outpath + "/full_labvitals.csv"
    data = pd.read_csv(file)
    onsetfile = path + "processed/onset_hours.csv"
    onset = pd.read_csv(onsetfile)
    ct = CompactTransform(data, onset, outpath)
    ct.calculation()
    ct.save()


def modular_main(outpath, onset_file_name, mr_features, extension_name):
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir)) + "/data/"
    if extension_name is not None:
        file = path + outpath + "/full_labvitals_{}.csv".format(extension_name)
    else:
        file = path + outpath + "/full_labvitals.csv"
    data = pd.read_csv(file)
    onsetfile = path + "processed/" + onset_file_name
    onset = pd.read_csv(onsetfile)
    ct = CompactTransform(data, onset, outpath)
    if mr_features:
        ct.calculation(features='mr_features')
    else:
        ct.calculation()
    ct.save(features=extension_name)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path",
                        choices=['test', 'val', 'train'],
                        help="where to save the output files. Choose ['test','val','train']")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    main(args)
