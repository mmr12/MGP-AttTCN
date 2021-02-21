import argparse
import os
import pickle
import sys

import pandas as pd

# appending head path
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(head)


class GPPreprocessingSecondRound:
    def __init__(self, split, n_features=44):
        self.n_features = n_features
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
        print('working out of the assumption that head is ', self.head)
        self.path = os.path.join(self.head, "data", split)

    def load_files(self):

        file_path = os.path.join(self.path, "GP_prep.pkl")
        stat_file_path = os.path.join(self.path, "full_static.csv")
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
        self.static_data = pd.read_csv(stat_file_path)

    def discard_useless_files(self):
        # keep: Y, T, ind_T, ind_K, num_obs, X, num_X
        # atm values, times, ind_lvs, ind_times,
        #     labels, num_rnn_grid_times, rnn_grid_times,
        #     num_obs_times, num_obs_values, onset_hour, ids
        Y, T, ind_Y, _, labels, len_X, X, len_T, _, onset_hour, ids = self.data
        self.data = [Y, T, ind_Y, len_T, X, len_X, labels, ids, onset_hour]

    def join_files(self):
        self.static_data.set_index("icustay_id", inplace=True)
        self.static_data = self.static_data.loc[self.data[-2]]
        self.static_data.drop(columns="Unnamed: 0", inplace=True)
        self.static_data = self.static_data.to_numpy()
        self.data.append(self.static_data)

    def save(self):
        file_path = os.path.join(self.path, "GP_prep_v2.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self.data, f)
