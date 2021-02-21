import os
import pickle
import random
import sys
import pandas as pd

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir,  os.pardir))
sys.path.append(head)

"""
Note: this file assumes that you have already generated all files in
make_hourly_data/static_data and inclusioncrit.sql
TODO: change file source for the files above

"""


class MakeSetsAndNormalise:

    def __init__(self, final_data_var_path, final_data_stat_path,
                 split_file='sets_split.pkl', new_split_file='sets_split_split_2.pkl'):
        self.stat_data_path = final_data_stat_path
        self.var_data_path = final_data_var_path
        self.split_file = split_file
        self.new_split_file = new_split_file
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
        print('working out of the assumption that head is ', self.head)

    def load_data(self):
        self.stat_data = pd.read_csv(self.stat_data_path)
        if "Unnamed: 0" in self.stat_data.columns:
            self.stat_data.drop(columns="Unnamed: 0")
        self.var_data = pd.read_csv(self.var_data_path)
        if "Unnamed: 0" in self.var_data.columns:
            self.var_data.drop(columns="Unnamed: 0")

    def split(self):
        file = os.path.join(self.head, 'data', self.split_file)
        all_icus = self.stat_data.icustay_id.unique().tolist()
        random.shuffle(all_icus)
        no_icus = len(all_icus)
        self.sets = {
            "train": all_icus[:int(no_icus * 0.8)],
            "val": all_icus[int(no_icus * 0.8): int(no_icus * 0.9)],
            "test": all_icus[int(no_icus * 0.9):]
        }
        f = open(file, "wb")
        pickle.dump(self.sets, f)
        f.close()

    def load_split(self):
        file = os.path.join(self.head, 'data', self.split_file)
        with open(file, "rb") as f:
            self.sets = pickle.load(f)

    def new_splits(self):
        all_non_test_icus = self.sets['train'] + self.sets['val']
        random.shuffle(all_non_test_icus)
        self.new_sets = {
            "train": all_non_test_icus[:len(self.sets['train'])],
            "val": all_non_test_icus[len(self.sets['train']): len(self.sets['train']) + len(self.sets['val'])],
            "test": self.sets['test']
        }
        file = os.path.join(self.head, 'data', self.new_split_file)
        f = open(file, "wb")
        pickle.dump(self.new_sets, f)
        f.close()
        self.sets = self.new_sets


    def normalise(self, file_name=None):
        # first re-order columns if needed
        cols = list(self.var_data.columns)
        start_col = ['label', 'icustay_id', 'chart_time', 'subject_id', 'sepsis_target']
        if cols[:5] != start_col:
            for col in start_col:
                cols.remove(col)
            cols = start_col + cols
            self.var_data = self.var_data[cols]
        mean = self.var_data.loc[(self.var_data.icustay_id.isin(self.sets["train"])), cols[5:]].mean(axis=0)
        std = (self.var_data.loc[(self.var_data.icustay_id.isin(self.sets["train"])), cols[5:]] - mean).std(axis=0)
        self.var_data[cols[5:]] = (self.var_data[cols[5:]] - mean) / std
        if file_name is not None:
            mean = self.stat_data.loc[(self.stat_data.icustay_id.isin(self.sets["train"])), "admission_age"].mean()
            std = (self.stat_data.loc[(self.stat_data.icustay_id.isin(self.sets["train"])), "admission_age"]
                   - mean).std()
            self.stat_data.admission_age = (self.stat_data.admission_age - mean) / std


    def save(self, file_name=None):
        path = os.path.join(self.head, 'data')
        sets_names = ["train", "val", "test"]
        if file_name is None:
            full_static = "full_static.csv"
            full_labvitals = "full_labvitals.csv"
        else:
            full_static = "full_static_{}.csv".format(file_name)
            full_labvitals = "full_labvitals_{}.csv".format(file_name)
        for set in sets_names:
            if not os.path.exists(path + set):
                os.makedirs(path + set)
            self.stat_data[self.stat_data.icustay_id.isin(self.sets[set])].to_csv(os.path.join(path, set, full_static))
            self.var_data[self.var_data.icustay_id.isin(self.sets[set])].to_csv(os.path.join(path, set,  full_labvitals))


