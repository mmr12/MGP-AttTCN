import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
sys.path.append(head)
from src.utils.debug import t_print


class DataPreprocessing:

    def __init__(self,
                 cas_f, cos_f, cav_f, cov_f, cal_f, col_f,
                 horizon=0,
                 na_thres=500,
                 variable_start_index=5,
                 min_length=None,
                 max_length=None,
                 ):
        """

        :param cas_f:                   case static file
        :param cos_f:                   control static file
        :param cav_f:                   case vital file
        :param cov_f:
        :param cal_f:                   case labs file
        :param col_f:
        :param horizon:                 prediction horizon: data after this date will be discarded TODO: this should only happen at test time no?
        :param na_thres:                drop variables that don't at least have na_thres many measurements
        :param variable_start_index:    TODO: ??
        :param min_length:              time series min length in hours
        :param max_length:              time series max length in hours
        """
        self.horizon = horizon
        self.na_thres = na_thres
        self.variable_start_index = variable_start_index
        self.min_length = min_length
        self.max_length = max_length
        self.case_static_file = cas_f
        self.control_static_file = cos_f
        self.case_vitals_file = cav_f
        self.control_vitals_file = cov_f
        self.case_labs_file = cal_f
        self.control_labs_file = col_f
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
        print('working out of the assumption that head is ', self.head)

    def drop_unnamed(self, df):
        if "Unnamed: 0" in df.columns:
            df.drop(columns="Unnamed: 0", inplace=True)

    def txt_to_date(self, df, columns):
        """
        changing str to date in dataframe
        :param df: dataframe to convert
        :param columns: tables within dataframe to convert
        :return: converted dataframe
        """
        for t in columns:  # convert string (of times) to datetime objects
            df[t] = df[t].apply(str)
            df[t] = df[t].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    def load_static(self):
        self.case_static = pd.read_csv(self.case_static_file)
        self.txt_to_date(self.case_static, ['intime', 'sepsis_onset'])

        self.control_static = pd.read_csv(self.control_static_file)
        self.txt_to_date(self.control_static, ['intime', 'control_onset_time'])

    def load_vitals(self):
        self.case_vitals = pd.read_csv(self.case_vitals_file)
        self.txt_to_date(self.case_vitals, ["chart_time"])

        self.control_vitals = pd.read_csv(self.control_vitals_file)
        self.txt_to_date(self.control_vitals, ["chart_time"])

    def load_labs(self):
        self.case_labs = pd.read_csv(self.case_labs_file)
        self.txt_to_date(self.case_labs, ["chart_time"])

        self.control_labs = pd.read_csv(self.control_labs_file)
        # MM: had to include this dropna row as few severly incomplete records in labevents table
        self.control_labs = self.control_labs.dropna(subset=['chart_time'])
        # MM: re-enumerate the row index after removing few noisy rows missing chart_time (173)
        self.control_labs = self.control_labs.reset_index(drop=True)
        self.txt_to_date(self.control_labs, ["chart_time"])

    def drop_all_unnamed(self):
        dfs = [self.case_static, self.control_static,
               self.case_vitals, self.control_vitals,
               self.case_labs, self.control_labs]
        for df in dfs:
            self.drop_unnamed(df)

    def merge_labs_vitals(self):
        self.case_labvitals = pd.merge(self.case_vitals, self.case_labs, how='outer',
                                       left_on=['icustay_id', 'chart_time', 'subject_id', 'sepsis_target'],
                                       right_on=['icustay_id', 'chart_time', 'subject_id', 'sepsis_target'],
                                       sort=True)

        self.control_labvitals = pd.merge(self.control_vitals, self.control_labs, how='outer',
                                          left_on=['icustay_id', 'chart_time', 'subject_id', 'pseudo_target'],
                                          right_on=['icustay_id', 'chart_time', 'subject_id', 'pseudo_target'],
                                          sort=True)

    def get_onset_hour(self):
        self.onset_hours = pd.DataFrame(columns=['icustay_id',
                                                 'onset_hour'])
        # df to return mapping icustay_id to onset_hour (true sepsis onset or matched control onset)
        # prepare case and control info such that they can be joined into same df:
        case_hour = self.case_static[['icustay_id', 'sepsis_onset_hour']]
        control_hour = self.control_static[['icustay_id', 'control_onset_hour']]
        for dataset in [case_hour, control_hour]:
            new_cols = {x: y for x, y in zip(dataset.columns, self.onset_hours.columns)}
            dataset = dataset.rename(columns=new_cols)
            self.onset_hours = self.onset_hours.append(dataset, sort=False)
        path = os.path.join(self.head, 'data', 'processed')
        file = "onset_hours.csv"
        self.onset_hours.to_csv(os.path.join(path, file), index=False)

    def extract_window(self, data=None, static_data=None, onset_name=None, horizon=0):
        """
        :param data: table you want to filter (data table)
        :param static_data: table containing the max time you want to filter for
        :param onset_name: sepsis onset or equivalent
        :param horizon: horizon for prediction
        :return:
        """
        result = data.merge(static_data[["icustay_id", "intime", onset_name]], how="left", on="icustay_id")
        result = result[(result.intime <= result.chart_time) &
                        (result.chart_time <= result[onset_name] - pd.DateOffset(hours=horizon))]
        result["chart_time"] = (result["chart_time"] - result["intime"]) / np.timedelta64(1, 'h')
        result.drop(columns=["intime", onset_name], inplace=True)
        return result

    def filter_time_window(self):
        self.case_labvitals = self.extract_window(data=self.case_labvitals,
                                                  static_data=self.case_static,
                                                  onset_name='sepsis_onset',
                                                  horizon=self.horizon)
        self.control_labvitals = self.extract_window(data=self.control_labvitals,
                                                     static_data=self.control_static,
                                                     onset_name='control_onset_time',
                                                     horizon=self.horizon)

    def merge_case_control(self, file_name=None):
        # rename pseudo_target, such that case and controls can be appended to same df..
        self.control_labvitals = self.control_labvitals.rename(columns={'pseudo_target': 'sepsis_target'})

        # for joining label cases/controls with label: 1/0
        self.control_labvitals.insert(loc=0, column='label', value=np.repeat(0, len(self.control_labvitals)))
        self.case_labvitals.insert(loc=0, column='label', value=np.repeat(1, len(self.case_labvitals)))

        # append cases and controls, for spliting/standardizing:
        self.full_labvitals = self.case_labvitals.append(self.control_labvitals, sort=False)
        # drop chart_time index, so that on-the-fly df is identical with loaded one
        self.full_labvitals = self.full_labvitals.reset_index(drop=True)

        # save
        path = os.path.join(self.head, 'data', 'interim')
        if file_name is None:
            file = "full_labvitals_horizon_{}.csv".format(self.horizon)
        else:
            file = "full_labvitals_horizon_{}_{}.csv".format(self.horizon, file_name)

        self.full_labvitals.to_csv(os.path.join(path, file), index=False)

    def ts_length_checks(self, file_name=None):
        # drop variables that don't at least have na_thres many measurements..
        self.full_labvitals = self.full_labvitals.dropna(axis=1, thresh=self.na_thres)
        # drop too short time series samples.
        if self.min_length:
            cases_to_drop = self.case_static.loc[self.case_static.sepsis_onset_hour < self.min_length,
                                                 'icustay_id'].tolist()
            controls_to_drop = self.control_static.loc[self.control_static.control_onset_hour < self.min_length,
                                                       'icustay_id'].tolist()
            to_drop = cases_to_drop + controls_to_drop
            self.full_labvitals = self.full_labvitals[~self.full_labvitals.icustay_id.isin(to_drop)]
        # drop too long time series
        if self.max_length:
            to_drop = self.full_labvitals.loc[self.full_labvitals.chart_time > self.max_length, "icustay_id"].tolist()
            self.full_labvitals = self.full_labvitals[~self.full_labvitals.icustay_id.isin(to_drop)]
        path = os.path.join(self.head, 'data', 'processed')
        if file_name is None:
            file = "full_labvitals_horizon_{}_last.csv".format(self.horizon)
        else:
            file = "full_labvitals_horizon_{}_{}_last.csv".format(self.horizon, file_name)
        self.full_labvitals.to_csv(os.path.join(path, file), index=False)

    def static_prep(self):
        static_vars = ['icustay_id', 'gender', 'admission_age', 'ethnicity', 'first_careunit']
        all_stats = self.case_static[static_vars].append(self.control_static[static_vars], sort=False)
        # unify ethnicites
        black = ['BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 'BLACK/AFRICAN', 'BLACK/HAITIAN']
        white = ['WHITE', 'WHITE - RUSSIAN', 'WHITE - BRAZILIAN', 'PORTUGUESE']
        asian = ['ASIAN', 'ASIAN - VIETNAMESE', 'ASIAN - CHINESE', 'ASIAN - CAMBODIAN',
                 'ASIAN - ASIAN INDIAN', 'ASIAN - OTHER', 'ASIAN - FILIPINO']
        hispanic = ['HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC OR LATINO', 'HISPANIC/LATINO - SALVADORAN',
                    'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - MEXICAN', 'HISPANIC/LATINO - GUATEMALAN']
        notgiven = ['na', 'UNKNOWN/NOT SPECIFIED', 'UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER']
        middleeastern = ['MIDDLE EASTERN']
        other = ['NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'OTHER', 'MULTI RACE ETHNICITY']
        all_stats.loc[all_stats.ethnicity.isin(black), "ethnicity"] = 'black'
        all_stats.loc[all_stats.ethnicity.isin(white), "ethnicity"] = 'white'
        all_stats.loc[all_stats.ethnicity.isin(asian), "ethnicity"] = 'asian'
        all_stats.loc[all_stats.ethnicity.isin(hispanic), "ethnicity"] = 'hispanic'
        all_stats.loc[all_stats.ethnicity.isin(notgiven), "ethnicity"] = 'notgiven'
        all_stats.loc[all_stats.ethnicity.isin(middleeastern), "ethnicity"] = 'middle eastern'
        all_stats.loc[all_stats.ethnicity.isin(other), "ethnicity"] = 'other'
        # unify ages
        all_stats.admission_age = all_stats.admission_age.round(-1)
        all_stats.loc[all_stats.admission_age > 90, "admission_age"] = 90
        all_stats.admission_age = all_stats.admission_age.apply(str)
        # create one-hot vector
        self.full_static = pd.get_dummies(all_stats[static_vars[1:]])
        self.full_static.insert(loc=0, column='icustay_id', value=all_stats.icustay_id.tolist())
        path = os.path.join(self.head, 'data', 'processed')
        file = "full_static.csv"
        self.full_static.to_csv(os.path.join(path, file), index=False)

    def mr_static_prep(self):
        static_vars = ['icustay_id', 'admission_age', 'gender', 'first_careunit']
        all_stats = self.case_static[static_vars].append(self.control_static[static_vars], sort=False)
        all_stats.loc[all_stats.admission_age > 90, "admission_age"] = 90
        self.full_static = pd.get_dummies(all_stats[static_vars[2:]])
        self.full_static.insert(loc=0, column='admission_age', value=all_stats.admission_age.tolist())
        self.full_static.insert(loc=0, column='icustay_id', value=all_stats.icustay_id.tolist())
        path = os.path.join(self.head, 'data', 'processed')
        file = "full_static_mr_features.csv"
        self.full_static.to_csv(os.path.join(path, file), index=False)


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir)) + "/data/interim/"
    files = ["static_variables.csv",
             "static_variables_cases.csv",
             "static_variables_controls.csv",
             "case_55h_hourly_vitals_ex1c.csv",
             "control_55h_hourly_vitals_ex1c.csv",
             "case_55h_hourly_labs_ex1c.csv",
             "control_55h_hourly_labs_ex1c.csv"]
    cas_f = path + files[1]
    cos_f = path + files[2]
    cav_f = path + files[3]
    cov_f = path + files[4]
    cal_f = path + files[5]
    col_f = path + files[6]
    horizon = 0
    na_thres = 500
    min_length = None
    max_length = None
    t_print("Initialising")
    dp = DataPreprocessing(cas_f, cos_f, cav_f, cov_f, cal_f, col_f,
                           horizon=horizon, na_thres=na_thres,
                           min_length=min_length, max_length=max_length)
    t_print("load_static")
    dp.load_static()
    t_print("load_labs")
    dp.load_labs()
    t_print("load_vitals")
    dp.load_vitals()
    t_print("dropping unnamed columns")
    dp.drop_all_unnamed()
    t_print("get onset 4 all")
    dp.get_onset_hour()
    t_print("merge l & v")
    dp.merge_labs_vitals()
    t_print("filter")
    dp.filter_time_window()
    t_print("merge ca & co")
    dp.merge_case_control()
    t_print("check ts lengths")
    dp.ts_length_checks()
    t_print("static_prep")
    dp.static_prep()


if __name__ == '__main__':
    main()
