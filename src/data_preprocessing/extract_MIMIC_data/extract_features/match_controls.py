'''
---------------------------------
Author: Michael Moor, 09.10.2018
---------------------------------

Summary: Script to match controls to cases based on icustay_ids and case/control ratio.
Input: it takes a 
    - cases csv, that links case icustay_id to sepsis_onset_hour (relative time after icu-intime in hours)
    - control csv listing control icustay_ids and corresponding icu-intime
Output:
    - matched_controls.csv that list controls the following way:
        icustay_id, control_onset_time, control_onset_hour, matched_case_icustay_id  
Detailed Description:
    1. Load Input files
    2. Determine Control vs Case ratio p (e.g. 10/1)
    3. Loop:
        For each case:
            randomly select (without repe) p controls as matched_controls
            For each selected control:
                append to result df: icustay_id, control_onset_time, control_onset_hour, matched_case_icustay_id
                (icustay_id of this control, the cases sepsis_onset_hour as control_onset_hour and the absolute time as control_onset_time, and the matched_case_icustay_id)
    4. return result df as output
'''

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir, os.pardir))


def match_controls():

    np.random.seed(42)
    result = pd.DataFrame()

    # --------------------
    # 1. Load Input files
    # --------------------

    casepath = os.path.join(head, 'data', 'interim', 'q13_cases_hourly_ex1c.csv')
    controlpath = os.path.join(head, 'data', 'interim', 'q13_controls_hourly.csv')
    outpath = os.path.join(head, 'data', 'interim', 'q13_matched_controls.csv')

    start = time.time()  # Get current time

    cases = pd.read_csv(casepath)  # read file to pd.dataframe
    unnamed = "Unnamed: 0"
    if unnamed in cases.columns:
        cases = cases.drop(columns="Unnamed: 0").drop_duplicates()
    else:
        cases = cases.drop_duplicates()

    controls = pd.read_csv(controlpath)  # read file to pd.dataframe, it has many duplicates (why?), remove them
    if unnamed in controls.columns:
        controls = controls.drop(columns="Unnamed: 0").drop_duplicates()  # drop duplicate rows
    else:
        controls = controls.drop_duplicates()  # drop duplicate rows
    controls = controls.reset_index(drop=True)  # resetting row index for aesthetic reasons

    controls['intime'] = controls['intime'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))  # convert time string to datetime object

    case_ids = cases.sort_values(by="sepsis_onset_hour")['icustay_id'].unique()  # get unique ids
    control_ids = controls['icustay_id'].unique()

    # --------------------------------
    # 2. Determine Control/Case Ratio
    # --------------------------------

    ratio = len(control_ids) / float(len(case_ids))
    rf = int(np.floor(ratio))  # rf is the ratio floored, to receive the largest viable integer ratio

    # ---------------------------------------------
    # 3. For each case match 'ratio-many' controls 
    # ---------------------------------------------

    controls_s = controls.sort_values(
        by="length_of_stay")  # Shuffle controls dataframe rows, for random control selection

    for i, case_id in enumerate(case_ids):
        matched_controls = controls_s[
                           int(i * ratio):int(
                               ratio * (i + 1))]  # select the next batch of controls to match to current case
        matched_controls = matched_controls.drop(columns=['delta_score', 'sepsis_onset'])  # drop unnecessary cols

        onset_hour = float(
            cases[cases['icustay_id'] == case_id]['sepsis_onset_hour'])  # get float of current case onset hour

        matched_controls[
            'control_onset_hour'] = onset_hour  # use sepsis_onset_hour of current case as control_onset_hour
        matched_controls['control_onset_time'] = matched_controls['intime'] + pd.Timedelta(
            hours=onset_hour)  # compute control_onset time w.r.t. control icu-intime
        matched_controls[
            'matched_case_icustay_id'] = case_id  # so that each matched control can be mapped back to its matched case

        result = result.append(matched_controls, ignore_index=True)

    # ---------------------------------------------------------------------
    # 4. Return matched controls: here, write to csv (as next step in sql) 
    # ---------------------------------------------------------------------

    result.to_csv(outpath, sep=',', index=False)  # write to csv, but without row indices
    print('Matching Controls RUNTIME: {} seconds'.format(time.time() - start))

    print('Number of Cases: {}'.format(len(case_ids)))
    print('Number of Controls: {}'.format(len(control_ids)))
    print('Number of Matched Controls: {}'.format(len(result)))
    print('Matching Ratio: {}, floored: {}'.format(ratio, rf))

