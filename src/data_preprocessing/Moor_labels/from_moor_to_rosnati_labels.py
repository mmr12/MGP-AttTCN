import os
import sys
import pandas as pd
import argparse

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
sys.path.append(head)
from src.data_preprocessing.features_preprocessing.stepI_data_prep import DataPreprocessing
from src.data_preprocessing.features_preprocessing.stepII_split_sets_n_normalise import MakeSetsAndNormalise
from src.data_preprocessing.features_preprocessing.stepIII_GP_prep import CompactTransform
from src.data_preprocessing.features_preprocessing.stepIV_GP_prep_part_II import GPPreprocessingSecondRound

def moor_labels_main():
    # merge extracted data, normalise TS length, run basic tests
    interim_path = os.path.join(head, 'data', 'moor' ,'interim')
    processed_path = os.path.join(head, 'data', 'moor' ,'processed')
    files = ["static_variables.csv",
             "static_variables_cases.csv",
             "static_variables_controls.csv",
             "case_55h_hourly_vitals_ex1c.csv",
             "control_55h_hourly_vitals_ex1c.csv",
             "case_55h_hourly_labs_ex1c.csv",
             "control_55h_hourly_labs_ex1c.csv", ]

    cas_f = os.path.join(interim_path, files[1])
    cos_f = os.path.join(interim_path, files[2])
    cav_f = os.path.join(interim_path, files[3])
    cov_f = os.path.join(interim_path, files[4])
    cal_f = os.path.join(interim_path, files[5])
    col_f = os.path.join(interim_path, files[6])

    print("Initialising", flush=True)
    first_processing = DataPreprocessing(cas_f, cos_f, cav_f, cov_f, cal_f, col_f, )
    print("load_static", flush=True)
    first_processing.load_static()
    print("load_labs", flush=True)
    first_processing.load_labs()
    print("load_vitals", flush=True)
    first_processing.load_vitals()
    print("dropping unnamed columns", flush=True)
    first_processing.drop_all_unnamed()
    print("get onset 4 all", flush=True)
    # TODO this breaks
    first_processing.get_onset_hour(savepath=processed_path)
    print("merge l & v", flush=True)
    first_processing.merge_labs_vitals()
    print("filter", flush=True)
    first_processing.filter_time_window()
    print("merge ca & co", flush=True)
    first_processing.merge_case_control(savepath=interim_path)
    print("check ts lengths", flush=True)
    first_processing.ts_length_checks(savepath=processed_path)
    print("static_prep", flush=True)
    first_processing.static_prep(savepath=processed_path)

    # normalise, separate sets
    moor_path = os.path.join(head, 'data', 'moor' )
    final_data_var_path = os.path.join(head, 'data', 'moor','processed', 'full_labvitals_horizon_0_last.csv')
    final_data_stat_path = os.path.join(head, 'data','moor', 'processed', 'full_static.csv')
    sets_n_norm = MakeSetsAndNormalise(final_data_var_path, final_data_stat_path)
    sets_n_norm.load_data()
    try:
        sets_n_norm.load_split(moor_path)
    except FileNotFoundError:
        sets_n_norm.split(moor_path)
    sets_n_norm.normalise()
    sets_n_norm.save(path=moor_path)

    # GP models features
    for outpath in ['train', 'val', 'test']:
        file = os.path.join(head, 'data', 'moor', outpath, 'full_labvitals.csv')
        data = pd.read_csv(file)
        onsetfile = os.path.join(head, 'data','moor', 'processed', 'onset_hours.csv')
        onset = pd.read_csv(onsetfile)
        GP_prepI = CompactTransform(data, onset, outpath)
        GP_prepI.calculation()
        GP_prepI.save(moor_path)
        GP_prepII = GPPreprocessingSecondRound(outpath)
        GP_prepII.path = os.path.join(moor_path, outpath)
        GP_prepII.load_files()
        GP_prepII.discard_useless_files()
        GP_prepII.join_files()
        GP_prepII.save()

if __name__ == "__main__":
    moor_labels_main()
