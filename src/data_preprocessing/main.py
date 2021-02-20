import os
import sys
import pandas as pd
import argparse

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(head)

from src.data_preprocessing.extract_MIMIC_data.extract_labels.make_labels import make_labels
from src.data_preprocessing.extract_MIMIC_data.extract_features.make_data import MakeData
from src.data_preprocessing.extract_MIMIC_data.extract_features.match_controls import match_controls
from src.data_preprocessing.features_preprocessing.stepI_data_prep import DataPreprocessing
from src.data_preprocessing.features_preprocessing.stepII_split_sets_n_normalise import MakeSetsAndNormalise
from src.data_preprocessing.features_preprocessing.stepIII_GP_prep import CompactTransform
from src.data_preprocessing.features_preprocessing.stepIV_GP_prep_part_II import GPPreprocessingSecondRound

def make_dirs():
    data_path = os.path.join(head, 'data')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    interim_path = os.path.join(data_path, 'interim')
    if not os.path.isdir(interim_path):
        os.mkdir(interim_path)
    processed_path = os.path.join(data_path, 'processed')
    if not os.path.isdir(processed_path):
        os.mkdir(processed_path)


def main(args):
    # create directories
    make_dirs()

    # generate sepsis labels
    labels = make_labels(args.connect_key, args.path)
    labels.generate_SI_data()
    labels.generate_SOFA_data()
    labels.generate_all_sepsis_onset()
    labels.filter_first_sepsis_onset()
    labels.save_to_postgres()
    labels.generate_sofa_delta_table()

    # generate data to feed in model
    data = MakeData(args.connect_key)
    data.step1_cohort()
    match_controls()
    data.step3_match_controls_to_sql()
    data.step4_extract_data()
    data.step4_extract_MR_data()

    # merge extracted data, normalise TS length, run basic tests
    interim_path = os.path.join(head, 'data', 'interim')
    files = ["static_variables.csv",
             "static_variables_cases.csv",
             "static_variables_controls.csv",
             "case_55h_hourly_vitals_ex1c.csv",
             "control_55h_hourly_vitals_ex1c.csv",
             "case_55h_hourly_labs_ex1c.csv",
             "control_55h_hourly_labs_ex1c.csv"]
    cas_f = os.path.join(interim_path, files[1])
    cos_f = os.path.join(interim_path, files[2])
    cav_f = os.path.join(interim_path, files[3])
    cov_f = os.path.join(interim_path, files[4])
    cal_f = os.path.join(interim_path, files[5])
    col_f = os.path.join(interim_path, files[6])

    print("Initialising", flush=True)
    first_processing = DataPreprocessing(cas_f, cos_f, cav_f, cov_f, cal_f, col_f,)
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
    first_processing.get_onset_hour()
    print("merge l & v", flush=True)
    first_processing.merge_labs_vitals()
    print("filter", flush=True)
    first_processing.filter_time_window()
    print("merge ca & co", flush=True)
    first_processing.merge_case_control()
    print("check ts lengths", flush=True)
    first_processing.ts_length_checks()
    print("static_prep", flush=True)
    first_processing.static_prep()

    # normalise, separate sets
    final_data_var_path = os.path.join(head, 'data', 'processed', 'full_labvitals_horizon_0_last.csv')
    final_data_stat_path = os.path.join(head, 'data', 'processed', 'full_static.csv')
    sets_n_norm = MakeSetsAndNormalise(final_data_var_path, final_data_stat_path)
    sets_n_norm.load_data()
    try:
        sets_n_norm.load_split()
    except FileNotFoundError:
        sets_n_norm.split()
    sets_n_norm.normalise()
    sets_n_norm.save()

    # GP models features
    for outpath in ['train', 'val', 'test']:
        file = os.path.join(head, 'data', outpath, 'full_labvitals.csv')
        data = pd.read_csv(file)
        onsetfile = os.path.join(head, 'data', 'processed', 'onset_hours.csv')
        onset = pd.read_csv(onsetfile)
        GP_prepI = CompactTransform(data, onset, outpath)
        GP_prepI.calculation()
        GP_prepI.save()
        GP_prepII = GPPreprocessingSecondRound(outpath)
        GP_prepII.load_files()
        GP_prepII.discard_useless_files()
        GP_prepII.join_files()
        GP_prepII.save()
    

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--connect_key",
                        default="dbname=mimic user=postgres password=postgres host=localhost options=--search_path=mimiciii",
                        help="key to enter the DB, eg: 'dbname=mimic user=postgres password=postgres options=--search_path=mimiciii'")
    parser.add_argument("-p", "--path", default="/cluster/home/mrosnat/MGP-AttTCN",
                        help="path to data folder - where you would like to save your data")

    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    main(args)
