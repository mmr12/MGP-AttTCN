import os
import sys

import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.types import Integer, DateTime, Numeric

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.append(head)
from src.utils.debug import *



class MakeData:

    def __init__(self, connect_key="dbname=mimic user=postgres host=localhost password=postgres options=--search_path=mimiciii",):
        """
        Initialise function
        :param sqluser:             user name
        :param schema_write_name:   schema with write access
        :param schema_read_name:    schema where mimic is saved
        """
        # specify user/password/where the database is
        self.connect_key = connect_key
        self.cwd = cwd
        self.dbname = connect_key.rsplit('dbname=')[1].rsplit(' ')[0]
        self.user = connect_key.rsplit('user=')[1].rsplit(' ')[0]
        self.password = connect_key.rsplit('password=')[1].rsplit(' ')[0]
        try:
            self.host = connect_key.rsplit('host=')[1].rsplit(' ')[0]
        except:
            self.host = 'localhost'
        try:
            self.port = connect_key.rsplit('port=')[1].rsplit(' ')[0]
        except:
            self.port = str(5432)
        print('working out of the assumption that cwd is ', self.cwd)

        self.engine = create_engine('postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}'.format(self.user,
                                                                                      self.password,
                                                                                      self.host,
                                                                                      self.port,
                                                                                      self.dbname))

    def create_table(self, sqlfile):
        conn = psycopg2.connect(self.connect_key)
        cur = conn.cursor()
        file = self.cwd + sqlfile
        with open(file, 'r') as openfile:
            query = openfile.read()
        openfile.close()
        cur.execute(query)
        conn.commit()
        conn.close()

    def build_df(self, q_text):
        conn = psycopg2.connect(self.connect_key)
        query = q_text
        return pd.read_sql_query(query, conn)

    def step1_cohort(self):
        t_print("welcome to step1")
        pre_file = "/sepsis3_cohort_mr.sql"
        file = "/hourly-cohort.sql"
        t_print("creating hourly cohort ...")
        start = time.time()
        self.create_table(pre_file)
        t_print(".. done 1/2 time: {}".format(time.time() - start))
        self.create_table(file)
        t_print(".. done 2/2 time: {}".format(time.time() - start))
        path = os.path.join(head, 'data', 'interim')
        file1 = "q13_cases_hourly_ex1c.csv"
        file2 = "q13_controls_hourly.csv"
        t_print("saving results ...")
        start = time.time()
        self.build_df("SELECT * FROM cases_hourly_ex1c").to_csv(os.path.join(path, file1))
        t_print("time : {}".format(time.time() - start))
        self.build_df("SELECT * FROM controls_hourly").to_csv(os.path.join(path, file2))
        t_print(".. done! time: {}".format(time.time() - start))

    def step3_match_controls_to_sql(self):
        path = os.path.join(head, 'data', 'interim')
        file = "q13_matched_controls.csv"
        t_print("reading csv..")
        mc = pd.read_csv(os.path.join(path, file))
        t_print("read")
        print_time()
        types = {"icustay_id": Integer(),
                 "hadm_id": Integer(),
                 "intime": DateTime(),
                 "outtime": DateTime(),
                 "length_of_stay": Numeric(),
                 "control_onset_hour": Numeric(),
                 "control_onset_time": DateTime(),
                 "matched_case_icustay_id": Integer()
                 }
        t_print("saving to SQL...")
        # somehow we cannot overwrite tables directly with "to_sql" so let's do that before
        conn = psycopg2.connect(self.connect_key)
        cur = conn.cursor()
        cur.execute("drop table IF EXISTS matched_controls_hourly cascade")
        conn.commit()
        mc[mc.columns].to_sql("matched_controls_hourly",
                              self.engine,
                              if_exists='append',
                              schema="mimiciii",
                              dtype=types)
        t_print("saved")

    def step4_extract_data(self):
        # read all SQL files
        files = ["/extract-55h-of-hourly-case-vital-series_ex1c.sql",
                 "/extract-55h-of-hourly-control-vital-series_ex1c.sql",
                 "/extract-55h-of-hourly-case-lab-series_ex1c.sql",
                 "/extract-55h-of-hourly-control-lab-series_ex1c.sql",
                 "/static-query.sql"]
        for file in files:
            print_time()
            t_print(file)
            self.create_table(file)
        path = os.path.join(head, 'data', 'interim')

        # save static files
        queries = ["select * from icustay_static",
             "select * from icustay_static st inner join cases_hourly_ex1c ch on st.icustay_id=ch.icustay_id",
             "select * from icustay_static st inner join matched_controls_hourly ch on st.icustay_id=ch.icustay_id",]
        files = ["static_variables.csv",
                 "static_variables_cases.csv",
                 "static_variables_controls.csv", ]
        for q, f in zip(queries, files):
            print_time()
            t_print(f)
            self.build_df(q).to_csv(os.path.join(path, f))

        # save time series files
        queries = ["""select 
                  icustay_id
                , subject_id
                , chart_time
                , sepsis_target
                , sysbp
                , diabp
                , meanbp
                , resprate
                , heartrate
                , spo2_pulsoxy
                , tempc
                from case_55h_hourly_vitals_ex1c cv order by cv.icustay_id, cv.chart_time""",
             """select 
                  icustay_id
                , subject_id
                , chart_time
                , pseudo_target
                , sysbp
                , diabp
                , meanbp
                , resprate
                , heartrate
                , spo2_pulsoxy
                , tempc
              from control_55h_hourly_vitals_ex1c cv order by cv.icustay_id, cv.chart_time""",
             """select 
                  icustay_id
                , subject_id
                , chart_time
                , sepsis_target
                , bicarbonate
                , creatinine
                , chloride
                , glucose
                , hematocrit
                , hemoglobin
                , lactate
                , platelet
                , potassium
                , ptt
                , inr
                , pt
                , sodium
                , bun
                , wbc
                , magnesium
                , ph_bloodgas
            from case_55h_hourly_labs_ex1c cl order by cl.icustay_id, cl.chart_time""",
             """select 
                  icustay_id
                , subject_id
                , chart_time
                , pseudo_target
                , bicarbonate
                , creatinine
                , chloride
                , glucose
                , hematocrit
                , hemoglobin
                , lactate
                , platelet
                , potassium
                , ptt
                , inr
                , pt
                , sodium
                , bun
                , wbc
                , magnesium
                , ph_bloodgas
                from control_55h_hourly_labs_ex1c cl order by cl.icustay_id, cl.chart_time"""
             ]

        files = ["vital_variables_cases.csv",
                 "vital_variables_controls.csv",
                 "lab_variables_cases.csv",
                 "lab_variables_controls.csv",]

        # then do data extraction: group together all readings per timestamp
        for q, f in zip(queries, files):
            print_time()
            t_print(f)
            temp = self.build_df(q)
            temp.groupby(["icustay_id", "chart_time"], as_index=False).mean().to_csv(os.path.join(path, f))

