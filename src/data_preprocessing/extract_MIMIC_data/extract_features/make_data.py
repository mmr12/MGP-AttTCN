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

    def __init__(self, sqluser='mrosnati',
                 schema_write_name='mimic3_mrosnati',
                 schema_read_name='mimic3'):
        """
        Initialise function
        :param sqluser:             user name
        :param schema_write_name:   schema with write access
        :param schema_read_name:    schema where mimic is saved
        """
        # specify user/password/where the database is
        self.sqluser = sqluser
        self.sqlpass = ''
        self.dbname = 'mimic3'
        self.host = 'lm-db-01.leomed.ethz.ch'
        self.query_schema = 'SET search_path to ' + schema_write_name + ',' + schema_read_name + ';'
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.engine = create_engine('postgresql+psycopg2://{0}:{1}@{2}:5432/mimic3'.format(self.sqluser,
                                                                                           self.sqlpass,
                                                                                           self.host))

    def create_table(self, sqlfile):
        ##
        conn = psycopg2.connect(dbname=self.dbname,
                                user=self.sqluser,
                                password=self.sqlpass,
                                host=self.host)
        cur = conn.cursor()
        file = self.cwd + sqlfile
        with open(file, 'r') as openfile:
            query = openfile.read()
        openfile.close()
        cur.execute(self.query_schema + query)
        conn.commit()
        conn.close()

    def build_df(self, q_text):
        con = psycopg2.connect(dbname=self.dbname,
                               user=self.sqluser,
                               password=self.sqlpass,
                               host=self.host)
        query = self.query_schema + q_text
        return pd.read_sql_query(query, con)

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
        mc = pd.read_csv(path + file)
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
        conn = psycopg2.connect(dbname=self.dbname,
                                user=self.sqluser,
                                password=self.sqlpass,
                                host=self.host)
        cur = conn.cursor()
        cur.execute(self.query_schema + "drop table IF EXISTS matched_controls_hourly cascade")
        conn.commit()
        mc[mc.columns].to_sql("matched_controls_hourly",
                              self.engine,
                              if_exists='append',
                              schema="mimic3_mrosnati",
                              dtype=types)
        t_print("saved")

    def step4_extract_data(self):

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

        q = ["select * from icustay_static",
             "select * from icustay_static st inner join cases_hourly_ex1c ch on st.icustay_id=ch.icustay_id",
             "select * from icustay_static st inner join matched_controls_hourly ch on st.icustay_id=ch.icustay_id",
             "select * from case_55h_hourly_vitals_ex1c cv order by cv.icustay_id, cv.chart_time",
             "select * from control_55h_hourly_vitals_ex1c cv order by cv.icustay_id, cv.chart_time",
             "select * from case_55h_hourly_labs_ex1c cl order by cl.icustay_id, cl.chart_time",
             "select * from control_55h_hourly_labs_ex1c cl order by cl.icustay_id, cl.chart_time"
             ]
        files = ["static_variables.csv",
                 "static_variables_cases.csv",
                 "static_variables_controls.csv",
                 "case_55h_hourly_vitals_ex1c.csv",
                 "control_55h_hourly_vitals_ex1c.csv",
                 "case_55h_hourly_labs_ex1c.csv",
                 "control_55h_hourly_labs_ex1c.csv"]

        # first do static files
        for i in range(3):
            print_time()
            t_print(files[i])
            self.build_df(q[i]).to_csv(os.path.join(path, files[i]))

        # then do data extraction: group together all readings per timestamp
        for i in range(3, len(files)):
            print_time()
            t_print(files[i])
            temp = self.build_df(q[i])
            temp.groupby(["icustay_id", "chart_time"], as_index=False).mean().to_csv(os.path.join(path, files[i]))

    def step4_extract_MR_data(self):
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
        q = ["""select 
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

        files = ["case_55h_hourly_vitals_mr_features.csv",
                 "control_55h_hourly_vitals_mr_features.csv",
                 "case_55h_hourly_labs_mr_features.csv",
                 "control_55h_hourly_labs_mr_features.csv"]

        # then do data extraction: group together all readings per timestamp
        for i in range(len(files)):
            print_time()
            t_print(files[i])
            temp = self.build_df(q[i])
            temp.groupby(["icustay_id", "chart_time"], as_index=False).mean().to_csv(os.path.join(path, files[i]))

