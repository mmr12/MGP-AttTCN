import os
import sys

import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.types import Integer, DateTime

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir, os.pardir))
sys.path.append(head)
from src.utils.debug import *


class make_labels:

    def __init__(self, connect_key="dbname=mimic user=postgres password=postgres options=--search_path=mimiciii",
                 path='/cluster//home/mrosnat/MGP-AttTCN'):
        """
        Initialise function
        :param sqluser:             user name
        :param schema_write_name:   schema with write access
        :param schema_read_name:    schema where mimic is saved
        """
        # specify user/password/where the database is
        self.connect_key = connect_key
        self.path = path
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

    def generate_all_sepsis_onset(self):
        """
        Generate sepsis onset time for all hadm_id
        :return:
        """
        # step I : filter SOFA within SI window
        q = """
        select si.hadm_id
        , SOFA
        , SOFAresp
        , SOFAcoag
        , SOFAliv
        , SOFAcardio
        , SOFAgcs
        , SOFAren
        , so.hlos as h_from_admission
        , ha.admittime + so.hlos * interval '1 hour' as sepsis_time
        from SOFAperhour so 
        join SI_flag si 
        on si.hadm_id = so.hadm_id
        left join admissions ha 
        on ha.hadm_id = so.hadm_id
        where  ha.admittime + so.hlos * interval '1 hour' between si_start - interval '1 hour' and si_end
        and so.hlos>= 0
        order by hadm_id, sepsis_time
        """
        self.sofa_within_si = self.build_df(q)
        t_print("built sofa_within_si")
        # translate date into pandas format
        self.sofa_within_si["sepsis_time"] = self.sofa_within_si["sepsis_time"]. \
            apply(str).apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        # calculate the first derivative of each SOFA score part
        #       first look for the previous value

        self.sofa_within_si["h_in_SI_window"] = self.sofa_within_si.sort_values(['hadm_id', 'sepsis_time'],
                                                                                ascending=[True, True]) \
                                                    .groupby(['hadm_id']) \
                                                    .cumcount() - 1

        columns = ['sofa', 'sofaresp', 'sofacoag', 'sofaliv', 'sofacardio', 'sofagcs', 'sofaren']
        #       then calculate the delta

        t_print("calculating deltas ...")
        for col in columns:
            t_print("for col {}".format(col))
            self.sofa_within_si[col + "_temp"] = self.sofa_within_si[col]
            self.sofa_within_si[col + "_temp"].fillna(value=0, inplace=True)
            self.sofa_within_si[col + "_min"] = self.sofa_within_si.sort_values(['hadm_id', 'sepsis_time'],
                                                                                ascending=[True, True]) \
                .groupby(['hadm_id']) \
                .cummin()[col + "_temp"]
            self.sofa_within_si[col + "_delta"] = self.sofa_within_si[col + "_temp"] - self.sofa_within_si[col + "_min"]
            self.sofa_within_si.drop(columns=[col + "_temp"], inplace=True)

        self.sofa_within_si["sepsis_onset"] = 0
        self.sofa_within_si.loc[self.sofa_within_si.sofa_delta >= 2, "sepsis_onset"] = 1

        # save
        path = self.path + "/data/interim/"
        self.sofa_within_si.to_csv(path + "19-06-12-detailed-case-labels.csv")

    def filter_first_sepsis_onset(self):
        # rank occurrences of positive sepsis onset per hadm_id by timestamp
        self.sofa_within_si.loc[self.sofa_within_si.sepsis_onset == 1, "ranked_onsets"] = \
            self.sofa_within_si[self.sofa_within_si.sepsis_onset == 1].groupby("hadm_id").cumcount() + 1
        # filter by first occurrence
        self.sofa_within_si = self.sofa_within_si[(self.sofa_within_si.sepsis_onset == 1)
                                                  & (self.sofa_within_si.ranked_onsets == 1)]
        path = self.path + "/data/interim/"
        self.sofa_within_si.to_csv(path + "19-06-12-sepsis_onsets.csv")

    def save_to_postgres(self):
        engine = create_engine('postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}'.format(self.user,
                                                                                      self.password,
                                                                                      self.host,
                                                                                      self.port,
                                                                                      self.dbname))
        self.sofa_within_si.rename(columns={"sofa_delta": "delta_score"}, inplace=True)
        # somehow we cannot overwrite tables directly with "to_sql" so let's do that before
        conn = psycopg2.connect(self.connect_key)
        cur = conn.cursor()
        cur.execute("drop table IF EXISTS sepsis_onset cascade")
        conn.commit()
        # now let's fill it again
        self.sofa_within_si[['hadm_id',
                             'sofa',
                             'sofaresp',
                             'sofacoag',
                             'sofaliv',
                             'sofacardio',
                             'sofagcs',
                             'sofaren',
                             'sepsis_time',
                             'sepsis_onset',
                             'delta_score',
                             'sofaresp_delta', 'sofacoag_delta', 'sofaliv_delta',
                             'sofacardio_delta', 'sofagcs_delta', 'sofaren_delta']] \
            .to_sql("sepsis_onset",
                    engine,
                    if_exists='append',
                    schema="mimiciii",
                    dtype={"hadm_id": Integer(),
                           "sofa": Integer(),
                           'sofaresp': Integer(),
                           'sofacoag': Integer(),
                           'sofaliv': Integer(),
                           'sofacardio': Integer(),
                           'sofagcs': Integer(),
                           'sofaren': Integer(),
                           "sepsis_time": DateTime(),
                           "delta_score": Integer(),
                           'sofaresp_delta': Integer(),
                           'sofacoag_delta': Integer(),
                           'sofaliv_delta': Integer(),
                           'sofacardio_delta': Integer(),
                           'sofagcs_delta': Integer(),
                           'sofaren_delta': Integer()})
        self.create_table(sqlfile="/sofa_delta.sql")

    def generate_sofa_delta_table(self):
        self.create_table(sqlfile="/sofa_delta.sql")
        path = self.path + "/data/interim/"
        self.build_df("select * from sofa_delta").to_csv(path + "19-07-02-sofa-delta.csv")


    def generate_SI_data(self):
        """
        Generate tables to calculate suspicion of infection
        :return:
        """
        # generate list of antibiotics
        self.create_table(sqlfile="/SQL-SI/abx_poe_list.sql")
        # generate table with hadm_id and all suspicion of infection times
        self.create_table(sqlfile="/SQL-SI/abx_micro_poe.sql")
        # filter for first suspicion of infection time
        self.create_table(sqlfile="/SQL-SI/SI.sql")

    def generate_SOFA_data(self):
        start = time.time()
        print("Calculating SOFA score")
        print("Calculating cardio contribution ..")
        self.SOFA_cardio()
        print(".. done. Time taken: {} sec".format(time.time() - start))
        start = time.time()
        print("Calculating GCS contribution ..")
        self.SOFA_c_n_s()
        print(".. done. Time taken: {} sec".format(time.time() - start))
        start = time.time()
        print("Calculating coagulation contribution ..")
        self.SOFA_coag()
        print(".. done. Time taken: {} sec".format(time.time() - start))
        start = time.time()
        print("Calculating liver contribution ..")
        self.SOFA_liv()
        print(".. done. Time taken: {} sec".format(time.time() - start))
        start = time.time()
        print("Calculating renal contribution ..")
        self.SOFA_ren()
        print(".. done. Time taken: {} sec".format(time.time() - start))
        start = time.time()
        print("Calculating respiratory contribution ..")
        self.SOFA_resp()
        print(".. done. Time taken: {} sec".format(time.time() - start))
        self.SOFA_last_steps()

    def SOFA_cardio(self):
        path = "/SQL-SOFA/cardiovascular/"
        self.create_table(sqlfile=path + "echo.sql")
        self.create_table(sqlfile=path + "vitalsperhour.sql")
        self.create_table(sqlfile=path + "cardio_SOFA.sql")

    def SOFA_c_n_s(self):
        path = "/SQL-SOFA/central_nervous_system/"
        self.create_table(sqlfile=path + "gcsperhour.sql")

    def SOFA_coag(self):
        path = "/SQL-SOFA/coagulation/"
        self.create_table(sqlfile=path + "labsperhour.sql")

    def SOFA_liv(self):
        path = "/SQL-SOFA/liver/"
        self.create_table(sqlfile=path + "labsperhour.sql")

    def SOFA_ren(self):
        path = "/SQL-SOFA/renal/"
        self.create_table(sqlfile=path + "labsperhour.sql")
        self.create_table(sqlfile=path + "uoperhour.sql")
        self.create_table(sqlfile=path + "runninguo24h.sql")

    def SOFA_resp(self):
        path = "/SQL-SOFA/respiration/"
        self.create_table(sqlfile=path + "ventsettings.sql")
        self.create_table(sqlfile=path + "ventdurations.sql")
        self.create_table(sqlfile=path + "bloodgasfirstday.sql")
        self.create_table(sqlfile=path + "bloodgasfirstdayarterial.sql")
        self.create_table(sqlfile=path + "resp_SOFA.sql")

    def SOFA_last_steps(self):
        start = time.time()
        print("Combining score ..")
        path = "/SQL-SOFA/"
        self.create_table(sqlfile=path + "hourly_table.sql")
        self.create_table(sqlfile=path + "SOFA.sql")
        print(".. done. Time taken: {} sec".format(time.time() - start))
        # print("Creating SOFA increase flag ..")
        # start = time.time()
        # self.create_table(sqlfile=path + "SOFA_flag.sql")
        # print(".. done. Time taken: {} sec".format(time.time() - start))

    def create_table(self, sqlfile):
        ##
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
        con = psycopg2.connect(self.connect_key)
        query = q_text
        return pd.read_sql_query(query, con)

