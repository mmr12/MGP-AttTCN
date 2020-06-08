
-- The aim of this query is to pivot entries related to blood gases and
-- chemistry values which were found in LABEVENTS

-- things to check:
--  when a mixed venous/arterial blood sample are taken at the same time, is the store time different?

DROP MATERIALIZED VIEW IF EXISTS resp_bloodgasfirstday CASCADE;
create materialized view resp_bloodgasfirstday as
with pvt as
( -- begin query that extracts the data
  select ha.subject_id, ha.hadm_id
  -- here we assign labels to ITEMIDs
  -- this also fuses together multiple ITEMIDs containing the same data
      , case
        when itemid = 50800 then 'SPECIMEN' -- KEEP
        when itemid = 50801 then 'AADO2' -- KEEP
        when itemid = 50803 then 'BICARBONATE' -- KEEP
        when itemid = 50804 then 'TOTALCO2' -- KEEP
        when itemid = 50811 then 'HEMOGLOBIN' -- KEEP
        when itemid = 50813 then 'LACTATE' -- KEEP
        when itemid = 50815 then 'O2FLOW' -- KEEP
        when itemid = 50816 then 'FIO2' -- KEEP
        when itemid = 50817 then 'SO2' -- OXYGENSATURATION -- KEEP
        when itemid = 50818 then 'PCO2' -- KEEP
        when itemid = 50820 then 'PH' -- KEEP
        when itemid = 50821 then 'PO2' -- KEEP
        else null
        end as label
        , charttime
        , value
        -- add in some sanity checks on the values
        , case
          when valuenum <= 0 then null
          when itemid = 50810 and valuenum > 100 then null -- hematocrit
          -- ensure FiO2 is a valid number between 21-100
          -- mistakes are rare (<100 obs out of ~100,000)
          -- there are 862 obs of valuenum == 20 - some people round down!
          -- rather than risk imputing garbage data for FiO2, we simply NULL invalid values
          when itemid = 50816 and valuenum < 20 then null
          when itemid = 50816 and valuenum > 100 then null
          when itemid = 50817 and valuenum > 100 then null -- O2 sat
          when itemid = 50815 and valuenum >  70 then null -- O2 flow
          when itemid = 50821 and valuenum > 800 then null -- PO2
           -- conservative upper limit
        else valuenum
        end as valuenum

    from admissions ha
    left join labevents le
      on le.subject_id = ha.subject_id and le.hadm_id = ha.hadm_id
      and le.charttime between ha.admittime - interval '1' day and ha.dischtime -- MR add
      and le.ITEMID in
      -- blood gases
      (
        50800, 50801, 50803, 50804, 50811, 50813, 50815, 50816, 50817, 50818
        , 50820, 50821
      )
)
select pvt.SUBJECT_ID, pvt.HADM_ID, pvt.CHARTTIME
-- SPECIMEN
, max(case when label = 'SPECIMEN'          then value else null end) as SPECIMEN
-- SPECIMEN PROB
, max(case when label = 'AADO2'             then valuenum else null end) as AADO2 -- KEEP
, max(case when label = 'BICARBONATE'       then valuenum else null end) as BICARBONATE -- KEEP
, max(case when label = 'TOTALCO2'          then valuenum else null end) as TOTALCO2 -- KEEP
, max(case when label = 'HEMOGLOBIN'        then valuenum else null end) as HEMOGLOBIN -- KEEP
, max(case when label = 'LACTATE'           then valuenum else null end) as LACTATE -- KEEP
, max(case when label = 'O2FLOW'            then valuenum else null end) as O2FLOW -- KEEP
, max(case when label = 'SO2'               then valuenum else null end) as SO2 -- OXYGENSATURATION -- KEEP
, max(case when label = 'PCO2'              then valuenum else null end) as PCO2 -- KEEP
, max(case when label = 'PH'                then valuenum else null end) as PH -- KEEP
-- RESPIRATION VALS
, max(case when label = 'PO2'               then valuenum else null end) as PO2 -- KEEP
, max(case when label = 'FIO2'              then valuenum else null end) as FIO2 -- KEEP

from pvt
group by pvt.subject_id, pvt.hadm_id, pvt.CHARTTIME
order by pvt.subject_id, pvt.hadm_id, pvt.CHARTTIME;