DROP TABLE IF EXISTS sofa_delta CASCADE;
CREATE TABLE sofa_delta AS
with case_cohort as (
    select
      so.hadm_id
    , so.sofa
    , so.sofaresp
    , so.sofacoag
    , so.sofaliv
    , so.sofacardio
    , so.sofagcs
    , so.sofaren
    , so.sepsis_time as sepsis_onset
    , so.delta_score
    , so.sofaresp_delta
    , so.sofacoag_delta
    , so.sofaliv_delta
    , so.sofacardio_delta
    , so.sofagcs_delta
    , so.sofaren_delta
    , ie.icustay_id
    , ie.intime
    , ie.outtime
    , (date_part('year', age(sepsis_time, intime))*365 * 24
        + date_part('month', age(sepsis_time, intime))*365/12 * 24
        + date_part('day', age(sepsis_time, intime))* 24
        + date_part('hour', age(sepsis_time, intime))
        + round(date_part('minute', age(sepsis_time, intime))/60)) as h_from_intime
    from sepsis_onset so
    left join icustays ie
    on so.hadm_id = ie.hadm_id
    where sepsis_time between intime and outtime

)
select C.*
, case when sepsis_onset is null then 0 else 1 end as septic
, case when h_from_intime < 7 then 1 else 0 end as excluded
, case when dg.icd9_code in ('78552','99591','99592') then 1 else 0 end as ICD_positive
from case_cohort C
left join icustays ie
  on C.icustay_id = ie.icustay_id
left join diagnoses_icd dg
  on C.hadm_id = dg.hadm_id