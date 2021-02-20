-- ------------------------------------------------------------------
-- Purpose: Create a view of the urine output for each ICUSTAY_ID over the first 24 hours.
-- ------------------------------------------------------------------

DROP MATERIALIZED VIEW IF EXISTS SOFA_uoperhour CASCADE;
create materialized view SOFA_uoperhour as

select
  -- patient identifiers
  ha.subject_id, ha.hadm_id, ie.icustay_id

  -- volumes associated with urine output ITEMIDs
  , sum(
      -- we consider input of GU irrigant as a negative volume
      case
        when oe.itemid = 227488 and oe.value > 0 then -1*oe.value
        else oe.value
    end) as UrineOutput
  , (date_part('year', age(oe.charttime, ha.admittime))*365 * 24
    + date_part('month', age(oe.charttime, ha.admittime))*365/12 * 24
    + date_part('day', age(oe.charttime, ha.admittime))* 24
    + date_part('hour', age(oe.charttime, ha.admittime))
    + round(date_part('minute', age(oe.charttime, ha.admittime))/60)) as HLOS

  , (date_part('year', age(oe.charttime, ie.intime))*365 * 24
    + date_part('month', age(oe.charttime, ie.intime))*365/12 * 24
    + date_part('day', age(oe.charttime, ie.intime))* 24
    + date_part('hour', age(oe.charttime, ie.intime))
    + round(date_part('minute', age(oe.charttime, ie.intime))/60)) as ICULOS

from admissions ha
-- Join to the outputevents table to get urine output
left join outputevents oe
-- join on all patient identifiers
on ha.subject_id = oe.subject_id and ha.hadm_id = oe.hadm_id
left join icustays ie
  on ie.icustay_id = oe.icustay_id
-- and ensure the data occurs during the first day
and oe.charttime between (ha.admittime - interval '1' day) AND ha.dischtime
where itemid in
(
-- these are the most frequently occurring urine output observations in CareVue
40055, -- "Urine Out Foley"
43175, -- "Urine ."
40069, -- "Urine Out Void"
40094, -- "Urine Out Condom Cath"
40715, -- "Urine Out Suprapubic"
40473, -- "Urine Out IleoConduit"
40085, -- "Urine Out Incontinent"
40057, -- "Urine Out Rt Nephrostomy"
40056, -- "Urine Out Lt Nephrostomy"
40405, -- "Urine Out Other"
40428, -- "Urine Out Straight Cath"
40086,--	Urine Out Incontinent
40096, -- "Urine Out Ureteral Stent #1"
40651, -- "Urine Out Ureteral Stent #2"

-- these are the most frequently occurring urine output observations in MetaVision
226559, -- "Foley"
226560, -- "Void"
226561, -- "Condom Cath"
226584, -- "Ileoconduit"
226563, -- "Suprapubic"
226564, -- "R Nephrostomy"
226565, -- "L Nephrostomy"
226567, --	Straight Cath
226557, -- R Ureteral Stent
226558, -- L Ureteral Stent
227488, -- GU Irrigant Volume In
227489  -- GU Irrigant/Urine Volume Out
)
group by ha.subject_id, ha.hadm_id, ie.icustay_id, HLOS, ICULOS
order by ha.subject_id, ha.hadm_id, ie.icustay_id, HLOS, ICULOS