-- This query pivots the vital signs for the first 24 hours of a patient's stay
-- Vital signs include heart rate, blood pressure, respiration rate, and temperature

DROP MATERIALIZED VIEW IF EXISTS cardio_vitalsperhour CASCADE;
create materialized view cardio_vitalsperhour as
SELECT pvt.subject_id, pvt.hadm_id, pvt.HLOS

-- Easier names
, min(case when VitalID = 4 then valuenum else null end) as MinBP

FROM  (
  select ha.subject_id, ha.hadm_id
  , valuenum
  , case
    when itemid in (456,52,6702,443,220052,220181,225312) and valuenum > 0 and valuenum < 300 then 4 -- MeanBP
    else null end as VitalID
      -- convert F to C
  , (date_part('year', age(ce.charttime, ha.admittime))*365 * 24
    + date_part('month', age(ce.charttime, ha.admittime))*365/12 * 24
    + date_part('day', age(ce.charttime, ha.admittime))* 24
    + date_part('hour', age(ce.charttime, ha.admittime))
    + round(date_part('minute', age(ce.charttime, ha.admittime))/60)) as HLOS
  from admissions ha
  left join chartevents ce
  on ha.subject_id = ce.subject_id and ha.hadm_id = ce.hadm_id
  AND ce.charttime BETWEEN (ha.admittime - interval '1' day) AND ha.dischtime
  -- exclude rows marked as error
  and ce.error IS DISTINCT FROM 1
  where ce.itemid in
  (
  -- MEAN ARTERIAL PRESSURE
  456, --"NBP Mean"
  52, --"Arterial BP Mean"
  6702, --	Arterial BP Mean #2
  443, --	Manual BP Mean(calc)
  220052, --"Arterial Blood Pressure mean"
  220181, --"Non Invasive Blood Pressure mean"
  225312 --"ART BP mean"
  )
) pvt
group by pvt.subject_id, pvt.hadm_id, pvt.HLOS
order by pvt.subject_id, pvt.hadm_id, pvt.HLOS;