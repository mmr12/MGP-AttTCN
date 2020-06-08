-- This query generates a row for every hour the patient is in the ICU.
-- The hours are based on clock-hours (i.e. 02:00, 03:00).
-- The hour clock starts 24 hours before the first heart rate measurement.
-- Note that the time of the first heart rate measurement is ceilinged to the hour.

-- this query extracts the cohort and every possible hour they were in the ICU
-- this table can be to other tables on ICUSTAY_ID and (ENDTIME - 1 hour,ENDTIME]
DROP MATERIALIZED VIEW IF EXISTS hadms_hours CASCADE;
CREATE MATERIALIZED VIEW hadms_hours as
-- get first/last measurement time
with all_hours as
(
  select
    ha.hadm_id

    -- ceiling the intime to the nearest hour by adding 59 minutes then truncating
    , date_trunc('hour', ha.admittime + interval '59' minute) as endtime

    -- create integers for each charttime in hours from admission
    -- so 0 is admission time, 1 is one hour after admission, etc, up to ICU disch
    , generate_series
    (
      -- allow up to 24 hours before ICU admission (to grab labs before admit)
      -24,
      ceil(extract(EPOCH from ha.dischtime - ha.admittime)/60.0/60.0)::INTEGER
    ) as hr

  from admissions ha
)
SELECT
  ah.hadm_id
  , ah.hr
  -- add the hr series
  -- endtime now indexes the end time of every hour for each patient
  , ah.endtime + ah.hr*interval '1' hour as endtime
from all_hours ah
order by ah.hadm_id, ah.hr;