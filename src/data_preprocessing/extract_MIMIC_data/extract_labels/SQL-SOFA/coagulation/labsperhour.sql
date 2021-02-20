-- This query pivots lab values taken in the first 24 hours of a patient's stay

-- Have already confirmed that the unit of measurement is always the same: null or the correct unit

DROP MATERIALIZED VIEW IF EXISTS coag_labsperhour CASCADE;
CREATE materialized VIEW coag_labsperhour AS
SELECT
  pvt.subject_id, pvt.hadm_id, pvt.HLOS
  , min(CASE WHEN label = 'PLATELET' THEN valuenum ELSE null END) as PLATELET

FROM
( -- begin query that extracts the data
  SELECT ha.subject_id, ha.hadm_id
  -- here we assign labels to ITEMIDs
  -- this also fuses together multiple ITEMIDs containing the same data
  , CASE
        WHEN itemid = 51265 THEN 'PLATELET'
      ELSE null
    END AS label
  , -- add in some sanity checks on the values
  -- the where clause below requires all valuenum to be > 0, so these are only upper limit checks
    CASE
      WHEN itemid = 51265 and valuenum > 10000 THEN null -- K/uL 'PLATELET'
    ELSE le.valuenum
    END AS valuenum

  , (date_part('year', age(le.charttime, ha.admittime))*365 * 24
    + date_part('month', age(le.charttime, ha.admittime))*365/12 * 24
    + date_part('day', age(le.charttime, ha.admittime))* 24
    + date_part('hour', age(le.charttime, ha.admittime))
    + round(date_part('minute', age(le.charttime, ha.admittime))/60)) as HLOS

  FROM admissions ha

  LEFT JOIN labevents le
    ON le.hadm_id = ha.hadm_id
    AND le.charttime BETWEEN (ha.admittime - interval '1' day) AND ha.dischtime
    AND le.ITEMID in
    (
      -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
      51265 -- PLATELET COUNT | HEMATOLOGY | BLOOD | 778444
    )
    AND valuenum IS NOT null AND valuenum > 0 -- lab values cannot be 0 and cannot be negative
) pvt
GROUP BY pvt.subject_id, pvt.hadm_id, pvt.HLOS
ORDER BY pvt.subject_id, pvt.hadm_id, pvt.HLOS;