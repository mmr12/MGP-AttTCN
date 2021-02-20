-- This code extracts structured data from echocardiographies
-- You can join it to the text notes using ROW_ID
-- Just note that ROW_ID will differ across versions of MIMIC-III.

DROP MATERIALIZED VIEW IF EXISTS ECHODATA CASCADE;
CREATE MATERIALIZED VIEW ECHODATA AS
select ROW_ID
  , subject_id, hadm_id
  , chartdate

  -- charttime is always null for echoes..
  -- however, the time is available in the echo text, e.g.:
  -- , substring(ne.text, 'Date/Time: [\[\]0-9*-]+ at ([0-9:]+)') as TIMESTAMP
  -- we can therefore impute it and re-create charttime
  , cast(to_timestamp( (to_char( chartdate, 'DD-MM-YYYY' ) || substring(ne.text, 'Date/Time: [\[\]0-9*-]+ at ([0-9:]+)')),
            'DD-MM-YYYYHH24:MI') as timestamp without time zone)
    as charttime

  , case
      when substring(ne.text, 'Weight \(lb\): (.*?)\n') like '%*%'
        then null
      else cast(substring(ne.text, 'Weight \(lb\): (.*?)\n') as numeric)
    end as Weight

from noteevents ne
where category = 'Echo'
