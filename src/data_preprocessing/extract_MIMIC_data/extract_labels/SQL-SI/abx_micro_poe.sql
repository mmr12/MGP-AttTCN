/*
- MODIFIED VERSION
- SOURCE: https://github.com/alistairewj/sepsis3-mimic/blob/master/query/tbls/abx-micro-prescription.sql
- DOWNLOADED on 8th February 2018
*/

-- only works for metavision as carevue does not accurately document antibiotics
DROP TABLE IF EXISTS abx_micro_poe CASCADE;
CREATE TABLE abx_micro_poe as
-- mv tells us how many antibiotics were prescribed and when
with mv as
(
  select hadm_id
  , count(mv.drug) as no_antibiotic
  , startdate as antibiotic_time
  from prescriptions mv
  inner join abx_poe_list ab
      on mv.drug = ab.drug
  group by hadm_id, antibiotic_time
)
-- me tells us when cultures were taken
, me as
(
  select hadm_id
    , chartdate, charttime
    , spec_type_desc
    -- , max(case when org_name is not null and org_name != '' then 1 else 0 end) as PositiveCulture
  from microbiologyevents
  group by hadm_id, chartdate, charttime, spec_type_desc
)
-- ab_fnl checks whether a culture was taken either 72h prior or 24h after administration of antibiotics
-- conditions on there being more than 1 antibiotic administered
-- (see: Sepsis 3 Seymour paper attachment)
, ab_fnl as
(
  select
      mv.hadm_id
    -- , mv.no_antibiotic
    , mv.antibiotic_time
    , coalesce(me72.charttime,me72.chartdate) as last72_charttime
    , coalesce(me24.charttime,me24.chartdate) as next24_charttime
    , case when me72.charttime is null then 'date' else 'time' end as last72
    , case when me24.charttime is null then 'date' else 'time' end as next24

    --, me72.positiveculture as last72_positiveculture
    --, me72.spec_type_desc as last72_specimen
    --, me24.positiveculture as next24_positiveculture
    --, me24.spec_type_desc as next24_specimen
  from mv
  -- blood culture in last 72 hours
  left join me me72
    on mv.hadm_id = me72.hadm_id
    and mv.antibiotic_time is not null
    and
    (
      -- if charttime is available, use it
      (
          mv.antibiotic_time >= me72.charttime
      and mv.antibiotic_time <= me72.charttime + interval '72' hour
      )
      OR
      (
      -- if charttime is not available, use chartdate
          me72.charttime is null
      and mv.antibiotic_time >= me72.chartdate
      and mv.antibiotic_time <= me72.chartdate + interval '3' day
      )
    )
  -- blood culture in subsequent 24 hours
  left join me me24
    on mv.hadm_id = me24.hadm_id
    and mv.antibiotic_time is not null
    -- and me24.charttime is not null -- this probably takes away quite a few options
    and
    (
      -- if charttime is available, use it
      (
          mv.antibiotic_time >= me24.charttime - interval '24' hour
      and mv.antibiotic_time <= me24.charttime
      )
      OR
      (
      -- if charttime is not available, use chartdate
          me24.charttime is null
      and mv.antibiotic_time >= me24.chartdate - interval '1' day
      and mv.antibiotic_time <= me24.chartdate
      )
    )
    -- added the 19.09.05 - apparently this happens sometimes
    where coalesce(me72.charttime,me72.chartdate,me24.charttime,me24.chartdate) is not null

  -- where no_antibiotic > 1 -- see: https://github.com/alistairewj/sepsis3-mimic/issues/12
)
, abx_micro_poe_temp as (
select
    hadm_id
  -- , antibiotic_name
  , antibiotic_time
  , last72_charttime
  , next24_charttime

  -- suspected_infection flag: redundant with suspected_infection_time
  /*
  , case
      when coalesce(last72_charttime,next24_charttime) is null
        then 0
      else 1 end as suspected_infection
  */
  -- time of suspected infection: either the culture time (if before antibiotic), or the antibiotic time
  , case
      when coalesce(last72_charttime, next24_charttime) is null
        then null
      else least(coalesce(last72_charttime, next24_charttime), antibiotic_time)
      end as suspected_infection_time
  -- to calculate time of SI, we don't care about which specimen was cultured or whether it was a positive culture
  /*
  -- the specimen that was cultured
  , case
      when last72_charttime is not null
        then last72_specimen
      when next24_charttime is not null
        then next24_specimen
    else null
  end as specimen

  -- whether the cultured specimen ended up being positive or not
  , case
      when last72_charttime is not null
        then last72_positiveculture
      when next24_charttime is not null
        then next24_positiveculture
    else null
  end as positiveculture
  */
from ab_fnl
)
select
a.hadm_id,
ad.admittime,
a.antibiotic_time,
extract(EPOCH from a.antibiotic_time - ad.admittime)
          / 60.0 / 60.0 as abx_h,
a.last72_charttime,
extract(EPOCH from a.last72_charttime - ad.admittime)
          / 60.0 / 60.0 as l72_h,
a.next24_charttime,
extract(EPOCH from a.next24_charttime - ad.admittime)
          / 60.0 / 60.0 as n24_h,
a.suspected_infection_time,
extract(EPOCH from a.suspected_infection_time - ad.admittime)
          / 60.0 / 60.0 as si_h
from abx_micro_poe_temp a
left join admissions ad
    on a.hadm_id=ad.hadm_id

