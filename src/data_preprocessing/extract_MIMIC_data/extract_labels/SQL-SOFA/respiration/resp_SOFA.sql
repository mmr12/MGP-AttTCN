DROP MATERIALIZED VIEW IF EXISTS SOFA_PaO2FiO2 CASCADE;
CREATE materialized VIEW SOFA_PaO2FiO2 AS

-- adding hadm_id to ventduations
with vd as (
select vd.* , ie.hadm_id
from ventdurations vd
left join icustays ie
on vd.icustay_id = ie.icustay_id
)
-- combining ventilation and respiration data
, pafi1 as (
  select bg.hadm_id
	, bg.charttime
	, PaO2FiO2
	, case when sum(vd.icustay_id) is not null
			then 1
            when sum(vd.icustay_id) =0 then -1
			else 0 end as IsVent

 	from resp_bloodgasfirstdayarterial bg
	left join vd
		on bg.hadm_id = vd.hadm_id
		and bg.charttime >= vd.starttime
		and bg.charttime <= vd.endtime
	group by bg.hadm_id,  bg.charttime, PaO2FiO2
	order by bg.hadm_id, bg.charttime
)

-- because pafi has an interaction between vent/PaO2:FiO2, we need two columns for the score
-- it can happen that the lowest unventilated PaO2/FiO2 is 68, but the lowest ventilated PaO2/FiO2 is 120
-- in this case, the SOFA score is 3, *not* 4.
select pf.hadm_id
-- , charttime
, min( case when IsVent = 0
      then PaO2FiO2
      else null end) as PaO2FiO2_novent_min
, min( case when IsVent = 1
      then PaO2FiO2
      else null end) as PaO2FiO2_vent_min
, (date_part('year', age(pf.charttime, ha.admittime))*365 * 24
+ date_part('month', age(pf.charttime, ha.admittime))*365/12 * 24
+ date_part('day', age(pf.charttime, ha.admittime))* 24
+ date_part('hour', age(pf.charttime, ha.admittime))
+ round(date_part('minute', age(pf.charttime, ha.admittime))/60)) as HLOS

from pafi1 pf
left join admissions ha
  on ha.hadm_id = pf.hadm_id
group by pf.hadm_id, HLOS
