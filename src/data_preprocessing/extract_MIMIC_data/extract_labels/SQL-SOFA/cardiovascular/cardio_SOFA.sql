DROP MATERIALIZED VIEW IF EXISTS SOFA_cardio CASCADE;
CREATE materialized VIEW SOFA_cardio AS

-- calculate weight
with wt as	(
	select ha.hadm_id
	-- average weight
	, avg(
		case
			-- kg
			when itemid in (762, 763, 3723, 3580, 226512)
			then valuenum
			 -- lbs
			when itemid in (3581)
			then valuenum * 0.45359237
			-- oz
	        when itemid IN (3582)
	        then valuenum * 0.0283495231
	        else null
	        end) as weight

 	from admissions ha
	left join chartevents c
		on ha.hadm_id = c.hadm_id
	where valuenum is not null
	and valuenum != 0
	and itemid in
		(
		-- cv
	    762, 763, 3723, 3580,                     -- Weight Kg
	    3581,                                     -- Weight lb
	    3582,                                     -- Weight oz
	    -- mv
	    226512 									  -- Weight Kg
	  	)
	-- note that the timeframe below assumes weight does not change as a funrion of time
	-- verified from the fact that mv only has an admission weight and no dynamic weighting
	and charttime between ha.admittime - interval '2' month and ha.dischtime
	-- some rows are marked as error, let's inglore them
	and c.error is distinct from 1
	group by ha.hadm_id
)
-- calculate weight indirectly through echography weight
, echo2 as (
	select ha.hadm_id, avg(weight * 0.45359237) as weight
	from admissions ha
	left join echodata echo
		on ha.hadm_id = echo.hadm_id
		and echo.charttime > ha.admittime - interval '7' day
		and echo.charttime < ha.dischtime
	group by ha.hadm_id
)
-- calculate rates for carevue
, vaso_cv as (
	select ha.hadm_id
	, max( case
			when itemid = 30047 then rate / coalesce(wt.weight, ec.weight)
			when itemid = 30120 then rate
			else null end ) as rate_norepinephrine
	, max( case
			when itemid = 30044 then rate / coalesce(wt.weight, ec.weight)
			when itemid in (30119, 30309) then rate
			else null end ) as rate_epinephrine
 	, max( case when itemid in (30043, 30307) then rate end) as rate_dopamine
 	, max( case when itemid in (30042, 30306) then rate end) as rate_dobutamine
 	, (date_part('year', age(cv.charttime, ha.admittime ))*365 * 24
		+ date_part('month', age(cv.charttime, ha.admittime ))*365/12 * 24
		+ date_part('day', age(cv.charttime, ha.admittime ))* 24
		+ date_part('hour', age(cv.charttime, ha.admittime ))
		+ round(date_part('minute', age(cv.charttime, ha.admittime ))/60)) as HLOS

  	from admissions ha
 	inner join inputevents_cv cv
 		on ha.hadm_id = cv.hadm_id
 		and cv.charttime between ha.admittime  - interval '1' day and ha.dischtime
 	left join wt
 		on ha.hadm_id = wt.hadm_id
 	left join echo2 ec
 		on ha.hadm_id = ec.hadm_id
 	where itemid in (30047,30120,30044,30119,30309,30043,30307,30042,30306)
 	and rate is not null
 	group by ha.hadm_id, HLOS
 )
 -- calculate rates for metavision
, vaso_mv as (
	select ha.hadm_id
	, max(case when itemid = 221906 then rate end) as rate_norepinephrine
  , max(case when itemid = 221289 then rate end) as rate_epinephrine
  , max(case when itemid = 221662 then rate end) as rate_dopamine
  , max(case when itemid = 221653 then rate end) as rate_dobutamine
  , (date_part('year', age(mv.starttime, ha.admittime))*365 * 24
  + date_part('month', age(mv.starttime, ha.admittime))*365/12 * 24
  + date_part('day', age(mv.starttime, ha.admittime))* 24
  + date_part('hour', age(mv.starttime, ha.admittime))
  + round(date_part('minute', age(mv.starttime, ha.admittime))/60)) as HLOS
    from admissions ha
    inner join inputevents_mv mv
    	on ha.hadm_id = mv.hadm_id
    	and mv.starttime between ha.admittime - interval '1' day and ha.dischtime
    where itemid in (221906,221289,221662,221653)
    -- 'Rewritten' orders are not delivered to the patient
    and statusdescription != 'Rewritten'
    group by ha.hadm_id , HLOS
)
-- join everything
select
  ha.hadm_id, vaso.hlos
	, coalesce(cv.rate_norepinephrine, mv.rate_norepinephrine) as rate_norepinephrine
	, coalesce(cv.rate_epinephrine, mv.rate_epinephrine) as rate_epinephrine
	, coalesce(cv.rate_dopamine, mv.rate_dopamine) as rate_dopamine
	, coalesce(cv.rate_dobutamine, mv.rate_dobutamine) as rate_dobutamine
	from admissions ha
	left join (
	    select hadm_id, hlos from vaso_cv union
	    select hadm_id, hlos from vaso_mv
	)	as vaso
        on ha.hadm_id = vaso.hadm_id
	left join vaso_cv cv
		on vaso.hadm_id = cv.hadm_id
        and vaso.hlos = cv.hlos
	left join vaso_mv mv
		on vaso.hadm_id = mv.hadm_id
        and vaso.hlos = mv.hlos
