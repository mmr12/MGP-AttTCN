DROP MATERIALIZED VIEW IF EXISTS SOFAperhour CASCADE;
CREATE materialized VIEW SOFAperhour AS
-- get all the data in one place
with scorecomp as (
	select 	
	-- General info	
	  ha.hadm_id
    , u.HLOS as HLOS

  -- Respiration
  , pf.PaO2FiO2_novent_min
	, pf.PaO2FiO2_vent_min

	-- Coagulation
	, cl.Platelet

	-- Liver
	, ll.Bilirubin

	-- Cardiovascular
	, c.rate_norepinephrine
	, c.rate_epinephrine
	, c.rate_dopamine
	, c.rate_dobutamine
  , cv.MinBP

	-- Central nervous system
	, gcs.GCS

	-- Renal
	, uo.running_uo_24h as UrineOutput24h
	, rl.Creatinine

 	from admissions ha
    left join (	
	    select hadm_id, hlos from SOFA_PaO2FiO2 union
	    select hadm_id, hlos from coag_labsperhour union
	    select hadm_id, hlos from liv_labsperhour union
	    select hadm_id, hlos from SOFA_cardio union
	    select hadm_id, hlos from cardio_vitalsperhour union
	    select hadm_id, hlos from SOFA_gcsperhour union
	    select hadm_id, hlos from SOFA_runninguo24h union
	    select hadm_id, hlos from ren_labsperhour
    ) as u	
        on ha.hadm_id = u.hadm_id
      left join SOFA_PaO2FiO2 pf
        on u.hadm_id = pf.hadm_id
        and u.hlos = pf.hlos
      left join coag_labsperhour cl
        on u.hadm_id = cl.hadm_id
        and u.hlos = cl.hlos
      left join liv_labsperhour ll
        on u.hadm_id = ll.hadm_id
        and u.hlos = ll.hlos
      left join SOFA_cardio c
        on u.hadm_id = c.hadm_id
        and u.hlos = c.hlos
      left join cardio_vitalsperhour cv
        on u.hadm_id = cv.hadm_id
        and u.hlos = cv.hlos
      left join SOFA_gcsperhour gcs
        on u.hadm_id = gcs.hadm_id
        and u.hlos = gcs.hlos
      left join SOFA_runninguo24h uo
        on u.hadm_id = uo.hadm_id
        and u.hlos = uo.hlos
      left join ren_labsperhour rl
        on u.hadm_id = rl.hadm_id
        and u.hlos = rl.hlos
)
-- calculating all the variables
, SOFA as (	
	select hadm_id, HLOS
	-- Respiration	
	, case	
		when PaO2FiO2_vent_min < 100 then 4
		when PaO2FiO2_vent_min < 200 then 3	
		when coalesce(PaO2FiO2_novent_min, PaO2FiO2_vent_min) < 300 then 2
		when coalesce(PaO2FiO2_novent_min, PaO2FiO2_vent_min) < 400 then 1
		when coalesce(PaO2FiO2_vent_min, PaO2FiO2_novent_min) is null then null	
		else 0 end as respiration	
	-- Coagulation	
	, case	
		when Platelet <20 then 4	
		when Platelet <50 then 3	
		when Platelet < 100 then 2	
		when Platelet < 150 then 1	
		when Platelet is null then null	
		else 0 end as coagulation	
	-- Liver	
	, case	
		when Bilirubin >= 12.0 then 4	
		when Bilirubin >= 6.0 then 3	
		when Bilirubin >= 2.0 then 2	
		when Bilirubin >= 1.2 then 1	
		when Bilirubin is null then null	
		else 0 end as liver	
	-- Cardiovascular	
	, case	
		when rate_dopamine > 15	
			or rate_epinephrine > 0.1	
			or rate_norepinephrine > 0.1	
			then 4	
		when rate_dopamine > 5	
			or rate_epinephrine <= 0.1 	
			or rate_norepinephrine <= 0.1	
			then 3	
		when rate_dopamine > 0 	
			or rate_dobutamine >0	
			then 2	
		when MinBP < 70 then 1
		when coalesce(MinBP,
						rate_dopamine,	
						rate_dobutamine,	
						rate_epinephrine,	
						rate_norepinephrine) 	
						is null then null	
		else 0 end as Cardiovascular	
		-- Neurological failure (GCS)	
	, case		
		when (GCS >= 13 and GCS <= 14) then 1	
		when (GCS >= 10 and GCS <= 12) then 2	
		when (GCS >= 6 and GCS <= 9) then 3	
		when GCS < 6 then 4	
		else 0 end as CNS	
		-- Renal failure	-- TODO: this  becomes wrong once you look over the past 24h, given urine is already over 24h :(
	, case	
		when (Creatinine >= 5) then 4	
		-- when (UrineOutput24h < 200 and HLOS > 24) then 4
		when (Creatinine >= 3.5 and Creatinine < 5.0) then 3
		-- when (UrineOutput24h < 500 and HLOS > 24) then 3
		when (Creatinine >= 2.0 and Creatinine < 3.5) then 2
		when (Creatinine >= 1.2 and Creatinine < 2.0) then 1
		when coalesce(UrineOutput24h, Creatinine) is null then null
		else 0 end as renal_labs
	, case
	  when (UrineOutput24h < 200 and HLOS > 24) then 4
	  when (UrineOutput24h < 500 and HLOS > 24) then 3
	  else 0 end as renal_uo

 	from scorecomp
)
-- making an hourly table
, SOFA_per_hour as (
  select
  hah.hadm_id
  , hah.hr as hlos
  , hah.endtime
  , SOFA.respiration as SOFAresp
  , SOFA.coagulation as SOFAcoag
  , SOFA.liver as SOFAliv
  , SOFA.Cardiovascular as SOFAcardio
  , SOFA.CNS as SOFAgcs
  , SOFA.renal_labs as SOFAren
  from hadms_hours hah
  left join SOFA
    on SOFA.hadm_id = hah.hadm_id
    and SOFA.HLOS = hah.hr
  order by hadm_id, hr
)
-- maximum value for each component of SOFA over the past 24 hours
, SOFA_per_hour_looking_back as (
  SELECT
  sofa1.hadm_id
  , sofa1.HLOS
  , max(sofa2.SOFAresp)   as SOFAresp
  , max(sofa2.SOFAcoag)   as SOFAcoag
  , max(sofa2.SOFAliv)    as SOFAliv
  , max(sofa2.SOFAcardio) as SOFAcardio
  , max(sofa2.SOFAgcs)    as SOFAgcs
  , max(sofa2.SOFAren)    as SOFAren

  FROM SOFA_per_hour sofa1
  JOIN SOFA_per_hour sofa2 ON
      sofa1.hadm_id = sofa2.hadm_id and
      sofa2.HLOS between sofa1.HLOS -24 and sofa1.HLOS
  group by sofa1.hadm_id, sofa1.HLOS
  order by sofa1.hadm_id, sofa1.HLOS
)
SELECT s1.hadm_id
, s1.hlos
, SOFAresp
, SOFAcoag
, SOFAliv
, SOFAcardio
, SOFAgcs
, GREATEST(SOFAren, s2.renal_uo) as SOFAren
, coalesce(SOFAresp, 0)
  + coalesce(SOFAcoag, 0)
  + coalesce(SOFAliv, 0)
  + coalesce(SOFAcardio, 0)
  + coalesce(SOFAgcs, 0)
  + coalesce(GREATEST(SOFAren, s2.renal_uo), 0) as SOFA
  from SOFA_per_hour_looking_back s1
  left join  SOFA s2
    on s1.hadm_id = s2.hadm_id
    and s1.HLOS = s2.hlos
