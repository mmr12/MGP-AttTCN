DROP MATERIALIZED VIEW IF EXISTS resp_bloodgasfirstdayarterial CASCADE;
CREATE MATERIALIZED VIEW resp_bloodgasfirstdayarterial AS
-- export from chartevents SoP2 data
with stg_spo2 as
(
  select SUBJECT_ID, HADM_ID, CHARTTIME
    -- max here is just used to group SpO2 by charttime
    , max(case when valuenum <= 0 or valuenum > 100 then null else valuenum end) as SpO2
  from CHARTEVENTS
  -- o2 sat
  where ITEMID in
  (
    646 -- SpO2
  , 220277 -- O2 saturation pulseoxymetry
  )
  group by SUBJECT_ID, HADM_ID, CHARTTIME
)
-- export from chartevents FiO2 data
, stg_fio2 as
(
  select SUBJECT_ID, HADM_ID, CHARTTIME
    -- pre-process the FiO2s to ensure they are between 21-100%
    , max(
        case
          when itemid = 223835
            then case
              when valuenum > 0 and valuenum <= 1
                then valuenum * 100
              -- improperly input data - looks like O2 flow in litres
              when valuenum > 1 and valuenum < 21
                then null
              when valuenum >= 21 and valuenum <= 100
                then valuenum
              else null end -- unphysiological
        when itemid in (3420, 3422)
        -- all these values are well formatted
            then valuenum
        when itemid = 190 and valuenum > 0.20 and valuenum < 1
        -- well formatted but not in %
            then valuenum * 100
      else null end
    ) as fio2_chartevents -- keep
  from CHARTEVENTS
  where ITEMID in
  (
    3420 -- FiO2
  , 190 -- FiO2 set
  , 223835 -- Inspired O2 Fraction (FiO2)
  , 3422 -- FiO2 [measured]
  )
  -- exclude rows marked as error
  and error IS DISTINCT FROM 1
  group by SUBJECT_ID, HADM_ID, CHARTTIME
)
-- extract first time SpO2 is recorded / sampled
, stg2 as
(
select bg.*
  , ROW_NUMBER() OVER (partition by bg.hadm_id, bg.charttime order by s1.charttime DESC) as lastRowSpO2 -- keep
  , s1.spo2 -- keep
from resp_bloodgasfirstday bg
left join stg_spo2 s1
  -- same patient
  on  bg.hadm_id = s1.hadm_id
  -- spo2 occurred at most 2 hours before this blood gas
  and s1.charttime between bg.charttime - interval '2' hour and bg.charttime
where bg.po2 is not null
)
-- extract first time FiO2 is recorded / sampled + specimen prediction (?)
, stg3 as
(
select bg.subject_id, bg.hadm_id, bg.charttime
  , bg.SPECIMEN
  , bg.PO2
  , ROW_NUMBER() OVER (partition by bg.hadm_id, bg.charttime order by greatest(bg2.charttime, s2.charttime) DESC) as lastRowFiO2 -- KEEP
  , case
      when coalesce(bg2.charttime, s2.charttime) is null then null
      when bg2.charttime is null then s2.fio2_chartevents
      when s2.charttime is null then bg2.FIO2
      when bg2.charttime >= s2.charttime then coalesce(bg2.FIO2, s2.fio2_chartevents)
      else coalesce(s2.fio2_chartevents, bg2.FIO2) end
      as FIO2_val

  -- create our specimen prediction
  -- data conditioned on this for some reason
  ,  1/(1+exp(-(-0.02544
  +    0.04598 * bg.po2
  + coalesce(-0.15356 * bg.spo2             , -0.15356 *   97.49420 +    0.13429)
  + coalesce( 0.00621 * fio2_chartevents ,  0.00621 *   51.49550 +   -0.24958)
  + coalesce( 0.10559 * bg.hemoglobin       ,  0.10559 *   10.32307 +    0.05954)
  + coalesce( 0.13251 * bg.so2              ,  0.13251 *   93.66539 +   -0.23172)
  + coalesce(-0.01511 * bg.pco2             , -0.01511 *   42.08866 +   -0.01630)
  + coalesce( 0.01480 * bg.fio2             ,  0.01480 *   63.97836 +   -0.31142)
  + coalesce(-0.00200 * bg.aado2            , -0.00200 *  442.21186 +   -0.01328)
  + coalesce(-0.03220 * bg.bicarbonate      , -0.03220 *   22.96894 +   -0.06535)
  + coalesce( 0.05384 * bg.totalco2         ,  0.05384 *   24.72632 +   -0.01405)
  + coalesce( 0.08202 * bg.lactate          ,  0.08202 *    3.06436 +    0.06038)
  + coalesce( 0.10956 * bg.ph               ,  0.10956 *    7.36233 +   -0.00617)
  + coalesce( 0.00848 * bg.o2flow           ,  0.00848 *    7.59362 +   -0.35803)
  ))) as SPECIMEN_PROB -- keep
from stg2 bg
left join stg_fio2 s2
  -- same patient
  on  bg.hadm_id = s2.hadm_id
  -- fio2 occurred at most 4 hours before this blood gas
  and s2.charttime between bg.charttime - interval '4' hour and bg.charttime
left join stg2 bg2
  -- same patient
  on  bg.hadm_id = bg2.hadm_id
  -- fio2 occurred at most 4 hours before this blood gas
  and bg2.charttime between bg.charttime - interval '4' hour and bg.charttime
where bg.lastRowSpO2 = 1 -- only the row with the most recent SpO2 (if no SpO2 found lastRowSpO2 = 1)
)
-- calculate PaO2FiO2
select subject_id, hadm_id, charttime
, case
    when PO2 is not null and FIO2_val is not null
     -- multiply by 100 because FiO2 is in a % but should be a fraction
      then 100*PO2/FIO2_val
    else null
  end as PaO2FiO2
from stg3
where lastRowFiO2 = 1 -- only the most recent FiO2
-- restrict it to *only* arterial samples
and (SPECIMEN = 'ART' or SPECIMEN_PROB > 0.75)
order by hadm_id, charttime;