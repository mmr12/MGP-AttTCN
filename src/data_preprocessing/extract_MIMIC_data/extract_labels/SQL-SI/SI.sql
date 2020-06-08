DROP TABLE IF EXISTS SI_flag CASCADE;
CREATE TABLE SI_flag as
with abx as
(
  select hadm_id
    , suspected_infection_time
    , ROW_NUMBER() OVER
    (
      PARTITION BY hadm_id
      ORDER BY suspected_infection_time
    ) as rn
  from abx_micro_poe
)
select
  hadm_id
  , suspected_infection_time
  , suspected_infection_time - interval '48' hour as si_start
  , suspected_infection_time + interval '24' hour as si_end
from abx
where abx.rn = 1
order by hadm_id