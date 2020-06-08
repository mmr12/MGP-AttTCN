DROP MATERIALIZED VIEW IF EXISTS SOFAflag CASCADE;
CREATE materialized VIEW SOFAflag AS
-- create flag
with SOFA_flag as (
  select t0.hadm_id
  , t0.hlos
  , case
      when tn2.SOFA - t0.SOFA >=2 then 1
      else 0 end as SOFAflag
  from SOFAperhour tn2
  join SOFAperhour t0
    on tn2.hadm_id = t0.hadm_id
    and tn2.hlos = t0.hlos - 1
    where t0.hlos >= 0
)
-- adding admissions info, calculating hour of onset
select S.*
, ha.subject_id
, ha.admittime
, ha.admittime + S.hlos * interval '1 hour' as SOFAtime
from SOFA_flag S
left join admissions ha
  on S.hadm_id = ha.hadm_id