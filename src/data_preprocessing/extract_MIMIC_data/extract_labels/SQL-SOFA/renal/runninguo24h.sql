DROP MATERIALIZED VIEW IF EXISTS SOFA_runninguo24h CASCADE;
create materialized view SOFA_runninguo24h as
SELECT 
uo_1.hadm_id, uo_1.HLOS

, SUM(uo_4.UrineOutput)  as running_uo_24h
FROM SOFA_uoperhour uo_1
JOIN SOFA_uoperhour uo_4 ON
    uo_1.hadm_id = uo_4.hadm_id and
    uo_4.HLOS between uo_1.HLOS -24 and uo_1.HLOS

where uo_4.ICULOS >= 0 and uo_1.ICULOS  >= 24

group by uo_1.hadm_id, uo_1.HLOS
order by uo_1.hadm_id, uo_1.HLOS
