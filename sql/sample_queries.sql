-- Example queries
-- Batches overview
SELECT b.batch_id, b.kettle_id, b.process_card_id, q.pass_flag, q.viscosity, q.free_hcho
FROM batch b
LEFT JOIN qc_result q USING(batch_id)
ORDER BY b.start_ts DESC
LIMIT 20;

-- A simple feature: time to reach 95C in step 2 (approx via first crossing)
WITH t AS (
  SELECT ts, batch_id, value AS temp
  FROM ts_signal
  WHERE tag='T'
),
t2 AS (
  SELECT batch_id, MIN(ts) FILTER (WHERE temp >= 95) AS t95,
         MIN(ts) AS t0
  FROM t
  GROUP BY batch_id
)
SELECT batch_id, EXTRACT(EPOCH FROM (t95 - t0))/60.0 AS minutes_to_95C
FROM t2;
