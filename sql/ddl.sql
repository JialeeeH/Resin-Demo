-- Core tables
CREATE TABLE IF NOT EXISTS batch (
  batch_id TEXT PRIMARY KEY,
  kettle_id TEXT,
  process_card_id TEXT,
  start_ts TIMESTAMPTZ,
  end_ts TIMESTAMPTZ,
  shift TEXT,
  team TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS qc_result (
  batch_id TEXT REFERENCES batch(batch_id),
  viscosity DOUBLE PRECISION,
  free_hcho DOUBLE PRECISION,
  moisture DOUBLE PRECISION,
  dextrin DOUBLE PRECISION,
  sec_cut_2h DOUBLE PRECISION,
  sec_cut_24h DOUBLE PRECISION,
  hardness DOUBLE PRECISION,
  penetration DOUBLE PRECISION,
  pass_flag BOOLEAN,
  PRIMARY KEY (batch_id)
);

CREATE TABLE IF NOT EXISTS ts_signal (
  ts TIMESTAMPTZ NOT NULL,
  batch_id TEXT NOT NULL,
  tag TEXT NOT NULL,
  value DOUBLE PRECISION,
  PRIMARY KEY (ts, batch_id, tag)
);

-- If you enable TimescaleDB, uncomment:
-- SELECT create_hypertable('ts_signal','ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS op_event (
  id BIGSERIAL PRIMARY KEY,
  batch_id TEXT NOT NULL,
  step INT,
  action TEXT,
  param JSONB,
  ts TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_material (
  material TEXT,
  lot TEXT,
  specs JSONB,
  PRIMARY KEY(material, lot)
);
