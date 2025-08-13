# Industrial AI — Process Optimization (POC Skeleton)

This repo contains a minimal, end-to-end scaffold for **batch process optimization** with
synthetic data resembling an 8-step recipe (heat/hold/cool, pH adjust, vacuum dehydration ≤16000 mL).

## What's inside
- **sql/** — DDL to create tables on PostgreSQL/TimescaleDB and sample queries.
- **data/synthetic/** — CSVs with synthetic batches, signals, events, QC results, raw materials.
- **etl/** — scripts for stage segmentation, feature building, and synthetic data generation.
- **training/** — LightGBM-based training pipeline (classification+regression) + evaluation.
- **service/** — FastAPI (state/advice) skeleton + golden curve builder.
- **utils/** — shared helpers.
- **docker-compose.yml** — optional local infra (Postgres/Timescale).

## Quick start (local, minimal)
1. (Optional) `docker compose up -d` to start Postgres/Timescale.
2. `pip install -r requirements.txt`
3. Load DDL: `psql -h localhost -U postgres -f sql/ddl.sql`
4. (Optional) Ingest synthetic CSVs from `data/synthetic/`.
5. Validate CSVs (schema & basic sanity): `python etl/validation.py`
6. Build features: `python etl/build_features.py`
7. Train: `python training/train_gbm.py`
8. Serve: `uvicorn service.app:app --reload`

> You can also use the CSVs and Python scripts directly **without** Postgres for the POC.

## Data validation

Great Expectations suites verify that the core CSV files match the expected
schema before any processing occurs.  Validation runs automatically at the
start of `etl/stage_segmentation.py` and `etl/build_features.py`, and can also
be executed standalone:

```bash
python etl/validation.py
```

Sample output:

```
batch.csv: True {'evaluated_expectations': 4, 'successful_expectations': 4, 'unsuccessful_expectations': 0, 'success_percent': 100.0}
ts_signal.csv: True {'evaluated_expectations': 6, 'successful_expectations': 6, 'unsuccessful_expectations': 0, 'success_percent': 100.0}
qc_result.csv: True {'evaluated_expectations': 12, 'successful_expectations': 12, 'unsuccessful_expectations': 0, 'success_percent': 100.0}
```

If any expectation fails the scripts will raise an error and stop execution,
allowing issues to be addressed early in the pipeline.
