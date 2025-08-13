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
5. Build features: `python etl/build_features.py`
6. Train: `python training/train_gbm.py`
7. Serve: `uvicorn service.app:app --reload`

> You can also use the CSVs and Python scripts directly **without** Postgres for the POC.
