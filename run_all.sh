#!/usr/bin/env bash
# Rebuilds everything from the raw CSVs. Requires PostgreSQL running with the database/user from .env.example.
set -euo pipefail
python -m src.data_processing.clean          # raw -> data/processed + reports/cleaning_log.json
python database/load_data.py                 # schema.sql + COPY load
python -m src.features.build_features        # customer-level feature table
python -m src.models.train                   # 3 models + scores
python -m src.database.load_ml_tables        # publish scores/features to PostgreSQL
python -m src.llm.insights                   # recompute the findings the copilot uses
python -m src.database.run_queries           # run + validate the 37 SQL questions
python -m src.features.eda                   # 24 EDA charts
python scripts/build_notebooks.py            # execute the 6 notebooks
python -m pytest tests -q                    # 44 tests
echo "Done. Launch the app:  streamlit run app/dashboard.py"
