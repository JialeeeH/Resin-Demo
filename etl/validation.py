"""Great Expectations validation suites for ETL inputs.

This module defines simple expectation suites for the synthetic CSV files
used throughout the pipeline.  Each function returns the validation result
dictionary returned by Great Expectations.  A convenience ``validate_all``
function validates all known datasets and raises ``ValueError`` if any
expectations are not met.

The functions are intentionally lightweight so they can run quickly at the
start of the ETL stages (``stage_segmentation`` and ``build_features``).
"""

from __future__ import annotations

import great_expectations as ge
from utils.constants import TAGS
from utils.io import read_csv


def _ge_dataset(df):
    """Return a Great Expectations dataset from a Pandas DataFrame."""

    try:  # GE <1.0
        return ge.dataset.PandasDataset(df)
    except AttributeError:  # GE >=1.0 exposes ``from_pandas``
        return ge.from_pandas(df)


def validate_batch(df=None):
    """Expectation suite for ``batch.csv``."""

    df = read_csv("batch.csv") if df is None else df
    gdf = _ge_dataset(df)
    expected_cols = {
        "batch_id",
        "kettle_id",
        "process_card_id",
        "start_ts",
        "end_ts",
        "shift",
        "team",
    }
    gdf.expect_table_columns_to_match_set(expected_cols)
    for col in ["batch_id", "start_ts", "end_ts"]:
        gdf.expect_column_values_to_not_be_null(col)
    return gdf.validate()


def validate_ts_signal(df=None):
    """Expectation suite for ``ts_signal.csv``."""

    df = read_csv("ts_signal.csv") if df is None else df
    gdf = _ge_dataset(df)
    expected_cols = {"ts", "batch_id", "tag", "value"}
    gdf.expect_table_columns_to_match_set(expected_cols)
    for col in expected_cols:
        gdf.expect_column_values_to_not_be_null(col)
    gdf.expect_column_values_to_be_in_set("tag", TAGS)
    return gdf.validate()


def validate_qc_result(df=None):
    """Expectation suite for ``qc_result.csv``."""

    df = read_csv("qc_result.csv") if df is None else df
    gdf = _ge_dataset(df)
    expected_cols = {
        "batch_id",
        "viscosity",
        "free_hcho",
        "moisture",
        "dextrin",
        "sec_cut_2h",
        "sec_cut_24h",
        "hardness",
        "penetration",
        "pass_flag",
    }
    gdf.expect_table_columns_to_match_set(expected_cols)
    for col in expected_cols:
        gdf.expect_column_values_to_not_be_null(col)
    gdf.expect_column_values_to_be_of_type("pass_flag", "bool")
    return gdf.validate()


def validate_all():
    """Validate all known datasets; raise ``ValueError`` on failure."""

    funcs = [
        ("batch.csv", validate_batch),
        ("ts_signal.csv", validate_ts_signal),
        ("qc_result.csv", validate_qc_result),
    ]
    results = {}
    for name, func in funcs:
        res = func()
        results[name] = res
        stats = res.get("statistics", {})
        print(f"{name}: {res['success']} {stats}")
    if not all(r["success"] for r in results.values()):
        raise ValueError("Data validation failed")
    return results


if __name__ == "__main__":
    validate_all()

