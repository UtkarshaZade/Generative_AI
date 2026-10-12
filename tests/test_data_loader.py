"""
Unit tests for data loader module.
"""

import os
import pytest
import pandas as pd
from src.data.loader import get_available_dataset_files, load_schema, load_vmtable, load_cpu_readings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_get_available_dataset_files():
    files_status = get_available_dataset_files(PROJECT_ROOT)
    assert isinstance(files_status, dict)
    assert "schema" in files_status
    assert "vmtable" in files_status
    assert files_status["schema"]["exists"] is True


def test_load_schema():
    schema_df, meta = load_schema(PROJECT_ROOT)
    assert schema_df is not None
    assert meta["loaded"] is True
    assert "filepattern" in schema_df.columns
    assert "content" in schema_df.columns


def test_load_vmtable_sample():
    vmtable_df, meta = load_vmtable(PROJECT_ROOT, sample_size=50)
    assert vmtable_df is not None
    assert meta["loaded"] is True
    assert meta["is_sample"] is True
    assert len(vmtable_df) == 50
    assert "vmid" in vmtable_df.columns
    assert "maxcpu" in vmtable_df.columns
    assert "vmcategory" in vmtable_df.columns


def test_load_cpu_readings_sample():
    cpu_df, meta = load_cpu_readings(PROJECT_ROOT, sample_size=50)
    if meta["loaded"]:
        assert cpu_df is not None
        assert len(cpu_df) == 50
        assert "timestamp" in cpu_df.columns
        assert "vmid" in cpu_df.columns
        assert "avgcpu" in cpu_df.columns
