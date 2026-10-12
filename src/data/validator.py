"""
Deterministic Data Validation Module.
Performs empirical sanity checks on dataset fields without modifying data.
"""

import pandas as pd
from typing import Dict, Any


VALID_CATEGORIES = {'Interactive', 'Delay-insensitive', 'Unknown'}
EXPECTED_CORE_BUCKETS = {'0-2', '2-4', '4-8', '8-12', '12-24', '>24', '2', '4', '8', '12', '24', '30', '>64', 2, 4, 8, 12, 24, 30}
EXPECTED_MEMORY_BUCKETS = {'0-2', '2-4', '4-8', '8-32', '32-64', '>64', '2', '4', '8', '32', '64', '70', 2, 4, 8, 32, 64, 70}


def validate_vmtable(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform deterministic validation on vmtable DataFrame.
    
    Returns structured results dictionary containing missing counts,
    duplicates, CPU range violations, invalid categories, and timestamp inconsistencies.
    """
    total_records = len(df)
    if total_records == 0:
        return {"passed": False, "total_records": 0, "error": "Empty dataframe provided"}

    # 1. Missing Values
    missing_counts = df.isnull().sum().to_dict()

    # 2. Duplicates
    duplicate_vmids = int(df['vmid'].duplicated().sum()) if 'vmid' in df.columns else 0

    # 3. CPU Range Checks [0.0, 100.0]
    invalid_cpu_records = {}
    for cpu_col in ['avgcpu', 'maxcpu', 'p95maxcpu']:
        if cpu_col in df.columns:
            out_of_bounds = ((df[cpu_col] < 0.0) | (df[cpu_col] > 100.0)).sum()
            invalid_cpu_records[cpu_col] = int(out_of_bounds)

    # CPU Hierarchy Check: maxcpu >= avgcpu
    hierarchy_violations = 0
    if 'maxcpu' in df.columns and 'avgcpu' in df.columns:
        hierarchy_violations = int((df['maxcpu'] < df['avgcpu']).sum())

    # 4. Category Validation
    invalid_categories = 0
    unknown_category_count = 0
    if 'vmcategory' in df.columns:
        categories_found = set(df['vmcategory'].dropna().astype(str).unique())
        invalid_cat_mask = ~df['vmcategory'].astype(str).isin(VALID_CATEGORIES)
        invalid_categories = int(invalid_cat_mask.sum())
        unknown_category_count = int((df['vmcategory'] == 'Unknown').sum())

    # 5. Timestamp & Lifetime Consistency
    timestamp_negative_count = 0
    deletion_before_creation_count = 0
    if 'vmcreated' in df.columns and 'vmdeleted' in df.columns:
        timestamp_negative_count = int(((df['vmcreated'] < 0) | (df['vmdeleted'] < 0)).sum())
        deletion_before_creation_count = int((df['vmdeleted'] < df['vmcreated']).sum())

    # 6. Core and Memory Bucket Validation
    unusual_core_buckets = []
    if 'vmcorecountbucket' in df.columns:
        found_cores = set(df['vmcorecountbucket'].astype(str).unique())
        unusual_core_buckets = [b for b in found_cores if b not in EXPECTED_CORE_BUCKETS]

    unusual_memory_buckets = []
    if 'vmmemorybucket' in df.columns:
        found_mem = set(df['vmmemorybucket'].astype(str).unique())
        unusual_memory_buckets = [b for b in found_mem if b not in EXPECTED_MEMORY_BUCKETS]

    passed = (
        sum(missing_counts.values()) == 0 and
        duplicate_vmids == 0 and
        sum(invalid_cpu_records.values()) == 0 and
        hierarchy_violations == 0 and
        invalid_categories == 0 and
        timestamp_negative_count == 0 and
        deletion_before_creation_count == 0
    )

    return {
        "passed": passed,
        "total_records": total_records,
        "missing_counts": missing_counts,
        "duplicate_vmids": duplicate_vmids,
        "invalid_cpu_records": invalid_cpu_records,
        "cpu_hierarchy_violations": hierarchy_violations,
        "invalid_categories": invalid_categories,
        "unknown_category_count": unknown_category_count,
        "timestamp_negative_count": timestamp_negative_count,
        "deletion_before_creation_count": deletion_before_creation_count,
        "unusual_core_buckets": unusual_core_buckets,
        "unusual_memory_buckets": unusual_memory_buckets
    }


def validate_cpu_readings(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate 5-minute sampled CPU readings dataset."""
    total_records = len(df)
    if total_records == 0:
        return {"passed": False, "total_records": 0, "error": "Empty dataframe"}

    missing_counts = df.isnull().sum().to_dict()
    invalid_cpu = {}

    for cpu_col in ['mincpu', 'maxcpu', 'avgcpu']:
        if cpu_col in df.columns:
            out_of_bounds = ((df[cpu_col] < 0.0) | (df[cpu_col] > 100.0)).sum()
            invalid_cpu[cpu_col] = int(out_of_bounds)

    hierarchy_violations = 0
    if 'maxcpu' in df.columns and 'mincpu' in df.columns:
        hierarchy_violations = int((df['maxcpu'] < df['mincpu']).sum())

    passed = (sum(missing_counts.values()) == 0) and (sum(invalid_cpu.values()) == 0) and (hierarchy_violations == 0)

    return {
        "passed": passed,
        "total_records": total_records,
        "missing_counts": missing_counts,
        "invalid_cpu_records": invalid_cpu,
        "cpu_hierarchy_violations": hierarchy_violations
    }
