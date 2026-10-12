"""
Dataset Loader Module for Azure 2019 Public Dataset V2.
Provides modular, safe, and memory-efficient data loading with chunking/sampling support.
"""

import os
import pandas as pd
from typing import Tuple, Dict, List, Optional


VMTABLE_HEADERS = [
    'vmid',
    'subscriptionid',
    'deploymentid',
    'vmcreated',
    'vmdeleted',
    'maxcpu',
    'avgcpu',
    'p95maxcpu',
    'vmcategory',
    'vmcorecountbucket',
    'vmmemorybucket'
]

CPU_READINGS_HEADERS = [
    'timestamp',
    'vmid',
    'mincpu',
    'maxcpu',
    'avgcpu'
]


def resolve_file_path(base_dir: str, primary_rel: str, fallback_rels: List[str]) -> Optional[str]:
    """Check primary and fallback relative paths within base_dir."""
    candidate = os.path.join(base_dir, primary_rel)
    if os.path.exists(candidate) and os.path.isfile(candidate):
        return candidate
    
    for rel in fallback_rels:
        candidate = os.path.join(base_dir, rel)
        if os.path.exists(candidate) and os.path.isfile(candidate):
            return candidate
    return None


def get_available_dataset_files(base_dir: str = ".") -> Dict[str, Dict]:
    """
    Inspect the workspace directory to determine which dataset files actually exist.
    Does NOT assume missing files exist.
    """
    known_files = {
        "schema": {
            "rel": "schema.csv",
            "fallbacks": [],
            "description": "Master schema CSV file"
        },
        "vmtable": {
            "rel": os.path.join("trace_data_vmtable_vmtable.csv", "trace_data_vmtable_vmtable.csv"),
            "fallbacks": ["vmtable.csv", "trace_data_vmtable_vmtable.csv"],
            "description": "Full Virtual Machine metadata and aggregate CPU table"
        },
        "vm_cpu_readings_chunk1": {
            "rel": os.path.join("trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv", "trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv"),
            "fallbacks": ["trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv", "vm_cpu_readings-file-1-of-195.csv"],
            "description": "5-minute sampled CPU readings chunk 1 of 195"
        },
        "azure2019_category": {
            "rel": "azure2019_data_category.txt",
            "fallbacks": [],
            "description": "VM category benchmark distribution text file"
        },
        "azure2019_cores": {
            "rel": "azure2019_data_cores.txt",
            "fallbacks": [],
            "description": "VM core count bucket benchmark distribution text file"
        },
        "azure2019_memory": {
            "rel": "azure2019_data_memory.txt",
            "fallbacks": [],
            "description": "VM memory bucket benchmark distribution text file"
        },
        "azure2019_lifetime": {
            "rel": "azure2019_data_lifetime.txt",
            "fallbacks": [],
            "description": "VM lifetime cumulative CDF distribution text file"
        },
        "notebook_analysis": {
            "rel": "Azure.2019.Public.Dataset.V2.-.Trace.Analysis.ipynb",
            "fallbacks": [],
            "description": "Trace Analysis Jupyter notebook"
        },
        "links_manifest": {
            "rel": "AzurePublicDatasetLinksV2.txt",
            "fallbacks": [],
            "description": "Official dataset download URL links manifest"
        }
    }

    status = {}
    for key, info in known_files.items():
        found_path = resolve_file_path(base_dir, info["rel"], info["fallbacks"])
        if found_path:
            size_bytes = os.path.getsize(found_path)
            status[key] = {
                "exists": True,
                "path": found_path,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "description": info["description"]
            }
        else:
            status[key] = {
                "exists": False,
                "path": os.path.join(base_dir, info["rel"]),
                "size_bytes": 0,
                "size_mb": 0.0,
                "description": info["description"]
            }

    return status


def load_schema(base_dir: str = ".") -> Tuple[Optional[pd.DataFrame], Dict]:
    """Load schema.csv file and return dataframe with metadata."""
    schema_path = resolve_file_path(base_dir, "schema.csv", [])
    if not schema_path:
        return None, {"loaded": False, "error": "schema.csv not found"}
    
    df = pd.read_csv(schema_path)
    metadata = {
        "loaded": True,
        "path": schema_path,
        "record_count": len(df),
        "columns": list(df.columns)
    }
    return df, metadata


def load_vmtable(
    base_dir: str = ".",
    sample_size: Optional[int] = 50000,
    use_cols: Optional[List[str]] = None,
    random_state: int = 42
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Load vmtable.csv efficiently with sampling/chunking support.
    
    Parameters:
        base_dir: Root directory of workspace
        sample_size: Number of records to load/sample (None for complete dataset loading)
        use_cols: Subset of columns to read
        random_state: Seed for reproducible sampling
        
    Returns:
        (DataFrame, metadata_dict)
    """
    file_info = get_available_dataset_files(base_dir).get("vmtable", {})
    if not file_info.get("exists"):
        return None, {
            "loaded": False,
            "error": "vmtable dataset file not found",
            "is_sample": False,
            "sample_size": 0
        }

    file_path = file_info["path"]

    cols_to_read = use_cols if use_cols else VMTABLE_HEADERS
    col_indices = [VMTABLE_HEADERS.index(col) for col in cols_to_read if col in VMTABLE_HEADERS]

    if sample_size is not None:
        # Load sample using pandas nrows for fast research inspection
        df = pd.read_csv(
            file_path,
            header=None,
            names=VMTABLE_HEADERS,
            usecols=col_indices if use_cols else None,
            nrows=sample_size,
            delimiter=','
        )
        is_sample = True
        record_count = len(df)
    else:
        # Load complete file
        df = pd.read_csv(
            file_path,
            header=None,
            names=VMTABLE_HEADERS,
            usecols=col_indices if use_cols else None,
            delimiter=','
        )
        is_sample = False
        record_count = len(df)

    metadata = {
        "loaded": True,
        "file_path": file_path,
        "file_size_mb": file_info["size_mb"],
        "is_sample": is_sample,
        "sample_size": sample_size if is_sample else record_count,
        "record_count": record_count,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "columns_loaded": list(df.columns)
    }

    return df, metadata


def load_category_stratified_vmtable(
    base_dir: str = ".",
    target_sample_size: int = 50000,
    random_state: int = 42
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Draw a category-stratified sample from vmtable.csv matching benchmark file azure2019_data_category.txt.
    
    Benchmark Assessment:
        Benchmark file azure2019_data_category.txt specifies the distribution by VM count
        (51.0% Delay-insensitive, 43.0% Interactive, 6.0% Unknown).
        
    Targets for 50,000 sample:
        - Delay-insensitive: 25,500 VMs (51%)
        - Interactive: 21,500 VMs (43%)
        - Unknown: 3,000 VMs (6%)
    """
    # Load default 50K sample or initial chunk
    df, meta = load_vmtable(base_dir=base_dir, sample_size=100000, random_state=random_state)
    if df is None or 'vmcategory' not in df.columns:
        return df, meta

    targets = {
        'Delay-insensitive': int(target_sample_size * 0.51),
        'Interactive': int(target_sample_size * 0.43),
        'Unknown': int(target_sample_size * 0.06)
    }

    sampled_dfs = []
    for cat, target_n in targets.items():
        cat_df = df[df['vmcategory'] == cat]
        if len(cat_df) >= target_n:
            sampled_dfs.append(cat_df.sample(n=target_n, random_state=random_state))
        else:
            sampled_dfs.append(cat_df)

    strat_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    meta["is_stratified"] = True
    meta["benchmark_assessment"] = "azure2019_data_category.txt specifies category distribution by VM count (51% Delay-insensitive, 43% Interactive, 6% Unknown)."
    meta["sample_size"] = len(strat_df)
    meta["category_breakdown"] = strat_df['vmcategory'].value_counts().to_dict()

    return strat_df, meta


def load_cpu_readings(
    base_dir: str = ".",
    sample_size: Optional[int] = 100000,
    target_vm_ids: Optional[List[str]] = None
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Load trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv efficiently.
    
    Parameters:
        base_dir: Root directory of workspace
        sample_size: Number of initial rows to load (None for complete file)
        target_vm_ids: If provided, filters readings for specific VM IDs
        
    Returns:
        (DataFrame, metadata_dict)
    """
    file_info = get_available_dataset_files(base_dir).get("vm_cpu_readings_chunk1", {})
    if not file_info.get("exists"):
        return None, {
            "loaded": False,
            "error": "vm_cpu_readings chunk 1 file not found",
            "is_sample": False
        }

    file_path = file_info["path"]

    if sample_size is not None:
        df = pd.read_csv(
            file_path,
            header=None,
            names=CPU_READINGS_HEADERS,
            nrows=sample_size,
            delimiter=','
        )
        is_sample = True
    else:
        df = pd.read_csv(
            file_path,
            header=None,
            names=CPU_READINGS_HEADERS,
            delimiter=','
        )
        is_sample = False

    if target_vm_ids is not None and len(target_vm_ids) > 0:
        df = df[df['vmid'].isin(target_vm_ids)].copy()

    metadata = {
        "loaded": True,
        "file_path": file_path,
        "file_size_mb": file_info["size_mb"],
        "is_sample": is_sample,
        "sample_size": sample_size if is_sample else len(df),
        "record_count": len(df),
        "unique_vms_in_sample": df['vmid'].nunique() if 'vmid' in df.columns else 0,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }

    return df, metadata
