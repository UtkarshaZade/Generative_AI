"""
Exploratory Data Analysis and Statistics Module.
Provides statistical calculations, feature relevance matrix, missing data audit, and deterministic observations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def get_dataset_overview(
    df: pd.DataFrame,
    file_status: Dict[str, Any],
    validation_results: Dict[str, Any],
    is_sample: bool = True,
    sample_size: int = 50000
) -> Dict[str, Any]:
    """Generate high-level dataset overview statistics."""
    total_loaded = len(df)
    unique_vms = df['vmid'].nunique() if 'vmid' in df.columns else 0
    unique_subs = df['subscriptionid'].nunique() if 'subscriptionid' in df.columns else 0
    unique_deps = df['deploymentid'].nunique() if 'deploymentid' in df.columns else 0

    col_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_sum = int(df.isnull().sum().sum())
    duplicate_count = validation_results.get('duplicate_vmids', 0)

    return {
        "is_sample": is_sample,
        "subset_description": f"Sample of {total_loaded:,} records" if is_sample else f"Complete dataset ({total_loaded:,} records)",
        "total_records_loaded": total_loaded,
        "unique_vms": unique_vms,
        "unique_subscriptions": unique_subs,
        "unique_deployments": unique_deps,
        "columns": list(df.columns),
        "column_dtypes": col_dtypes,
        "missing_values_total": missing_sum,
        "duplicate_vmid_count": duplicate_count,
        "file_status": file_status
    }


def analyze_cpu_utilisation(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate descriptive statistics for avgcpu, maxcpu, and p95maxcpu."""
    stats = {}
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    
    for cpu_col in ['avgcpu', 'maxcpu', 'p95maxcpu']:
        if cpu_col in df.columns:
            s = df[cpu_col].dropna()
            stats[cpu_col] = {
                "count": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median()),
                "percentiles": {f"p{int(p*100)}": float(s.quantile(p)) for p in percentiles}
            }
            
    return stats


def analyze_lifetime(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate statistical distribution of VM lifetime (in hours)."""
    if 'lifetime' not in df.columns:
        return {"error": "lifetime column not present in dataframe"}

    s = df['lifetime'].dropna()
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    
    return {
        "count": int(len(s)),
        "mean_hours": float(s.mean()),
        "std_hours": float(s.std()),
        "min_hours": float(s.min()),
        "max_hours": float(s.max()),
        "median_hours": float(s.median()),
        "percentiles_hours": {f"p{int(p*100)}": float(s.quantile(p)) for p in percentiles},
        "derived_formula": "lifetime = max(vmdeleted - vmcreated, 300) / 3600"
    }


def analyze_category(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze frequency and percentage distribution of vmcategory."""
    if 'vmcategory' not in df.columns:
        return pd.DataFrame()

    counts = df['vmcategory'].value_counts(dropna=False)
    pcts = df['vmcategory'].value_counts(dropna=False, normalize=True) * 100

    res = pd.DataFrame({
        'Frequency': counts,
        'Percentage': pcts.round(2)
    })
    res.index.name = 'VM Category'
    return res


def analyze_memory_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze frequency and percentage distribution of vmmemorybucket."""
    if 'vmmemorybucket' not in df.columns:
        return pd.DataFrame()

    counts = df['vmmemorybucket'].astype(str).value_counts()
    pcts = df['vmmemorybucket'].astype(str).value_counts(normalize=True) * 100

    res = pd.DataFrame({
        'Frequency': counts,
        'Percentage': pcts.round(2)
    })
    res.index.name = 'Memory Bucket (GB)'
    return res


def analyze_core_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze frequency and percentage distribution of vmcorecountbucket."""
    if 'vmcorecountbucket' not in df.columns:
        return pd.DataFrame()

    counts = df['vmcorecountbucket'].astype(str).value_counts()
    pcts = df['vmcorecountbucket'].astype(str).value_counts(normalize=True) * 100

    res = pd.DataFrame({
        'Frequency': counts,
        'Percentage': pcts.round(2)
    })
    res.index.name = 'Core Count Bucket'
    return res


def analyze_time_series_cpu(
    cpu_df: pd.DataFrame,
    sample_vm_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Analyze time-series CPU telemetry samples."""
    if cpu_df is None or len(cpu_df) == 0:
        return {"loaded": False, "error": "No CPU readings data available"}

    df_sub = cpu_df
    if sample_vm_ids and len(sample_vm_ids) > 0:
        df_sub = cpu_df[cpu_df['vmid'].isin(sample_vm_ids)]

    overall_mean = float(df_sub['avgcpu'].mean()) if 'avgcpu' in df_sub.columns else 0.0
    overall_max = float(df_sub['maxcpu'].max()) if 'maxcpu' in df_sub.columns else 0.0
    overall_min = float(df_sub['mincpu'].min()) if 'mincpu' in df_sub.columns else 0.0

    return {
        "loaded": True,
        "total_time_series_records": len(df_sub),
        "unique_vms_in_sample": int(df_sub['vmid'].nunique()) if 'vmid' in df_sub.columns else 0,
        "overall_mean_cpu": round(overall_mean, 2),
        "overall_max_cpu": round(overall_max, 2),
        "overall_min_cpu": round(overall_min, 2)
    }


def get_feature_relevance_matrix() -> pd.DataFrame:
    """
    Generate initial feature relevance assessment matrix for future optimization agents.
    
    Ratings: High, Medium, Low, None based on technical domain reasoning.
    """
    matrix = [
        {"Feature": "avgcpu", "Rightsizing": "High", "Pricing": "Medium", "Autoscaling": "High", "Reason": "Indicates mean workload utilization and baseline resource usage"},
        {"Feature": "maxcpu", "Rightsizing": "High", "Pricing": "Low", "Autoscaling": "High", "Reason": "Measures peak capacity demand and spike headroom requirement"},
        {"Feature": "p95maxcpu", "Rightsizing": "High", "Pricing": "Low", "Autoscaling": "High", "Reason": "Key indicator for overprovisioning; excludes single transient spikes"},
        {"Feature": "vmcorecountbucket", "Rightsizing": "High", "Pricing": "High", "Autoscaling": "Medium", "Reason": "Current vCPU capacity baseline and unit cost multiplier"},
        {"Feature": "vmmemorybucket", "Rightsizing": "High", "Pricing": "High", "Autoscaling": "Medium", "Reason": "Current RAM capacity baseline and SKU family determinant"},
        {"Feature": "vmcategory", "Rightsizing": "Medium", "Pricing": "High", "Autoscaling": "High", "Reason": "Differentiates interactive services from delay-insensitive spot tasks"},
        {"Feature": "lifetime", "Rightsizing": "Medium", "Pricing": "High", "Autoscaling": "High", "Reason": "Distinguishes short-lived jobs from steady-state reserved instance candidates"},
        {"Feature": "mincpu/maxcpu time-series", "Rightsizing": "Medium", "Pricing": "Low", "Autoscaling": "High", "Reason": "5-minute telemetry enables pattern forecasting and dynamic scaling triggers"},
        {"Feature": "subscriptionid", "Rightsizing": "Low", "Pricing": "High", "Autoscaling": "Low", "Reason": "Groups enterprise tenant VMs for volume commitment negotiations"},
        {"Feature": "deploymentid", "Rightsizing": "Low", "Pricing": "Medium", "Autoscaling": "High", "Reason": "Identifies co-allocated application clusters for group scaling policies"}
    ]
    
    df = pd.DataFrame(matrix)
    return df


from .preprocessing import apply_lifetime_eligibility_filter


def analyze_lifetime_eligibility(
    df: pd.DataFrame,
    min_lifetime_hours: float = 24.0
) -> Dict[str, Any]:
    """Calculate detailed statistics on lifetime eligibility filtering."""
    _, summary = apply_lifetime_eligibility_filter(df, min_lifetime_hours=min_lifetime_hours)
    return summary


def get_data_gaps() -> List[Dict[str, str]]:
    """
    List information NOT directly provided by the Azure 2019 dataset
    that must later be synthetically modeled.
    """
    return [
        {"Category": "Cloud Pricing", "Item": "On-Demand Hourly Rates ($/hr)", "Impact": "Required to calculate baseline monetary spending"},
        {"Category": "Cloud Pricing", "Item": "Reserved Instance Discounts (1-yr/3-yr %)", "Impact": "Required to evaluate long-term commitment savings"},
        {"Category": "Cloud Pricing", "Item": "Spot Instance Rates & Eviction Risk", "Impact": "Required to negotiate fault-tolerant spot adoption"},
        {"Category": "SLA & Constraints", "Item": "Latency & Response Time Limits (ms)", "Impact": "Required to enforce performance safety guardrails"},
        {"Category": "SLA & Constraints", "Item": "Maximum CPU Throttling % Threshold", "Impact": "Required to prevent interactive service degradation"},
        {"Category": "SLA & Constraints", "Item": "Interruption Tolerance Limits", "Impact": "Required to bound spot eviction risk for critical workloads"},
        {"Category": "Operational Costs", "Item": "VM Resizing Reboot Cost ($ / downtime)", "Impact": "Required to weigh downsizing gains against migration disruption"},
        {"Category": "Operational Costs", "Item": "SLA Violation Penalty Costs ($/hr)", "Impact": "Required to penalize aggressive under-provisioning during negotiation"},
        {"Category": "Human Governance", "Item": "Approval Thresholds & Risk Preferences", "Impact": "Required to customize automated negotiator aggressiveness"}
    ]


def get_deterministic_observations(df: pd.DataFrame, cpu_df: Optional[pd.DataFrame] = None) -> List[str]:
    """Generate empirical observations directly supported by loaded data."""
    obs = []
    
    if 'avgcpu' in df.columns:
        median_avg_cpu = df['avgcpu'].median()
        mean_avg_cpu = df['avgcpu'].mean()
        obs.append(f"In the loaded sample ({len(df):,} VMs), the median average CPU utilization is {median_avg_cpu:.2f}% (mean: {mean_avg_cpu:.2f}%).")

    if 'p95maxcpu' in df.columns:
        p95_below_20 = (df['p95maxcpu'] < 20.0).mean() * 100
        obs.append(f"{p95_below_20:.1f}% of loaded VMs exhibit a 95th percentile max CPU utilization below 20%.")

    if 'vmcorecountbucket' in df.columns:
        top_core = df['vmcorecountbucket'].astype(str).value_counts().index[0]
        top_core_pct = df['vmcorecountbucket'].astype(str).value_counts(normalize=True).iloc[0] * 100
        obs.append(f"The most common vCPU core count bucket is '{top_core}', representing {top_core_pct:.1f}% of loaded instances.")

    if 'vmmemorybucket' in df.columns:
        top_mem = df['vmmemorybucket'].astype(str).value_counts().index[0]
        top_mem_pct = df['vmmemorybucket'].astype(str).value_counts(normalize=True).iloc[0] * 100
        obs.append(f"The most common memory bucket is '{top_mem}' GB, representing {top_mem_pct:.1f}% of loaded instances.")

    if 'vmcategory' in df.columns:
        cats = df['vmcategory'].value_counts(normalize=True) * 100
        cats_str = ", ".join([f"{cat}: {pct:.1f}%" for cat, pct in cats.items()])
        obs.append(f"Workload category distribution in sample: {cats_str}.")

    if 'lifetime' in df.columns:
        med_life = df['lifetime'].median()
        obs.append(f"Median VM lifetime across loaded instances is {med_life:.2f} hours.")
        
        _, filter_summary = apply_lifetime_eligibility_filter(df, min_lifetime_hours=24.0, min_scheduling_hours=48.0)
        rp_s = filter_summary['rightsizing_pricing_filter']
        sch_s = filter_summary['scheduling_filter']
        obs.append(
            f"At a {rp_s['threshold_hours']:.1f}-hour threshold, {rp_s['retained_vms']:,} VMs ({rp_s['retained_pct']}%) "
            f"are retained for Rightsizing & Pricing agents ({rp_s['excluded_vms']:,} excluded as short-lived <24h). "
            f"At a {sch_s['threshold_hours']:.1f}-hour observation window, {sch_s['retained_vms']:,} VMs ({sch_s['retained_pct']}%) "
            f"are eligible for the Scheduling agent."
        )

    if cpu_df is not None and len(cpu_df) > 0:
        obs.append(f"Time-series CPU telemetry chunk 1 contains {len(cpu_df):,} 5-minute sampling readings across {cpu_df['vmid'].nunique():,} unique VMs.")

    return obs

