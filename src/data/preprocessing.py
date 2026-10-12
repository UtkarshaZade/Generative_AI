"""
Data Preprocessing and Feature Derivation Module.
Computes derived metrics, handles bucket mappings, and applies agent-specific eligibility filters.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def add_derived_lifetime(df: pd.DataFrame, min_lifetime_sec: int = 300) -> pd.DataFrame:
    """
    Derive VM lifetime in hours based on creation and deletion timestamps.
    
    Formula:
        lifetime_hours = max((vmdeleted - vmcreated), 300) / 3600
        
    Note:
        A lower bound floor of 300 seconds (5 minutes) is applied as defined in the
        official Azure 2019 trace notebook to prevent zero-division or near-zero lifetime artifacts.
    """
    df_copy = df.copy()
    if 'vmcreated' in df_copy.columns and 'vmdeleted' in df_copy.columns:
        diff_seconds = df_copy['vmdeleted'] - df_copy['vmcreated']
        df_copy['lifetime'] = np.maximum(diff_seconds, min_lifetime_sec) / 3600.0
    return df_copy


def apply_lifetime_eligibility_filter(
    df: pd.DataFrame,
    min_lifetime_hours: float = 24.0,
    min_scheduling_hours: float = 48.0,
    min_lifetime_sec: int = 300
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply agent-specific lifetime & observation window eligibility filters to the dataset.

    Formula:
        lifetime_hours = max((vmdeleted - vmcreated), 300) / 3600

    Rules:
        1. eligible_rightsizing_pricing (boolean):
           - True for VMs with lifetime_hours >= min_lifetime_hours (default 24.0 hours).
           - Excludes short-lived / ephemeral VMs where SKU resizing reboot overhead or long-term
             pricing commitments are inappropriate.
        
        2. scheduling_eligible (boolean):
           - True for VMs with lifetime_hours >= min_scheduling_hours (default 48.0 hours).
           - Excludes VMs below a minimum observation window specifically for the Scheduling agent
             (which requires sufficient time-series history to observe workload periodicity/diurnal patterns).
           - Maintained separately from eligible_rightsizing_pricing.

    Returns:
        (df_copy with 'eligible_rightsizing_pricing' and 'scheduling_eligible' boolean columns, summary_dict)
    """
    df_copy = add_derived_lifetime(df, min_lifetime_sec=min_lifetime_sec)
    
    if 'lifetime' in df_copy.columns:
        df_copy['eligible_rightsizing_pricing'] = df_copy['lifetime'] >= min_lifetime_hours
        df_copy['scheduling_eligible'] = df_copy['lifetime'] >= min_scheduling_hours
    else:
        df_copy['eligible_rightsizing_pricing'] = True
        df_copy['scheduling_eligible'] = True

    total_vms = len(df_copy)
    
    # Rightsizing & Pricing Eligibility Metrics
    retained_rp = int(df_copy['eligible_rightsizing_pricing'].sum())
    excluded_rp = total_vms - retained_rp
    retained_rp_pct = round((retained_rp / total_vms * 100.0), 2) if total_vms > 0 else 0.0
    excluded_rp_pct = round((excluded_rp / total_vms * 100.0), 2) if total_vms > 0 else 0.0

    # Scheduling Eligibility Metrics
    retained_sched = int(df_copy['scheduling_eligible'].sum())
    excluded_sched = total_vms - retained_sched
    retained_sched_pct = round((retained_sched / total_vms * 100.0), 2) if total_vms > 0 else 0.0
    excluded_sched_pct = round((excluded_sched / total_vms * 100.0), 2) if total_vms > 0 else 0.0

    summary = {
        "total_vms": total_vms,
        "rightsizing_pricing_filter": {
            "threshold_hours": min_lifetime_hours,
            "retained_vms": retained_rp,
            "retained_pct": retained_rp_pct,
            "excluded_vms": excluded_rp,
            "excluded_pct": excluded_rp_pct,
            "target_agents": ["Rightsizing Agent", "Pricing Agent"]
        },
        "scheduling_filter": {
            "threshold_hours": min_scheduling_hours,
            "retained_vms": retained_sched,
            "retained_pct": retained_sched_pct,
            "excluded_vms": excluded_sched,
            "excluded_pct": excluded_sched_pct,
            "target_agents": ["Scheduling & Autoscaling Agent"]
        }
    }

    return df_copy, summary


def get_agent_specific_dataset(
    df: pd.DataFrame,
    agent_type: str = "rightsizing",
    min_lifetime_hours: float = 24.0,
    min_scheduling_hours: float = 48.0
) -> pd.DataFrame:
    """
    Retrieve agent-specific eligible VM dataset based on lifetime & observation window filters.

    Parameters:
        df: DataFrame containing VM data.
        agent_type: 'rightsizing', 'pricing', or 'scheduling'/'autoscaling'.
        min_lifetime_hours: Threshold for Rightsizing/Pricing (default 24.0h).
        min_scheduling_hours: Observation window threshold for Scheduling (default 48.0h).
    """
    df_filtered, _ = apply_lifetime_eligibility_filter(
        df,
        min_lifetime_hours=min_lifetime_hours,
        min_scheduling_hours=min_scheduling_hours
    )
    
    agent_clean = agent_type.lower().strip()
    if agent_clean in ['scheduling', 'autoscaling', 'scheduling_agent', 'autoscaling_agent']:
        # Scheduling agent filters by scheduling_eligible flag (default >= 48h)
        return df_filtered[df_filtered['scheduling_eligible']].copy()
    else:
        # Rightsizing & Pricing agents filter by eligible_rightsizing_pricing flag (default >= 24h)
        return df_filtered[df_filtered['eligible_rightsizing_pricing']].copy()


def map_core_memory_buckets(df: pd.DataFrame, apply_approximation: bool = False) -> pd.DataFrame:
    """
    Handle VM core count and memory buckets.
    
    Semantics:
        - By default (apply_approximation=False), original string bucket labels (e.g. '>24', '>64') are preserved.
        - If apply_approximation=True, numerical approximations are generated:
            - core count bucket '>24' is mapped to 30
            - memory bucket '>64' is mapped to 70
          as performed in Azure.2019.Public.Dataset.V2.-.Trace.Analysis.ipynb for computing core-hour totals.
          
    Analytical Formula:
        corehour = lifetime * vmcorecountbucket_num
    """
    df_copy = df.copy()
    
    if not apply_approximation:
        # Preserve original bucket labels
        return df_copy

    # Apply documented numerical approximations
    if 'vmcorecountbucket' in df_copy.columns:
        df_copy['vmcorecountbucket_num'] = (
            df_copy['vmcorecountbucket']
            .astype(str)
            .replace({'>24': '30'})
            .astype(float)
        )

    if 'vmmemorybucket' in df_copy.columns:
        df_copy['vmmemorybucket_num'] = (
            df_copy['vmmemorybucket']
            .astype(str)
            .replace({'>64': '70'})
            .astype(float)
        )

    if 'lifetime' in df_copy.columns and 'vmcorecountbucket_num' in df_copy.columns:
        df_copy['corehour'] = df_copy['lifetime'] * df_copy['vmcorecountbucket_num']

    return df_copy
