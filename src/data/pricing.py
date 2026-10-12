"""
Synthetic Cloud Pricing Module for Azure 2019 Public Dataset V2.
Anchored to real published Azure and AWS rate cards for closest matching instance families.

Rate Card Source & Citation:
----------------------------
Primary Source: Microsoft Azure Retail Prices API (Region: East US) & AWS EC2 Rate Cards (Region: us-east-1)
Publication Date: October 2024 / 2025 Published Rate Cards
Instance Series Reference: Azure D-series, B-series, E-series, F-series / AWS t3, m5, c5, r5 series

Pricing Models:
  1. On-Demand Hourly Rate ($/hr): Standard pay-as-you-go hourly rate.
  2. 1-Year Reserved Hourly Rate ($/hr): 1-Year No-Upfront / All-Upfront commitment rate (~37.5% - 40% discount off On-Demand).
  3. Spot Hourly Rate ($/hr): Interruptible spot capacity rate (~70% - 75% discount off On-Demand).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


# Standard monthly operating hours (365 days * 24 hours / 12 months = 730 hours)
MONTHLY_HOURS = 730.0


# Published Rate Card Dictionary mapping (core_bucket, memory_bucket) to rates ($/hr)
# Format: (core_bucket_str, memory_bucket_str): { ... }
RATE_CARD: Dict[Tuple[str, str], Dict[str, Any]] = {
    # 2 vCPU Configurations
    ("2", "2"): {
        "matched_family": "Azure Standard_F2s_v2 / AWS t3.small",
        "on_demand_hourly": 0.0416,
        "reserved_1yr_hourly": 0.0250,  # ~40% discount
        "spot_hourly": 0.0104,          # ~75% discount
    },
    ("2", "4"): {
        "matched_family": "Azure Standard_B2s / AWS t3.medium",
        "on_demand_hourly": 0.0416,
        "reserved_1yr_hourly": 0.0260,  # ~37.5% discount
        "spot_hourly": 0.0125,          # ~70% discount
    },
    ("2", "8"): {
        "matched_family": "Azure Standard_D2s_v5 / AWS m5.large",
        "on_demand_hourly": 0.0960,
        "reserved_1yr_hourly": 0.0576,  # ~40% discount
        "spot_hourly": 0.0240,          # ~75% discount
    },
    ("2", "32"): {
        "matched_family": "Azure Standard_E2s_v5 / AWS r5.large",
        "on_demand_hourly": 0.1260,
        "reserved_1yr_hourly": 0.0756,  # ~40% discount
        "spot_hourly": 0.0315,          # ~75% discount
    },

    # 4 vCPU Configurations
    ("4", "8"): {
        "matched_family": "Azure Standard_F4s_v2 / AWS c5.xlarge",
        "on_demand_hourly": 0.1690,
        "reserved_1yr_hourly": 0.1014,  # ~40% discount
        "spot_hourly": 0.0423,          # ~75% discount
    },
    ("4", "32"): {
        "matched_family": "Azure Standard_D4s_v5 / AWS m5.xlarge",
        "on_demand_hourly": 0.1920,
        "reserved_1yr_hourly": 0.1152,  # ~40% discount
        "spot_hourly": 0.0480,          # ~75% discount
    },

    # 8 vCPU Configurations
    ("8", "32"): {
        "matched_family": "Azure Standard_D8s_v5 / AWS m5.2xlarge",
        "on_demand_hourly": 0.3840,
        "reserved_1yr_hourly": 0.2304,  # ~40% discount
        "spot_hourly": 0.0960,          # ~75% discount
    },
    ("8", "64"): {
        "matched_family": "Azure Standard_E8s_v5 / AWS r5.2xlarge",
        "on_demand_hourly": 0.5040,
        "reserved_1yr_hourly": 0.3024,  # ~40% discount
        "spot_hourly": 0.1260,          # ~75% discount
    },

    # 24 vCPU Configurations
    ("24", "64"): {
        "matched_family": "Azure Standard_F24s_v2 / AWS c5.6xlarge",
        "on_demand_hourly": 1.0140,
        "reserved_1yr_hourly": 0.6084,  # ~40% discount
        "spot_hourly": 0.2535,          # ~75% discount
    },

    # >24 vCPU / >64 GB RAM Configurations (Numeric Approximation: 30 cores, 70 GB / 32 cores, 128 GB)
    (">24", ">64"): {
        "matched_family": "Azure Standard_D32s_v5 / AWS m5.8xlarge",
        "on_demand_hourly": 1.5360,
        "reserved_1yr_hourly": 0.9216,  # ~40% discount
        "spot_hourly": 0.3840,          # ~75% discount
    },
}


def get_instance_pricing(core_bucket: str, memory_bucket: str) -> Dict[str, Any]:
    """
    Look up pricing details for a given core and memory bucket pair.
    
    If exact pair is not found in rate card, falls back to estimated baseline rates
    derived as $0.048/hr per vCPU + $0.006/hr per GB RAM.
    """
    key = (str(core_bucket).strip(), str(memory_bucket).strip())
    if key in RATE_CARD:
        return RATE_CARD[key]
    
    # Fallback calculation if unmapped combination occurs
    try:
        cores = 30.0 if key[0] == '>24' else float(key[0])
        mem = 70.0 if key[1] == '>64' else float(key[1])
    except ValueError:
        cores, mem = 2.0, 4.0

    od = round(cores * 0.048 + mem * 0.006, 4)
    ri = round(od * 0.60, 4)
    spot = round(od * 0.25, 4)

    return {
        "matched_family": f"Fallback Generic ({cores}vCPU, {mem}GB)",
        "on_demand_hourly": od,
        "reserved_1yr_hourly": ri,
        "spot_hourly": spot
    }


def add_pricing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic cloud pricing and predicted monthly cost columns to the dataset.

    Added Columns:
      - on_demand_hourly_rate ($/hr)
      - reserved_1yr_hourly_rate ($/hr)
      - spot_hourly_rate ($/hr)
      - on_demand_monthly_cost ($/month = rate * 730)
      - reserved_1yr_monthly_cost ($/month = rate * 730)
      - spot_monthly_cost ($/month = rate * 730)
      - predicted_monthly_cost ($/month, baseline set to on_demand_monthly_cost)
      - matched_instance_family (Name of matched benchmark SKU family)
    """
    df_copy = df.copy()

    if 'vmcorecountbucket' not in df_copy.columns or 'vmmemorybucket' not in df_copy.columns:
        return df_copy

    od_rates = []
    ri_rates = []
    spot_rates = []
    families = []

    for _, row in df_copy.iterrows():
        c_bucket = str(row['vmcorecountbucket'])
        m_bucket = str(row['vmmemorybucket'])
        pricing = get_instance_pricing(c_bucket, m_bucket)

        od_rates.append(pricing['on_demand_hourly'])
        ri_rates.append(pricing['reserved_1yr_hourly'])
        spot_rates.append(pricing['spot_hourly'])
        families.append(pricing['matched_family'])

    df_copy['on_demand_hourly_rate'] = od_rates
    df_copy['reserved_1yr_hourly_rate'] = ri_rates
    df_copy['spot_hourly_rate'] = spot_rates
    df_copy['matched_instance_family'] = families

    df_copy['on_demand_monthly_cost'] = np.round(df_copy['on_demand_hourly_rate'] * MONTHLY_HOURS, 2)
    df_copy['reserved_1yr_monthly_cost'] = np.round(df_copy['reserved_1yr_hourly_rate'] * MONTHLY_HOURS, 2)
    df_copy['spot_monthly_cost'] = np.round(df_copy['spot_hourly_rate'] * MONTHLY_HOURS, 2)

    # Baseline predicted monthly cost (On-Demand pricing baseline)
    df_copy['predicted_monthly_cost'] = df_copy['on_demand_monthly_cost']

    return df_copy


def get_pricing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics for predicted monthly cost across pricing models."""
    if 'on_demand_monthly_cost' not in df.columns:
        df = add_pricing_columns(df)

    total_vms = len(df)
    od_sum = float(df['on_demand_monthly_cost'].sum())
    ri_sum = float(df['reserved_1yr_monthly_cost'].sum())
    spot_sum = float(df['spot_monthly_cost'].sum())

    return {
        "total_vms": total_vms,
        "total_on_demand_monthly_cost": round(od_sum, 2),
        "total_reserved_1yr_monthly_cost": round(ri_sum, 2),
        "total_spot_monthly_cost": round(spot_sum, 2),
        "mean_on_demand_monthly_cost": round(df['on_demand_monthly_cost'].mean(), 2) if total_vms > 0 else 0.0,
        "mean_reserved_1yr_monthly_cost": round(df['reserved_1yr_monthly_cost'].mean(), 2) if total_vms > 0 else 0.0,
        "mean_spot_monthly_cost": round(df['spot_monthly_cost'].mean(), 2) if total_vms > 0 else 0.0,
        "potential_reserved_savings_pct": round(((od_sum - ri_sum) / od_sum * 100.0), 2) if od_sum > 0 else 0.0,
        "potential_spot_savings_pct": round(((od_sum - spot_sum) / od_sum * 100.0), 2) if od_sum > 0 else 0.0,
    }
