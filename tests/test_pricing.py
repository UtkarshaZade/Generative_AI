"""
Unit tests for synthetic cloud pricing module and scheduling eligibility filter.
"""

import pytest
import pandas as pd
from src.data.pricing import (
    RATE_CARD,
    get_instance_pricing,
    add_pricing_columns,
    get_pricing_summary,
)
from src.data.preprocessing import (
    apply_lifetime_eligibility_filter,
    get_agent_specific_dataset,
)


@pytest.fixture
def sample_pricing_vmtable():
    return pd.DataFrame({
        'vmid': ['vm1', 'vm2', 'vm3', 'vm4'],
        'vmcreated': [0, 0, 0, 0],
        'vmdeleted': [1800, 90000, 180000, 360000],  # 0.5h, 25.0h, 50.0h, 100.0h
        'vmcorecountbucket': ['2', '4', '8', '>24'],
        'vmmemorybucket': ['4', '32', '64', '>64']
    })


def test_rate_card_structure():
    assert ("2", "4") in RATE_CARD
    assert ("4", "32") in RATE_CARD
    assert RATE_CARD[("2", "4")]["on_demand_hourly"] > 0
    assert RATE_CARD[("2", "4")]["reserved_1yr_hourly"] < RATE_CARD[("2", "4")]["on_demand_hourly"]
    assert RATE_CARD[("2", "4")]["spot_hourly"] < RATE_CARD[("2", "4")]["reserved_1yr_hourly"]


def test_get_instance_pricing():
    pricing_2_4 = get_instance_pricing("2", "4")
    assert pricing_2_4["on_demand_hourly"] == 0.0416

    # Test fallback
    fallback = get_instance_pricing("16", "128")
    assert fallback["on_demand_hourly"] > 0


def test_add_pricing_columns(sample_pricing_vmtable):
    df_priced = add_pricing_columns(sample_pricing_vmtable)
    
    assert "on_demand_hourly_rate" in df_priced.columns
    assert "reserved_1yr_hourly_rate" in df_priced.columns
    assert "spot_hourly_rate" in df_priced.columns
    assert "on_demand_monthly_cost" in df_priced.columns
    assert "reserved_1yr_monthly_cost" in df_priced.columns
    assert "spot_monthly_cost" in df_priced.columns
    assert "predicted_monthly_cost" in df_priced.columns

    # Verify predicted_monthly_cost equals on_demand_monthly_cost by default
    assert df_priced.loc[0, "predicted_monthly_cost"] == df_priced.loc[0, "on_demand_monthly_cost"]
    # Check rate formula: 0.0416 * 730 = 30.37
    assert df_priced.loc[0, "on_demand_monthly_cost"] == 30.37


def test_get_pricing_summary(sample_pricing_vmtable):
    summary = get_pricing_summary(sample_pricing_vmtable)
    assert summary["total_vms"] == 4
    assert summary["total_on_demand_monthly_cost"] > 0
    assert summary["potential_reserved_savings_pct"] > 0
    assert summary["potential_spot_savings_pct"] > summary["potential_reserved_savings_pct"]


def test_scheduling_eligible_flag(sample_pricing_vmtable):
    df_filtered, summary = apply_lifetime_eligibility_filter(
        sample_pricing_vmtable,
        min_lifetime_hours=24.0,
        min_scheduling_hours=48.0
    )

    assert "eligible_rightsizing_pricing" in df_filtered.columns
    assert "scheduling_eligible" in df_filtered.columns

    # vm1 (0.5h) -> False for both
    # vm2 (25.0h) -> True for Rightsizing/Pricing (>= 24h), False for Scheduling (< 48h)
    # vm3 (50.0h) -> True for both (>= 24h & >= 48h)
    # vm4 (100.0h) -> True for both
    assert bool(df_filtered.loc[0, "eligible_rightsizing_pricing"]) is False
    assert bool(df_filtered.loc[0, "scheduling_eligible"]) is False

    assert bool(df_filtered.loc[1, "eligible_rightsizing_pricing"]) is True
    assert bool(df_filtered.loc[1, "scheduling_eligible"]) is False

    assert bool(df_filtered.loc[2, "eligible_rightsizing_pricing"]) is True
    assert bool(df_filtered.loc[2, "scheduling_eligible"]) is True

    # Test get_agent_specific_dataset
    rights_df = get_agent_specific_dataset(sample_pricing_vmtable, agent_type="rightsizing", min_lifetime_hours=24.0, min_scheduling_hours=48.0)
    sched_df = get_agent_specific_dataset(sample_pricing_vmtable, agent_type="scheduling", min_lifetime_hours=24.0, min_scheduling_hours=48.0)

    assert len(rights_df) == 3  # vm2, vm3, vm4
    assert len(sched_df) == 2   # vm3, vm4
