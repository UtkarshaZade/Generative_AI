"""
Unit tests for data validator module and preprocessing helper functions.
"""

import pytest
import pandas as pd
from src.data.validator import validate_vmtable, validate_cpu_readings
from src.data.preprocessing import (
    add_derived_lifetime,
    map_core_memory_buckets,
    apply_lifetime_eligibility_filter,
    get_agent_specific_dataset,
)


@pytest.fixture
def sample_valid_vmtable():
    return pd.DataFrame({
        'vmid': ['vm1', 'vm2', 'vm3', 'vm4'],
        'subscriptionid': ['sub1', 'sub1', 'sub2', 'sub2'],
        'deploymentid': ['dep1', 'dep1', 'dep2', 'dep2'],
        'vmcreated': [0, 0, 0, 0],
        'vmdeleted': [1800, 3600, 90000, 180000],  # 0.5h, 1.0h, 25.0h, 50.0h
        'maxcpu': [80.5, 45.0, 95.0, 50.0],
        'avgcpu': [12.3, 10.0, 30.0, 20.0],
        'p95maxcpu': [25.0, 15.0, 70.0, 40.0],
        'vmcategory': ['Interactive', 'Delay-insensitive', 'Unknown', 'Interactive'],
        'vmcorecountbucket': ['4', '8', '>24', '4'],
        'vmmemorybucket': ['8', '32', '>64', '16']
    })


def test_validate_vmtable_valid(sample_valid_vmtable):
    res = validate_vmtable(sample_valid_vmtable)
    assert res['passed'] is True
    assert res['total_records'] == 4
    assert res['duplicate_vmids'] == 0
    assert res['invalid_categories'] == 0


def test_validate_vmtable_invalid_cpu(sample_valid_vmtable):
    df_invalid = sample_valid_vmtable.copy()
    df_invalid.loc[0, 'avgcpu'] = 150.0  # Invalid > 100%
    res = validate_vmtable(df_invalid)
    assert res['passed'] is False
    assert res['invalid_cpu_records']['avgcpu'] == 1


def test_validate_vmtable_invalid_category(sample_valid_vmtable):
    df_invalid = sample_valid_vmtable.copy()
    df_invalid.loc[0, 'vmcategory'] = 'NonExistentCategory'
    res = validate_vmtable(df_invalid)
    assert res['passed'] is False
    assert res['invalid_categories'] == 1


def test_validate_vmtable_timestamp_inconsistency(sample_valid_vmtable):
    df_invalid = sample_valid_vmtable.copy()
    df_invalid.loc[0, 'vmdeleted'] = 0  # Deletion before creation
    df_invalid.loc[0, 'vmcreated'] = 1000
    res = validate_vmtable(df_invalid)
    assert res['passed'] is False
    assert res['deletion_before_creation_count'] == 1


def test_add_derived_lifetime(sample_valid_vmtable):
    df_lifetime = add_derived_lifetime(sample_valid_vmtable)
    assert 'lifetime' in df_lifetime.columns
    # vmcreated=0, vmdeleted=3600 -> diff=3600s -> 1.0 hour
    assert df_lifetime.loc[1, 'lifetime'] == 1.0


def test_apply_lifetime_eligibility_filter(sample_valid_vmtable):
    df_filtered, summary = apply_lifetime_eligibility_filter(
        sample_valid_vmtable,
        min_lifetime_hours=24.0,
        min_scheduling_hours=48.0
    )
    
    assert 'eligible_rightsizing_pricing' in df_filtered.columns
    assert 'scheduling_eligible' in df_filtered.columns

    # vm1 (0.5h) and vm2 (1.0h) -> False (< 24.0h)
    # vm3 (25.0h) and vm4 (50.0h) -> True (>= 24.0h)
    assert bool(df_filtered.loc[0, 'eligible_rightsizing_pricing']) is False
    assert bool(df_filtered.loc[1, 'eligible_rightsizing_pricing']) is False
    assert bool(df_filtered.loc[2, 'eligible_rightsizing_pricing']) is True
    assert bool(df_filtered.loc[3, 'eligible_rightsizing_pricing']) is True

    rp_summary = summary['rightsizing_pricing_filter']
    assert summary['total_vms'] == 4
    assert rp_summary['retained_vms'] == 2
    assert rp_summary['excluded_vms'] == 2
    assert rp_summary['retained_pct'] == 50.0
    assert rp_summary['excluded_pct'] == 50.0

    sched_summary = summary['scheduling_filter']
    assert sched_summary['retained_vms'] == 1  # vm4 (50.0h >= 48.0h)
    assert sched_summary['excluded_vms'] == 3  # vm1, vm2, vm3 (< 48.0h)


def test_get_agent_specific_dataset(sample_valid_vmtable):
    # Rightsizing & Pricing agents exclude VMs below threshold (retains 2 VMs: vm3, vm4)
    rights_df = get_agent_specific_dataset(sample_valid_vmtable, agent_type="rightsizing", min_lifetime_hours=24.0, min_scheduling_hours=48.0)
    pricing_df = get_agent_specific_dataset(sample_valid_vmtable, agent_type="pricing", min_lifetime_hours=24.0, min_scheduling_hours=48.0)
    assert len(rights_df) == 2
    assert len(pricing_df) == 2
    assert set(rights_df['vmid']) == {'vm3', 'vm4'}

    # Scheduling agent excludes VMs below observation window (retains 1 VM: vm4 >= 48h)
    sched_df = get_agent_specific_dataset(sample_valid_vmtable, agent_type="scheduling", min_lifetime_hours=24.0, min_scheduling_hours=48.0)
    assert len(sched_df) == 1
    assert set(sched_df['vmid']) == {'vm4'}

    # When scheduling threshold is 0.0h, scheduling agent retains all 4 VMs
    sched_all_df = get_agent_specific_dataset(sample_valid_vmtable, agent_type="scheduling", min_lifetime_hours=24.0, min_scheduling_hours=0.0)
    assert len(sched_all_df) == 4


def test_map_core_memory_buckets(sample_valid_vmtable):
    # Test preserving original buckets
    df_preserved = map_core_memory_buckets(sample_valid_vmtable, apply_approximation=False)
    assert 'vmcorecountbucket_num' not in df_preserved.columns
    assert df_preserved.loc[2, 'vmcorecountbucket'] == '>24'

    # Test analytical mapping approximation
    df_mapped = map_core_memory_buckets(sample_valid_vmtable, apply_approximation=True)
    assert 'vmcorecountbucket_num' in df_mapped.columns
    assert df_mapped.loc[2, 'vmcorecountbucket_num'] == 30.0
    assert df_mapped.loc[2, 'vmmemorybucket_num'] == 70.0

