"""
Unit tests for optimization agents (RightsizingAgent, PricingAgent, SchedulingAgent) and multi-agent runner.
"""

import pytest
import pandas as pd
from src.agents import (
    Proposal,
    RightsizingAgent,
    PricingAgent,
    SchedulingAgent,
    run_all_agents
)
from src.data.preprocessing import apply_lifetime_eligibility_filter
from src.data.pricing import add_pricing_columns


@pytest.fixture
def sample_agent_dataset():
    df = pd.DataFrame({
        'vmid': ['vm_overprovisioned', 'vm_interactive', 'vm_delay_insensitive', 'vm_short_lived'],
        'vmcreated': [0, 0, 0, 0],
        'vmdeleted': [2520000, 2520000, 2520000, 1800],  # 700h, 700h, 700h, 0.5h
        'maxcpu': [18.0, 25.0, 40.0, 90.0],
        'avgcpu': [5.0, 25.0, 8.0, 45.0],
        'p95maxcpu': [12.0, 60.0, 15.0, 85.0],
        'vmcategory': ['Interactive', 'Interactive', 'Delay-insensitive', 'Unknown'],
        'vmcorecountbucket': ['8', '4', '4', '2'],
        'vmmemorybucket': ['32', '32', '8', '4']
    })
    df = add_pricing_columns(df)
    df, _ = apply_lifetime_eligibility_filter(df, min_lifetime_hours=24.0, min_scheduling_hours=48.0)
    return df


@pytest.fixture
def sample_cpu_readings_df():
    # Generate 576 5-minute readings (48 hours) for vm_overprovisioned and vm_delay_insensitive
    records = []
    for vm_id in ['vm_overprovisioned', 'vm_delay_insensitive']:
        for i in range(576):
            ts = i * 300  # 5 minute intervals in seconds
            hour = (ts // 3600) % 24
            # Hours 0-7 are off-peak (< 10% CPU), remaining are 12% CPU
            cpu_val = 4.0 if hour < 8 else 12.0
            records.append({
                'timestamp': ts,
                'vmid': vm_id,
                'mincpu': cpu_val - 1.0,
                'maxcpu': cpu_val + 2.0,
                'avgcpu': cpu_val
            })
    return pd.DataFrame(records)


def test_rightsizing_agent(sample_agent_dataset):
    agent = RightsizingAgent(p95_threshold=20.0)
    proposals = agent.generate_proposals(sample_agent_dataset)

    # Eligible & P95 CPU < 20%: vm_overprovisioned (p95=12.0%), vm_delay_insensitive (p95=15.0%)
    # vm_short_lived is excluded by eligible_rightsizing_pricing=False (<24h)
    assert len(proposals) == 2
    vm_ids = {p.vm_id for p in proposals}
    assert vm_ids == {'vm_overprovisioned', 'vm_delay_insensitive'}

    for p in proposals:
        assert p.agent_name == "RightsizingAgent"
        assert p.action_type == "DOWNSIZE_SKU"
        assert p.predicted_monthly_savings > 0
        assert 0.0 <= p.confidence <= 1.0


def test_pricing_agent(sample_agent_dataset):
    agent = PricingAgent(min_reserved_lifetime_hours=600.0, max_cpu_variance=100.0)
    proposals = agent.generate_proposals(sample_agent_dataset)

    # Eligible VMs with lifetime >= 600h: vm_overprovisioned (700h, low var), vm_interactive (700h, low var), vm_delay_insensitive (Spot)
    assert len(proposals) == 3

    prop_map = {p.vm_id: p for p in proposals}
    
    # vm_delay_insensitive -> Spot pricing proposal
    assert prop_map['vm_delay_insensitive'].action_type == "SWITCH_TO_SPOT"
    assert prop_map['vm_delay_insensitive'].proposed_configuration['pricing_model'] == "Spot"

    # vm_interactive -> 1-Year Reserved Instance proposal (lifetime >= 600h and low variance)
    assert prop_map['vm_interactive'].action_type == "SWITCH_TO_RESERVED_1YR"
    assert prop_map['vm_interactive'].proposed_configuration['pricing_model'] == "1-Year Reserved"


def test_scheduling_agent(sample_agent_dataset, sample_cpu_readings_df):
    agent = SchedulingAgent(avg_cpu_threshold=15.0)
    proposals = agent.generate_proposals(sample_agent_dataset, cpu_readings_df=sample_cpu_readings_df)

    # Eligible for scheduling (lifetime >= 48h and 48h readings with >=6h < 10% CPU): vm_overprovisioned, vm_delay_insensitive (2 VMs)
    assert len(proposals) == 2
    vm_ids = {p.vm_id for p in proposals}
    assert vm_ids == {'vm_overprovisioned', 'vm_delay_insensitive'}

    for p in proposals:
        assert p.agent_name == "SchedulingAgent"
        assert p.action_type == "SCHEDULE_OFF_PEAK_PAUSE"
        assert p.predicted_monthly_savings > 0


def test_run_all_agents_and_conflicts(sample_agent_dataset, sample_cpu_readings_df):
    agent_proposals, summary = run_all_agents(
        sample_agent_dataset,
        cpu_readings_df=sample_cpu_readings_df,
        p95_threshold=20.0,
        avg_cpu_threshold=15.0,
        min_lifetime_hours=24.0,
        min_scheduling_hours=48.0,
        min_reserved_lifetime_hours=600.0,
        max_cpu_variance=100.0
    )

    p_counts = summary['proposal_counts']
    assert p_counts['RightsizingAgent'] == 2
    assert p_counts['PricingAgent'] == 3
    assert p_counts['SchedulingAgent'] == 2
    assert p_counts['total_proposals'] == 7

    assert summary['unique_vms_with_proposals'] == 3

    c_metrics = summary['conflict_metrics']
    assert c_metrics['conflict_vm_count'] == 2
    assert c_metrics['conflict_counts_by_type']['order_dependent'] == 2
    assert c_metrics['conflict_counts_by_type']['mutually_exclusive'] == 2
    assert c_metrics['conflict_counts_by_type']['compatible'] == 2
    assert summary['total_predicted_monthly_savings'] > 0
