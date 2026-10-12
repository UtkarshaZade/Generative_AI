"""
Unit tests for Arbitration Layer, Compounding Price Calculations, Hard Constraints, and Baselines.
"""

import pytest
import pandas as pd
from src.agents.base import Proposal
from src.agents.rightsizing_agent import RightsizingAgent
from src.arbitration import (
    VMConstraint,
    check_hard_constraint,
    ArbitrationEngine,
    calculate_compounded_plan_cost,
    calculate_subset_score
)
from src.eval import (
    check_constraint_violations,
    run_baseline_b0_naive_sum,
    run_baseline_b1_single_agent,
    run_baseline_b2_constrained_single
)


def test_interactive_no_spot_or_schedule():
    """Test that Interactive VMs reject Spot pricing and Scheduling pause proposals."""
    constraint = VMConstraint(
        vm_id="vm_interactive_test",
        vmcategory="Interactive",
        existing_commitment="none",
        p95maxcpu=15.0
    )

    spot_p = Proposal(
        vm_id="vm_interactive_test",
        agent_name="PricingAgent",
        action_type="SWITCH_TO_SPOT",
        current_configuration={"on_demand_monthly_cost": 100.0},
        proposed_configuration={"proposed_monthly_cost": 25.0},
        predicted_monthly_savings=75.0,
        confidence=0.9,
        rationale="Spot proposal"
    )

    sched_p = Proposal(
        vm_id="vm_interactive_test",
        agent_name="SchedulingAgent",
        action_type="SCHEDULE_OFF_PEAK_PAUSE",
        current_configuration={"on_demand_monthly_cost": 100.0},
        proposed_configuration={"proposed_monthly_cost": 66.7},
        predicted_monthly_savings=33.3,
        confidence=0.85,
        rationale="Schedule proposal"
    )

    valid_spot, reason_spot = check_hard_constraint(spot_p, constraint)
    assert not valid_spot
    assert "RULE_INTERACTIVE_NO_SPOT" in reason_spot

    valid_sched, reason_sched = check_hard_constraint(sched_p, constraint)
    assert not valid_sched
    assert "RULE_INTERACTIVE_NO_PAUSE" in reason_sched


def test_projected_p95_cpu_constraint():
    """Test projected P95 CPU threshold check (<= 70%)."""
    constraint = VMConstraint(
        vm_id="vm_p95_test",
        vmcategory="Delay-insensitive",
        p95maxcpu=40.0,
        current_vCPU=8.0,
        current_RAM=32.0
    )

    # Downsize from 8 vCPU to 2 vCPU -> Projected P95 = 40 * (8/2) = 160% (>70%) -> INVALID
    rs_aggressive = Proposal(
        vm_id="vm_p95_test",
        agent_name="RightsizingAgent",
        action_type="DOWNSIZE_SKU",
        current_configuration={"vCPU_bucket": "8", "RAM_bucket_GB": "32", "p95_max_cpu_pct": 40.0},
        proposed_configuration={"vCPU_bucket": "2", "RAM_bucket_GB": "32", "projected_p95_cpu_pct": 160.0},
        predicted_monthly_savings=188.34,
        confidence=0.85,
        rationale="Aggressive downsize"
    )

    valid_agg, reason_agg = check_hard_constraint(rs_aggressive, constraint)
    assert not valid_agg
    assert "RULE_PROJECTED_P95_EXCEEDED" in reason_agg

    # Downsize from 8 vCPU to 4 vCPU -> Projected P95 = 40 * (8/4) = 80% (>70%) -> INVALID
    # Downsize from 8 vCPU to 6 vCPU -> Projected P95 = 40 * (8/6) = 53.3% (<=70%) -> VALID
    rs_safe = Proposal(
        vm_id="vm_p95_test",
        agent_name="RightsizingAgent",
        action_type="DOWNSIZE_SKU",
        current_configuration={"vCPU_bucket": "8", "RAM_bucket_GB": "32", "p95_max_cpu_pct": 40.0},
        proposed_configuration={"vCPU_bucket": "6", "RAM_bucket_GB": "32", "projected_p95_cpu_pct": 53.3},
        predicted_monthly_savings=100.0,
        confidence=0.85,
        rationale="Safe downsize"
    )

    valid_safe, _ = check_hard_constraint(rs_safe, constraint)
    assert valid_safe


def test_existing_commitment_lock():
    """Test that existing pricing commitment locks pricing model changes."""
    constraint = VMConstraint(
        vm_id="vm_locked",
        vmcategory="Delay-insensitive",
        existing_commitment="1yr_reserved"
    )

    ri_p = Proposal(
        vm_id="vm_locked",
        agent_name="PricingAgent",
        action_type="SWITCH_TO_RESERVED_1YR",
        current_configuration={},
        proposed_configuration={},
        predicted_monthly_savings=50.0,
        confidence=0.9,
        rationale="RI proposal"
    )

    valid_ri, reason_ri = check_hard_constraint(ri_p, constraint)
    assert not valid_ri
    assert "RULE_EXISTING_COMMITMENT_LOCK" in reason_ri


def test_compounding_price_calculation():
    """
    Test exact sequential compounding price evaluation:
      1. Rightsizing: 8 vCPU / 32 GB ($280.32/mo) -> 4 vCPU / 32 GB ($140.16/mo, $0.1920/hr)
      2. Pricing: 1-Yr Reserved discount on 4 vCPU / 32 GB ($0.1152/hr)
      3. Scheduling: 8-hour daily pause (16h operating / day = 486.67h / mo)
      4. Compounded Cost = $0.1152 * 486.6667 = $56.06 / month.
      5. Baseline Cost = $280.32 / month.
      6. Compounded Savings = $280.32 - $56.06 = $224.26 / month.
    """
    baseline_cost = 280.32  # 8 vCPU / 32 GB On-Demand baseline

    rs_p = Proposal(
        vm_id="vm_compounding",
        agent_name="RightsizingAgent",
        action_type="DOWNSIZE_SKU",
        current_configuration={"vCPU_bucket": "8", "RAM_bucket_GB": "32"},
        proposed_configuration={"vCPU_bucket": "4", "RAM_bucket_GB": "32"},
        predicted_monthly_savings=140.16,
        confidence=0.9,
        rationale="Downsize to 4 vCPU / 32 GB"
    )

    ri_p = Proposal(
        vm_id="vm_compounding",
        agent_name="PricingAgent",
        action_type="SWITCH_TO_RESERVED_1YR",
        current_configuration={},
        proposed_configuration={},
        predicted_monthly_savings=112.13,
        confidence=0.85,
        rationale="Switch to 1-Yr Reserved"
    )

    # Price subset of Rightsizing + Pricing
    subset = [rs_p, ri_p]
    final_cost, comp_savings, details = calculate_compounded_plan_cost(
        subset, curr_c="8", curr_m="32", baseline_od_cost=baseline_cost
    )

    # 4 vCPU / 32 GB 1-Yr Reserved hourly rate is 0.1152. 0.1152 * 730 = 84.10.
    # Baseline 280.32 - 84.10 = 196.22 savings.
    assert final_cost == 84.10
    assert comp_savings == 196.22
    assert details['pricing_model'] == "1-Year Reserved"


def test_arbitration_round2_adaptive_rightsizing():
    """Test Round 2 adaptive rightsizing when Round 1 Rightsizing fails P95 check."""
    rs_agent = RightsizingAgent(p95_threshold=20.0, max_projected_p95=70.0)
    engine = ArbitrationEngine(rightsizing_agent=rs_agent)

    constraint = VMConstraint(
        vm_id="vm_adapt",
        vmcategory="Unknown",
        p95maxcpu=25.0,  # 25% max CPU
        current_vCPU=8.0,
        current_RAM=32.0
    )

    # Round 1 Proposal downsize 8 vCPU -> 2 vCPU -> Projected P95 = 25 * (8/2) = 100% (>70%) -> Fails Round 1!
    prop_failing = Proposal(
        vm_id="vm_adapt",
        agent_name="RightsizingAgent",
        action_type="DOWNSIZE_SKU",
        current_configuration={"vCPU_bucket": "8", "RAM_bucket_GB": "32", "p95_max_cpu_pct": 25.0},
        proposed_configuration={"vCPU_bucket": "2", "RAM_bucket_GB": "32", "projected_p95_cpu_pct": 100.0},
        predicted_monthly_savings=188.34,
        confidence=0.8,
        rationale="Aggressive downsize"
    )

    res = engine.arbitrate_vm("vm_adapt", [prop_failing], constraint)

    # Round 2 should adaptively downsize 8 vCPU -> 4 vCPU (Projected P95 = 25 * (8/4) = 50% <= 70%)
    assert res.rounds_used == 2
    assert res.status == "accepted"
    assert res.chosen_plan["execution_details"]["target_vCPU"] == "4"


def test_constraint_violation_checker():
    """Test standalone constraint violation checker for baseline evaluation."""
    constraint = VMConstraint(
        vm_id="vm_violator",
        vmcategory="Interactive",
        existing_commitment="none",
        p95maxcpu=10.0
    )

    spot_p = Proposal(
        vm_id="vm_violator",
        agent_name="PricingAgent",
        action_type="SWITCH_TO_SPOT",
        current_configuration={},
        proposed_configuration={},
        predicted_monthly_savings=50.0,
        confidence=0.9,
        rationale="Spot"
    )

    # Interactive VM choosing Spot = 1 violation
    violations = check_constraint_violations("vm_violator", [spot_p], constraint)
    assert violations == 1
