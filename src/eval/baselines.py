"""
Evaluation Baselines and Fair Constraint Violation Checker Module.
Implements:
  - B0 (Naive Sum Baseline): Additive sum of all proposals per VM, no constraints.
  - B1 (Single-Agent Baseline): Highest-savings single proposal per VM, no constraints.
  - B2 (Constrained Single-Agent Baseline): Highest-savings single proposal per VM passing hard constraints.
  - Constraint Violation Checker: Evaluates constraint violations fairly across all methods.
"""

from typing import List, Dict, Any, Tuple, Optional
from src.agents.base import Proposal
from src.arbitration.constraints import VMConstraint, check_hard_constraint
from src.arbitration.config import MAX_PROJECTED_P95_CPU


def check_constraint_violations(
    vm_id: str,
    chosen_proposals: List[Proposal],
    constraint: VMConstraint
) -> int:
    """
    Count constraint violations for any method's chosen actions using exact hard constraint rules.

    Violations Checked:
      1. Interactive VM taking Spot pricing.
      2. Interactive VM taking Scheduling pause.
      3. Downsizing resulting in projected P95 CPU > 70%.
      4. Pricing change on VM with active commitment.
      5. Chosen actions containing a mutually_exclusive pair (Reserved+Schedule, Spot+Schedule).

    Returns:
        total_violations (int >= 0)
    """
    violations = 0
    actions = [p.action_type for p in chosen_proposals]

    # Individual hard constraint checks
    for p in chosen_proposals:
        valid, reason = check_hard_constraint(p, constraint)
        if not valid:
            violations += 1

    # Mutual exclusivity check
    if 'SCHEDULE_OFF_PEAK_PAUSE' in actions and ('SWITCH_TO_RESERVED_1YR' in actions or 'SWITCH_TO_SPOT' in actions):
        violations += 1

    return violations


def run_baseline_b0_naive_sum(
    vm_id: str,
    proposals: List[Proposal],
    constraint: VMConstraint
) -> Dict[str, Any]:
    """
    B0 (Naive Sum Baseline):
    Takes every proposal generated for the VM and sums predicted savings without constraint checks or combining.
    """
    if not proposals:
        return {
            "vm_id": vm_id,
            "method": "B0_NaiveSum",
            "acted_on": False,
            "chosen_actions": [],
            "total_monthly_savings": 0.0,
            "constraint_violations": 0,
            "escalations": 0
        }

    total_savings = sum(p.predicted_monthly_savings for p in proposals)
    violations = check_constraint_violations(vm_id, proposals, constraint)

    return {
        "vm_id": vm_id,
        "method": "B0_NaiveSum",
        "acted_on": True,
        "chosen_actions": [p.action_type for p in proposals],
        "total_monthly_savings": round(total_savings, 2),
        "constraint_violations": violations,
        "escalations": 0
    }


def run_baseline_b1_single_agent(
    vm_id: str,
    proposals: List[Proposal],
    constraint: VMConstraint
) -> Dict[str, Any]:
    """
    B1 (Single-Agent Baseline):
    Picks the single proposal with the highest predicted savings for the VM, ignoring hard constraints.
    """
    if not proposals:
        return {
            "vm_id": vm_id,
            "method": "B1_SingleAgent",
            "acted_on": False,
            "chosen_actions": [],
            "total_monthly_savings": 0.0,
            "constraint_violations": 0,
            "escalations": 0
        }

    best_prop = max(proposals, key=lambda p: p.predicted_monthly_savings)
    chosen = [best_prop]
    violations = check_constraint_violations(vm_id, chosen, constraint)

    return {
        "vm_id": vm_id,
        "method": "B1_SingleAgent",
        "acted_on": True,
        "chosen_actions": [best_prop.action_type],
        "total_monthly_savings": round(best_prop.predicted_monthly_savings, 2),
        "constraint_violations": violations,
        "escalations": 0
    }


def run_baseline_b2_constrained_single(
    vm_id: str,
    proposals: List[Proposal],
    constraint: VMConstraint
) -> Dict[str, Any]:
    """
    B2 (Constrained Single-Agent Baseline):
    Picks the single proposal with the highest predicted savings for the VM that passes hard constraints.
    No negotiation, combination, or compounding.
    """
    valid_props = []
    for p in proposals:
        valid, _ = check_hard_constraint(p, constraint)
        if valid:
            valid_props.append(p)

    if not valid_props:
        return {
            "vm_id": vm_id,
            "method": "B2_ConstrainedSingle",
            "acted_on": False,
            "chosen_actions": [],
            "total_monthly_savings": 0.0,
            "constraint_violations": 0,
            "escalations": 0
        }

    best_prop = max(valid_props, key=lambda p: p.predicted_monthly_savings)
    chosen = [best_prop]
    violations = check_constraint_violations(vm_id, chosen, constraint)

    return {
        "vm_id": vm_id,
        "method": "B2_ConstrainedSingle",
        "acted_on": True,
        "chosen_actions": [best_prop.action_type],
        "total_monthly_savings": round(best_prop.predicted_monthly_savings, 2),
        "constraint_violations": violations,
        "escalations": 0
    }
