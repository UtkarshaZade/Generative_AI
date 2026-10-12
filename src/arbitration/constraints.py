"""
Arbitration Hard Constraints Module.
Deterministic hard constraint validation (no LLM).
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from src.agents.base import Proposal
from src.arbitration.config import MAX_PROJECTED_P95_CPU


@dataclass
class VMConstraint:
    """
    Constraint specification object for a Virtual Machine.
    
    Attributes:
        vm_id: Anonymized VM identifier.
        vmcategory: Workload classification ('Interactive', 'Unknown', 'Delay-insensitive').
        existing_commitment: Existing cloud pricing contract ('none', '1yr_reserved', '3yr_reserved'). Defaults to 'none'.
        p95maxcpu: Historical 95th percentile max CPU utilization %.
        current_vCPU: Baseline vCPU count (numeric).
        current_RAM: Baseline RAM GB (numeric).
    """
    vm_id: str
    vmcategory: str = "Unknown"
    existing_commitment: str = "none"
    p95maxcpu: float = 0.0
    current_vCPU: float = 2.0
    current_RAM: float = 4.0


def check_hard_constraint(
    proposal: Proposal,
    constraint: VMConstraint,
    max_projected_p95: float = MAX_PROJECTED_P95_CPU
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate a single proposal against hard constraints.

    Returns:
        (is_valid: bool, rejection_reason: Optional[str])
    """
    action = proposal.action_type
    category = constraint.vmcategory.strip()
    commitment = constraint.existing_commitment.strip().lower()

    # Rule 1: Interactive VMs cannot take Spot or Scheduling pauses
    if category == 'Interactive':
        if action == 'SWITCH_TO_SPOT':
            return False, "RULE_INTERACTIVE_NO_SPOT: Interactive VMs cannot be converted to interruptible Spot instances."
        if action == 'SCHEDULE_OFF_PEAK_PAUSE':
            return False, "RULE_INTERACTIVE_NO_PAUSE: Interactive VMs cannot be subjected to automated off-peak pause schedules."

    # Rule 2: Projected P95 CPU after resize must be <= 70%
    if action == 'DOWNSIZE_SKU':
        proj_p95 = proposal.proposed_configuration.get('projected_p95_cpu_pct')
        if proj_p95 is None:
            # Calculate from proposal metadata if not explicitly set
            curr_c = float(proposal.current_configuration.get('vCPU_bucket', constraint.current_vCPU))
            tgt_c = float(proposal.proposed_configuration.get('vCPU_bucket', curr_c))
            p95_cpu = float(proposal.current_configuration.get('p95_max_cpu_pct', constraint.p95maxcpu))
            proj_p95 = p95_cpu * (curr_c / tgt_c) if tgt_c > 0 else 100.0

        if proj_p95 > max_projected_p95:
            return False, f"RULE_PROJECTED_P95_EXCEEDED: Projected P95 CPU post-resize ({proj_p95:.1f}%) exceeds safety threshold ({max_projected_p95:.1f}%)."

    # Rule 3: No pricing change on a VM with an existing commitment
    if commitment != 'none':
        if action in ['SWITCH_TO_SPOT', 'SWITCH_TO_RESERVED_1YR']:
            return False, f"RULE_EXISTING_COMMITMENT_LOCK: Cannot alter pricing model on VM with active '{commitment}' commitment."

    return True, None
