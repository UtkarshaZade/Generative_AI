"""
Arbitration Engine Module.
Multi-round constraint resolution, mutual-exclusivity filtering, compounding price evaluation, scoring, and escalation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import itertools
from src.agents.base import Proposal
from src.agents.rightsizing_agent import RightsizingAgent, get_instance_pricing
from src.arbitration.config import RISK_PENALTY_BY_CATEGORY, ESCALATION_MARGIN_PCT, MAX_PROJECTED_P95_CPU
from src.arbitration.constraints import VMConstraint, check_hard_constraint


@dataclass
class ArbitrationResult:
    """
    Structured outcome of Arbitration negotiation for a single VM.
    
    Attributes:
        vm_id: Target VM identifier.
        status: Decision status ('accepted', 'do_nothing', or 'escalate').
        chosen_plan: Dictionary describing winning action set, compounded savings, cost, and score.
        runner_up_plan: Optional dictionary describing runner-up action set.
        rejected_proposals: List of proposal rejections with rule name and round.
        rounds_used: Number of constraint resolution rounds executed (1 or 2).
    """
    vm_id: str
    status: str
    chosen_plan: Dict[str, Any]
    runner_up_plan: Optional[Dict[str, Any]]
    rejected_proposals: List[Dict[str, Any]]
    rounds_used: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vm_id": self.vm_id,
            "status": self.status,
            "chosen_plan": self.chosen_plan,
            "runner_up_plan": self.runner_up_plan,
            "rejected_proposals": self.rejected_proposals,
            "rounds_used": self.rounds_used
        }


def calculate_compounded_plan_cost(
    subset: List[Proposal],
    curr_c: str,
    curr_m: str,
    baseline_od_cost: float
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Price a proposal subset in exact sequential compounding order:
      1. Rightsizing first (establishes target SKU)
      2. Pricing second (applies Reserved/Spot rate on target SKU)
      3. Scheduling third (applies pause hours reduction to resulting rate)

    Returns:
        (final_monthly_cost, compounded_monthly_savings, execution_details)
    """
    # Step 1: Rightsizing
    rs_proposal = next((p for p in subset if p.action_type == 'DOWNSIZE_SKU'), None)
    if rs_proposal:
        tgt_c = str(rs_proposal.proposed_configuration['vCPU_bucket'])
        tgt_m = str(rs_proposal.proposed_configuration['RAM_bucket_GB'])
    else:
        tgt_c, tgt_m = curr_c, curr_m

    sku_pricing = get_instance_pricing(tgt_c, tgt_m)
    od_hourly = sku_pricing['on_demand_hourly']

    # Step 2: Pricing Model
    pricing_proposal = next((p for p in subset if p.action_type in ['SWITCH_TO_SPOT', 'SWITCH_TO_RESERVED_1YR']), None)
    if pricing_proposal:
        if pricing_proposal.action_type == 'SWITCH_TO_SPOT':
            effective_hourly = sku_pricing['spot_hourly']
            pricing_model = 'Spot'
        else:
            effective_hourly = sku_pricing['reserved_1yr_hourly']
            pricing_model = '1-Year Reserved'
    else:
        effective_hourly = od_hourly
        pricing_model = 'On-Demand'

    # Step 3: Scheduling
    sched_proposal = next((p for p in subset if p.action_type == 'SCHEDULE_OFF_PEAK_PAUSE'), None)
    if sched_proposal:
        pause_h_daily = float(sched_proposal.proposed_configuration.get('pause_hours_per_day', 8.0))
        operating_hours = (24.0 - pause_h_daily) * (730.0 / 24.0)
        pause_hours_monthly = pause_h_daily * (730.0 / 24.0)
    else:
        operating_hours = 730.0
        pause_hours_monthly = 0.0

    final_monthly_cost = round(effective_hourly * operating_hours, 2)
    compounded_savings = round(baseline_od_cost - final_monthly_cost, 2)

    details = {
        "target_vCPU": tgt_c,
        "target_RAM": tgt_m,
        "matched_family": sku_pricing['matched_family'],
        "effective_hourly_rate": effective_hourly,
        "pricing_model": pricing_model,
        "operating_hours_per_month": round(operating_hours, 1),
        "pause_hours_per_month": round(pause_hours_monthly, 1)
    }

    return final_monthly_cost, max(0.0, compounded_savings), details


def calculate_subset_score(
    subset: List[Proposal],
    compounded_savings: float,
    vmcategory: str
) -> Tuple[float, float, float]:
    """
    Calculate arbitration score for a proposal subset.
    Score = Savings * mean_confidence - Risk_Penalty

    Returns:
        (score, mean_confidence, risk_penalty)
    """
    if not subset:
        return 0.0, 1.0, 0.0

    mean_conf = sum(p.confidence for p in subset) / len(subset)

    base_penalty = RISK_PENALTY_BY_CATEGORY.get(vmcategory, 20.0)

    # Determine risk multiplier based on action types in subset
    actions = {p.action_type for p in subset}
    if 'SWITCH_TO_SPOT' in actions:
        multiplier = 1.0
    elif 'SCHEDULE_OFF_PEAK_PAUSE' in actions:
        multiplier = 0.8
    elif 'SWITCH_TO_RESERVED_1YR' in actions:
        multiplier = 0.4
    else:  # Downsize only
        multiplier = 0.2

    risk_penalty = round(base_penalty * multiplier, 2)
    score = round((compounded_savings * mean_conf) - risk_penalty, 2)

    return score, mean_conf, risk_penalty


def is_valid_subset(subset: List[Proposal]) -> bool:
    """Check if subset contains any mutually_exclusive proposal pairs."""
    actions = {p.action_type for p in subset}
    if 'SCHEDULE_OFF_PEAK_PAUSE' in actions and ('SWITCH_TO_RESERVED_1YR' in actions or 'SWITCH_TO_SPOT' in actions):
        return False
    return True


class ArbitrationEngine:
    """
    Arbitration Engine for resolving multi-agent proposal conflicts.
    """

    def __init__(self, rightsizing_agent: Optional[RightsizingAgent] = None):
        self.rightsizing_agent = rightsizing_agent

    def arbitrate_vm(
        self,
        vm_id: str,
        proposals: List[Proposal],
        constraint: VMConstraint
    ) -> ArbitrationResult:
        """
        Execute 2-round arbitration pipeline for a single VM.
        """
        rejected_log: List[Dict[str, Any]] = []
        surviving_proposals: List[Proposal] = []
        rounds_used = 1

        # baseline On-Demand monthly cost from first proposal or rate lookup
        if proposals:
            curr_c = str(proposals[0].current_configuration.get('vCPU_bucket', constraint.current_vCPU))
            curr_m = str(proposals[0].current_configuration.get('RAM_bucket_GB', constraint.current_RAM))
            curr_pricing = get_instance_pricing(curr_c, curr_m)
            baseline_od_cost = round(curr_pricing['on_demand_hourly'] * 730.0, 2)
        else:
            curr_c = str(constraint.current_vCPU)
            curr_m = str(constraint.current_RAM)
            curr_pricing = get_instance_pricing(curr_c, curr_m)
            baseline_od_cost = round(curr_pricing['on_demand_hourly'] * 730.0, 2)

        # -------------------------------------------------------------
        # ROUND 1: Hard Constraint Validation
        # -------------------------------------------------------------
        p95_rejected_rs_proposal: Optional[Proposal] = None

        for p in proposals:
            valid, reason = check_hard_constraint(p, constraint)
            if valid:
                surviving_proposals.append(p)
            else:
                rejected_log.append({
                    "proposal_agent": p.agent_name,
                    "action_type": p.action_type,
                    "rule": reason,
                    "round": 1
                })
                if p.action_type == 'DOWNSIZE_SKU' and 'RULE_PROJECTED_P95_EXCEEDED' in str(reason):
                    p95_rejected_rs_proposal = p

        # -------------------------------------------------------------
        # ROUND 2: Adaptive Rightsizing (if P95 exceeded in Round 1)
        # -------------------------------------------------------------
        if p95_rejected_rs_proposal and self.rightsizing_agent:
            rounds_used = 2
            p95_cpu = float(p95_rejected_rs_proposal.current_configuration.get('p95_max_cpu_pct', constraint.p95maxcpu))
            # Retrieve all RAM-safe downsize candidates
            candidates = self.rightsizing_agent.get_candidate_skus(curr_c, curr_m, p95_cpu, ignore_p95_limit=True)

            # Evaluate less aggressive candidates that pass projected P95 <= 70% constraint
            for cand_c, cand_m, cand_sav, cand_proj_p95 in candidates:
                cand_pricing = get_instance_pricing(cand_c, cand_m)
                cand_monthly = round(cand_pricing['on_demand_hourly'] * 730.0, 2)
                
                adaptive_p = Proposal(
                    vm_id=vm_id,
                    agent_name=p95_rejected_rs_proposal.agent_name,
                    action_type="DOWNSIZE_SKU",
                    current_configuration=p95_rejected_rs_proposal.current_configuration,
                    proposed_configuration={
                        "vCPU_bucket": cand_c,
                        "RAM_bucket_GB": cand_m,
                        "matched_family": cand_pricing['matched_family'],
                        "on_demand_hourly_rate": cand_pricing['on_demand_hourly'],
                        "on_demand_monthly_cost": cand_monthly,
                        "projected_p95_cpu_pct": cand_proj_p95
                    },
                    predicted_monthly_savings=cand_sav,
                    confidence=p95_rejected_rs_proposal.confidence,
                    rationale=f"Round 2 Adaptive Rightsizing: downsize to less aggressive SKU {cand_c} vCPU / {cand_m} GB RAM (Projected P95: {cand_proj_p95:.1f}%)."
                )

                v_ok, v_reason = check_hard_constraint(adaptive_p, constraint)
                if v_ok:
                    surviving_proposals.append(adaptive_p)
                    break
                else:
                    rejected_log.append({
                        "proposal_agent": adaptive_p.agent_name,
                        "action_type": adaptive_p.action_type,
                        "rule": v_reason,
                        "round": 2
                    })

        # -------------------------------------------------------------
        # SUBSET ENUMERATION & MUTUAL EXCLUSIVITY FILTERING
        # -------------------------------------------------------------
        candidate_subsets: List[List[Proposal]] = []
        for r in range(len(surviving_proposals) + 1):
            for combo in itertools.combinations(surviving_proposals, r):
                sub_list = list(combo)
                if is_valid_subset(sub_list):
                    candidate_subsets.append(sub_list)

        # -------------------------------------------------------------
        # PRICING, SAVINGS COMPOUNDING, & SCORING
        # -------------------------------------------------------------
        evaluated_plans = []

        for sub in candidate_subsets:
            final_cost, comp_savings, exec_details = calculate_compounded_plan_cost(
                sub, curr_c, curr_m, baseline_od_cost
            )
            score, mean_conf, risk_pen = calculate_subset_score(
                sub, comp_savings, constraint.vmcategory
            )

            plan_dict = {
                "actions": [p.action_type for p in sub],
                "proposals": [p.to_dict() for p in sub],
                "baseline_monthly_cost": baseline_od_cost,
                "final_monthly_cost": final_cost,
                "compounded_monthly_savings": comp_savings,
                "mean_confidence": round(mean_conf, 3),
                "risk_penalty": risk_pen,
                "score": score,
                "execution_details": exec_details
            }
            evaluated_plans.append(plan_dict)

        # Sort plans by score descending
        evaluated_plans.sort(key=lambda x: x['score'], reverse=True)

        winning_plan = evaluated_plans[0]
        runner_up = evaluated_plans[1] if len(evaluated_plans) > 1 else None

        # -------------------------------------------------------------
        # ESCALATION CHECK
        # -------------------------------------------------------------
        status = "accepted" if winning_plan["actions"] else "do_nothing"

        if runner_up and winning_plan["score"] > 0 and runner_up["score"] > 0:
            score_diff = winning_plan["score"] - runner_up["score"]
            rel_margin = score_diff / winning_plan["score"]
            if rel_margin <= ESCALATION_MARGIN_PCT:
                status = "escalate"

        return ArbitrationResult(
            vm_id=vm_id,
            status=status,
            chosen_plan=winning_plan,
            runner_up_plan=runner_up,
            rejected_proposals=rejected_log,
            rounds_used=rounds_used
        )
