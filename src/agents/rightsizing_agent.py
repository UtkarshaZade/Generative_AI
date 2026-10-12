"""
Rightsizing Agent Module.
Flags overprovisioned VMs with p95maxcpu below threshold and proposes downsized SKU configurations.
Memory Safety Rule: Memory (RAM) must NEVER be reduced (new_RAM >= current_RAM).
Performance Safety Rule: Projected P95 CPU (p95maxcpu * current_vCPU / new_vCPU) <= 70.0%.
"""

import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from .base import BaseAgent, Proposal
from src.data.pricing import get_instance_pricing, MONTHLY_HOURS, RATE_CARD


def parse_bucket_num(val: str) -> float:
    """Parse bucket string to numeric float approximation."""
    s = str(val).strip()
    if s == '>24':
        return 30.0
    if s == '>64':
        return 70.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def get_valid_downsize_candidates(
    curr_c: str,
    curr_m: str,
    p95_cpu: float,
    max_projected_p95: float = 70.0,
    ignore_p95_limit: bool = False
) -> List[Tuple[str, str, float, float]]:
    """
    Find all candidate SKUs in RATE_CARD matching:
      1. new_vCPU < curr_vCPU
      2. new_RAM >= curr_RAM (Memory must NEVER be reduced)
      3. projected_p95_cpu = p95_cpu * (curr_vCPU / new_vCPU) <= max_projected_p95 (unless ignore_p95_limit=True)
      4. rate-card On-Demand monthly cost < current On-Demand monthly cost

    Returns list of tuples: (cand_c, cand_m, savings, projected_p95)
    Sorted by savings descending (most aggressive downsize first).
    """
    curr_c_num = parse_bucket_num(curr_c)
    curr_m_num = parse_bucket_num(curr_m)
    
    if curr_c_num <= 0 or curr_m_num <= 0:
        return []

    curr_pricing = get_instance_pricing(curr_c, curr_m)
    curr_monthly = round(curr_pricing['on_demand_hourly'] * MONTHLY_HOURS, 2)

    candidates = []
    for (cand_c, cand_m) in RATE_CARD.keys():
        cand_c_num = parse_bucket_num(cand_c)
        cand_m_num = parse_bucket_num(cand_m)

        # 1. new_vCPU < curr_vCPU
        if cand_c_num >= curr_c_num:
            continue

        # 2. new_RAM >= curr_RAM
        if cand_m_num < curr_m_num:
            continue

        # 3. Projected P95 CPU check
        proj_p95 = round(p95_cpu * (curr_c_num / cand_c_num), 2)
        if not ignore_p95_limit and proj_p95 > max_projected_p95:
            continue

        # 4. Savings check
        cand_pricing = get_instance_pricing(cand_c, cand_m)
        cand_monthly = round(cand_pricing['on_demand_hourly'] * MONTHLY_HOURS, 2)
        savings = round(curr_monthly - cand_monthly, 2)

        if savings > 0 and cand_pricing['on_demand_hourly'] < curr_pricing['on_demand_hourly']:
            candidates.append((cand_c, cand_m, savings, proj_p95))

    # Sort by savings descending (highest savings / most aggressive first)
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


class RightsizingAgent(BaseAgent):
    """
    Rightsizing Agent for identifying overprovisioned cloud capacity.
    
    Rules:
      1. Memory (RAM) must NEVER be reduced (new_RAM >= current_RAM).
      2. Projected P95 CPU after resize must be <= max_projected_p95 (default 70.0%).
      3. Skips VMs where no candidate SKU satisfies these constraints.
    """
    
    def __init__(
        self,
        name: str = "RightsizingAgent",
        p95_threshold: float = 20.0,
        max_projected_p95: float = 70.0
    ):
        super().__init__(name=name)
        self.p95_threshold = p95_threshold
        self.max_projected_p95 = max_projected_p95

    def get_candidate_skus(
        self,
        curr_c: str,
        curr_m: str,
        p95_cpu: float,
        ignore_p95_limit: bool = False
    ) -> List[Tuple[str, str, float, float]]:
        """Public helper to retrieve candidate downsized SKUs sorted by aggressiveness."""
        return get_valid_downsize_candidates(
            curr_c, curr_m, p95_cpu, max_projected_p95=self.max_projected_p95, ignore_p95_limit=ignore_p95_limit
        )

    def generate_proposals(
        self,
        df: pd.DataFrame,
        cpu_readings_df: Optional[pd.DataFrame] = None
    ) -> List[Proposal]:
        proposals: List[Proposal] = []
        
        # Filter for eligible VMs
        if 'eligible_rightsizing_pricing' in df.columns:
            eligible_df = df[df['eligible_rightsizing_pricing']].copy()
        else:
            eligible_df = df.copy()

        if 'p95maxcpu' not in eligible_df.columns:
            return proposals

        # Flag VMs below P95 max CPU threshold
        overprovisioned_df = eligible_df[eligible_df['p95maxcpu'] < self.p95_threshold].copy()

        for _, row in overprovisioned_df.iterrows():
            vm_id = str(row['vmid'])
            curr_c = str(row['vmcorecountbucket'])
            curr_m = str(row['vmmemorybucket'])
            p95_cpu = float(row['p95maxcpu'])
            avg_cpu = float(row['avgcpu']) if 'avgcpu' in row else 0.0

            # Optimistic Proposing (Part 1a): Retrieve highest-savings RAM-safe SKU WITHOUT checking projected P95 limit
            # Memory safety rule (new_RAM >= current_RAM) is strictly enforced.
            # The projected P95 CPU check lives ONLY in Arbitration (Round 1 reject if > 70%, Round 2 step-down).
            candidates = self.get_candidate_skus(curr_c, curr_m, p95_cpu, ignore_p95_limit=True)
            if not candidates:
                # No RAM-safe candidate SKU available (RAM must NEVER be reduced)
                continue

            # Select highest-savings RAM-safe candidate SKU
            tgt_c, tgt_m, savings, proj_p95 = candidates[0]

            curr_pricing = get_instance_pricing(curr_c, curr_m)
            tgt_pricing = get_instance_pricing(tgt_c, tgt_m)

            curr_monthly = round(curr_pricing['on_demand_hourly'] * MONTHLY_HOURS, 2)
            tgt_monthly = round(tgt_pricing['on_demand_hourly'] * MONTHLY_HOURS, 2)

            # Confidence score inversely proportional to P95 max CPU
            confidence = round(min(0.95, max(0.70, 1.0 - (p95_cpu / 50.0))), 3)

            rationale = (
                f"P95 max CPU is {p95_cpu:.2f}% (below {self.p95_threshold:.1f}% threshold). "
                f"Optimistically proposing downsize from {curr_c} vCPU / {curr_m} GB RAM ({curr_pricing['matched_family']}) "
                f"to {tgt_c} vCPU / {tgt_m} GB RAM ({tgt_pricing['matched_family']}), preserving RAM allocation. "
                f"Projected P95 CPU: {proj_p95:.1f}% (subject to Arbitration headroom validation). "
                f"Reduces On-Demand cost from ${curr_monthly:.2f}/mo to ${tgt_monthly:.2f}/mo (saving ${savings:.2f}/mo)."
            )

            proposal = Proposal(
                vm_id=vm_id,
                agent_name=self.name,
                action_type="DOWNSIZE_SKU",
                current_configuration={
                    "vCPU_bucket": curr_c,
                    "RAM_bucket_GB": curr_m,
                    "matched_family": curr_pricing['matched_family'],
                    "on_demand_hourly_rate": curr_pricing['on_demand_hourly'],
                    "on_demand_monthly_cost": curr_monthly,
                    "p95_max_cpu_pct": p95_cpu,
                    "avg_cpu_pct": avg_cpu
                },
                proposed_configuration={
                    "vCPU_bucket": tgt_c,
                    "RAM_bucket_GB": tgt_m,
                    "matched_family": tgt_pricing['matched_family'],
                    "on_demand_hourly_rate": tgt_pricing['on_demand_hourly'],
                    "on_demand_monthly_cost": tgt_monthly,
                    "projected_p95_cpu_pct": proj_p95
                },
                predicted_monthly_savings=savings,
                confidence=confidence,
                rationale=rationale
            )

            proposals.append(proposal)

        return proposals

