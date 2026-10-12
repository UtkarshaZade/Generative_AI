"""
Reserved / Spot Pricing Agent Module.
Evaluates pricing model trade-offs (On-Demand vs 1-Year Reserved vs Spot) based on workload category, lifetime, and CPU variance.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from .base import BaseAgent, Proposal
from src.data.pricing import get_instance_pricing, MONTHLY_HOURS


class PricingAgent(BaseAgent):
    """
    Reserved / Spot Pricing Agent for negotiating cloud purchasing models.
    
    Rule:
      - Spot pricing: Proposed ONLY for 'Delay-insensitive' workloads (~75% rate discount).
      - 1-Year Reserved pricing: Proposed ONLY for VMs with lifetime >= min_reserved_lifetime_hours (default 600.0h)
        and low CPU variance (cpu_variance <= max_cpu_variance, default 100.0).
      - Otherwise: NO PROPOSAL.
    """
    
    def __init__(
        self,
        name: str = "PricingAgent",
        min_reserved_lifetime_hours: float = 600.0,
        max_cpu_variance: float = 100.0
    ):
        super().__init__(name=name)
        self.min_reserved_lifetime_hours = min_reserved_lifetime_hours
        self.max_cpu_variance = max_cpu_variance

    def _calculate_cpu_variance(self, row: pd.Series, ts_data: Optional[pd.DataFrame] = None) -> float:
        """Calculate CPU variance from time-series readings or aggregate metric spread proxy."""
        if ts_data is not None and len(ts_data) > 1 and 'avgcpu' in ts_data.columns:
            var_val = float(ts_data['avgcpu'].var())
            if not np.isnan(var_val):
                return var_val

        # Fallback aggregate proxy: ((maxcpu - avgcpu) / 2.0)^2
        avg_cpu = float(row.get('avgcpu', 0.0))
        max_cpu = float(row.get('maxcpu', avg_cpu))
        return float(((max_cpu - avg_cpu) / 2.0) ** 2)

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

        # Build lookup for time-series telemetry if provided
        vm_ts_map: Dict[str, pd.DataFrame] = {}
        if cpu_readings_df is not None and 'vmid' in cpu_readings_df.columns:
            for vmid, group in cpu_readings_df.groupby('vmid'):
                vm_ts_map[str(vmid)] = group

        for _, row in eligible_df.iterrows():
            vm_id = str(row['vmid'])
            curr_c = str(row['vmcorecountbucket'])
            curr_m = str(row['vmmemorybucket'])
            category = str(row.get('vmcategory', 'Unknown')).strip()
            lifetime_h = float(row.get('lifetime', 0.0))

            ts_data = vm_ts_map.get(vm_id)
            cpu_var = self._calculate_cpu_variance(row, ts_data)

            pricing = get_instance_pricing(curr_c, curr_m)
            od_rate = pricing['on_demand_hourly']
            ri_rate = pricing['reserved_1yr_hourly']
            spot_rate = pricing['spot_hourly']

            od_monthly = round(od_rate * MONTHLY_HOURS, 2)
            ri_monthly = round(ri_rate * MONTHLY_HOURS, 2)
            spot_monthly = round(spot_rate * MONTHLY_HOURS, 2)

            # Decision Logic:
            # 1. Spot ONLY for Delay-insensitive workloads
            # 2. 1-Year Reserved ONLY for VMs with lifetime >= min_reserved_lifetime_hours (600h) and low CPU variance
            # 3. Otherwise: no proposal
            if category == 'Delay-insensitive':
                proposed_model = "Spot"
                proposed_rate = spot_rate
                proposed_monthly = spot_monthly
                action_type = "SWITCH_TO_SPOT"
                confidence = 0.90
                savings = round(od_monthly - spot_monthly, 2)
                rationale = (
                    f"Workload is classified as Delay-insensitive (fault-tolerant). "
                    f"Proposing Spot Instance pricing (${spot_rate:.4f}/hr vs ${od_rate:.4f}/hr On-Demand), "
                    f"saving ${savings:.2f}/mo (current: ${od_monthly:.2f}/mo, proposed: ${spot_monthly:.2f}/mo)."
                )

            elif lifetime_h >= self.min_reserved_lifetime_hours and cpu_var <= self.max_cpu_variance:
                proposed_model = "1-Year Reserved"
                proposed_rate = ri_rate
                proposed_monthly = ri_monthly
                action_type = "SWITCH_TO_RESERVED_1YR"
                confidence = 0.85
                savings = round(od_monthly - ri_monthly, 2)
                rationale = (
                    f"Workload lifetime is {lifetime_h:.1f}h (>= {self.min_reserved_lifetime_hours:.0f}h threshold) "
                    f"with low CPU variance ({cpu_var:.2f} <= {self.max_cpu_variance:.1f}). "
                    f"Proposing 1-Year Reserved Instance commitment (${ri_rate:.4f}/hr vs ${od_rate:.4f}/hr On-Demand), "
                    f"saving ${savings:.2f}/mo (current: ${od_monthly:.2f}/mo, proposed: ${ri_monthly:.2f}/mo)."
                )

            else:
                # Otherwise: NO PROPOSAL
                continue

            if savings <= 0:
                continue

            proposal = Proposal(
                vm_id=vm_id,
                agent_name=self.name,
                action_type=action_type,
                current_configuration={
                    "vCPU_bucket": curr_c,
                    "RAM_bucket_GB": curr_m,
                    "matched_family": pricing['matched_family'],
                    "pricing_model": "On-Demand",
                    "hourly_rate": od_rate,
                    "on_demand_monthly_cost": od_monthly,
                    "category": category,
                    "lifetime_hours": lifetime_h,
                    "cpu_variance": round(cpu_var, 2)
                },
                proposed_configuration={
                    "pricing_model": proposed_model,
                    "hourly_rate": proposed_rate,
                    "proposed_monthly_cost": proposed_monthly
                },
                predicted_monthly_savings=savings,
                confidence=confidence,
                rationale=rationale
            )

            proposals.append(proposal)

        return proposals

