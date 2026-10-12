"""
Scheduling & Autoscaling Agent Module.
Detects recurring low-utilization windows in 5-minute CPU telemetry and proposes off-peak pause schedules.
Rule:
  - Mean CPU < 15% alone DOES NOT trigger a proposal.
  - Requires a detected recurring low window (>= 6 h/day with mean CPU < 10%) computed from 5-minute CPU readings.
  - Requires at least 48 h of CPU readings for the VM.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from .base import BaseAgent, Proposal
from src.data.pricing import get_instance_pricing, MONTHLY_HOURS


class SchedulingAgent(BaseAgent):
    """
    Scheduling & Autoscaling Agent for dynamic capacity management.
    
    Rule:
      For VMs with scheduling_eligible is True (observation window >= 48.0h)
      and at least 48h of CPU telemetry, proposes an off-peak pause schedule IF AND ONLY IF:
        A recurring off-peak low-utilization window is detected (>= 6 hours/day with mean CPU < 10%).
      
      Mean CPU < 15% alone without a detected recurring low window DOES NOT trigger a proposal.
      Savings = hourly_rate * hours_paused_per_month
    """
    
    def __init__(
        self,
        name: str = "SchedulingAgent",
        min_recurring_low_hours: int = 6,
        low_cpu_threshold: float = 10.0,
        default_pause_hours: float = 8.0,
        data_source_name: str = "trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv",
        avg_cpu_threshold: Optional[float] = None
    ):
        super().__init__(name=name)
        self.min_recurring_low_hours = min_recurring_low_hours
        self.low_cpu_threshold = low_cpu_threshold
        self.default_pause_hours = default_pause_hours
        self.data_source_name = data_source_name
        self.avg_cpu_threshold = avg_cpu_threshold

    def generate_proposals(
        self,
        df: pd.DataFrame,
        cpu_readings_df: Optional[pd.DataFrame] = None
    ) -> List[Proposal]:
        proposals: List[Proposal] = []
        
        # Filter for scheduling eligible VMs (lifetime >= 48h)
        if 'scheduling_eligible' in df.columns:
            eligible_df = df[df['scheduling_eligible']].copy()
        else:
            eligible_df = df.copy()

        # Telemetry is strictly required for recurring low window detection
        if cpu_readings_df is None or 'vmid' not in cpu_readings_df.columns:
            print(f"[{self.name}] No CPU readings data provided. Generated 0 proposals.")
            return proposals

        # Build lookup for time-series telemetry
        vm_ts_map: Dict[str, pd.DataFrame] = {}
        for vmid, group in cpu_readings_df.groupby('vmid'):
            vm_ts_map[str(vmid)] = group

        window_proposals_count = 0

        for _, row in eligible_df.iterrows():
            vm_id = str(row['vmid'])
            curr_c = str(row['vmcorecountbucket'])
            curr_m = str(row['vmmemorybucket'])
            avg_cpu = float(row.get('avgcpu', 0.0))
            max_cpu = float(row.get('maxcpu', 0.0))
            lifetime_h = float(row.get('lifetime', 48.0))

            # Must have CPU readings in telemetry
            if vm_id not in vm_ts_map:
                continue

            ts_data = vm_ts_map[vm_id]

            # Require at least 48 hours of telemetry span or >= 576 readings
            min_ts = ts_data['timestamp'].min()
            max_ts = ts_data['timestamp'].max()
            telemetry_span_h = (max_ts - min_ts) / 3600.0 if 'timestamp' in ts_data.columns else 0.0

            if telemetry_span_h < 48.0 and len(ts_data) < 576:
                # Telemetry observation window less than 48h
                continue

            # Group readings by hour of day (timestamp in seconds -> hour = (ts // 3600) % 24)
            ts_copy = ts_data.copy()
            ts_copy['hour_of_day'] = (ts_copy['timestamp'] // 3600) % 24
            hourly_means = ts_copy.groupby('hour_of_day')['avgcpu'].mean()
            
            # Count recurring hours with mean CPU < 10%
            recurring_low_hours = int((hourly_means < self.low_cpu_threshold).sum())

            # Rule: MUST have a detected recurring low window (>= 6 h/day < 10% CPU)
            if recurring_low_hours < self.min_recurring_low_hours:
                continue

            # Calculate pause duration based on detected low window
            pause_hours = float(min(12.0, max(6.0, recurring_low_hours)))
            confidence = 0.90
            detected_pattern = (
                f"5-minute telemetry ({self.data_source_name}) detected {recurring_low_hours} recurring "
                f"off-peak low-utilization hours/day (<{self.low_cpu_threshold:.0f}% CPU) across {telemetry_span_h:.1f}h window"
            )

            pricing = get_instance_pricing(curr_c, curr_m)
            od_rate = pricing['on_demand_hourly']
            od_monthly = round(od_rate * MONTHLY_HOURS, 2)

            # Calculate savings = hourly rate x hours paused per month
            hours_paused_per_month = pause_hours * (MONTHLY_HOURS / 24.0)
            savings = round(od_rate * hours_paused_per_month, 2)

            if savings <= 0:
                continue

            pause_ratio = pause_hours / 24.0

            rationale = (
                f"Workload exhibits recurring off-peak low window ({detected_pattern}, avg CPU {avg_cpu:.2f}%). "
                f"Proposing automated {pause_hours:.0f}-hour daily pause schedule ({hours_paused_per_month:.1f} hours/mo off-peak pause), "
                f"reducing monthly operating cost from ${od_monthly:.2f}/mo to ${od_monthly - savings:.2f}/mo "
                f"(saving ${savings:.2f}/mo = ${od_rate:.4f}/hr x {hours_paused_per_month:.1f}h, ~{pause_ratio * 100:.1f}% cost reduction)."
            )

            proposal = Proposal(
                vm_id=vm_id,
                agent_name=self.name,
                action_type="SCHEDULE_OFF_PEAK_PAUSE",
                current_configuration={
                    "vCPU_bucket": curr_c,
                    "RAM_bucket_GB": curr_m,
                    "matched_family": pricing['matched_family'],
                    "hourly_rate": od_rate,
                    "operating_hours_per_day": 24.0,
                    "on_demand_monthly_cost": od_monthly,
                    "avg_cpu_pct": avg_cpu,
                    "max_cpu_pct": max_cpu,
                    "lifetime_hours": lifetime_h
                },
                proposed_configuration={
                    "schedule_action": f"Pause {pause_hours:.0f} hours daily during off-peak window",
                    "operating_hours_per_day": 24.0 - pause_hours,
                    "pause_hours_per_day": pause_hours,
                    "hours_paused_per_month": round(hours_paused_per_month, 2),
                    "proposed_monthly_cost": round(od_monthly - savings, 2)
                },
                predicted_monthly_savings=savings,
                confidence=confidence,
                rationale=rationale
            )

            proposals.append(proposal)
            window_proposals_count += 1

        print(
            f"[{self.name}] Data Source: '{self.data_source_name}' | "
            f"Proposals from Recurring Low Window Rule (>= 6h/day < 10% CPU with >= 48h readings): {window_proposals_count}"
        )

        return proposals


