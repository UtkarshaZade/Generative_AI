"""
Multi-Agent Orchestration & Conflict Analysis Runner Module.
Executes Rightsizing, Reserved/Spot Pricing, and Scheduling agents, aggregating proposal counts and conflict metrics.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from .base import Proposal
from .rightsizing_agent import RightsizingAgent
from .pricing_agent import PricingAgent
from .scheduling_agent import SchedulingAgent
from src.data.preprocessing import apply_lifetime_eligibility_filter
from src.data.pricing import add_pricing_columns


import random
import pandas as pd
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from .base import Proposal
from .rightsizing_agent import RightsizingAgent
from .pricing_agent import PricingAgent
from .scheduling_agent import SchedulingAgent
from src.data.preprocessing import apply_lifetime_eligibility_filter
from src.data.pricing import add_pricing_columns


def run_all_agents(
    df: pd.DataFrame,
    cpu_readings_df: Optional[pd.DataFrame] = None,
    p95_threshold: float = 20.0,
    avg_cpu_threshold: float = 15.0,
    min_lifetime_hours: float = 24.0,
    min_scheduling_hours: float = 48.0,
    min_reserved_lifetime_hours: float = 600.0,
    max_cpu_variance: float = 100.0
) -> Tuple[Dict[str, List[Proposal]], Dict[str, Any]]:
    """
    Execute all optimization agents on the dataset and analyze multi-agent proposal overlaps & conflicts.

    Returns:
        (agent_proposals_dict, summary_report_dict)
    """
    # Ensure dataset is preprocessed, eligibility-flagged, and priced
    df_prep = df.copy()
    if 'on_demand_monthly_cost' not in df_prep.columns:
        df_prep = add_pricing_columns(df_prep)

    df_prep, _ = apply_lifetime_eligibility_filter(
        df_prep,
        min_lifetime_hours=min_lifetime_hours,
        min_scheduling_hours=min_scheduling_hours
    )

    rightsizing_agent = RightsizingAgent(p95_threshold=p95_threshold)
    pricing_agent = PricingAgent(
        min_reserved_lifetime_hours=min_reserved_lifetime_hours,
        max_cpu_variance=max_cpu_variance
    )
    scheduling_agent = SchedulingAgent(avg_cpu_threshold=avg_cpu_threshold)

    rightsizing_proposals = rightsizing_agent.generate_proposals(df_prep, cpu_readings_df)
    pricing_proposals = pricing_agent.generate_proposals(df_prep, cpu_readings_df)
    scheduling_proposals = scheduling_agent.generate_proposals(df_prep, cpu_readings_df)

    agent_proposals = {
        "RightsizingAgent": rightsizing_proposals,
        "PricingAgent": pricing_proposals,
        "SchedulingAgent": scheduling_proposals
    }

    # Map VM ID to set of agent names proposing actions
    vm_agent_map: Dict[str, Set[str]] = defaultdict(set)
    vm_proposal_map: Dict[str, List[Proposal]] = defaultdict(list)

    all_proposals: List[Proposal] = []
    for agent_name, prop_list in agent_proposals.items():
        for prop in prop_list:
            vm_agent_map[prop.vm_id].add(agent_name)
            vm_proposal_map[prop.vm_id].append(prop)
            all_proposals.append(prop)

    # Calculate Proposal Counts
    rightsizing_count = len(rightsizing_proposals)
    pricing_count = len(pricing_proposals)
    scheduling_count = len(scheduling_proposals)
    total_proposals = len(all_proposals)
    unique_vms_with_proposals = len(vm_agent_map)

    # Multi-Agent Overlap Classification Rules:
    # ----------------------------------------
    # 1. order_dependent: Rightsizing + Pricing
    #    Rule: Rightsizing changes the baseline SKU rate card, which must be evaluated before applying Reserved/Spot pricing discounts.
    # 2. mutually_exclusive: Reserved + Scheduling OR Spot + Scheduling
    #    Rule: 1-Year Reserved commitments pre-pay for 24/7 capacity (making off-peak pause schedules financially redundant);
    #          Spot instances risk eviction during scheduled restarts.
    # 3. compatible: Rightsizing + Scheduling
    #    Rule: Rightsizing (reduces hourly rate) and Scheduling (reduces operating hours) can be combined; total savings compound multiplicatively.

    conflict_vm_count = 0
    order_dependent_count = 0
    mutually_exclusive_count = 0
    compatible_count = 0

    rightsizing_and_pricing = 0
    rightsizing_and_scheduling = 0
    pricing_and_scheduling = 0
    all_three_agents = 0

    for vm_id, agents in vm_agent_map.items():
        if len(agents) > 1:
            conflict_vm_count += 1
            
            has_r = "RightsizingAgent" in agents
            has_p = "PricingAgent" in agents
            has_s = "SchedulingAgent" in agents

            if has_r and has_p:
                rightsizing_and_pricing += 1
                order_dependent_count += 1

            if has_p and has_s:
                pricing_and_scheduling += 1
                mutually_exclusive_count += 1

            if has_r and has_s:
                rightsizing_and_scheduling += 1
                compatible_count += 1

            if has_r and has_p and has_s:
                all_three_agents += 1

    total_predicted_savings = sum(p.predicted_monthly_savings for p in all_proposals)

    summary = {
        "total_sample_vms": len(df_prep),
        "proposal_counts": {
            "RightsizingAgent": rightsizing_count,
            "PricingAgent": pricing_count,
            "SchedulingAgent": scheduling_count,
            "total_proposals": total_proposals
        },
        "unique_vms_with_proposals": unique_vms_with_proposals,
        "proposal_coverage_pct": round((unique_vms_with_proposals / len(df_prep) * 100.0), 2) if len(df_prep) > 0 else 0.0,
        "conflict_metrics": {
            "conflict_vm_count": conflict_vm_count,
            "conflict_vm_pct_of_proposal_vms": round((conflict_vm_count / unique_vms_with_proposals * 100.0), 2) if unique_vms_with_proposals > 0 else 0.0,
            "conflict_counts_by_type": {
                "order_dependent": order_dependent_count,
                "mutually_exclusive": mutually_exclusive_count,
                "compatible": compatible_count
            },
            "conflict_breakdown": {
                "rightsizing_and_pricing": rightsizing_and_pricing,
                "rightsizing_and_scheduling": rightsizing_and_scheduling,
                "pricing_and_scheduling": pricing_and_scheduling,
                "all_three_agents": all_three_agents
            }
        },
        "total_predicted_monthly_savings": round(total_predicted_savings, 2)
    }

    return agent_proposals, summary


def print_agent_analysis_summary(
    summary: Dict[str, Any],
    agent_proposals: Dict[str, List[Proposal]],
    sample_seed: int = 42
) -> None:
    """Print formatted summary of agent proposals, conflicts, and sample hand-checks."""
    counts = summary["proposal_counts"]
    conflict_m = summary["conflict_metrics"]
    by_type = conflict_m["conflict_counts_by_type"]

    print("=" * 80)
    print("                MULTI-AGENT PROPOSAL & CONFLICT ANALYSIS REPORT              ")
    print("=" * 80)
    print(f"Total Dataset Sample VMs : {summary['total_sample_vms']:,}")
    print(f"Unique VMs with Proposals: {summary['unique_vms_with_proposals']:,} ({summary['proposal_coverage_pct']}%)")
    print(f"VMs with >1 Proposal    : {conflict_m['conflict_vm_count']:,} ({conflict_m['conflict_vm_pct_of_proposal_vms']}% of proposal VMs)")
    print("-" * 80)
    print("PROPOSALS PER AGENT:")
    print(f"  - RightsizingAgent: {counts['RightsizingAgent']:,} proposals")
    print(f"  - PricingAgent    : {counts['PricingAgent']:,} proposals")
    print(f"  - SchedulingAgent : {counts['SchedulingAgent']:,} proposals")
    print(f"  - Total Proposals : {counts['total_proposals']:,}")
    print("-" * 80)
    print("CONFLICT COUNT BY TYPE:")
    print(f"  - order_dependent    (Rightsizing + Pricing)   : {by_type['order_dependent']:,} conflicts")
    print(f"  - mutually_exclusive (Reserved/Spot + Schedule): {by_type['mutually_exclusive']:,} conflicts")
    print(f"  - compatible         (Rightsizing + Schedule)  : {by_type['compatible']:,} conflicts")
    print("-" * 80)

    # Sample 3 Rightsizing proposals for manual hand checking
    rs_props = agent_proposals.get("RightsizingAgent", [])
    print("SAMPLE RIGHTSIZING PROPOSALS FOR MANUAL HAND-CHECKING:")
    if len(rs_props) > 0:
        rng = random.Random(sample_seed)
        sampled = rng.sample(rs_props, min(3, len(rs_props)))
        for idx, prop in enumerate(sampled, 1):
            curr_c = prop.current_configuration["vCPU_bucket"]
            curr_m = prop.current_configuration["RAM_bucket_GB"]
            curr_sku = f"{curr_c} vCPU / {curr_m} GB ({prop.current_configuration['matched_family']})"
            curr_cost = prop.current_configuration["on_demand_monthly_cost"]

            tgt_c = prop.proposed_configuration["vCPU_bucket"]
            tgt_m = prop.proposed_configuration["RAM_bucket_GB"]
            tgt_sku = f"{tgt_c} vCPU / {tgt_m} GB ({prop.proposed_configuration['matched_family']})"
            tgt_cost = prop.proposed_configuration["on_demand_monthly_cost"]
            savings = prop.predicted_monthly_savings
            calc_diff = round(curr_cost - tgt_cost, 2)

            print(f"\n  [Sample {idx}] VM ID: {prop.vm_id}")
            print(f"    Current SKU  : {curr_sku}")
            print(f"    Proposed SKU : {tgt_sku}")
            print(f"    Current Cost : ${curr_cost:.2f} / month")
            print(f"    Proposed Cost: ${tgt_cost:.2f} / month")
            print(f"    Savings      : ${savings:.2f} / month (Verification Check: ${curr_cost:.2f} - ${tgt_cost:.2f} = ${calc_diff:.2f})")
    else:
        print("  (No Rightsizing proposals available to sample)")

    print("=" * 80)

