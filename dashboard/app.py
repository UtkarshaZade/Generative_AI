"""
Dataset Explorer & Multi-Agent Evaluation Dashboard (Streamlit & CLI Interface).
Phase 2B & Agent Module Implementation for AI Cloud Waste Negotiator.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import get_available_dataset_files, load_vmtable, load_cpu_readings, load_schema
from src.data.validator import validate_vmtable, validate_cpu_readings
from src.data.preprocessing import (
    add_derived_lifetime,
    map_core_memory_buckets,
    apply_lifetime_eligibility_filter,
    get_agent_specific_dataset,
)
from src.data.pricing import (
    add_pricing_columns,
    get_pricing_summary,
    RATE_CARD,
)
from src.data.analysis import (
    get_dataset_overview,
    analyze_cpu_utilisation,
    analyze_lifetime,
    analyze_category,
    analyze_memory_buckets,
    analyze_core_buckets,
    analyze_time_series_cpu,
    get_feature_relevance_matrix,
    get_data_gaps,
    get_deterministic_observations,
)
from src.agents import (
    RightsizingAgent,
    PricingAgent,
    SchedulingAgent,
    run_all_agents,
)


def run_cli_analysis():
    """Run CLI-based exploratory data analysis and multi-agent conflict evaluation report."""
    print("=" * 80)
    print("AI Cloud Waste Negotiator — Dataset Explorer & Multi-Agent Report")
    print("=" * 80)

    # 1. Dataset Files Availability
    files_status = get_available_dataset_files(PROJECT_ROOT)
    print("\n--- 1. DATASET FILES AVAILABILITY ---")
    for key, info in files_status.items():
        status_str = f"FOUND ({info['size_mb']} MB)" if info['exists'] else "MISSING / NOT PRESENT LOCALLY"
        print(f"  [{key}]: {status_str} -> {info['description']}")

    # 2. Data Loading & Validation
    print("\n--- 2. DATA LOADING & VALIDATION ---")
    sample_size = 50000
    print(f"Loading vmtable sample ({sample_size:,} records)...")
    vmtable_df, meta = load_vmtable(PROJECT_ROOT, sample_size=sample_size)

    if vmtable_df is None:
        print("ERROR: Failed to load vmtable dataset!")
        return

    vmtable_df = add_derived_lifetime(vmtable_df)
    vmtable_df = map_core_memory_buckets(vmtable_df, apply_approximation=False)
    vmtable_df = add_pricing_columns(vmtable_df)
    vmtable_df, life_filter_res = apply_lifetime_eligibility_filter(
        vmtable_df,
        min_lifetime_hours=24.0,
        min_scheduling_hours=48.0
    )

    val_res = validate_vmtable(vmtable_df)
    print(f"Validation Result: {'PASSED' if val_res['passed'] else 'FAILED / ATTENTION REQUIRED'}")
    print(f"  Total records loaded: {val_res['total_records']:,}")
    print(f"  Duplicate VM IDs: {val_res['duplicate_vmids']}")
    print(f"  Missing values count: {sum(val_res['missing_counts'].values())}")

    # 3. Overview
    overview = get_dataset_overview(vmtable_df, files_status, val_res, is_sample=meta['is_sample'], sample_size=meta['sample_size'])
    print(f"  Dataset designation: {overview['subset_description']}")
    print(f"  Unique VMs: {overview['unique_vms']:,}")
    print(f"  Unique Subscriptions: {overview['unique_subscriptions']:,}")
    print(f"  Unique Deployments: {overview['unique_deployments']:,}")

    # 4. CPU Utilisation
    print("\n--- 3. CPU UTILISATION STATISTICS ---")
    cpu_stats = analyze_cpu_utilisation(vmtable_df)
    for cpu_field, stats in cpu_stats.items():
        print(f"\n  [{cpu_field.upper()}]")
        print(f"    Mean: {stats['mean']:.2f}% | Median: {stats['median']:.2f}% | Std: {stats['std']:.2f}%")
        print(f"    Min: {stats['min']:.2f}% | Max: {stats['max']:.2f}%")
        print(f"    Percentiles: P25={stats['percentiles']['p25']:.2f}%, P50={stats['percentiles']['p50']:.2f}%, P75={stats['percentiles']['p75']:.2f}%, P95={stats['percentiles']['p95']:.2f}%")

    # 5. VM Lifetime & Eligibility Filters
    print("\n--- 4. VM LIFETIME ANALYSIS & AGENT ELIGIBILITY ---")
    life_stats = analyze_lifetime(vmtable_df)
    print(f"  Formula: {life_stats['derived_formula']}")
    print(f"  Mean lifetime: {life_stats['mean_hours']:.2f} hours")
    print(f"  Median lifetime: {life_stats['median_hours']:.2f} hours")

    rp_filter = life_filter_res['rightsizing_pricing_filter']
    sched_filter = life_filter_res['scheduling_filter']

    print(f"\n  [a] Rightsizing & Pricing Agent Filter (Threshold >= {rp_filter['threshold_hours']:.1f}h):")
    print(f"      Retained (Eligible): {rp_filter['retained_vms']:,} ({rp_filter['retained_pct']}%)")
    print(f"      Excluded (Short-lived): {rp_filter['excluded_vms']:,} ({rp_filter['excluded_pct']}%)")

    print(f"\n  [b] Scheduling & Autoscaling Agent Filter (Observation Window >= {sched_filter['threshold_hours']:.1f}h):")
    print(f"      Retained (Eligible): {sched_filter['retained_vms']:,} ({sched_filter['retained_pct']}%)")
    print(f"      Excluded (Short Observation): {sched_filter['excluded_vms']:,} ({sched_filter['excluded_pct']}%)")

    # 6. Synthetic Cloud Pricing Summary
    print("\n--- 5. SYNTHETIC CLOUD PRICING & MONTHLY COST SUMMARY ---")
    pricing_summary = get_pricing_summary(vmtable_df)
    print(f"  Total Sample VMs Analyzed: {pricing_summary['total_vms']:,}")
    print(f"  Total On-Demand Monthly Cost:  ${pricing_summary['total_on_demand_monthly_cost']:,.2f} (Mean: ${pricing_summary['mean_on_demand_monthly_cost']:.2f}/mo)")
    print(f"  Total 1-Yr Reserved Monthly:   ${pricing_summary['total_reserved_1yr_monthly_cost']:,.2f} (Potential Savings: {pricing_summary['potential_reserved_savings_pct']}%)")
    print(f"  Total Spot Monthly Cost:       ${pricing_summary['total_spot_monthly_cost']:,.2f} (Potential Savings: {pricing_summary['potential_spot_savings_pct']}%)")

    # 7. Time-series Telemetry Loading for Agents
    cpu_readings_df, cpu_meta = load_cpu_readings(PROJECT_ROOT, sample_size=100000)

    # 8. Multi-Agent Optimization Proposal & Conflict Evaluation
    print("\n--- 6. MULTI-AGENT PROPOSALS & CONFLICT ANALYSIS ---")
    agent_props, agent_summary = run_all_agents(vmtable_df, cpu_readings_df)

    p_counts = agent_summary['proposal_counts']
    c_metrics = agent_summary['conflict_metrics']
    c_breakdown = c_metrics['conflict_breakdown']

    print(f"  Rightsizing Agent Proposals:  {p_counts['RightsizingAgent']:,}")
    print(f"  Pricing Agent Proposals:      {p_counts['PricingAgent']:,}")
    print(f"  Scheduling Agent Proposals:   {p_counts['SchedulingAgent']:,}")
    print(f"  ------------------------------------------------")
    print(f"  Total Agent Proposals:        {p_counts['total_proposals']:,}")
    print(f"  Unique Proposal VMs:          {agent_summary['unique_vms_with_proposals']:,} ({agent_summary['proposal_coverage_pct']}% of sample)")
    print(f"  Total Predicted Monthly Savings: ${agent_summary['total_predicted_monthly_savings']:,.2f}/month")
    print(f"\n  🔥 CONFLICT COUNT (VMs with proposals from >1 agent): {c_metrics['conflict_vm_count']:,} ({c_metrics['conflict_vm_pct_of_proposal_vms']}% of proposal VMs)")
    print(f"    • Rightsizing + Pricing Conflict:   {c_breakdown['rightsizing_and_pricing']:,}")
    print(f"    • Pricing + Scheduling Conflict:     {c_breakdown['pricing_and_scheduling']:,}")
    print(f"    • Rightsizing + Scheduling Conflict: {c_breakdown['rightsizing_and_scheduling']:,}")
    print(f"    • All 3 Agents Conflict:            {c_breakdown['all_three_agents']:,}")

    # 9. VM Category Distribution
    print("\n--- 7. VM CATEGORY DISTRIBUTION ---")
    cat_df = analyze_category(vmtable_df)
    print(cat_df.to_string())

    # 10. Memory & Cores
    print("\n--- 8. RESOURCE CONFIGURATION BUCKETS ---")
    print(analyze_memory_buckets(vmtable_df).to_string())
    print(analyze_core_buckets(vmtable_df).to_string())

    # 11. Feature Relevance & Data Gaps
    print("\n--- 9. INITIAL FEATURE RELEVANCE MATRIX ---")
    print(get_feature_relevance_matrix().to_string(index=False))

    print("\n--- 10. MISSING DOMAIN INFORMATION (DATA GAPS) ---")
    for gap in get_data_gaps():
        print(f"  • [{gap['Category']}] {gap['Item']}: {gap['Impact']}")

    # 12. Empirical Observations
    print("\n--- 11. DETERMINISTIC RESEARCH OBSERVATIONS ---")
    obs_list = get_deterministic_observations(vmtable_df, cpu_readings_df)
    for idx, obs in enumerate(obs_list, 1):
        print(f"  {idx}. {obs}")

    print("\n" + "=" * 80)
    print("Report completed successfully!")
    print("=" * 80)


def run_streamlit_dashboard():
    """Run interactive Streamlit EDA & Multi-Agent Evaluation Dashboard."""
    import streamlit as st
    import matplotlib.pyplot as plt

    st.set_page_config(
        page_title="Azure 2019 Dataset Explorer",
        page_icon="☁️",
        layout="wide"
    )

    st.title("☁️ AI Cloud Waste Negotiator — Multi-Agent & Dataset Explorer")
    st.markdown("**Phase 2B & Multi-Agent Module Evaluation**")

    # Sidebar Controls
    st.sidebar.header("📁 Data Loader & Agent Controls")
    sample_size = st.sidebar.slider("vmtable Sample Size", min_value=10000, max_value=200000, value=50000, step=10000)
    p95_thresh = st.sidebar.slider("Rightsizing P95 Max CPU Threshold (%)", min_value=5.0, max_value=50.0, value=20.0, step=1.0)
    lifetime_threshold = st.sidebar.slider("Rightsizing/Pricing Threshold (Hours)", min_value=1.0, max_value=168.0, value=24.0, step=1.0)
    scheduling_threshold = st.sidebar.slider("Scheduling Observation Window (Hours)", min_value=1.0, max_value=168.0, value=48.0, step=1.0)
    load_ts = st.sidebar.checkbox("Load CPU Time-Series Telemetry Sample", value=True)

    # Load Data
    with st.spinner("Loading dataset sample and evaluating optimization agents..."):
        file_status = get_available_dataset_files(PROJECT_ROOT)
        vmtable_df, meta = load_vmtable(PROJECT_ROOT, sample_size=sample_size)

        if vmtable_df is None:
            st.error("❌ Failed to load `vmtable.csv`. Check workspace dataset files.")
            return

        vmtable_df = add_derived_lifetime(vmtable_df)
        vmtable_df = map_core_memory_buckets(vmtable_df, apply_approximation=False)
        vmtable_df = add_pricing_columns(vmtable_df)
        vmtable_df, life_filter_res = apply_lifetime_eligibility_filter(
            vmtable_df,
            min_lifetime_hours=lifetime_threshold,
            min_scheduling_hours=scheduling_threshold
        )
        val_res = validate_vmtable(vmtable_df)

        cpu_df, cpu_meta = (None, {})
        if load_ts and file_status.get("vm_cpu_readings_chunk1", {}).get("exists"):
            cpu_df, cpu_meta = load_cpu_readings(PROJECT_ROOT, sample_size=100000)

        # Run Agents
        agent_props, agent_summary = run_all_agents(
            vmtable_df,
            cpu_df,
            p95_threshold=p95_thresh,
            min_lifetime_hours=lifetime_threshold,
            min_scheduling_hours=scheduling_threshold
        )

    # Dashboard Tabs
    tabs = st.tabs([
        "1. Multi-Agent Proposals & Conflicts",
        "2. Dataset Overview",
        "3. CPU Utilisation",
        "4. VM Lifetime & Eligibility",
        "5. Synthetic Cloud Pricing",
        "6. VM Category & Buckets",
        "7. Time-Series Telemetry",
        "8. Feature Relevance & Gaps",
        "9. Empirical Observations"
    ])

    # Tab 1: Multi-Agent Proposals & Conflicts
    with tabs[0]:
        st.header("1. Multi-Agent Proposals & Conflict Evaluation")
        st.caption("Rightsizing, Pricing, and Scheduling agents operate independently on preprocessed eligible VMs.")

        p_counts = agent_summary['proposal_counts']
        c_metrics = agent_summary['conflict_metrics']
        c_breakdown = c_metrics['conflict_breakdown']

        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Rightsizing Proposals", f"{p_counts['RightsizingAgent']:,}")
        a2.metric("Pricing Proposals", f"{p_counts['PricingAgent']:,}")
        a3.metric("Scheduling Proposals", f"{p_counts['SchedulingAgent']:,}")
        a4.metric("Total Predicted Savings", f"${agent_summary['total_predicted_monthly_savings']:,.2f}/mo")
        a5.metric("Conflict Count (VMs)", f"{c_metrics['conflict_vm_count']:,}", f"{c_metrics['conflict_vm_pct_of_proposal_vms']}% of proposal VMs")

        st.subheader("Multi-Agent Proposal Conflict Breakdown")
        st.info("🔥 **Conflict Definition:** A VM receives proposals from more than one agent (e.g. Rightsizing downsize + Pricing model switch). These conflicts feed directly into the Negotiation & Arbitration Layer.")

        cb1, cb2, cb3, cb4 = st.columns(4)
        cb1.metric("Rightsizing + Pricing", f"{c_breakdown['rightsizing_and_pricing']:,}")
        cb2.metric("Pricing + Scheduling", f"{c_breakdown['pricing_and_scheduling']:,}")
        cb3.metric("Rightsizing + Scheduling", f"{c_breakdown['rightsizing_and_scheduling']:,}")
        cb4.metric("All 3 Agents Conflict", f"{c_breakdown['all_three_agents']:,}")

        st.subheader("Sample Agent Proposals")
        selected_agent = st.selectbox("Select Agent to View Proposals", ["RightsizingAgent", "PricingAgent", "SchedulingAgent"])
        sample_props = agent_props[selected_agent][:10]
        if sample_props:
            prop_df = pd.DataFrame([p.to_dict() for p in sample_props])
            st.dataframe(prop_df[['vm_id', 'action_type', 'predicted_monthly_savings', 'confidence', 'rationale']], use_container_width=True)

    # Tab 2: Dataset Overview
    with tabs[1]:
        st.header("2. Dataset Overview & Data Quality Audit")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Loaded Records", f"{len(vmtable_df):,}")
        m2.metric("Unique VMs", f"{vmtable_df['vmid'].nunique():,}")
        m3.metric("Unique Subscriptions", f"{vmtable_df['subscriptionid'].nunique():,}")
        m4.metric("Unique Deployments", f"{vmtable_df['deploymentid'].nunique():,}")

        st.markdown(f"**Dataset Designation:** ⚠️ `{meta['file_size_mb']} MB File Chunk / Sample of {len(vmtable_df):,} records` (Complete dataset: 2,695,548 VMs)")
        
        st.subheader("Workspace Dataset Files Audit")
        files_df = pd.DataFrame([
            {
                "File Key": k,
                "Status": "✅ Present" if v["exists"] else "❌ Missing",
                "Size (MB)": v["size_mb"],
                "Description": v["description"]
            }
            for k, v in file_status.items()
        ])
        st.dataframe(files_df, use_container_width=True)

    # Tab 3: CPU Utilisation
    with tabs[2]:
        st.header("3. Aggregate CPU Utilisation Analysis")
        cpu_stats = analyze_cpu_utilisation(vmtable_df)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Descriptive Statistics (%)")
            cpu_summary_df = pd.DataFrame(cpu_stats).T
            st.dataframe(cpu_summary_df[['mean', 'std', 'min', 'median', 'max']], use_container_width=True)

            st.subheader("Percentiles (%)")
            p_df = pd.DataFrame({k: v['percentiles'] for k, v in cpu_stats.items()}).T
            st.dataframe(p_df, use_container_width=True)

        with col2:
            st.subheader("CPU Distribution CDF Plot")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            
            for col, color, label in [('avgcpu', 'blue', 'Average CPU'), ('maxcpu', 'red', 'Max CPU'), ('p95maxcpu', 'green', 'P95 Max CPU')]:
                if col in vmtable_df.columns:
                    sorted_data = np.sort(vmtable_df[col].dropna())
                    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1) * 100
                    ax.plot(sorted_data, yvals, label=label, color=color, linewidth=2)
                    
            ax.set_xlabel("CPU Utilisation (%)")
            ax.set_ylabel("Cumulative Percentage (CDF %)")
            ax.set_title("CPU Utilisation CDF Distribution")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            st.pyplot(fig)

    # Tab 4: VM Lifetime & Eligibility Filters
    with tabs[3]:
        st.header("4. VM Lifetime & Agent Eligibility Filters")
        st.markdown("**Derived Formula:** `lifetime_hours = max(vmdeleted - vmcreated, 300) / 3600`")
        
        life_stats = analyze_lifetime(vmtable_df)
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Mean Lifetime", f"{life_stats['mean_hours']:.1f} hrs")
        l2.metric("Median Lifetime", f"{life_stats['median_hours']:.1f} hrs")
        l3.metric("Min Lifetime", f"{life_stats['min_hours']:.2f} hrs")
        l4.metric("Max Lifetime", f"{life_stats['max_hours']:.1f} hrs")

        rp_info = life_filter_res['rightsizing_pricing_filter']
        sched_info = life_filter_res['scheduling_filter']

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.subheader("Rightsizing & Pricing Agent Filter")
            st.info(f"🎯 Threshold: **{lifetime_threshold:.1f} Hours** (Excludes short-lived VMs where resizing reboot downtime or commitments are unsuitable).")
            st.metric("Retained (Eligible)", f"{rp_info['retained_vms']:,} ({rp_info['retained_pct']}%)")
            st.metric("Excluded (Short-lived)", f"{rp_info['excluded_vms']:,} ({rp_info['excluded_pct']}%)")

        with f_col2:
            st.subheader("Scheduling & Autoscaling Agent Filter")
            st.info(f"⚡ Minimum Observation Window: **{scheduling_threshold:.1f} Hours** (Requires history to observe diurnal periodicity).")
            st.metric("Retained (Eligible)", f"{sched_info['retained_vms']:,} ({sched_info['retained_pct']}%)")
            st.metric("Excluded (Short Window)", f"{sched_info['excluded_vms']:,} ({sched_info['excluded_pct']}%)")

    # Tab 5: Synthetic Cloud Pricing
    with tabs[4]:
        st.header("5. Synthetic Cloud Pricing & Predicted Monthly Costs")
        st.caption("Rate Card Source: Microsoft Azure Retail Prices API & AWS EC2 Published Rates (East US / us-east-1, Oct 2024/2025)")

        pricing_sum = get_pricing_summary(vmtable_df)
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Total On-Demand Monthly Cost", f"${pricing_sum['total_on_demand_monthly_cost']:,.2f}")
        pc2.metric("Total 1-Yr Reserved Monthly Cost", f"${pricing_sum['total_reserved_1yr_monthly_cost']:,.2f}", f"-{pricing_sum['potential_reserved_savings_pct']}% Savings")
        pc3.metric("Total Spot Monthly Cost", f"${pricing_sum['total_spot_monthly_cost']:,.2f}", f"-{pricing_sum['potential_spot_savings_pct']}% Savings")

        st.subheader("Published Rate Card Anchor Matrix")
        rate_matrix_df = pd.DataFrame([
            {
                "vCPU Bucket": k[0],
                "RAM Bucket (GB)": k[1],
                "Matched Family": v["matched_family"],
                "On-Demand ($/hr)": f"${v['on_demand_hourly']:.4f}",
                "1-Yr Reserved ($/hr)": f"${v['reserved_1yr_hourly']:.4f}",
                "Spot ($/hr)": f"${v['spot_hourly']:.4f}",
                "On-Demand Monthly ($/mo)": f"${round(v['on_demand_hourly'] * 730, 2):,.2f}"
            }
            for k, v in RATE_CARD.items()
        ])
        st.dataframe(rate_matrix_df, use_container_width=True)

    # Tab 6: VM Category & Buckets
    with tabs[5]:
        st.header("6. Workload Category & Resource Buckets")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.subheader("Core Count Buckets")
            st.dataframe(analyze_core_buckets(vmtable_df), use_container_width=True)
        with mc2:
            st.subheader("Memory Buckets (GB)")
            st.dataframe(analyze_memory_buckets(vmtable_df), use_container_width=True)

    # Tab 7: Time-series
    with tabs[6]:
        st.header("7. Time-Series CPU Telemetry Sample")
        if cpu_df is not None and len(cpu_df) > 0:
            st.success(f"Loaded {len(cpu_df):,} 5-minute sampling telemetry records across {cpu_df['vmid'].nunique():,} unique VMs.")
            sample_vms = list(cpu_df['vmid'].unique()[:5])
            selected_vm = st.selectbox("Select VM ID for Time-Series Inspection", sample_vms)
            
            vm_ts = cpu_df[cpu_df['vmid'] == selected_vm].sort_values('timestamp')
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(vm_ts['timestamp'] / 3600.0, vm_ts['avgcpu'], label='Avg CPU %', color='blue')
            ax.plot(vm_ts['timestamp'] / 3600.0, vm_ts['maxcpu'], label='Max CPU %', color='red', alpha=0.6)
            ax.set_xlabel("Time (Hours from trace start)")
            ax.set_ylabel("CPU Utilisation (%)")
            ax.set_title(f"Time-Series Telemetry for VM: {selected_vm[:16]}...")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            st.pyplot(fig)

    # Tab 8: Feature Relevance & Gaps
    with tabs[7]:
        st.header("8. Feature Relevance & Data Gaps")
        st.dataframe(get_feature_relevance_matrix(), use_container_width=True)
        st.subheader("Missing Domain Information (Data Gaps)")
        st.dataframe(pd.DataFrame(get_data_gaps()), use_container_width=True)

    # Tab 9: Empirical Observations
    with tabs[8]:
        st.header("9. Deterministic Research Observations")
        for idx, obs in enumerate(get_deterministic_observations(vmtable_df, cpu_df), 1):
            st.markdown(f"**{idx}.** {obs}")


def is_running_with_streamlit() -> bool:
    """Check if script is executing inside a Streamlit runner context."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return True
    except Exception:
        pass
    
    # Fallback check for command line args or environment
    for arg in sys.argv:
        if "streamlit" in str(arg).lower():
            return True
    return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli_analysis()
    elif is_running_with_streamlit() or (len(sys.argv) > 1 and sys.argv[1] == "--streamlit"):
        run_streamlit_dashboard()
    else:
        # Default when run via python without flags: run CLI report, but inform user how to launch Streamlit UI
        run_cli_analysis()
