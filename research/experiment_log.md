# Research Experiment Log
**Project:** AI Cloud Waste Negotiator: A Multi-Agent Negotiation Framework for Explainable Cloud Cost Optimization  

---

## Experiment Entry: EXP-004 (Phase 2B Update — Multi-Agent Optimization & Conflict Analysis)

* **Experiment/Activity:** Implementation of Specialized Optimization Agents & Multi-Agent Conflict Evaluation
* **Date:** 2026-10-08
* **Phase:** Phase 2B — Multi-Agent Generation & Conflict Diagnostics

### Objective:
1. Implement the three domain-specific optimization agents (`RightsizingAgent`, `PricingAgent`, `SchedulingAgent`) operating on the preprocessed, eligibility-flagged, priced dataset.
2. Quantify proposal generation, monthly savings predictions, agent coverage, and multi-agent proposal conflicts (VMs receiving competing recommendations from >1 agent) across the 50,000 VM sample dataset.

### Agent Logic Specifications:
* **Rightsizing Agent (`src/agents/rightsizing_agent.py`):**
  * Target: `eligible_rightsizing_pricing == True` and `p95maxcpu < 20.0%` (configurable).
  * Action: Downsize vCPU/memory step-wise (e.g. 8 vCPU / 32 GB -> 4 vCPU / 16 GB).
  * Metrics: Calculates monthly savings from reduced On-Demand rate; sets confidence based on sample depth and max CPU peak headroom.
* **Pricing Agent (`src/agents/pricing_agent.py`):**
  * Target: `eligible_rightsizing_pricing == True`.
  * Action: Compare On-Demand vs 1-Year Reserved vs Spot monthly rates. Evaluates VM workload risk (`vmcategory`). Proposes Spot for `Delay-insensitive`, 1-Year Reserved for `Interactive` / `Unknown`.
  * Metrics: Savings based on rate difference; confidence based on workload tolerance alignment.
* **Scheduling Agent (`src/agents/scheduling_agent.py`):**
  * Target: `scheduling_eligible == True` (observation window >= 48h).
  * Action: Detects low utilization variance or low mean CPU (< 15%) across observation windows, proposing off-peak automated pause schedules (e.g., 12h/day off-peak shutdown = 50% runtime savings).
  * Metrics: Calculates 50% monthly cost savings on On-Demand rate; confidence based on utilization variance stability.

### Data Used:
* `trace_data_vmtable_vmtable.csv/trace_data_vmtable_vmtable.csv` (Sample of 50,000 records)

### Empirical Results (50,000 VM Sample):
* **Total Dataset VMs:** 50,000
* **Rightsizing Agent Proposals:** 1,484 proposals
* **Pricing Agent Proposals:** 6,472 proposals
* **Scheduling Agent Proposals:** 1,881 proposals
* **Total Proposals Generated:** 9,837 proposals
* **Unique VMs Receiving Proposals:** 6,472 VMs (12.94% coverage)
* **Conflict Count (VMs with >1 Proposal):** **2,900 VMs** (**44.81%** of proposal VMs!)
* **Conflict Breakdown:**
  * Rightsizing + Pricing Conflict: 1,484 VMs
  * Pricing + Scheduling Conflict: 1,881 VMs
  * Rightsizing + Scheduling Conflict: 465 VMs
  * All 3 Agents Conflict: 465 VMs
* **Total Predicted Monthly Savings:** **$429,956.23 / month**

---

## Experiment Entry: EXP-003 (Phase 2B Update — Synthetic Pricing & Scheduling Observation Window)

* **Experiment/Activity:** Synthetic Cloud Pricing Module & Scheduling Observation Window Filter
* **Date:** 2026-10-08
* **Phase:** Phase 2B — Data Pipeline Feature Derivation, Pricing Ingestion & Agent Scoping

### Objective:
1. Build a synthetic cloud pricing module (`src/data/pricing.py`) mapping every real core/memory combination in the Azure dataset to On-Demand, 1-Year Reserved Instance, and Spot rates anchored to published Azure/AWS rate cards.
2. Add predicted monthly cost columns (`on_demand_monthly_cost`, `reserved_1yr_monthly_cost`, `spot_monthly_cost`, `predicted_monthly_cost`) to the dataset.
3. Implement a separate `scheduling_eligible` flag in `src/data/preprocessing.py` excluding VMs below a minimum observation window (default 48.0h) specifically for the Scheduling agent.

### Data Used:
* `trace_data_vmtable_vmtable.csv/trace_data_vmtable_vmtable.csv` (Sample of 50,000 records)
* Rate Card Citations: Microsoft Azure Retail Prices API (Region: East US) & AWS EC2 Published Rates (us-east-1), October 2024/2025.

### Method:
1. Created `src/data/pricing.py` defining `RATE_CARD` for core/memory pairs (`2/2`, `2/4`, `2/8`, `2/32`, `4/8`, `4/32`, `8/32`, `8/64`, `24/64`, `>24/>64`), citing exact rate card sources in top comments.
2. Implemented `add_pricing_columns(df)` appending hourly rates, monthly costs ($/mo = rate * 730), and baseline `predicted_monthly_cost`.
3. Updated `apply_lifetime_eligibility_filter(df)` in `src/data/preprocessing.py` to calculate both `eligible_rightsizing_pricing` (default >=24h) and `scheduling_eligible` (default >=48h).
4. Updated `get_agent_specific_dataset(df, agent_type)` to filter by targeted flag.
5. Integrated pricing metrics and dual eligibility breakdowns into `dashboard/app.py`.
6. Created unit test suite `tests/test_pricing.py` (17 total unit tests pass in 0.58s).

### Logged Results (50,000 VM Sample):
* **Rightsizing & Pricing Filter (>= 24.0h):** Retained: 6,472 VMs (12.94%) | Excluded: 43,528 VMs (87.06%)
* **Scheduling Observation Window (>= 48.0h):** Retained: 5,446 VMs (10.89%) | Excluded: 44,554 VMs (89.11%)
* **Total Baseline On-Demand Monthly Cost:** $5,956,997.60/month (Mean: $119.14/month per VM)
* **Total 1-Year Reserved Monthly Cost:** $3,580,147.57/month (Potential Savings: 39.9%)
* **Total Spot Monthly Cost:** $1,500,873.60/month (Potential Savings: 74.8%)

---

## Experiment Entry: EXP-002 (Phase 2B Update — Lifetime Eligibility Filter)

* **Experiment/Activity:** Implementation of Agent-Specific Lifetime Hours Eligibility Filter
* **Date:** 2026-10-05
* **Phase:** Phase 2B — Data Pipeline Feature Derivation & Agent Scoping

### Objective:
Implement a configurable VM lifetime eligibility filter (`min_lifetime_hours`, default 24.0h) to scope VMs for Rightsizing and Reserved/Spot Pricing agents specifically, while retaining all VMs for the Scheduling & Autoscaling agent.

### Data Used:
* `trace_data_vmtable_vmtable.csv/trace_data_vmtable_vmtable.csv` (Sample of 50,000 records)

### Method:
1. Implemented `apply_lifetime_eligibility_filter(df, min_lifetime_hours=24.0)` in `src/data/preprocessing.py` deriving `lifetime = max(vmdeleted - vmcreated, 300) / 3600` and attaching boolean flag `eligible_rightsizing_pricing`.
2. Implemented `get_agent_specific_dataset(df, agent_type, min_lifetime_hours)` to supply filtered DataFrames based on targeted agent module.
3. Updated statistical analysis engine (`src/data/analysis.py`) and dashboard UI/CLI (`dashboard/app.py`) to log filter breakdown.
4. Added unit tests in `tests/test_validator.py` verifying threshold behavior and agent-specific dataset retrieval.

### Logged Filter Results (50,000 VM Sample at 24.0h Threshold):
* **Total Sample VMs:** 50,000
* **Retained for Rightsizing & Pricing Agents (Lifetime >= 24.0h):** 6,472 VMs (12.94%)
* **Excluded from Rightsizing & Pricing Agents (Short-lived < 24.0h):** 43,528 VMs (87.06%)
* **Scheduling & Autoscaling Agent:** Retains 50,000 VMs (100.0%)

---

## Experiment Entry: EXP-001 (Phase 2B Initial)

* **Experiment/Activity:** Azure Dataset Exploratory Analysis & Dataset Explorer Implementation
* **Date:** 2026-09-12
* **Phase:** Phase 2B — Exploratory Data Analysis and Data Foundation

### Objective:
Understand and validate the Azure 2019 Public Dataset V2 before building optimization agents.

### Data Used:
* `schema.csv`
* `trace_data_vmtable_vmtable.csv/trace_data_vmtable_vmtable.csv` (Sample of 50,000 records)
* `trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv/trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv` (100,000 time-series readings)
* Benchmark distribution files (`azure2019_data_category.txt`, `azure2019_data_cores.txt`, `azure2019_data_memory.txt`, `azure2019_data_lifetime.txt`)

### Method:
1. Implemented modular data loader (`src/data/loader.py`) with explicit file resolution and pandas chunking/sampling.
2. Created deterministic data validator (`src/data/validator.py`) testing missing values, duplicate IDs, CPU bounds [0, 100], category compliance, and timestamp hierarchy.
3. Created preprocessing module (`src/data/preprocessing.py`) deriving VM lifetime (`max(vmdeleted - vmcreated, 300) / 3600`) and handling bucket mappings.
4. Created statistical analysis engine (`src/data/analysis.py`) producing CPU percentiles, feature relevance ratings, data gaps audit, and empirical observations.
5. Built dual Streamlit/CLI Dataset Explorer dashboard (`dashboard/app.py`).
6. Implemented automated unit test suite (`tests/test_data_loader.py`, `tests/test_validator.py`).
