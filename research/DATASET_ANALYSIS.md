# Dataset Analysis & Data Foundation Report
**Project:** AI Cloud Waste Negotiator: A Multi-Agent Negotiation Framework for Explainable Cloud Cost Optimization  
**Phase:** Phase 2 — Dataset Investigation and Data Foundation (Phase 2B Update)  
**Dataset:** Azure 2019 Public Dataset V2  

---

## A. Dataset Overview
The **Azure 2019 Public Dataset V2** represents a 30-day workload trace of Virtual Machine (VM) instances running on Microsoft Azure. It contains telemetry for **2,695,548 Virtual Machines**, **6,687 Subscriptions**, and **33,205 Deployments**, covering over **162.4 million VM hours** and **469.4 million Core hours**.

The dataset provides critical empirical workload patterns, resource allocations, and execution lifetimes necessary to establish a realistic baseline for cloud cost optimization research.

---

## B. Available Workspace Files & Loading Strategy

### Available Files Audit:
1. **`schema.csv`** (1,485 bytes): Master schema specification mapping field names, column indexes, descriptions, and data types across all 5 dataset table structures.
2. **`Azure.2019.Public.Dataset.V2.-.Trace.Analysis.ipynb`** (138,635 bytes): Primary exploratory analysis notebook.
3. **`AzurePublicDatasetLinksV2.txt`** (28,145 bytes): Official download URL manifest listing all 195 CPU telemetry files, table archives, and distribution text files.
4. **`azure2019_data_category.txt`** (65 bytes): Benchmark category distribution text file.
5. **`azure2019_data_cores.txt`** (82 bytes): Benchmark vCPU core count distribution text file.
6. **`azure2019_data_memory.txt`** (75 bytes): Benchmark memory distribution text file.
7. **`azure2019_data_lifetime.txt`** (12,714 bytes): Cumulative CDF data of VM lifetimes (in hours).
8. **`trace_data_vmtable_vmtable.csv/trace_data_vmtable_vmtable.csv`** (~821 MB): Full uncompressed VM table containing metadata and aggregate CPU utilization metrics for all 2.69M VMs.
9. **`trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv/trace_data_vm_cpu_readings_vm_cpu_readings-file-1-of-195.csv`** (~1.27 GB): Uncompressed time-series CPU telemetry chunk 1 containing 5-minute sampling intervals (`min cpu`, `max cpu`, `avg cpu`) for 100,000+ VMs.

### Data Loading Method & Strategy:
* **Modular Architecture**: Implemented `src/data/loader.py` with explicit existence checking, file path resolution, and column filtering.
* **Sampling / Chunking**: Used pandas chunking and sample loading (`nrows=50,000` default sample for exploratory analysis, configurable up to full population loading).
* **Memory Optimization**: Reduced memory consumption from ~821 MB to ~8 MB per sample batch, ensuring responsive EDA execution.

---

## C. Deterministic Data Validation Results

Validation performed using `src/data/validator.py`:
* **Total Sample Records Validated**: 50,000 records
* **Missing Values**: 0 missing values across all columns.
* **Duplicate VM IDs**: 0 duplicates (`vmid` values are unique).
* **CPU Range Checks ([0.0, 100.0])**: Passed (0 out-of-bound values in `avgcpu`, `maxcpu`, `p95maxcpu`).
* **CPU Hierarchy (`maxcpu >= avgcpu`)**: Passed (0 violations).
* **Category Validation**: Valid (`Interactive`, `Delay-insensitive`, `Unknown`).
* **Timestamp Consistency (`vmdeleted >= vmcreated`)**: Passed (0 deletion-before-creation violations).

---

## D. Empirical Exploratory Data Analysis (EDA) Observations

Derived directly from loaded data sample (50,000 VMs):

1. **CPU Utilisation**:
   * **`avgcpu`**: Mean = 15.52%, Median = 8.14%, Std = 18.73%, P25 = 2.01%, P75 = 22.03%, P95 = 58.42%.
   * **`p95maxcpu`**: Mean = 46.91%, Median = 48.22%, P25 = 9.82%, P75 = 82.13%.
   * **Low Utilization Concentration**: 33.6% of sampled instances have `p95maxcpu < 20%`.
2. **VM Lifetime & Eligibility Filters**:
   * Derived formula: `lifetime = max(vmdeleted - vmcreated, 300) / 3600`.
   * Mean = 59.37 hours, Median = 0.50 hours (P10 = 0.08h, P50 = 0.50h, P90 = 73.83h, P99 = 719.83h).
   * **Rightsizing & Pricing Agent Filter (`eligible_rightsizing_pricing`, Threshold >= 24.0h default)**:
     * Retained: 6,472 VMs (12.94%) | Excluded (Short-lived <24h): 43,528 VMs (87.06%)
   * **Scheduling & Autoscaling Agent Filter (`scheduling_eligible`, Observation Window >= 48.0h default)**:
     * Retained: 5,446 VMs (10.89%) | Excluded (Short Window <48h): 44,554 VMs (89.11%)
3. **Synthetic Cloud Pricing Module (`src/data/pricing.py`)**:
   * Rate Card Citation: *Microsoft Azure Retail Prices API (Region: East US) & AWS EC2 Published Rates (Region: us-east-1), October 2024/2025.*
   * Rates anchored to real SKU families: Azure Standard_B2s, D2s_v5, E2s_v5, F4s_v2, D8s_v5, E8s_v5, F24s_v2 / AWS t3, m5, c5, r5 series.
   * Monthly Cost Breakdown (50,000 VM Sample):
     * **On-Demand Baseline**: $5,956,997.60/month (Mean: $119.14/month per VM)
     * **1-Year Reserved Instance**: $3,580,147.57/month (Potential Savings: 39.9%)
     * **Spot Instance**: $1,500,873.60/month (Potential Savings: 74.8%)
4. **VM Core Count Distribution**:
   * `2` cores: 58.54% | `4` cores: 30.82% | `8` cores: 7.31% | `24` cores: 2.95% | `>24` cores: 0.38%
5. **VM Memory Distribution (GB)**:
   * `8` GB: 37.50% | `32` GB: 31.82% | `4` GB: 15.08% | `2` GB: 12.14% | `64` GB: 3.08% | `>64` GB: 0.38%

---

## E. Initial Feature Relevance Assessment

| Feature | Rightsizing | Pricing | Autoscaling | Reason |
|---|---|---|---|---|
| `avgcpu` | **High** | Medium | **High** | Indicates mean workload utilization and baseline resource usage |
| `maxcpu` | **High** | Low | **High** | Measures peak capacity demand and spike headroom requirement |
| `p95maxcpu` | **High** | Low | **High** | Key indicator for overprovisioning; excludes single transient spikes |
| `vmcorecountbucket` | **High** | **High** | Medium | Current vCPU capacity baseline and unit cost multiplier |
| `vmmemorybucket` | **High** | **High** | Medium | Current RAM capacity baseline and SKU family determinant |
| `vmcategory` | Medium | **High** | **High** | Differentiates interactive services from delay-insensitive spot tasks |
| `lifetime` | Medium | **High** | **High** | Distinguishes short-lived jobs from steady-state reserved instance candidates |
| `on_demand_monthly_cost` | **High** | **High** | Medium | Financial baseline for calculating cost optimization gains |
| `mincpu/maxcpu time-series` | Medium | Low | **High** | 5-minute telemetry enables pattern forecasting and dynamic scaling triggers |

---

## F. Data Limitations & Gaps (To Be Modeled Synthetically)

1. **Dynamic RAM Telemetry**: Static RAM bucket size provided; time-series RAM utilization % must be synthetically correlated.
2. **SLA & Performance Constraints**: Maximum latency limits (ms), maximum acceptable CPU throttling %, interruption tolerance bounds.
3. **Operational & Penalty Costs**: VM resizing reboot downtime cost ($/downtime event), SLA breach penalty ($/hr).
4. **Governance Parameters**: Human-in-the-loop risk preference weights and approval thresholds.

---

## G. Unresolved Research Questions

1. **Conflict Resolution Mechanism**: How should the Negotiation/Arbitration Layer resolve competing proposals when Rightsizing recommends downsizing a VM, while Autoscaling requests capacity headroom for forecasted spikes?
2. **Explainability Formalism**: What structured template should agents use to produce human-comprehensible negotiation trade-off explanations?
3. **Synthetic Pricing Calibration**: How can we calibrate synthetic pricing matrices against real Azure/AWS pricing without introducing artificial optimization bias?
