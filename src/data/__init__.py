"""
Data loading, validation, preprocessing, pricing, and analysis module.
"""

from .loader import load_vmtable, load_cpu_readings, load_schema, get_available_dataset_files
from .validator import validate_vmtable, validate_cpu_readings
from .preprocessing import (
    add_derived_lifetime,
    map_core_memory_buckets,
    apply_lifetime_eligibility_filter,
    get_agent_specific_dataset,
)
from .pricing import (
    RATE_CARD,
    get_instance_pricing,
    add_pricing_columns,
    get_pricing_summary,
)
from .analysis import (
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

__all__ = [
    "load_vmtable",
    "load_cpu_readings",
    "load_schema",
    "get_available_dataset_files",
    "validate_vmtable",
    "validate_cpu_readings",
    "add_derived_lifetime",
    "map_core_memory_buckets",
    "apply_lifetime_eligibility_filter",
    "get_agent_specific_dataset",
    "RATE_CARD",
    "get_instance_pricing",
    "add_pricing_columns",
    "get_pricing_summary",
    "get_dataset_overview",
    "analyze_cpu_utilisation",
    "analyze_lifetime",
    "analyze_category",
    "analyze_memory_buckets",
    "analyze_core_buckets",
    "analyze_time_series_cpu",
    "get_feature_relevance_matrix",
    "get_data_gaps",
    "get_deterministic_observations",
]
