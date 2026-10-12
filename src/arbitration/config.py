"""
Arbitration Layer Configuration & Risk Penalties.
"""

from typing import Dict, Any

# Risk penalties in $/month by VM category
# Interactive: high penalty ($50/mo), Unknown: medium penalty ($20/mo), Delay-insensitive: low penalty ($5/mo)
RISK_PENALTY_BY_CATEGORY: Dict[str, float] = {
    "Interactive": 50.0,
    "Unknown": 20.0,
    "Delay-insensitive": 5.0,
}

# Maximum allowed projected P95 CPU percentage after downsizing
MAX_PROJECTED_P95_CPU: float = 70.0

# Score difference percentage threshold for triggering human escalation (5.0%)
ESCALATION_MARGIN_PCT: float = 0.05
