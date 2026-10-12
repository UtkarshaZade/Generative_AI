"""
Base Agent and Proposal Data Structures.
Defines the unified Proposal schema and BaseAgent abstract class.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd


@dataclass
class Proposal:
    """
    Unified Proposal schema output by optimization agents.
    
    Attributes:
        vm_id: Anonymized identifier of target VM.
        agent_name: Name of the originating agent.
        action_type: Category of proposed action (DOWNSIZE_SKU, SWITCH_PRICING_MODEL, SCHEDULE_OFF_PEAK_PAUSE).
        current_configuration: Dictionary describing baseline SKU, rate, and cost.
        proposed_configuration: Dictionary describing target SKU, rate, model, or schedule.
        predicted_monthly_savings: Estimated monetary savings in $/month compared to baseline On-Demand cost.
        confidence: Confidence score in range [0.0, 1.0].
        rationale: Explainable natural language rationale.
    """
    vm_id: str
    agent_name: str
    action_type: str
    current_configuration: Dict[str, Any]
    proposed_configuration: Dict[str, Any]
    predicted_monthly_savings: float
    confidence: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vm_id": self.vm_id,
            "agent_name": self.agent_name,
            "action_type": self.action_type,
            "current_configuration": self.current_configuration,
            "proposed_configuration": self.proposed_configuration,
            "predicted_monthly_savings": round(float(self.predicted_monthly_savings), 2),
            "confidence": round(float(self.confidence), 3),
            "rationale": self.rationale
        }


class BaseAgent:
    """Abstract Base Class for optimization agents."""
    
    def __init__(self, name: str):
        self.name = name

    def generate_proposals(
        self,
        df: pd.DataFrame,
        cpu_readings_df: Optional[pd.DataFrame] = None
    ) -> List[Proposal]:
        """Generate optimization proposals for eligible VMs in dataset."""
        raise NotImplementedError("Subclasses must implement generate_proposals")
