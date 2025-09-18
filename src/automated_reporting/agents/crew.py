"""Crew scripts and data models for the automated reporting system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Dict, Any, Optional
from enum import Enum


class StakeholderType(Enum):
    """Types of report stakeholders."""
    EXECUTIVE = "executive"
    OPERATIONS = "operations"
    CUSTOMER = "customer"
    ENGINEERING = "engineering"


class ReportFormat(Enum):
    """Available report formats."""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    EMAIL = "email"


class InsightType(Enum):
    """Types of insights that can be generated."""
    TREND = "trend"
    ANOMALY = "anomaly"
    THRESHOLD = "threshold"
    COMPARISON = "comparison"
    FORECAST = "forecast"


@dataclass
class Stakeholder:
    """Represents a report stakeholder."""
    id: str
    name: str
    type: StakeholderType
    email: str
    preferred_format: ReportFormat
    notification_frequency: str  # daily, weekly, on_alert, etc.
    sla_requirements: Optional[Dict[str, Any]] = None


@dataclass
class Insight:
    """Represents an analytical insight from the data."""
    index: int
    title: str
    description: str
    insight_type: InsightType
    confidence: float
    metric_name: str
    current_value: float
    baseline_value: Optional[float] = None
    threshold_breached: bool = False
    business_impact: str = ""
    recommended_action: str = ""
    supporting_data: Optional[Dict[str, Any]] = None
    citation: str = ""


@dataclass
class Report:
    """Represents a generated report."""
    report_id: str
    title: str
    generated_at: str
    stakeholder_type: StakeholderType
    format: ReportFormat
    insights: List[Insight]
    executive_summary: str
    detailed_analysis: str
    recommendations: List[str]
    visualizations: List[str]  # paths to generated charts
    metadata: Dict[str, Any]
    delivery_status: str = "pending"


@dataclass
class DeliveryStatus:
    """Tracks report delivery status."""
    report_id: str
    stakeholder_id: str
    delivery_method: str
    status: str  # pending, sent, delivered, failed
    attempted_at: Optional[str] = None
    delivered_at: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ReportMessage:
    """Represents a message in the reporting workflow transcript."""
    speaker: str
    text: str
    timestamp: Optional[str] = None

    def render(self) -> str:
        prefix = f"[{self.timestamp}] " if self.timestamp else ""
        return f"{prefix}[{self.speaker}] {self.text}"


class ReportTranscriptBuilder:
    """Constructs a deterministic transcript for the automated reporting workflow."""

    def build(self, insights: Sequence[Insight], stakeholder_type: StakeholderType, seed_data_id: str) -> List[str]:
        """Build a workflow transcript for the reporting process."""
        messages: List[ReportMessage] = [
            ReportMessage("Telemetry Listener", f"Processing KPIs and events from data source {seed_data_id}"),
            ReportMessage("Telemetry Listener", "Noise filter applied → clean dataset prepared for analysis"),
            ReportMessage("Stakeholder Classifier", f"Target audience identified: {stakeholder_type.value}"),
        ]

        # Add template selection message
        if stakeholder_type == StakeholderType.EXECUTIVE:
            messages.append(ReportMessage("Stakeholder Classifier", "Executive template selected → high-level summary focus"))
        elif stakeholder_type == StakeholderType.OPERATIONS:
            messages.append(ReportMessage("Stakeholder Classifier", "Operations template selected → actionable details focus"))
        elif stakeholder_type == StakeholderType.CUSTOMER:
            messages.append(ReportMessage("Stakeholder Classifier", "Customer template selected → impact communication focus"))
        else:
            messages.append(ReportMessage("Stakeholder Classifier", "Technical template selected → detailed analysis focus"))

        # Context enrichment
        messages.append(ReportMessage("Context Enricher", "Gathering SLA targets, changelog, and historical context"))
        messages.append(ReportMessage("Context Enricher", "Business context enrichment complete → metadata stored"))

        # Insight mining
        insight_count = len(insights)
        messages.append(ReportMessage("Insight Miner", f"Analyzing {insight_count} metrics for trends and anomalies"))

        for insight in insights:
            confidence_indicator = "✓" if insight.confidence >= 0.8 else "⚠" if insight.confidence >= 0.6 else "?"
            impact_level = "HIGH" if insight.threshold_breached else "MEDIUM" if insight.confidence >= 0.7 else "LOW"

            messages.append(
                ReportMessage(
                    "Insight Miner",
                    f"Insight {insight.index}: {insight.title} ({insight.confidence:.2f} confidence) {confidence_indicator} Impact: {impact_level}"
                )
            )

            # Add validation messages for key insights
            if insight.insight_type == InsightType.ANOMALY and insight.confidence >= 0.8:
                messages.append(
                    ReportMessage(
                        "Validator: trend_analyzer",
                        f"Anomaly confirmed: {insight.metric_name} deviation from baseline"
                    )
                )
            elif insight.insight_type == InsightType.THRESHOLD and insight.threshold_breached:
                messages.append(
                    ReportMessage(
                        "Validator: threshold_monitor",
                        f"SLA breach detected: {insight.metric_name} exceeds threshold"
                    )
                )

        # Report writing
        messages.append(ReportMessage("Report Writer", "Assembling narrative sections from insights and context"))
        messages.append(ReportMessage("Visualization Bot", "Generating charts and visual assets for key metrics"))
        messages.append(ReportMessage("Visualization Bot", "Asset registry updated → chart paths stored"))

        # Quality validation
        messages.append(ReportMessage("Consistency Checker", "Cross-validating numbers, units, and narrative consistency"))

        # Check if high-confidence insights exist
        high_confidence_insights = [i for i in insights if i.confidence >= 0.8]
        if high_confidence_insights:
            messages.append(ReportMessage("Consistency Checker", "Quality gate passed ✓ → report approved for delivery"))
        else:
            messages.append(ReportMessage("Consistency Checker", "Low confidence detected → flagging for human review"))

        # Delivery orchestration
        messages.append(ReportMessage("Delivery Orchestrator", f"Personalizing content for {stakeholder_type.value} audience"))
        messages.append(ReportMessage("Delivery Orchestrator", "Scheduling delivery → notification queued"))
        messages.append(ReportMessage("Delivery Orchestrator", "Delivery status logged for audit trail"))

        # Knowledge capture
        messages.append(ReportMessage("KB Logger", "Updating dashboards with latest insights"))
        messages.append(ReportMessage("KB Logger", "Knowledge base updated → available for future comparisons"))

        return [message.render() for message in messages]


class ReportingWorkflowState:
    """Manages state for the automated reporting workflow."""

    def __init__(self):
        self.raw_telemetry: Dict[str, Any] = {}
        self.filtered_dataset: Dict[str, Any] = {}
        self.stakeholder_profile: Optional[Stakeholder] = None
        self.context_metadata: Dict[str, Any] = {}
        self.insights: List[Insight] = []
        self.generated_charts: List[str] = []
        self.report_draft: Optional[str] = None
        self.validation_results: Dict[str, Any] = {}
        self.final_reports: List[Report] = []
        self.delivery_statuses: List[DeliveryStatus] = []
        self.knowledge_updates: List[str] = []

    def add_insight(self, insight: Insight) -> None:
        """Add an insight to the workflow state."""
        self.insights.append(insight)

    def get_high_confidence_insights(self, threshold: float = 0.8) -> List[Insight]:
        """Get insights above the confidence threshold."""
        return [insight for insight in self.insights if insight.confidence >= threshold]

    def get_insights_by_type(self, insight_type: InsightType) -> List[Insight]:
        """Get insights of a specific type."""
        return [insight for insight in self.insights if insight.insight_type == insight_type]

    def has_threshold_breaches(self) -> bool:
        """Check if any insights indicate threshold breaches."""
        return any(insight.threshold_breached for insight in self.insights)


__all__ = [
    "ReportTranscriptBuilder",
    "ReportMessage",
    "Report",
    "Insight",
    "Stakeholder",
    "DeliveryStatus",
    "ReportingWorkflowState",
    "StakeholderType",
    "ReportFormat",
    "InsightType"
]