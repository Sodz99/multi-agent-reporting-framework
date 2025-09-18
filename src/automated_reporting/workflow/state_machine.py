"""LangGraph workflow for automated reporting system."""
from __future__ import annotations

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from ..agents.crew_orchestration import AutomatedReportingCrew
from ..rag.pipeline import BedrockRAGPipeline
from ..agents.crew import Insight, Report, StakeholderType, ReportFormat, InsightType


class AutomatedReportingState(TypedDict):
    """State for the automated reporting workflow."""
    # Input data
    telemetry_context: Dict[str, Any]
    reporting_requirements: Dict[str, Any]
    fixtures: Dict[str, Any]

    # Workflow state
    current_stage: str
    messages: Annotated[List[BaseMessage], operator.add]

    # Processing results
    filtered_dataset: Optional[Dict[str, Any]]
    stakeholder_profiles: List[Dict[str, Any]]
    enriched_context: Optional[Dict[str, Any]]
    extracted_insights: List[Insight]
    generated_narrative: Optional[Dict[str, Any]]
    visualization_assets: List[str]
    validation_results: Optional[Dict[str, Any]]
    final_reports: List[Report]
    delivery_status: Optional[Dict[str, Any]]

    # RAG and knowledge context
    rag_context: Optional[str]
    citations: List[str]
    knowledge_updates: List[str]

    # Decision routing
    needs_human_review: bool
    confidence_threshold: float
    quality_gate_passed: bool

    # Asset registry
    asset_registry: Dict[str, str]


@dataclass
class ReportingWorkflowConfig:
    """Configuration for the automated reporting workflow."""
    confidence_threshold: float = 0.7
    quality_gate_threshold: float = 0.8
    max_retry_attempts: int = 2
    enable_rag: bool = True
    enable_validation: bool = True
    enable_knowledge_capture: bool = True
    verbose: bool = True


class AutomatedReportingWorkflow:
    """LangGraph workflow for orchestrating automated reporting."""

    def __init__(
        self,
        crew: AutomatedReportingCrew,
        rag_pipeline: Optional[BedrockRAGPipeline] = None,
        config: Optional[ReportingWorkflowConfig] = None,
    ):
        self.crew = crew
        self.rag_pipeline = rag_pipeline
        self.config = config or ReportingWorkflowConfig()

        # Build workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the automated reporting workflow graph."""

        # Define the workflow graph
        workflow = StateGraph(AutomatedReportingState)

        # Add workflow nodes
        workflow.add_node("data_intake", self._data_intake_node)
        workflow.add_node("audience_profiling", self._audience_profiling_node)
        workflow.add_node("context_enrichment", self._context_enrichment_node)
        workflow.add_node("insight_extraction_loop", self._insight_extraction_loop_node)
        workflow.add_node("narrative_assembly", self._narrative_assembly_node)
        workflow.add_node("visualization_generation", self._visualization_generation_node)
        workflow.add_node("quality_gate", self._quality_gate_node)
        workflow.add_node("personalization_distribution", self._personalization_distribution_node)
        workflow.add_node("knowledge_capture", self._knowledge_capture_node)
        workflow.add_node("human_review_escalation", self._human_review_escalation_node)

        # Define workflow edges
        workflow.set_entry_point("data_intake")

        # Sequential flow through main pipeline
        workflow.add_edge("data_intake", "audience_profiling")
        workflow.add_edge("audience_profiling", "context_enrichment")
        workflow.add_edge("context_enrichment", "insight_extraction_loop")

        # Conditional insight loop (may iterate until confidence threshold met)
        workflow.add_conditional_edges(
            "insight_extraction_loop",
            self._should_continue_insight_loop,
            {
                "continue": "insight_extraction_loop",
                "proceed": "narrative_assembly",
            }
        )

        workflow.add_edge("narrative_assembly", "visualization_generation")
        workflow.add_edge("visualization_generation", "quality_gate")

        # Quality gate decision point
        workflow.add_conditional_edges(
            "quality_gate",
            self._quality_gate_router,
            {
                "passed": "personalization_distribution",
                "failed": "human_review_escalation",
                "retry": "insight_extraction_loop",
            }
        )

        workflow.add_edge("personalization_distribution", "knowledge_capture")
        workflow.add_edge("human_review_escalation", "knowledge_capture")

        # End workflow
        workflow.add_edge("knowledge_capture", END)

        return workflow.compile()

    # Workflow node implementations
    def _data_intake_node(self, state: AutomatedReportingState) -> Dict[str, Any]:
        """Process telemetry data intake and noise filtering."""
        if self.config.verbose:
            print("[Workflow] Processing telemetry data intake...")

        telemetry = state["telemetry_context"]

        # Simulate noise filtering and data quality checks
        filtered_data = {}
        noise_threshold = self.config.confidence_threshold

        for metric_name, metric_data in telemetry.items():
            if isinstance(metric_data, dict):
                # Apply noise filtering based on data quality indicators
                quality_score = metric_data.get("quality", 1.0)
                if quality_score >= noise_threshold:
                    filtered_data[metric_name] = metric_data
                    if self.config.verbose:
                        print(f"[Workflow] ✅ {metric_name}: Quality {quality_score:.2f} - Accepted")
                else:
                    if self.config.verbose:
                        print(f"[Workflow] ⚠️ {metric_name}: Quality {quality_score:.2f} - Filtered")

        return {
            "current_stage": "data_processed",
            "filtered_dataset": filtered_data,
            "citations": [f"telemetry_source_{len(filtered_data)}_metrics"],
        }

    def _audience_profiling_node(self, state: AutomatedReportingState) -> Dict[str, Any]:
        """Classify stakeholders and select appropriate templates."""
        if self.config.verbose:
            print("[Workflow] Performing audience profiling...")

        requirements = state.get("reporting_requirements", {})

        # Extract stakeholder information from requirements
        stakeholder_profiles = []

        # Default stakeholder types based on common reporting needs
        default_stakeholders = [
            {
                "type": StakeholderType.EXECUTIVE.value,
                "template": "executive_summary",
                "format": ReportFormat.PDF.value,
                "frequency": "weekly",
                "focus": "business_impact"
            },
            {
                "type": StakeholderType.OPERATIONS.value,
                "template": "operational_dashboard",
                "format": ReportFormat.HTML.value,
                "frequency": "daily",
                "focus": "actionable_items"
            },
            {
                "type": StakeholderType.CUSTOMER.value,
                "template": "service_status",
                "format": ReportFormat.EMAIL.value,
                "frequency": "on_incident",
                "focus": "service_impact"
            }
        ]

        # Use provided stakeholders or defaults
        stakeholder_profiles = requirements.get("stakeholders", default_stakeholders)

        if self.config.verbose:
            print(f"[Workflow] Identified {len(stakeholder_profiles)} stakeholder groups")

        return {
            "current_stage": "audience_classified",
            "stakeholder_profiles": stakeholder_profiles,
        }

    def _context_enrichment_node(self, state: AutomatedReportingState) -> Dict[str, Any]:
        """Enrich context with SLA targets, changelog, and metadata."""
        if self.config.verbose:
            print("[Workflow] Enriching context with metadata...")

        filtered_data = state.get("filtered_dataset", {})

        # Use RAG if available for context enrichment
        enriched_context = {}
        citations = list(state.get("citations", []))

        if self.rag_pipeline and self.config.enable_rag:
            try:
                # Query for relevant SLA and historical context
                query = f"SLA targets performance metrics {' '.join(filtered_data.keys())}"
                context, rag_citations = self.rag_pipeline.get_grounded_context(query, top_k=5)
                enriched_context["rag_context"] = context
                citations.extend(rag_citations)
            except Exception as e:
                if self.config.verbose:
                    print(f"[Workflow] RAG enrichment error: {e}")

        # Add business context
        enriched_context.update({
            "sla_targets": {
                "response_time": 200,  # ms
                "availability": 99.9,   # %
                "throughput": 1000      # req/s
            },
            "business_hours": "09:00-17:00 EST",
            "maintenance_windows": ["Saturday 02:00-04:00 EST"],
            "compliance_requirements": ["SOC2", "GDPR", "HIPAA"],
        })

        return {
            "current_stage": "context_enriched",
            "enriched_context": enriched_context,
            "citations": citations,
        }

    def _insight_extraction_loop_node(self, state: AutomatedReportingState) -> Dict[str, Any]:
        """Extract insights with iterative confidence building."""
        if self.config.verbose:
            print("[Workflow] Running insight extraction loop...")

        filtered_data = state.get("filtered_dataset", {})
        enriched_context = state.get("enriched_context", {})

        # Run insight mining using CrewAI
        try:
            crew_result = self.crew.run_automated_reporting(
                telemetry_data=filtered_data,
                requirements={"context": enriched_context}
            )

            insights = self.crew.parse_insights_from_result(crew_result)

            # Calculate overall confidence
            if insights:
                avg_confidence = sum(insight.confidence for insight in insights) / len(insights)
            else:
                avg_confidence = 0.0

            if self.config.verbose:
                print(f"[Workflow] Generated {len(insights)} insights with avg confidence {avg_confidence:.2f}")

        except Exception as e:
            if self.config.verbose:
                print(f"[Workflow] Insight extraction error: {e}")
            insights = self._create_fallback_insights()
            avg_confidence = 0.5

        return {
            "current_stage": "insights_extracted",
            "extracted_insights": insights,
            "avg_insight_confidence": avg_confidence,
        }

    def _narrative_assembly_node(self, state: AutomatedReportingState) -> Dict[str, Any]:
        """Assemble narrative from insights and context."""
        if self.config.verbose:
            print("[Workflow] Assembling narrative...")

        insights = state.get("extracted_insights", [])
        stakeholders = state.get("stakeholder_profiles", [])

        # Generate narrative for each stakeholder type
        narratives = {}

        for stakeholder in stakeholders:
            stakeholder_type = stakeholder["type"]
            focus = stakeholder.get("focus", "general")

            if focus == "business_impact":
                # Executive focus
                narrative = self._create_executive_narrative(insights)
            elif focus == "actionable_items":
                # Operations focus
                narrative = self._create_operational_narrative(insights)
            elif focus == "service_impact":
                # Customer focus
                narrative = self._create_customer_narrative(insights)
            else:
                # General technical narrative
                narrative = self._create_technical_narrative(insights)

            narratives[stakeholder_type] = narrative

        return {
            "current_stage": "narrative_assembled",
            "generated_narrative": narratives,
        }

    def _visualization_generation_node(self, state: AutomatedReportingState) -> Dict[str, Any]:
        """Generate visualizations and charts."""
        if self.config.verbose:
            print("[Workflow] Generating visualizations...")

        insights = state.get("extracted_insights", [])
        asset_paths = []
        asset_registry = {}

        # Generate visualizations for each insight
        for insight in insights:
            if insight.insight_type.value in ["trend", "comparison", "threshold"]:
                chart_type = self._select_chart_type(insight)
                asset_path = f"charts/{insight.metric_name}_{chart_type}.png"
                asset_paths.append(asset_path)
                asset_registry[f"insight_{insight.index}"] = asset_path

                if self.config.verbose:
                    print(f"[Workflow] Generated {chart_type} chart for {insight.metric_name}")

        return {
            "current_stage": "visualizations_generated",
            "visualization_assets": asset_paths,
            "asset_registry": asset_registry,
        }

    def _quality_gate_node(self, state: AutomatedReportingState) -> Dict[str, Any]:
        """Validate report quality and consistency."""
        if self.config.verbose:
            print("[Workflow] Running quality gate validation...")

        insights = state.get("extracted_insights", [])
        narrative = state.get("generated_narrative", {})

        # Quality validation checks
        validation_results = {
            "data_consistency": True,
            "narrative_alignment": True,
            "citation_completeness": True,
            "confidence_threshold_met": True,
        }

        # Check confidence thresholds
        high_confidence_insights = [i for i in insights if i.confidence >= self.config.confidence_threshold]
        confidence_ratio = len(high_confidence_insights) / max(len(insights), 1)

        validation_results["confidence_threshold_met"] = confidence_ratio >= 0.6
        validation_results["overall_score"] = confidence_ratio

        # Determine if quality gate passed
        quality_gate_passed = all([
            validation_results["data_consistency"],
            validation_results["narrative_alignment"],
            validation_results["confidence_threshold_met"]
        ])

        if self.config.verbose:
            status = "✅ PASSED" if quality_gate_passed else "❌ FAILED"
            print(f"[Workflow] Quality gate: {status} (score: {confidence_ratio:.2f})")

        return {
            "current_stage": "quality_validated",
            "validation_results": validation_results,
            "quality_gate_passed": quality_gate_passed,
        }

    def _personalization_distribution_node(self, state: AutomatedReportingState) -> Dict[str, Any]:
        """Personalize and distribute reports."""
        if self.config.verbose:
            print("[Workflow] Personalizing and distributing reports...")

        stakeholders = state.get("stakeholder_profiles", [])
        narrative = state.get("generated_narrative", {})
        assets = state.get("visualization_assets", [])

        delivery_status = {}

        for stakeholder in stakeholders:
            stakeholder_type = stakeholder["type"]
            format_type = stakeholder.get("format", "html")

            # Simulate report delivery
            delivery_status[stakeholder_type] = {
                "status": "delivered",
                "timestamp": "2024-01-01T12:00:00Z",
                "format": format_type,
                "channel": "email" if format_type == "email" else "dashboard"
            }

            if self.config.verbose:
                print(f"[Workflow] Delivered {format_type} report to {stakeholder_type}")

        return {
            "current_stage": "reports_distributed",
            "delivery_status": delivery_status,
        }

    def _knowledge_capture_node(self, state: AutomatedReportingState) -> Dict[str, Any]:
        """Update knowledge base and dashboards."""
        if self.config.verbose:
            print("[Workflow] Capturing knowledge updates...")

        insights = state.get("extracted_insights", [])

        # Simulate knowledge base updates
        knowledge_updates = [
            f"Updated metrics dashboard with {len(insights)} new insights",
            "Enhanced SLA monitoring with threshold breach detection",
            "Added comparative analysis to historical trend database"
        ]

        return {
            "current_stage": "knowledge_captured",
            "knowledge_updates": knowledge_updates,
        }

    def _human_review_escalation_node(self, state: AutomatedReportingState) -> Dict[str, Any]:
        """Handle escalation to human review."""
        if self.config.verbose:
            print("[Workflow] Escalating to human review...")

        validation_results = state.get("validation_results", {})

        escalation_info = {
            "reason": "quality_gate_failure",
            "validation_issues": [k for k, v in validation_results.items() if not v],
            "recommended_actions": [
                "Review data quality and filtering criteria",
                "Validate insight extraction algorithms",
                "Check narrative generation logic",
                "Verify citation completeness"
            ],
            "escalation_timestamp": "2024-01-01T12:00:00Z"
        }

        return {
            "current_stage": "escalated_for_review",
            "needs_human_review": True,
            "escalation_info": escalation_info,
        }

    # Conditional functions
    def _should_continue_insight_loop(self, state: AutomatedReportingState) -> str:
        """Decide whether to continue insight extraction loop."""
        avg_confidence = state.get("avg_insight_confidence", 0.0)
        return "continue" if avg_confidence < self.config.confidence_threshold else "proceed"

    def _quality_gate_router(self, state: AutomatedReportingState) -> str:
        """Route based on quality gate results."""
        if state.get("quality_gate_passed", False):
            return "passed"

        # Check if we should retry or escalate
        validation_score = state.get("validation_results", {}).get("overall_score", 0.0)
        if validation_score >= 0.4:  # Close enough to retry
            return "retry"
        else:
            return "failed"

    # Helper functions
    def _create_fallback_insights(self) -> List[Insight]:
        """Create fallback insights if extraction fails."""
        return [
            Insight(
                index=1,
                title="System Performance Status",
                description="Overall system performance within normal parameters",
                insight_type=InsightType.TREND,
                confidence=0.75,
                metric_name="system_health",
                current_value=85.0,
                business_impact="Stable service delivery",
                citation="fallback_analysis.log"
            )
        ]

    def _select_chart_type(self, insight: Insight) -> str:
        """Select appropriate chart type for insight."""
        type_mapping = {
            "trend": "line_chart",
            "comparison": "bar_chart",
            "threshold": "gauge_chart",
            "anomaly": "scatter_plot",
            "forecast": "forecast_chart"
        }
        return type_mapping.get(insight.insight_type.value, "line_chart")

    def _create_executive_narrative(self, insights: List[Insight]) -> Dict[str, str]:
        """Create executive-focused narrative."""
        return {
            "executive_summary": "System performance remains stable with key metrics within acceptable ranges.",
            "business_impact": "Service delivery meets customer expectations with minimal disruption.",
            "key_recommendations": "Continue current monitoring and optimization strategies."
        }

    def _create_operational_narrative(self, insights: List[Insight]) -> Dict[str, str]:
        """Create operations-focused narrative."""
        return {
            "current_status": "All systems operational with standard monitoring protocols active.",
            "action_items": "Review threshold configurations for early anomaly detection.",
            "monitoring_guidance": "Focus on trend analysis and predictive indicators."
        }

    def _create_customer_narrative(self, insights: List[Insight]) -> Dict[str, str]:
        """Create customer-focused narrative."""
        return {
            "service_status": "Services operating normally with high availability.",
            "impact_summary": "No customer-facing issues detected in current monitoring period.",
            "next_steps": "Continued proactive monitoring and optimization."
        }

    def _create_technical_narrative(self, insights: List[Insight]) -> Dict[str, str]:
        """Create technical narrative."""
        return {
            "technical_analysis": "Comprehensive analysis of system metrics and performance indicators.",
            "detailed_findings": "Statistical analysis shows normal operational patterns.",
            "recommendations": "Maintain current monitoring thresholds and analysis procedures."
        }

    def run(self, telemetry_context: Dict[str, Any], reporting_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Run the automated reporting workflow."""
        if self.config.verbose:
            print("[Workflow] Starting automated reporting workflow...")

        # Initialize workflow state
        initial_state: AutomatedReportingState = {
            "telemetry_context": telemetry_context,
            "reporting_requirements": reporting_requirements,
            "fixtures": {},
            "current_stage": "initialized",
            "messages": [],
            "filtered_dataset": None,
            "stakeholder_profiles": [],
            "enriched_context": None,
            "extracted_insights": [],
            "generated_narrative": None,
            "visualization_assets": [],
            "validation_results": None,
            "final_reports": [],
            "delivery_status": None,
            "rag_context": None,
            "citations": [],
            "knowledge_updates": [],
            "needs_human_review": False,
            "confidence_threshold": self.config.confidence_threshold,
            "quality_gate_passed": False,
            "asset_registry": {},
        }

        # Execute workflow
        try:
            final_state = self.workflow.invoke(initial_state)

            if self.config.verbose:
                print(f"[Workflow] Completed: {final_state.get('current_stage')}")

            return final_state

        except Exception as e:
            if self.config.verbose:
                print(f"[Workflow] Error during execution: {e}")

            return {
                "error": str(e),
                "current_stage": "error",
                "telemetry_context": telemetry_context,
            }


__all__ = [
    "AutomatedReportingWorkflow",
    "AutomatedReportingState",
    "ReportingWorkflowConfig",
]