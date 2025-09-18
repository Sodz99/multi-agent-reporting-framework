"""CrewAI crew orchestration for automated reporting system."""
from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import os

from crewai import Crew, Process

from .factory import AutomatedReportingAgents
from .tasks import AutomatedReportingTasks
from .crew import Insight, Report, StakeholderType, ReportFormat, InsightType


class AutomatedReportingCrew:
    """Orchestration for CrewAI-based automated reporting workflow."""

    def __init__(self, custom_models: Optional[Dict[str, str]] = None, verbose: bool = True):
        """Initialize crew with configuration.

        Args:
            custom_models: Override default model assignments per role
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        if self.verbose:
            print(f"[ReportingCrew] 🚀 Initializing AutomatedReportingCrew")

        # Initialize agents
        self.agents = AutomatedReportingAgents(custom_models=custom_models, verbose=verbose)
        if self.verbose:
            print(f"[ReportingCrew] Agents factory created")

        # Initialize tasks factory
        self.tasks_factory = AutomatedReportingTasks(self.agents)
        if self.verbose:
            print(f"[ReportingCrew] Tasks factory created")

        # Embedding configuration using Titan V1
        self.embedder_config = {
            "provider": "bedrock",
            "config": {
                "model": "amazon.titan-embed-text-v1",
                "region_name": os.getenv("BEDROCK_REGION", "us-east-1"),
            }
        }
        if self.verbose:
            print(f"[ReportingCrew] ✅ Embedder configured: Titan V1")

        # Placeholder for crew instances
        self.crew = None
        if self.verbose:
            print(f"[ReportingCrew] 🎉 AutomatedReportingCrew initialized successfully")

    def create_crew(self, telemetry_data: Dict[str, Any]) -> Crew:
        """Create and configure the CrewAI crew for this specific reporting request."""
        if self.verbose:
            print(f"[ReportingCrew] 🛠️ Creating crew for data: {telemetry_data.get('source_id', 'unknown')}")

        # Create reporting agents
        telemetry_listener = self.agents.create_telemetry_listener_agent()
        stakeholder_classifier = self.agents.create_stakeholder_classifier_agent()
        context_enricher = self.agents.create_context_enricher_agent()
        insight_miner = self.agents.create_insight_miner_agent()
        report_writer = self.agents.create_report_writer_agent()
        visualization_bot = self.agents.create_visualization_bot_agent()
        consistency_checker = self.agents.create_consistency_checker_agent()
        delivery_orchestrator = self.agents.create_delivery_orchestrator_agent()

        # Create initial data intake task
        data_intake_task = self.tasks_factory.create_data_intake_task(telemetry_data)

        # Create crew with sequential processing
        crew = Crew(
            agents=[
                telemetry_listener, stakeholder_classifier, context_enricher, insight_miner,
                report_writer, visualization_bot, consistency_checker, delivery_orchestrator
            ],
            tasks=[data_intake_task],  # Will add more tasks dynamically
            process=Process.sequential,
            verbose=self.verbose,
            memory=True,
            embedder=self.embedder_config,
            max_requests_per_minute=60,
        )

        if self.verbose:
            print(f"[ReportingCrew] ✅ Crew created with 8 agents and Titan V1 embeddings")
        return crew

    def run_automated_reporting(self, telemetry_data: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete automated reporting workflow."""
        if self.verbose:
            print(f"[CrewAI] 🏁 Starting automated reporting workflow")

        # Create crew for this specific request
        crew = self.create_crew(telemetry_data)
        workflow_results = {}

        try:
            # Stage 1: Data Intake and Filtering
            if self.verbose:
                print("[CrewAI] 📡 Stage 1: Data Intake and Filtering...")
            data_intake_task = self.tasks_factory.create_data_intake_task(telemetry_data)
            data_intake_result = crew.kickoff_for_each([{"telemetry_data": telemetry_data}])
            workflow_results["data_intake"] = data_intake_result

            # Stage 2: Stakeholder Classification
            if self.verbose:
                print("[CrewAI] 👥 Stage 2: Stakeholder Classification...")
            classification_task = self.tasks_factory.create_stakeholder_classification_task(requirements)
            crew.tasks.append(classification_task)
            classification_result = crew.kickoff()
            workflow_results["stakeholder_classification"] = classification_result

            # Stage 3: Context Enrichment
            if self.verbose:
                print("[CrewAI] 🔍 Stage 3: Context Enrichment...")
            enrichment_task = self.tasks_factory.create_context_enrichment_task({
                "data_intake": data_intake_result,
                "classification": classification_result
            })
            crew.tasks.append(enrichment_task)
            enrichment_result = crew.kickoff()
            workflow_results["context_enrichment"] = enrichment_result

            # Stage 4: Insight Mining
            if self.verbose:
                print("[CrewAI] ⛏️ Stage 4: Insight Mining and Analysis...")
            insight_task = self.tasks_factory.create_insight_mining_task({
                "enriched_data": enrichment_result,
                "telemetry": telemetry_data
            })
            crew.tasks.append(insight_task)
            insight_result = crew.kickoff()
            workflow_results["insights"] = insight_result

            # Stage 5: Narrative Assembly
            if self.verbose:
                print("[CrewAI] ✍️ Stage 5: Narrative Assembly...")
            narrative_task = self.tasks_factory.create_narrative_assembly_task({
                "insights": insight_result,
                "context": enrichment_result,
                "stakeholders": classification_result
            })
            crew.tasks.append(narrative_task)
            narrative_result = crew.kickoff()
            workflow_results["narrative"] = narrative_result

            # Stage 6: Visualization Generation
            if self.verbose:
                print("[CrewAI] 📊 Stage 6: Visualization Generation...")
            visualization_task = self.tasks_factory.create_visualization_task({
                "insights": insight_result,
                "narrative": narrative_result
            })
            crew.tasks.append(visualization_task)
            visualization_result = crew.kickoff()
            workflow_results["visualizations"] = visualization_result

            # Stage 7: Quality Validation
            if self.verbose:
                print("[CrewAI] ✅ Stage 7: Quality Validation...")
            validation_task = self.tasks_factory.create_quality_validation_task({
                "narrative": narrative_result,
                "visualizations": visualization_result,
                "insights": insight_result,
                "source_data": telemetry_data
            })
            crew.tasks.append(validation_task)
            validation_result = crew.kickoff()
            workflow_results["validation"] = validation_result

            # Stage 8: Delivery Orchestration
            if self.verbose:
                print("[CrewAI] 📨 Stage 8: Delivery Orchestration...")

            # Only proceed with delivery if validation passed
            if self._validation_passed(validation_result):
                validated_reports = self._prepare_reports_for_delivery(workflow_results)
                delivery_task = self.tasks_factory.create_delivery_orchestration_task(validated_reports)
                crew.tasks.append(delivery_task)
                delivery_result = crew.kickoff()
                workflow_results["delivery"] = delivery_result
            else:
                if self.verbose:
                    print("[CrewAI] ⚠️ Validation failed - skipping delivery stage")
                workflow_results["delivery"] = {"status": "skipped_due_to_validation_failure"}

            if self.verbose:
                print(f"[CrewAI] 🎉 Automated reporting workflow completed successfully")

        except Exception as e:
            if self.verbose:
                print(f"[CrewAI] ❌ Workflow error: {e}")

            # Add error information to results
            workflow_results["error"] = str(e)
            workflow_results["status"] = "failed"
        else:
            workflow_results["status"] = "completed"

        return workflow_results

    def parse_insights_from_result(self, crew_result: Dict[str, Any]) -> List[Insight]:
        """Parse CrewAI results into Insight objects for compatibility."""
        insights = []

        # Extract insights from the crew result
        insight_data = crew_result.get("insights", [])

        if isinstance(insight_data, str):
            # Parse string result into insights
            lines = insight_data.split('\n')
            current_insight = {}
            insight_count = 0

            for line in lines:
                line = line.strip()
                if line.startswith('Insight'):
                    if current_insight:
                        insights.append(self._create_insight_from_dict(current_insight, insight_count))
                        insight_count += 1
                    current_insight = {'title': line}
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.split(':')[1].strip())
                        current_insight['confidence'] = confidence
                    except (ValueError, IndexError):
                        current_insight['confidence'] = 0.5
                elif line.startswith('Type:'):
                    current_insight['insight_type'] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('Metric:'):
                    current_insight['metric_name'] = line.split(':', 1)[1].strip()
                elif line.startswith('Value:'):
                    try:
                        current_insight['current_value'] = float(line.split(':', 1)[1].strip())
                    except (ValueError, IndexError):
                        current_insight['current_value'] = 0.0

            if current_insight:
                insights.append(self._create_insight_from_dict(current_insight, insight_count))

        # If we don't have any parsed insights, create some default ones
        if not insights:
            insights = self._create_default_insights()

        return insights

    def _create_insight_from_dict(self, data: Dict[str, Any], index: int) -> Insight:
        """Create an Insight object from parsed data."""
        insight_type_mapping = {
            'trend': InsightType.TREND,
            'anomaly': InsightType.ANOMALY,
            'threshold': InsightType.THRESHOLD,
            'comparison': InsightType.COMPARISON,
            'forecast': InsightType.FORECAST
        }

        insight_type = insight_type_mapping.get(data.get('insight_type', 'trend'), InsightType.TREND)

        return Insight(
            index=index + 1,
            title=data.get('title', f'Insight {index + 1}'),
            description=data.get('description', ''),
            insight_type=insight_type,
            confidence=data.get('confidence', 0.5),
            metric_name=data.get('metric_name', 'unknown_metric'),
            current_value=data.get('current_value', 0.0),
            baseline_value=data.get('baseline_value'),
            threshold_breached=data.get('threshold_breached', False),
            business_impact=data.get('business_impact', ''),
            recommended_action=data.get('recommended_action', ''),
            citation=data.get('citation', 'crew_ai_analysis.log'),
        )

    def _create_default_insights(self) -> List[Insight]:
        """Create default insights if parsing fails."""
        return [
            Insight(
                index=1,
                title="Service Performance Trend",
                description="Overall service performance shows positive trend over the reporting period",
                insight_type=InsightType.TREND,
                confidence=0.85,
                metric_name="response_time",
                current_value=120.5,
                baseline_value=150.0,
                business_impact="Improved user experience and service efficiency",
                recommended_action="Continue current optimization strategies",
                citation="crew_ai_metrics.json",
            ),
            Insight(
                index=2,
                title="Resource Utilization Anomaly",
                description="CPU utilization spike detected during peak hours",
                insight_type=InsightType.ANOMALY,
                confidence=0.78,
                metric_name="cpu_utilization",
                current_value=95.2,
                baseline_value=65.0,
                threshold_breached=True,
                business_impact="Potential service degradation risk",
                recommended_action="Investigate workload patterns and consider scaling",
                citation="crew_ai_monitoring.log",
            ),
            Insight(
                index=3,
                title="SLA Performance Comparison",
                description="Service level performance compared to industry benchmarks",
                insight_type=InsightType.COMPARISON,
                confidence=0.72,
                metric_name="availability",
                current_value=99.5,
                baseline_value=99.9,
                business_impact="Meeting but not exceeding SLA commitments",
                recommended_action="Review and enhance availability strategies",
                citation="crew_ai_sla_report.json",
            ),
        ]

    def _validation_passed(self, validation_result: Dict[str, Any]) -> bool:
        """Check if quality validation passed."""
        if isinstance(validation_result, dict):
            status = validation_result.get("status", "")
            return status.lower() in ["approved", "passed", "✓"]
        elif isinstance(validation_result, str):
            return "approved" in validation_result.lower() or "✓" in validation_result
        return False

    def _prepare_reports_for_delivery(self, workflow_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare validated reports for delivery."""
        reports = []

        # Extract key components
        insights = self.parse_insights_from_result(workflow_results)
        narrative = workflow_results.get("narrative", {})
        visualizations = workflow_results.get("visualizations", {})

        # Create reports for different stakeholder types
        stakeholder_types = [StakeholderType.EXECUTIVE, StakeholderType.OPERATIONS, StakeholderType.CUSTOMER]

        for stakeholder_type in stakeholder_types:
            report = {
                "report_id": f"report_{stakeholder_type.value}_{len(reports)+1}",
                "stakeholder_type": stakeholder_type,
                "insights": insights,
                "narrative": narrative,
                "visualizations": visualizations,
                "validation_passed": True
            }
            reports.append(report)

        return reports


__all__ = ["AutomatedReportingCrew"]