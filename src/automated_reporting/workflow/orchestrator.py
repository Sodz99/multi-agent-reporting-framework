"""Automated reporting workflow orchestrator."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..agents.crew_orchestration import AutomatedReportingCrew
from ..data.fixtures import FixtureLoader
from .state_machine import AutomatedReportingWorkflow, ReportingWorkflowConfig
from ..rag.pipeline import BedrockRAGPipeline


@dataclass
class SimpleReportingWorkflowSettings:
    """Settings for the automated reporting workflow."""
    project_root: Path
    session_label: str | None = None
    use_rag: bool = True
    verbose: bool = True

    @property
    def data_dir(self) -> Path:
        return self.project_root / "fixtures"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"


class SimpleAutomatedReportingRunner:
    """Simple automated reporting runner for MVP prototype."""

    def __init__(self, settings: SimpleReportingWorkflowSettings) -> None:
        self.settings = settings

        # Initialize components
        self.rag_pipeline: Optional[BedrockRAGPipeline] = None
        self.crew: Optional[AutomatedReportingCrew] = None
        self.workflow: Optional[AutomatedReportingWorkflow] = None

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize workflow components."""

        # Initialize RAG pipeline if enabled
        if self.settings.use_rag:
            try:
                self.rag_pipeline = BedrockRAGPipeline(
                    collection_name="automated_reporting",
                    verbose=self.settings.verbose,
                )
                if self.settings.verbose:
                    print("[SimpleReporting] RAG pipeline initialized")
            except Exception as e:
                if self.settings.verbose:
                    print(f"[SimpleReporting] RAG pipeline initialization failed: {e}")
                self.rag_pipeline = None

        # Initialize CrewAI agents
        try:
            self.crew = AutomatedReportingCrew(verbose=self.settings.verbose)
            if self.settings.verbose:
                print("[SimpleReporting] CrewAI crew initialized")
        except Exception as e:
            if self.settings.verbose:
                print(f"[SimpleReporting] CrewAI initialization failed: {e}")
            self.crew = None

        # Initialize LangGraph workflow
        if self.crew:
            try:
                workflow_config = ReportingWorkflowConfig(
                    confidence_threshold=0.7,
                    quality_gate_threshold=0.8,
                    enable_rag=self.rag_pipeline is not None,
                    enable_validation=True,
                    enable_knowledge_capture=True,
                    verbose=self.settings.verbose,
                )
                self.workflow = AutomatedReportingWorkflow(
                    crew=self.crew,
                    rag_pipeline=self.rag_pipeline,
                    config=workflow_config,
                )
                if self.settings.verbose:
                    print("[SimpleReporting] LangGraph workflow initialized")
            except Exception as e:
                if self.settings.verbose:
                    print(f"[SimpleReporting] LangGraph workflow initialization failed: {e}")
                self.workflow = None

    def run(self) -> Dict:
        """Run the automated reporting workflow."""
        if self.settings.verbose:
            print("[SimpleReporting] Starting automated reporting workflow...")

        # Load fixtures
        loader = FixtureLoader(self.settings.data_dir)
        fixtures = list(loader.iter_fixtures())
        payloads = {}

        for fixture in fixtures:
            try:
                content = fixture.load()
                payloads[fixture.fixture_id] = content
                if self.settings.verbose:
                    print(f"[SimpleReporting] Loaded fixture: {fixture.fixture_id}")
            except Exception as e:
                if self.settings.verbose:
                    print(f"[SimpleReporting] Error loading {fixture.fixture_id}: {e}")

        # Load fixtures into RAG pipeline if available
        if self.rag_pipeline:
            try:
                stats = self.rag_pipeline.load_fixtures_to_vector_store(self.settings.data_dir)
                if self.settings.verbose:
                    print(f"[SimpleReporting] RAG: Loaded {stats.get('total_chunks', 0)} chunks")
            except Exception as e:
                if self.settings.verbose:
                    print(f"[SimpleReporting] RAG loading failed: {e}")

        # Prepare telemetry data
        telemetry_data = self._prepare_telemetry_data(payloads)

        # Prepare reporting requirements
        reporting_requirements = self._prepare_reporting_requirements()

        # Run workflow
        if self.workflow:
            try:
                workflow_results = self.workflow.run(telemetry_data, reporting_requirements)
                if self.settings.verbose:
                    print("[SimpleReporting] Workflow completed successfully")
                return workflow_results
            except Exception as e:
                if self.settings.verbose:
                    print(f"[SimpleReporting] Workflow error: {e}")
                return {"error": str(e), "telemetry_data": telemetry_data}
        else:
            # Fallback to basic crew execution
            if self.crew:
                try:
                    crew_results = self.crew.run_automated_reporting(telemetry_data, reporting_requirements)
                    if self.settings.verbose:
                        print("[SimpleReporting] Crew execution completed")
                    return crew_results
                except Exception as e:
                    if self.settings.verbose:
                        print(f"[SimpleReporting] Crew error: {e}")
                    return {"error": str(e), "telemetry_data": telemetry_data}
            else:
                return {"error": "No workflow components available", "telemetry_data": telemetry_data}

    def _prepare_telemetry_data(self, payloads: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare telemetry data from loaded fixtures."""
        telemetry_data = {}

        # Extract relevant metrics from fixtures
        for fixture_id, content in payloads.items():
            if isinstance(content, dict):
                # Look for metric data patterns
                if "metrics" in content:
                    telemetry_data[f"{fixture_id}_metrics"] = content["metrics"]
                elif "kpi" in fixture_id.lower():
                    telemetry_data[fixture_id] = content
                elif "performance" in fixture_id.lower():
                    telemetry_data[fixture_id] = content
                else:
                    # Try to extract numeric data
                    numeric_data = self._extract_numeric_data(content)
                    if numeric_data:
                        telemetry_data[fixture_id] = numeric_data

        # Add synthetic telemetry if no fixtures available
        if not telemetry_data:
            telemetry_data = self._generate_synthetic_telemetry()

        return telemetry_data

    def _extract_numeric_data(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract numeric metrics from content."""
        numeric_data = {}

        def extract_numbers(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    if isinstance(value, (int, float)):
                        numeric_data[new_key] = {
                            "value": value,
                            "quality": 1.0,
                            "timestamp": "2024-01-01T12:00:00Z"
                        }
                    elif isinstance(value, dict):
                        extract_numbers(value, new_key)

        extract_numbers(content)
        return numeric_data

    def _generate_synthetic_telemetry(self) -> Dict[str, Any]:
        """Generate synthetic telemetry data for demo purposes."""
        return {
            "response_time": {
                "value": 145.2,
                "quality": 0.95,
                "timestamp": "2024-01-01T12:00:00Z",
                "unit": "ms"
            },
            "throughput": {
                "value": 1250.0,
                "quality": 0.92,
                "timestamp": "2024-01-01T12:00:00Z",
                "unit": "requests/sec"
            },
            "error_rate": {
                "value": 0.3,
                "quality": 0.98,
                "timestamp": "2024-01-01T12:00:00Z",
                "unit": "percent"
            },
            "cpu_utilization": {
                "value": 68.5,
                "quality": 0.89,
                "timestamp": "2024-01-01T12:00:00Z",
                "unit": "percent"
            },
            "memory_usage": {
                "value": 72.1,
                "quality": 0.91,
                "timestamp": "2024-01-01T12:00:00Z",
                "unit": "percent"
            },
            "availability": {
                "value": 99.7,
                "quality": 1.0,
                "timestamp": "2024-01-01T12:00:00Z",
                "unit": "percent"
            }
        }

    def _prepare_reporting_requirements(self) -> Dict[str, Any]:
        """Prepare reporting requirements configuration."""
        return {
            "stakeholders": [
                {
                    "type": "executive",
                    "template": "executive_summary",
                    "format": "pdf",
                    "frequency": "weekly",
                    "focus": "business_impact"
                },
                {
                    "type": "operations",
                    "template": "operational_dashboard",
                    "format": "html",
                    "frequency": "daily",
                    "focus": "actionable_items"
                },
                {
                    "type": "customer",
                    "template": "service_status",
                    "format": "email",
                    "frequency": "on_incident",
                    "focus": "service_impact"
                }
            ],
            "sla_targets": {
                "response_time": {"threshold": 200, "unit": "ms"},
                "availability": {"threshold": 99.5, "unit": "percent"},
                "throughput": {"threshold": 1000, "unit": "requests/sec"}
            },
            "reporting_period": "last_24_hours",
            "confidence_threshold": 0.7,
            "quality_gate_threshold": 0.8,
            "delivery_channels": ["email", "dashboard", "api"],
            "compliance_requirements": ["SOC2", "GDPR"]
        }

    def get_status(self) -> Dict:
        """Get status of initialized components."""
        return {
            "rag_pipeline": self.rag_pipeline is not None,
            "crew": self.crew is not None,
            "workflow": self.workflow is not None,
        }


# Backward compatibility aliases
AutomatedReportingRunner = SimpleAutomatedReportingRunner
AutomatedReportingSettings = SimpleReportingWorkflowSettings

__all__ = [
    "SimpleAutomatedReportingRunner",
    "SimpleReportingWorkflowSettings",
    "AutomatedReportingRunner",
    "AutomatedReportingSettings",
]