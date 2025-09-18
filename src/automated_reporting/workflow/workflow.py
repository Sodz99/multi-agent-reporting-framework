"""Deterministic workflow implementation for the automated reporting demo."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from ..shared.console import SessionConsole
from ..shared.files import ensure_dir, write_lines

from ..artifacts.generators import (
    write_telemetry_snapshot,
    write_stakeholder_profiles,
    write_insight_board,
    write_report_bundle,
    write_visualization_assets,
    write_delivery_log,
    write_knowledge_updates,
    write_manifest,
    write_synthetic_data,
)
from ..agents.crew import ReportTranscriptBuilder, Insight, StakeholderType, InsightType
from ..data.fixtures import Fixture, FixtureLoader


@dataclass
class AutomatedReportingSettings:
    project_root: Path
    seed: int = 4242
    session_label: str | None = None
    demo: bool = True

    @property
    def data_dir(self) -> Path:
        return self.project_root / "fixtures"

    @property
    def artifacts_root(self) -> Path:
        return self.project_root / "outputs"


class AutomatedReportingRunner:
    def __init__(self, settings: AutomatedReportingSettings) -> None:
        self.settings = settings
        self.rng = random.Random(settings.seed)

    def run(self) -> Path:
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        session_name = f"{timestamp}"
        if self.settings.session_label:
            session_name = f"{session_name}_{self.settings.session_label}"

        session_dir = ensure_dir(self.settings.artifacts_root / "sessions" / session_name)
        console = SessionConsole()
        crew = ["Telemetry Listener", "Stakeholder Classifier", "Context Enricher",
                "Insight Miner", "Report Writer", "Visualization Bot",
                "Consistency Checker", "Delivery Orchestrator"]
        console.banner("Automated Reporting", crew, self.settings.seed, str(session_dir))
        console.stage_update("Launch", "Completed")

        loader = FixtureLoader(self.settings.data_dir)
        fixtures = list(loader.iter_fixtures())

        payloads: Dict[str, object] = {}
        rag_index: List[Dict[str, str]] = []
        console.stage_update("Data Intake", "Started")

        for fixture in fixtures:
            console.fixture_progress(fixture.fixture_id, fixture.description)
            content = fixture.load()
            payloads[fixture.fixture_id] = content
            rag_index.append(
                {
                    "id": fixture.fixture_id,
                    "path": str(fixture.path),
                    "description": fixture.description,
                    "preview": self._preview(content),
                }
            )

        # Create telemetry snapshot
        telemetry = self._prepare_telemetry_data(payloads)
        telemetry_path = write_telemetry_snapshot(session_dir, telemetry)
        console.stage_update("Data Intake", "Completed")

        # Audience profiling
        console.stage_update("Audience Profiling", "Started")
        stakeholder_profiles = self._build_stakeholder_profiles()
        profiles_path = write_stakeholder_profiles(session_dir, stakeholder_profiles)
        console.stage_update("Audience Profiling", "Completed")

        # Context enrichment
        console.stage_update("Context Enrichment", "Started")
        enriched_context = self._build_enriched_context(payloads)
        console.stage_update("Context Enrichment", "Completed")

        # Insight extraction
        console.stage_update("Insight Extraction Loop", "Started")
        insights = self._build_insights(telemetry, enriched_context)
        insight_board_path = write_insight_board(session_dir, insights)

        # Generate transcript
        seed_fixture = fixtures[0].fixture_id if fixtures else "telemetry_sample_001"
        transcript_lines = ReportTranscriptBuilder().build(
            insights, StakeholderType.EXECUTIVE, seed_fixture
        )
        console.stream(transcript_lines)
        console.stage_update("Insight Extraction Loop", "Completed")

        # Narrative assembly & visualization
        console.stage_update("Narrative Assembly", "Started")
        report_narratives = self._build_report_narratives(insights, stakeholder_profiles)
        report_paths = write_report_bundle(session_dir, report_narratives, insights)
        console.stage_update("Narrative Assembly", "Completed")

        console.stage_update("Visualization Generation", "Started")
        visualization_assets = self._build_visualization_assets(insights)
        viz_paths = write_visualization_assets(session_dir, visualization_assets)
        console.stage_update("Visualization Generation", "Completed")

        # Quality gate
        console.stage_update("Quality Gate", "Started")
        quality_results = self._run_quality_validation(insights, report_narratives)
        console.stage_update("Quality Gate", "Completed")

        # Distribution
        console.stage_update("Personalization & Distribution", "Started")
        delivery_results = self._execute_delivery(stakeholder_profiles, report_paths)
        delivery_log_path = write_delivery_log(session_dir, delivery_results)
        console.stage_update("Personalization & Distribution", "Completed")

        # Knowledge capture
        console.stage_update("Knowledge Capture", "Started")
        knowledge_updates = self._capture_knowledge_updates(insights)
        knowledge_path = write_knowledge_updates(session_dir, knowledge_updates)
        console.stage_update("Knowledge Capture", "Completed")

        # Wrap-up
        console.stage_update("Wrap-Up", "Started")
        synthetic_payload = {
            "telemetry": telemetry,
            "insights": [self._insight_to_dict(i) for i in insights],
            "stakeholders": stakeholder_profiles,
            "session": session_name,
        }
        synthetic_path = write_synthetic_data(
            self.settings.artifacts_root / "data",
            session_name,
            synthetic_payload
        )

        artifact_paths = [
            telemetry_path,
            profiles_path,
            insight_board_path,
            *report_paths,
            *viz_paths,
            delivery_log_path,
            knowledge_path,
            synthetic_path,
        ]

        session_log_path = session_dir / "session.log"
        manifest_payload = [*artifact_paths, session_log_path]
        manifest_path = write_manifest(session_dir, manifest_payload)
        manifest_payload.append(manifest_path)
        console.stage_update("Wrap-Up", "Completed")

        # Echo manifest for the operator
        console.stream(["", "Artifacts generated:"])
        for path in manifest_payload:
            console.stream([f" - {path}"])

        write_lines(session_log_path, console.lines)

        return session_dir

    def _prepare_telemetry_data(self, payloads: Dict[str, object]) -> Dict[str, object]:
        """Extract and prepare telemetry data from fixtures."""
        telemetry_data = {}

        # Look for KPI and metrics data
        for fixture_id, content in payloads.items():
            if isinstance(content, dict):
                if "kpi" in fixture_id.lower() or "metrics" in fixture_id.lower():
                    telemetry_data[fixture_id] = content
                elif any(key in str(content) for key in ["response_time", "throughput", "cpu", "memory"]):
                    telemetry_data[fixture_id] = content

        # Add synthetic telemetry if none found
        if not telemetry_data:
            telemetry_data = self._generate_synthetic_telemetry()

        return telemetry_data

    def _generate_synthetic_telemetry(self) -> Dict[str, object]:
        """Generate synthetic telemetry data for demo."""
        base_response_time = 150.0 + (self.rng.random() - 0.5) * 50
        base_throughput = 1200.0 + (self.rng.random() - 0.5) * 400
        base_error_rate = 0.5 + self.rng.random() * 2.0

        return {
            "system_metrics": {
                "response_time_ms": base_response_time,
                "throughput_rps": base_throughput,
                "error_rate_percent": base_error_rate,
                "availability_percent": 99.5 + self.rng.random() * 0.5,
                "cpu_utilization_percent": 60 + self.rng.random() * 30,
                "memory_usage_percent": 65 + self.rng.random() * 25
            },
            "business_metrics": {
                "active_users": 15000 + self.rng.randint(-2000, 3000),
                "transaction_volume": 50000 + self.rng.randint(-10000, 15000),
                "revenue_impact": 125000 + self.rng.randint(-20000, 30000)
            },
            "quality_indicators": {
                "data_completeness": 0.95 + self.rng.random() * 0.04,
                "measurement_accuracy": 0.92 + self.rng.random() * 0.06,
                "collection_timeliness": 0.88 + self.rng.random() * 0.08
            }
        }

    def _build_stakeholder_profiles(self) -> List[Dict[str, object]]:
        """Build stakeholder profiles for audience targeting."""
        return [
            {
                "type": "executive",
                "name": "Executive Leadership",
                "template": "executive_summary",
                "format": "pdf",
                "focus": "business_impact",
                "frequency": "weekly",
                "delivery_channels": ["email", "executive_dashboard"]
            },
            {
                "type": "operations",
                "name": "Operations Team",
                "template": "operational_report",
                "format": "html",
                "focus": "actionable_items",
                "frequency": "daily",
                "delivery_channels": ["dashboard", "slack_integration"]
            },
            {
                "type": "customer",
                "name": "Customer Communications",
                "template": "service_status",
                "format": "email",
                "focus": "service_impact",
                "frequency": "on_incident",
                "delivery_channels": ["customer_portal", "status_page"]
            }
        ]

    def _build_enriched_context(self, payloads: Dict[str, object]) -> Dict[str, object]:
        """Build enriched context from available data."""
        return {
            "sla_targets": {
                "response_time_ms": 200,
                "availability_percent": 99.5,
                "throughput_rps": 1000,
                "error_rate_percent": 1.0
            },
            "business_context": {
                "peak_hours": "09:00-17:00 EST",
                "maintenance_windows": ["Saturday 02:00-04:00 EST"],
                "critical_dependencies": ["payment_gateway", "user_authentication", "data_warehouse"]
            },
            "compliance_requirements": ["SOC2", "GDPR", "PCI_DSS"],
            "escalation_procedures": {
                "critical": "immediate_notification",
                "high": "within_15_minutes",
                "medium": "within_1_hour"
            }
        }

    def _build_insights(self, telemetry: Dict[str, object], context: Dict[str, object]) -> List[Insight]:
        """Build insights from telemetry data and context."""
        insights = []

        # Extract system metrics if available
        system_metrics = {}
        for key, data in telemetry.items():
            if isinstance(data, dict) and "system_metrics" in data:
                system_metrics = data["system_metrics"]
                break
            elif isinstance(data, dict):
                system_metrics.update(data)

        # Generate insights based on available data
        insight_index = 1

        # Response time insight
        response_time = system_metrics.get("response_time_ms", 150)
        sla_target = context.get("sla_targets", {}).get("response_time_ms", 200)

        if response_time < sla_target:
            confidence = 0.85 + self.rng.random() * 0.1
            insights.append(Insight(
                index=insight_index,
                title="Response Time Performance",
                description=f"Average response time of {response_time:.1f}ms is performing well below SLA target",
                insight_type=InsightType.TREND,
                confidence=confidence,
                metric_name="response_time_ms",
                current_value=response_time,
                baseline_value=sla_target,
                business_impact="Positive user experience and service efficiency",
                recommended_action="Continue monitoring and maintain optimization strategies",
                citation="system_metrics_analysis.json"
            ))
            insight_index += 1

        # Throughput insight
        throughput = system_metrics.get("throughput_rps", 1200)
        if throughput > 1000:
            confidence = 0.78 + self.rng.random() * 0.12
            insights.append(Insight(
                index=insight_index,
                title="High Throughput Performance",
                description=f"System handling {throughput:.0f} requests per second, exceeding baseline",
                insight_type=InsightType.COMPARISON,
                confidence=confidence,
                metric_name="throughput_rps",
                current_value=throughput,
                baseline_value=1000,
                business_impact="Strong capacity handling and system scalability",
                recommended_action="Monitor for potential capacity constraints during peak periods",
                citation="throughput_analysis.log"
            ))
            insight_index += 1

        # CPU utilization anomaly
        cpu_usage = system_metrics.get("cpu_utilization_percent", 65)
        if cpu_usage > 80:
            confidence = 0.82 + self.rng.random() * 0.08
            insights.append(Insight(
                index=insight_index,
                title="CPU Utilization Anomaly",
                description=f"CPU utilization at {cpu_usage:.1f}% indicates potential resource strain",
                insight_type=InsightType.ANOMALY,
                confidence=confidence,
                metric_name="cpu_utilization_percent",
                current_value=cpu_usage,
                baseline_value=60.0,
                threshold_breached=True,
                business_impact="Risk of performance degradation during peak loads",
                recommended_action="Investigate resource allocation and consider scaling options",
                citation="resource_monitoring.log"
            ))
            insight_index += 1

        # Availability threshold check
        availability = system_metrics.get("availability_percent", 99.5)
        sla_availability = context.get("sla_targets", {}).get("availability_percent", 99.5)

        if availability >= sla_availability:
            confidence = 0.92 + self.rng.random() * 0.06
            insights.append(Insight(
                index=insight_index,
                title="SLA Availability Compliance",
                description=f"Service availability at {availability:.2f}% meets SLA requirements",
                insight_type=InsightType.THRESHOLD,
                confidence=confidence,
                metric_name="availability_percent",
                current_value=availability,
                baseline_value=sla_availability,
                business_impact="Meeting customer expectations and contractual commitments",
                recommended_action="Maintain current reliability practices and monitoring",
                citation="availability_tracker.json"
            ))

        return insights

    def _build_report_narratives(self, insights: List[Insight], stakeholders: List[Dict[str, object]]) -> Dict[str, object]:
        """Build narrative content for different stakeholder groups."""
        narratives = {}

        high_confidence_insights = [i for i in insights if i.confidence >= 0.8]

        for stakeholder in stakeholders:
            stakeholder_type = stakeholder["type"]
            focus = stakeholder.get("focus", "general")

            if focus == "business_impact":
                # Executive narrative
                narratives[stakeholder_type] = {
                    "executive_summary": self._create_executive_summary(high_confidence_insights),
                    "key_findings": self._create_key_findings(insights),
                    "business_recommendations": self._create_business_recommendations(insights)
                }
            elif focus == "actionable_items":
                # Operations narrative
                narratives[stakeholder_type] = {
                    "operational_status": self._create_operational_status(insights),
                    "immediate_actions": self._create_immediate_actions(insights),
                    "monitoring_guidance": self._create_monitoring_guidance(insights)
                }
            elif focus == "service_impact":
                # Customer narrative
                narratives[stakeholder_type] = {
                    "service_overview": self._create_service_overview(insights),
                    "impact_assessment": self._create_impact_assessment(insights),
                    "resolution_status": self._create_resolution_status(insights)
                }

        return narratives

    def _create_executive_summary(self, insights: List[Insight]) -> str:
        """Create executive summary from insights."""
        if not insights:
            return "System performance remains within acceptable parameters with no critical issues identified."

        positive_trends = [i for i in insights if i.insight_type == InsightType.TREND and i.confidence > 0.8]
        issues = [i for i in insights if i.threshold_breached]

        if positive_trends and not issues:
            return "System performance shows positive trends with key metrics exceeding baseline expectations. Service delivery remains strong with opportunities for continued optimization."
        elif issues:
            return f"Performance monitoring identifies {len(issues)} areas requiring attention, with overall system stability maintained. Strategic focus recommended on resource optimization."
        else:
            return "System performance indicators remain stable with consistent service delivery. Continued monitoring recommended to maintain current performance levels."

    def _create_key_findings(self, insights: List[Insight]) -> List[str]:
        """Create key findings list."""
        findings = []
        for insight in insights[:3]:  # Top 3 insights
            findings.append(f"{insight.title}: {insight.description}")
        return findings

    def _create_business_recommendations(self, insights: List[Insight]) -> List[str]:
        """Create business recommendations."""
        recommendations = []
        for insight in insights:
            if insight.recommended_action and insight.confidence > 0.7:
                recommendations.append(insight.recommended_action)
        return recommendations[:5]  # Top 5 recommendations

    def _create_operational_status(self, insights: List[Insight]) -> str:
        """Create operational status summary."""
        critical_issues = [i for i in insights if i.threshold_breached]
        if critical_issues:
            return f"Operational status: {len(critical_issues)} items require immediate attention. System stability maintained with targeted interventions needed."
        return "Operational status: All systems operating within normal parameters. Standard monitoring protocols active."

    def _create_immediate_actions(self, insights: List[Insight]) -> List[str]:
        """Create immediate action items."""
        actions = []
        for insight in insights:
            if insight.threshold_breached or (insight.insight_type == InsightType.ANOMALY and insight.confidence > 0.8):
                actions.append(f"Address {insight.metric_name}: {insight.recommended_action}")
        return actions

    def _create_monitoring_guidance(self, insights: List[Insight]) -> List[str]:
        """Create monitoring guidance."""
        return [
            "Continue trend analysis on performance metrics",
            "Monitor threshold breach indicators for early detection",
            "Maintain alerting on anomaly detection systems",
            "Review capacity planning based on throughput patterns"
        ]

    def _create_service_overview(self, insights: List[Insight]) -> str:
        """Create service overview for customers."""
        availability_insights = [i for i in insights if "availability" in i.metric_name]
        if availability_insights and availability_insights[0].current_value > 99.0:
            return "Service availability remains high with strong performance across all key metrics."
        return "Service performance is being actively monitored with continuous optimization in progress."

    def _create_impact_assessment(self, insights: List[Insight]) -> str:
        """Create impact assessment."""
        issues = [i for i in insights if i.threshold_breached]
        if issues:
            return "Minor performance optimizations in progress with no customer-facing impact expected."
        return "No customer-facing impacts identified. Service delivery remains consistent."

    def _create_resolution_status(self, insights: List[Insight]) -> str:
        """Create resolution status."""
        return "Proactive monitoring and optimization activities continue to ensure service quality."

    def _build_visualization_assets(self, insights: List[Insight]) -> List[str]:
        """Build visualization assets for insights."""
        assets = []
        for insight in insights:
            if insight.insight_type in [InsightType.TREND, InsightType.COMPARISON]:
                chart_type = "line_chart" if insight.insight_type == InsightType.TREND else "bar_chart"
                asset_path = f"charts/{insight.metric_name}_{chart_type}.png"
                assets.append(asset_path)
        return assets

    def _run_quality_validation(self, insights: List[Insight], narratives: Dict[str, object]) -> Dict[str, object]:
        """Run quality validation checks."""
        validation_score = sum(i.confidence for i in insights) / max(len(insights), 1)
        return {
            "overall_score": validation_score,
            "passed": validation_score >= 0.7,
            "confidence_threshold_met": len([i for i in insights if i.confidence >= 0.7]) >= len(insights) * 0.6,
            "narrative_completeness": len(narratives) > 0
        }

    def _execute_delivery(self, stakeholders: List[Dict[str, object]], report_paths: List[Path]) -> Dict[str, object]:
        """Execute report delivery."""
        delivery_results = {}
        for stakeholder in stakeholders:
            stakeholder_type = stakeholder["type"]
            delivery_results[stakeholder_type] = {
                "status": "delivered",
                "timestamp": datetime.utcnow().isoformat(),
                "channels": stakeholder.get("delivery_channels", ["email"]),
                "format": stakeholder.get("format", "html")
            }
        return delivery_results

    def _capture_knowledge_updates(self, insights: List[Insight]) -> List[str]:
        """Capture knowledge updates."""
        return [
            f"Updated dashboard with {len(insights)} new insights",
            "Enhanced metric baseline calculations",
            "Added trend analysis to historical database",
            "Updated SLA monitoring thresholds"
        ]

    def _insight_to_dict(self, insight: Insight) -> Dict[str, object]:
        """Convert Insight object to dictionary."""
        return {
            "index": insight.index,
            "title": insight.title,
            "description": insight.description,
            "type": insight.insight_type.value,
            "confidence": insight.confidence,
            "metric_name": insight.metric_name,
            "current_value": insight.current_value,
            "baseline_value": insight.baseline_value,
            "threshold_breached": insight.threshold_breached,
            "business_impact": insight.business_impact,
            "recommended_action": insight.recommended_action,
            "citation": insight.citation
        }

    @staticmethod
    def _preview(content: object, limit: int = 120) -> str:
        try:
            text = json.dumps(content, ensure_ascii=False)
        except TypeError:
            text = str(content)
        return text[:limit]


__all__ = ["AutomatedReportingRunner", "AutomatedReportingSettings"]