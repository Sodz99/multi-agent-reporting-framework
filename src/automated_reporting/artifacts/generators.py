"""Artifact generation for the automated reporting workflow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Any

from ..shared.files import ensure_dir, write_json, write_lines, write_text
from ..agents.crew import Insight


def write_telemetry_snapshot(session_dir: Path, telemetry_data: Dict[str, object]) -> Path:
    """Write telemetry data snapshot."""
    path = session_dir / "telemetry_snapshot.json"
    write_json(path, telemetry_data)
    return path


def write_stakeholder_profiles(session_dir: Path, profiles: List[Dict[str, object]]) -> Path:
    """Write stakeholder profiles configuration."""
    path = session_dir / "stakeholder_profiles.json"
    write_json(path, {"stakeholder_profiles": profiles})
    return path


def write_insight_board(session_dir: Path, insights: Sequence[Insight]) -> Path:
    """Write insights board in markdown format."""
    path = session_dir / "insight_board.md"
    lines: List[str] = ["# Automated Reporting Insight Board", ""]

    for insight in insights:
        confidence_indicator = "✅" if insight.confidence >= 0.8 else "⚠️" if insight.confidence >= 0.6 else "❌"
        lines.append(f"## Insight {insight.index} — {insight.title} {confidence_indicator}")
        lines.append(f"**Type**: {insight.insight_type.value.title()}")
        lines.append(f"**Confidence**: {insight.confidence:.2f}")
        lines.append(f"**Metric**: {insight.metric_name}")
        lines.append(f"**Current Value**: {insight.current_value}")
        if insight.baseline_value:
            lines.append(f"**Baseline**: {insight.baseline_value}")
        lines.append(f"**Description**: {insight.description}")
        if insight.business_impact:
            lines.append(f"**Business Impact**: {insight.business_impact}")
        if insight.recommended_action:
            lines.append(f"**Recommended Action**: {insight.recommended_action}")
        if insight.threshold_breached:
            lines.append("🚨 **THRESHOLD BREACH DETECTED**")
        lines.append(f"**Citation**: {insight.citation}")
        lines.append("")

    write_lines(path, lines)
    return path


def write_report_bundle(session_dir: Path, narratives: Dict[str, object], insights: Sequence[Insight]) -> List[Path]:
    """Write comprehensive report bundle for different stakeholder types."""
    paths = []

    for stakeholder_type, narrative in narratives.items():
        # Create HTML report
        html_path = session_dir / f"{stakeholder_type}_report.html"
        html_content = _create_html_report(stakeholder_type, narrative, insights)
        write_text(html_path, html_content)
        paths.append(html_path)

        # Create JSON data
        json_path = session_dir / f"{stakeholder_type}_report.json"
        json_data = {
            "stakeholder_type": stakeholder_type,
            "generated_at": "2024-01-01T12:00:00Z",
            "narrative": narrative,
            "insights": [_insight_to_dict(insight) for insight in insights],
            "summary_stats": {
                "total_insights": len(insights),
                "high_confidence_insights": len([i for i in insights if i.confidence >= 0.8]),
                "threshold_breaches": len([i for i in insights if i.threshold_breached]),
                "avg_confidence": sum(i.confidence for i in insights) / len(insights) if insights else 0
            }
        }
        write_json(json_path, json_data)
        paths.append(json_path)

    return paths


def write_visualization_assets(session_dir: Path, asset_paths: List[str]) -> List[Path]:
    """Write visualization asset registry."""
    viz_dir = ensure_dir(session_dir / "visualizations")
    registry_path = viz_dir / "asset_registry.json"

    asset_registry = {
        "generated_at": "2024-01-01T12:00:00Z",
        "total_assets": len(asset_paths),
        "assets": []
    }

    # Create placeholder assets and registry entries
    created_paths = []
    for i, asset_path in enumerate(asset_paths):
        # Create placeholder visualization file
        asset_name = Path(asset_path).name
        placeholder_path = viz_dir / asset_name

        # Write placeholder content
        if asset_name.endswith('.png'):
            placeholder_content = f"# Placeholder for visualization: {asset_name}\n# Generated chart would be saved here"
        else:
            placeholder_content = f"Visualization placeholder: {asset_name}"

        write_text(placeholder_path, placeholder_content)
        created_paths.append(placeholder_path)

        # Add to registry
        asset_registry["assets"].append({
            "id": f"asset_{i+1}",
            "name": asset_name,
            "path": str(placeholder_path),
            "type": "chart" if asset_name.endswith('.png') else "data",
            "created_at": "2024-01-01T12:00:00Z"
        })

    write_json(registry_path, asset_registry)
    created_paths.append(registry_path)

    return created_paths


def write_delivery_log(session_dir: Path, delivery_results: Dict[str, object]) -> Path:
    """Write delivery execution log."""
    path = session_dir / "delivery_log.json"
    delivery_log = {
        "execution_timestamp": "2024-01-01T12:00:00Z",
        "total_deliveries": len(delivery_results),
        "successful_deliveries": len([r for r in delivery_results.values()
                                    if isinstance(r, dict) and r.get("status") == "delivered"]),
        "delivery_results": delivery_results,
        "audit_trail": [
            {
                "action": "personalization_started",
                "timestamp": "2024-01-01T12:00:00Z",
                "details": "Content adaptation for stakeholder groups"
            },
            {
                "action": "delivery_orchestration",
                "timestamp": "2024-01-01T12:00:10Z",
                "details": "Multi-channel distribution initiated"
            },
            {
                "action": "delivery_complete",
                "timestamp": "2024-01-01T12:00:30Z",
                "details": "All scheduled deliveries completed"
            }
        ]
    }
    write_json(path, delivery_log)
    return path


def write_knowledge_updates(session_dir: Path, updates: List[str]) -> Path:
    """Write knowledge capture updates."""
    path = session_dir / "knowledge_updates.md"
    lines = [
        "# Knowledge Capture Updates",
        "",
        "The following updates were made to the knowledge base and monitoring systems:",
        ""
    ]

    for i, update in enumerate(updates, 1):
        lines.append(f"{i}. {update}")

    lines.extend([
        "",
        "## Impact",
        "- Enhanced dashboard visibility into system metrics",
        "- Improved trend analysis capabilities",
        "- Updated alerting thresholds based on insights",
        "- Enriched historical data for future comparisons"
    ])

    write_lines(path, lines)
    return path


def write_manifest(session_dir: Path, artifact_paths: Sequence[Path]) -> Path:
    """Write session manifest with all generated artifacts."""
    path = session_dir / "session_manifest.json"
    manifest = {
        "session_type": "automated_reporting",
        "generated_at": "2024-01-01T12:00:00Z",
        "total_artifacts": len(artifact_paths),
        "artifacts": [
            {
                "name": p.name,
                "path": str(p),
                "type": _classify_artifact_type(p),
                "size_bytes": p.stat().st_size if p.exists() else 0
            }
            for p in artifact_paths
        ]
    }
    write_json(path, manifest)
    return path


def write_synthetic_data(data_dir: Path, session_name: str, payload: Dict[str, object]) -> Path:
    """Write synthetic data for testing and demos."""
    data_path = ensure_dir(data_dir) / f"synthetic_{session_name}.json"
    synthetic_payload = {
        "session_id": session_name,
        "generated_at": "2024-01-01T12:00:00Z",
        "data_type": "automated_reporting_synthetic",
        "payload": payload,
        "metadata": {
            "purpose": "Demo and testing data",
            "quality_indicators": {
                "completeness": 0.95,
                "accuracy": 0.92,
                "timeliness": 0.88
            }
        }
    }
    write_json(data_path, synthetic_payload)
    return data_path


def _create_html_report(stakeholder_type: str, narrative: object, insights: Sequence[Insight]) -> str:
    """Create HTML report content for stakeholder."""
    if not isinstance(narrative, dict):
        narrative = {"content": str(narrative)}

    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"    <title>Automated Reporting - {stakeholder_type.title()} Report</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; }",
        "        h1 { color: #2c3e50; }",
        "        h2 { color: #34495e; }",
        "        .insight { background: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }",
        "        .high-confidence { border-left: 4px solid #27ae60; }",
        "        .medium-confidence { border-left: 4px solid #f39c12; }",
        "        .low-confidence { border-left: 4px solid #e74c3c; }",
        "        .threshold-breach { background: #fadbd8; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>Automated Reporting - {stakeholder_type.title()} Report</h1>",
        "    <p>Generated by the automated reporting multi-agent system.</p>",
    ]

    # Add narrative sections
    for section_key, section_content in narrative.items():
        html_lines.append(f"    <h2>{section_key.replace('_', ' ').title()}</h2>")
        if isinstance(section_content, list):
            html_lines.append("    <ul>")
            for item in section_content:
                html_lines.append(f"      <li>{item}</li>")
            html_lines.append("    </ul>")
        else:
            html_lines.append(f"    <p>{section_content}</p>")

    # Add insights section
    if insights:
        html_lines.append("    <h2>Key Insights</h2>")
        for insight in insights:
            confidence_class = ("high-confidence" if insight.confidence >= 0.8
                              else "medium-confidence" if insight.confidence >= 0.6
                              else "low-confidence")
            breach_class = "threshold-breach" if insight.threshold_breached else ""

            html_lines.extend([
                f'    <div class="insight {confidence_class} {breach_class}">',
                f"        <h3>{insight.title}</h3>",
                f"        <p><strong>Type:</strong> {insight.insight_type.value.title()}</p>",
                f"        <p><strong>Confidence:</strong> {insight.confidence:.2f}</p>",
                f"        <p><strong>Metric:</strong> {insight.metric_name} = {insight.current_value}</p>",
                f"        <p>{insight.description}</p>",
            ])

            if insight.business_impact:
                html_lines.append(f"        <p><strong>Business Impact:</strong> {insight.business_impact}</p>")
            if insight.recommended_action:
                html_lines.append(f"        <p><strong>Recommended Action:</strong> {insight.recommended_action}</p>")
            if insight.threshold_breached:
                html_lines.append("        <p><strong>⚠️ THRESHOLD BREACH DETECTED</strong></p>")

            html_lines.append("    </div>")

    html_lines.extend([
        "</body>",
        "</html>"
    ])

    return "\n".join(html_lines)


def _insight_to_dict(insight: Insight) -> Dict[str, Any]:
    """Convert Insight object to dictionary for JSON serialization."""
    return {
        "index": insight.index,
        "title": insight.title,
        "description": insight.description,
        "insight_type": insight.insight_type.value,
        "confidence": insight.confidence,
        "metric_name": insight.metric_name,
        "current_value": insight.current_value,
        "baseline_value": insight.baseline_value,
        "threshold_breached": insight.threshold_breached,
        "business_impact": insight.business_impact,
        "recommended_action": insight.recommended_action,
        "citation": insight.citation
    }


def _classify_artifact_type(path: Path) -> str:
    """Classify artifact type based on file extension and name."""
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".json":
        if "manifest" in name:
            return "manifest"
        elif "log" in name:
            return "log"
        elif "profile" in name:
            return "configuration"
        else:
            return "data"
    elif suffix == ".html":
        return "report"
    elif suffix == ".md":
        return "documentation"
    elif suffix in [".png", ".jpg", ".svg"]:
        return "visualization"
    else:
        return "artifact"


__all__ = [
    "write_telemetry_snapshot",
    "write_stakeholder_profiles",
    "write_insight_board",
    "write_report_bundle",
    "write_visualization_assets",
    "write_delivery_log",
    "write_knowledge_updates",
    "write_manifest",
    "write_synthetic_data"
]