"""CrewAI task definitions for automated reporting workflow."""
from __future__ import annotations

from typing import Dict, Any, List
from crewai import Task
from .factory import AutomatedReportingAgents


class AutomatedReportingTasks:
    """Factory for creating automated reporting workflow tasks."""

    def __init__(self, agents: AutomatedReportingAgents):
        self.agents = agents

    def create_data_intake_task(self, telemetry_data: Dict[str, Any]) -> Task:
        """Create the telemetry data intake and filtering task."""
        return Task(
            description=f"""
            Process incoming telemetry data and filter out noise to create a clean dataset:

            Raw Telemetry Data: {telemetry_data}

            Your responsibilities:
            1. Parse and validate incoming KPI streams and event data
            2. Apply noise filtering algorithms to remove false positives and irrelevant signals
            3. Identify data quality issues and handle missing or corrupted values
            4. Create snapshots of clean, analyzable datasets
            5. Flag any anomalies that require immediate attention
            6. Store processed data in the shared report dataset

            Data quality checks:
            - Validate timestamp consistency and chronological order
            - Check for missing required fields and data completeness
            - Identify outliers and potential measurement errors
            - Ensure metric units and scales are standardized
            - Remove duplicate or redundant data points

            Output should include:
            - Clean, filtered dataset ready for analysis
            - Data quality metrics and validation results
            - List of anomalies or issues detected during filtering
            - Metadata about data sources and processing applied

            Focus on data integrity and signal clarity for downstream analysis.
            """,
            agent=self.agents.create_telemetry_listener_agent(),
            expected_output="Clean telemetry dataset with quality metrics, anomaly flags, and processing metadata",
        )

    def create_stakeholder_classification_task(self, reporting_requirements: Dict[str, Any]) -> Task:
        """Create the stakeholder classification and audience profiling task."""
        return Task(
            description=f"""
            Analyze reporting requirements and classify target stakeholders for appropriate content delivery:

            Reporting Requirements: {reporting_requirements}

            Your responsibilities:
            1. Identify all target stakeholder groups (executives, operations, customers, engineering)
            2. Analyze each group's information needs and communication preferences
            3. Select appropriate report templates and presentation formats
            4. Determine optimal delivery methods and scheduling for each audience
            5. Set tone and technical depth appropriate for each stakeholder type

            Stakeholder analysis criteria:
            - Executive level: High-level summaries, business impact focus, KPI trends
            - Operations teams: Actionable details, troubleshooting info, immediate actions needed
            - Customers: Impact communication, resolution status, service level information
            - Engineering: Technical details, root cause analysis, system health metrics

            Template selection factors:
            - Audience technical expertise level
            - Decision-making authority and required actions
            - Preferred communication channels and formats
            - Frequency of updates needed
            - Regulatory or compliance requirements

            Output should include:
            - Classified stakeholder groups with profiles
            - Recommended templates and formats per group
            - Delivery preferences and scheduling recommendations
            - Content customization guidelines for each audience

            Ensure messaging is tailored and actionable for each stakeholder type.
            """,
            agent=self.agents.create_stakeholder_classifier_agent(),
            expected_output="Stakeholder classification with audience profiles, template selections, and delivery preferences",
        )

    def create_context_enrichment_task(self, base_data: Dict[str, Any]) -> Task:
        """Create the context enrichment and metadata gathering task."""
        return Task(
            description=f"""
            Enrich the reporting context with relevant metadata and business intelligence:

            Base Data Context: {base_data}

            Your responsibilities:
            1. Gather SLA targets and performance baselines for relevant metrics
            2. Retrieve recent change logs and system modifications
            3. Collect historical incident patterns and resolution data
            4. Add business context including service dependencies and impact scope
            5. Include regulatory or compliance requirements affecting the metrics
            6. Prepare curated context blocks for downstream narrative generation

            Context sources to consult:
            - Service level agreements and performance targets
            - Change management records and deployment history
            - Historical incident database and pattern analysis
            - Business service maps and dependency information
            - Regulatory compliance requirements and audit trails
            - Customer impact assessments and escalation procedures

            Enrichment objectives:
            - Provide business context for technical metrics
            - Enable impact assessment and prioritization decisions
            - Support root cause analysis with historical patterns
            - Include actionable guidance based on similar incidents
            - Ensure compliance with regulatory reporting requirements

            Output should include:
            - Comprehensive metadata package with business context
            - SLA targets and threshold definitions
            - Historical patterns and comparative analysis
            - Business impact assessment and dependency mapping
            - Compliance and regulatory context where applicable

            Focus on actionable context that supports decision-making.
            """,
            agent=self.agents.create_context_enricher_agent(),
            expected_output="Enriched context package with SLA targets, historical patterns, business impact, and compliance metadata",
        )

    def create_insight_mining_task(self, enriched_data: Dict[str, Any]) -> Task:
        """Create the insight mining and analysis task."""
        return Task(
            description=f"""
            Analyze enriched telemetry data to extract meaningful insights and patterns:

            Enriched Data Package: {enriched_data}

            Your responsibilities:
            1. Run trend analysis across time series data to identify patterns
            2. Perform anomaly detection using statistical and ML techniques
            3. Check for threshold breaches and SLA violations
            4. Generate comparative analysis against historical baselines
            5. Create forecasting insights for predictive reporting
            6. Assign confidence scores based on data quality and analysis strength

            Analysis techniques to apply:
            - Time series decomposition for trend and seasonality analysis
            - Statistical process control for anomaly detection
            - Comparative analysis against SLA targets and historical performance
            - Correlation analysis between metrics and business events
            - Predictive modeling for forecast generation
            - Pattern recognition for incident similarity analysis

            Insight types to generate:
            - TREND: Long-term patterns and directional changes
            - ANOMALY: Statistical outliers and unexpected behaviors
            - THRESHOLD: SLA breaches and performance violations
            - COMPARISON: Performance vs baselines and peer services
            - FORECAST: Predictive insights and capacity planning

            For each insight provide:
            - Clear technical statement of the finding
            - Confidence score (0.0-1.0) based on data strength
            - Business impact assessment and severity classification
            - Supporting evidence with data citations
            - Recommended actions or follow-up investigations

            Focus on actionable insights that drive business decisions and operational improvements.
            """,
            agent=self.agents.create_insight_miner_agent(),
            expected_output="Comprehensive insights package with trends, anomalies, forecasts, confidence scores, and business impact assessments",
        )

    def create_narrative_assembly_task(self, insights_package: Dict[str, Any]) -> Task:
        """Create the narrative assembly and report writing task."""
        return Task(
            description=f"""
            Transform analytical insights into compelling narratives tailored for target audiences:

            Insights Package: {insights_package}

            Your responsibilities:
            1. Create executive summaries highlighting key findings and business impact
            2. Develop detailed technical sections with supporting evidence
            3. Craft actionable recommendations based on insights and context
            4. Structure content appropriate for each stakeholder audience
            5. Ensure narrative flow and logical progression of findings
            6. Maintain professional tone while making content accessible

            Content sections to develop:
            - Executive Summary: Key findings, business impact, critical actions needed
            - Current Status: Performance against SLAs, metric trends, service health
            - Key Insights: Detailed analysis of significant findings with evidence
            - Risk Assessment: Potential issues, impact scenarios, mitigation needs
            - Recommendations: Prioritized actions, timelines, resource requirements
            - Appendices: Supporting data, detailed metrics, technical references

            Narrative guidelines:
            - Lead with most critical findings and business impact
            - Use clear, jargon-free language appropriate for audience
            - Support claims with data citations and confidence indicators
            - Include visual references to charts and supporting evidence
            - Provide actionable next steps with clear ownership and timelines

            Audience-specific adaptations:
            - Executive: Focus on business outcomes, strategic implications, resource needs
            - Operations: Emphasize immediate actions, troubleshooting steps, monitoring guidance
            - Customer: Highlight service impact, resolution progress, preventive measures
            - Engineering: Include technical details, system metrics, diagnostic information

            Ensure narratives are professional, data-driven, and actionable for each audience.
            """,
            agent=self.agents.create_report_writer_agent(),
            expected_output="Complete narrative sections with executive summaries, detailed analysis, recommendations, and audience-specific content",
        )

    def create_visualization_task(self, analysis_results: Dict[str, Any]) -> Task:
        """Create the data visualization and chart generation task."""
        return Task(
            description=f"""
            Generate appropriate visualizations and charts to support the analytical narrative:

            Analysis Results: {analysis_results}

            Your responsibilities:
            1. Select optimal chart types for different data patterns and insights
            2. Create visualizations that enhance narrative comprehension
            3. Generate audience-appropriate visual complexity and detail levels
            4. Store generated assets in the asset registry with proper metadata
            5. Ensure visual consistency and professional presentation quality

            Visualization types to consider:
            - Time series charts for trends and historical comparisons
            - Threshold charts showing SLA performance and breach indicators
            - Heat maps for pattern recognition and correlation analysis
            - Dashboards for operational monitoring and real-time status
            - Comparative charts for baseline and peer performance analysis
            - Forecast charts with confidence intervals and prediction ranges

            Chart selection criteria:
            - Data type and dimensionality (time series, categorical, numerical)
            - Audience technical expertise and visualization literacy
            - Key message or insight being communicated
            - Available space and format constraints
            - Interactive vs static presentation requirements

            Visual design principles:
            - Clear axis labels, legends, and titles
            - Consistent color schemes and branding
            - Appropriate scale selection and data density
            - Highlighting of key findings and thresholds
            - Professional styling suitable for business reporting

            Asset management:
            - Generate high-resolution versions for different output formats
            - Store assets with descriptive metadata and usage guidelines
            - Create alternative text descriptions for accessibility
            - Maintain version control and update tracking
            - Ensure assets are properly referenced in narrative sections

            Focus on visual clarity and insight amplification.
            """,
            agent=self.agents.create_visualization_bot_agent(),
            expected_output="Generated visualization assets with metadata, stored in asset registry with proper references",
        )

    def create_quality_validation_task(self, report_draft: Dict[str, Any]) -> Task:
        """Create the quality validation and consistency checking task."""
        return Task(
            description=f"""
            Validate report quality through comprehensive consistency and accuracy checks:

            Report Draft: {report_draft}

            Your responsibilities:
            1. Cross-validate all numbers and calculations against source data
            2. Verify unit consistency and metric standardization throughout
            3. Check narrative alignment with underlying data and analysis
            4. Validate all citations and references to source materials
            5. Identify any logical inconsistencies or contradictory statements
            6. Flag low-confidence results that require additional review

            Validation criteria:
            - Data accuracy: All numbers match source data and calculations are correct
            - Unit consistency: Metrics use standardized units and scaling
            - Citation validity: All references point to actual source materials
            - Narrative coherence: Text aligns with data findings and analysis
            - Confidence thresholds: Results meet minimum confidence requirements
            - Completeness: All required sections and stakeholder needs addressed

            Quality checks to perform:
            - Reconcile summary statistics with detailed data
            - Verify threshold breach indicators against actual values
            - Check percentage calculations and ratio computations
            - Validate trend direction claims against time series data
            - Confirm forecast accuracy against model confidence intervals
            - Review recommendation feasibility and resource requirements

            Error handling procedures:
            - Flag discrepancies for immediate correction
            - Categorize issues by severity (critical, moderate, minor)
            - Recommend rerun of analysis for significant confidence gaps
            - Escalate to human review for complex validation failures
            - Document all validation results for audit trail

            Quality gates:
            - Data accuracy: >99% numerical consistency required
            - Confidence threshold: Insights must meet minimum confidence levels
            - Citation completeness: All claims must have supporting references
            - Narrative alignment: No contradictions between text and data

            Mark report status as: Approved ✓, Needs Revision ⚠, or Escalate ✗
            """,
            agent=self.agents.create_consistency_checker_agent(),
            expected_output="Quality validation report with approval status, error flags, confidence assessment, and correction recommendations",
        )

    def create_delivery_orchestration_task(self, validated_reports: List[Dict[str, Any]]) -> Task:
        """Create the delivery orchestration and distribution management task."""
        return Task(
            description=f"""
            Orchestrate personalized report delivery across multiple stakeholder channels:

            Validated Reports: {validated_reports}

            Your responsibilities:
            1. Personalize report content and format for each stakeholder group
            2. Schedule delivery according to stakeholder preferences and urgency
            3. Manage multi-channel distribution (email, API, dashboard, etc.)
            4. Track delivery status and handle failures or retries
            5. Log all delivery activities for audit and compliance purposes
            6. Monitor acknowledgment and follow-up requirements

            Personalization requirements:
            - Executive reports: Executive summary focus, PDF format, weekly cadence
            - Operations reports: Action-oriented content, real-time dashboards, immediate alerts
            - Customer communications: Impact focus, plain language, proactive notifications
            - Engineering reports: Technical detail, JSON/API delivery, on-demand access

            Delivery channels to support:
            - Email: Formatted reports with attachments and embedded visualizations
            - API endpoints: JSON payload delivery for automated systems integration
            - Dashboard updates: Real-time metric updates and alert status changes
            - Mobile notifications: Critical alert delivery to on-call personnel
            - Collaboration tools: Integration with Slack, Teams, or other messaging platforms

            Scheduling considerations:
            - Stakeholder time zones and availability preferences
            - Business hours vs off-hours delivery appropriateness
            - Escalation procedures for critical findings requiring immediate attention
            - Batch vs individual delivery optimization
            - Retry logic for delivery failures and network issues

            Delivery tracking:
            - Timestamp all delivery attempts and confirmations
            - Monitor email opens, API acknowledgments, dashboard access
            - Track stakeholder engagement and response patterns
            - Flag delivery failures for manual intervention
            - Maintain compliance audit trail for regulatory requirements

            Quality assurance:
            - Validate recipient lists and contact information
            - Test delivery channels before critical report distribution
            - Ensure content formatting is appropriate for each channel
            - Verify accessibility compliance for all delivery formats
            - Monitor delivery performance and optimize based on feedback

            Provide comprehensive delivery status reporting and audit trail maintenance.
            """,
            agent=self.agents.create_delivery_orchestrator_agent(),
            expected_output="Delivery execution report with personalized distributions, channel status, tracking logs, and audit trail",
        )


__all__ = ["AutomatedReportingTasks"]