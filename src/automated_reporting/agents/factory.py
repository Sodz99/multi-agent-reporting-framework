"""CrewAI agent definitions for automated reporting system."""
from __future__ import annotations

import os
from typing import Dict, Any, Optional
from crewai import Agent, LLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model configuration for automated reporting agents
REPORTING_MODEL_CONFIG = {
    "telemetry_listener": "bedrock/amazon.titan-text-express-v1",    # Fast data processing
    "stakeholder_classifier": "bedrock/amazon.titan-text-express-v1", # Quick classification
    "context_enricher": "bedrock/amazon.titan-text-express-v1",     # Context gathering
    "insight_miner": "bedrock/amazon.titan-text-premier-v1:0",       # Complex analysis
    "report_writer": "bedrock/amazon.titan-text-premier-v1:0",       # Narrative creation
    "visualization_bot": "bedrock/amazon.titan-text-express-v1",    # Chart generation
    "consistency_checker": "bedrock/amazon.titan-text-premier-v1:0", # Quality validation
    "delivery_orchestrator": "bedrock/amazon.titan-text-express-v1" # Distribution
}


def create_bedrock_llm(model_id: Optional[str] = None, agent_role: Optional[str] = None) -> LLM:
    """Create a Bedrock LLM instance using CrewAI's format.

    Args:
        model_id: Specific model ID (optional)
        agent_role: Agent role for model selection (optional)

    Returns:
        LLM: CrewAI LLM instance configured for Bedrock
    """
    if model_id is None:
        if agent_role and agent_role in REPORTING_MODEL_CONFIG:
            model_string = REPORTING_MODEL_CONFIG[agent_role]
        else:
            # Default to Titan Premier for general use
            model_string = "bedrock/amazon.titan-text-premier-v1:0"
    else:
        # Ensure proper bedrock/ prefix for LiteLLM format
        if not model_id.startswith("bedrock/"):
            model_string = f"bedrock/{model_id}"
        else:
            model_string = model_id

    return LLM(model=model_string)


class AutomatedReportingAgents:
    """Factory for creating automated reporting crew agents."""

    def __init__(self, custom_models: Optional[Dict[str, str]] = None, verbose: bool = True):
        """Initialize with model configuration.

        Args:
            custom_models: Override default model assignments per role
            verbose: Enable detailed logging
        """
        self.verbose = verbose

        # Use custom models or defaults
        self.model_config = custom_models or REPORTING_MODEL_CONFIG.copy()

        # Create role-specific LLM instances
        self.llms = {}
        for role, model_string in self.model_config.items():
            try:
                self.llms[role] = LLM(model=model_string)
                if self.verbose:
                    print(f"[Agents] ✅ Created {role} LLM: {model_string}")
            except Exception as e:
                if self.verbose:
                    print(f"[Agents] ❌ Failed to create {role} LLM: {e}")
                # Fallback to default Titan Premier
                self.llms[role] = LLM(model="bedrock/amazon.titan-text-premier-v1:0")

        if self.verbose:
            print(f"[Agents] 🚀 AutomatedReportingAgents initialized with {len(self.llms)} LLMs")

    def create_telemetry_listener_agent(self) -> Agent:
        """Create the Telemetry Listener agent for data intake."""
        if self.verbose:
            print(f"[Agents] 📡 Creating telemetry listener agent")

        agent = Agent(
            role="Telemetry Listener",
            goal=(
                "Stream KPIs and event data into the reporting system, filter noise and anomalies, "
                "and prepare clean datasets for analysis. Ensure data quality and completeness."
            ),
            backstory=(
                "You are a data ingestion specialist with deep expertise in telemetry systems, "
                "monitoring infrastructure, and data quality validation. You excel at processing "
                "high-volume streaming data, identifying signal from noise, and preparing clean "
                "datasets for analysis. Your role is critical for ensuring reliable reporting."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llms["telemetry_listener"],
            max_iter=2,
            max_execution_time=60,  # 1-minute timeout for data processing
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] ✅ Telemetry listener agent created")
        return agent

    def create_stakeholder_classifier_agent(self) -> Agent:
        """Create the Stakeholder Classifier agent for audience profiling."""
        if self.verbose:
            print(f"[Agents] 👥 Creating stakeholder classifier agent")

        agent = Agent(
            role="Stakeholder Classifier",
            goal=(
                "Analyze reporting requirements and classify target audiences (executives, operations, "
                "customers). Select appropriate templates, tone, and delivery methods for each stakeholder group."
            ),
            backstory=(
                "You are an expert in organizational communications and stakeholder management. "
                "You understand how different audiences consume information and can tailor content "
                "appropriately. Your expertise includes executive reporting, operational dashboards, "
                "and customer communications, ensuring each group gets relevant, actionable insights."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llms["stakeholder_classifier"],
            max_iter=2,
            max_execution_time=45,
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] ✅ Stakeholder classifier agent created")
        return agent

    def create_context_enricher_agent(self) -> Agent:
        """Create the Context Enricher agent for metadata collection."""
        if self.verbose:
            print(f"[Agents] 🔍 Creating context enricher agent")

        agent = Agent(
            role="Context Enricher",
            goal=(
                "Gather relevant metadata including SLA targets, change logs, historical incidents, "
                "and business context. Enrich reports with actionable context that helps stakeholders "
                "understand the business impact and required actions."
            ),
            backstory=(
                "You are a business intelligence analyst with expertise in data enrichment and "
                "contextual analysis. You understand how to connect technical metrics to business "
                "outcomes and can gather relevant metadata from multiple sources to provide complete "
                "situational awareness for decision makers."
            ),
            verbose=True,
            allow_delegation=True,  # Can delegate data gathering tasks
            llm=self.llms["context_enricher"],
            max_iter=3,
            max_execution_time=90,
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] ✅ Context enricher agent created")
        return agent

    def create_insight_miner_agent(self) -> Agent:
        """Create the Insight Miner agent for trend and anomaly detection."""
        if self.verbose:
            print(f"[Agents] ⛏️ Creating insight miner agent")

        agent = Agent(
            role="Insight Miner",
            goal=(
                "Analyze metrics and KPIs to identify trends, anomalies, and significant patterns. "
                "Generate data-driven insights with confidence scores and supporting evidence. "
                "Focus on actionable findings that drive business decisions."
            ),
            backstory=(
                "You are a senior data scientist and analytics expert with deep experience in "
                "time series analysis, anomaly detection, and statistical modeling. You excel at "
                "finding meaningful patterns in complex datasets and translating analytical findings "
                "into business-relevant insights that drive informed decision making."
            ),
            verbose=True,
            allow_delegation=True,  # Senior role can delegate
            llm=self.llms["insight_miner"],
            max_iter=4,
            max_execution_time=180,  # 3-minute timeout for complex analysis
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] ✅ Insight miner agent created")
        return agent

    def create_report_writer_agent(self) -> Agent:
        """Create the Report Writer agent for narrative assembly."""
        if self.verbose:
            print(f"[Agents] ✍️ Creating report writer agent")

        agent = Agent(
            role="Report Writer",
            goal=(
                "Transform insights and analysis into compelling narratives tailored for specific "
                "audiences. Create executive summaries, detailed technical sections, and actionable "
                "recommendations with proper structure and professional tone."
            ),
            backstory=(
                "You are an expert technical writer and business communications specialist with "
                "extensive experience creating reports for C-level executives, operations teams, "
                "and technical stakeholders. You excel at distilling complex analysis into clear, "
                "actionable narratives that drive business outcomes."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llms["report_writer"],
            max_iter=3,
            max_execution_time=120,  # 2-minute timeout for writing
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] ✅ Report writer agent created")
        return agent

    def create_visualization_bot_agent(self) -> Agent:
        """Create the Visualization Bot agent for chart generation."""
        if self.verbose:
            print(f"[Agents] 📊 Creating visualization bot agent")

        agent = Agent(
            role="Visualization Bot",
            goal=(
                "Generate appropriate charts, graphs, and visual assets for reports. Select optimal "
                "visualization types for different data patterns and audiences. Store generated assets "
                "in the asset registry with proper metadata."
            ),
            backstory=(
                "You are a data visualization expert with deep knowledge of chart design principles, "
                "visual analytics, and audience-appropriate presentation formats. You understand how "
                "to choose the right visualization for different data types and can create compelling "
                "visual narratives that enhance report comprehension."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llms["visualization_bot"],
            max_iter=2,
            max_execution_time=60,
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] ✅ Visualization bot agent created")
        return agent

    def create_consistency_checker_agent(self) -> Agent:
        """Create the Consistency Checker agent for quality validation."""
        if self.verbose:
            print(f"[Agents] ✅ Creating consistency checker agent")

        agent = Agent(
            role="Consistency Checker",
            goal=(
                "Validate report quality by cross-checking numbers, units, and narrative consistency "
                "against source data. Identify discrepancies, flag low-confidence results, and "
                "trigger reruns or human review when necessary."
            ),
            backstory=(
                "You are a quality assurance specialist and data validation expert with meticulous "
                "attention to detail. You excel at identifying inconsistencies, validating calculations, "
                "and ensuring data integrity across complex reports. Your role prevents errors from "
                "reaching stakeholders and maintains reporting credibility."
            ),
            verbose=True,
            allow_delegation=True,  # Can delegate validation tasks
            llm=self.llms["consistency_checker"],
            max_iter=3,
            max_execution_time=90,
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] ✅ Consistency checker agent created")
        return agent

    def create_delivery_orchestrator_agent(self) -> Agent:
        """Create the Delivery Orchestrator agent for distribution management."""
        if self.verbose:
            print(f"[Agents] 📨 Creating delivery orchestrator agent")

        agent = Agent(
            role="Delivery Orchestrator",
            goal=(
                "Manage report personalization and distribution across multiple channels. Schedule "
                "deliveries, adapt content for different stakeholders, and track delivery status "
                "for audit and follow-up purposes."
            ),
            backstory=(
                "You are a delivery and operations specialist with expertise in multi-channel "
                "communication systems, scheduling, and stakeholder management. You understand "
                "how to tailor content for different audiences and can orchestrate complex "
                "distribution workflows while maintaining delivery reliability and tracking."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llms["delivery_orchestrator"],
            max_iter=2,
            max_execution_time=60,
            memory=True,
        )
        if self.verbose:
            print(f"[Agents] ✅ Delivery orchestrator agent created")
        return agent


__all__ = [
    "AutomatedReportingAgents",
    "create_bedrock_llm",
    "REPORTING_MODEL_CONFIG",
]