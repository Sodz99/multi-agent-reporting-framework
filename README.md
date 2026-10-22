# Automated Reporting — Multi-Agent System

A multi-agent pipeline that turns raw telemetry/KPI data into stakeholder-specific reports (executive, operations, customer, technical). **LangGraph** runs the stateful workflow with branches and quality loops; **CrewAI** agents do the reasoning, writing and validation; a **ChromaDB RAG** layer enriches each report with context. LLMs and embeddings run on **AWS Bedrock** (Claude 3.5, Titan embeddings).

## Workflow
```
data intake → audience profiling → context enrichment (RAG)
   → insight extraction loop ⟲ (until all KPIs scored / confidence threshold met)
   → narrative assembly → visualization → quality gate
        ├─ pass → personalization & distribution → knowledge capture
        └─ fail → rerun insight loop, or escalate to human review
```

| Step | Agent (CrewAI role) | What it does |
|---|---|---|
| Data intake | Telemetry Listener | Loads KPI / event snapshots, filters noise |
| Audience profiling | Stakeholder Classifier | Tags the audience and picks the template + tone branch |
| Context enrichment | Context Enricher | Pulls SLA targets, changelog, prior incidents via RAG |
| Insight extraction | Insight Miner | Per-metric trend + anomaly analysis, looped until confident |
| Narrative | Report Writer | Drafts audience-specific sections |
| Visualization | Visualization Bot | Picks chart types and generates charts |
| Quality gate | Consistency Checker | Cross-checks numbers, units and narrative against data |
| Distribution | Delivery Orchestrator | Tailors length/tone and logs delivery for audit |

Design notes: [`Automated_reporting.md`](Automated_reporting.md).

## How to run
```bash
pip install -r requirements.txt
cp .env.example .env        # add AWS Bedrock credentials (OpenAI optional fallback)
python -c "import sys; sys.path.insert(0, 'src'); from automated_reporting.cli import main; main(['generate-reports', '--session', 'demo'])"
```
Flags: `--no-rag` disables retrieval, `--quiet` reduces output. Sample telemetry lives in `fixtures/` (business KPIs, system metrics, SLA performance, quality indicators, baselines).

## Project structure
```
src/automated_reporting/
  agents/      CrewAI agent factory, tasks, crew orchestration
  workflow/    LangGraph state machine + orchestrator
  rag/         ChromaDB + Bedrock embedding pipeline
  artifacts/   report + chart generators
  data/        fixture loading
  cli.py       entry point
fixtures/      sample telemetry datasets
```

## Tech stack
Python · LangGraph · CrewAI · LangChain · AWS Bedrock (Claude 3.5, Titan embeddings) · ChromaDB · Pydantic
