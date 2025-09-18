"""Command-line interface for the automated reporting application."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from .workflow.orchestrator import SimpleAutomatedReportingRunner, SimpleReportingWorkflowSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="automated-reporting", description="Automated reporting multi-agent system CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reporting_parser = subparsers.add_parser(
        "generate-reports",
        help="Run the automated reporting workflow",
    )
    reporting_parser.add_argument(
        "--session",
        metavar="NAME",
        default=None,
        help="Optional session label for the run",
    )
    reporting_parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG pipeline (for testing)",
    )
    reporting_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbose output",
    )

    return parser


def handle_generate_reports(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parent.parent.parent
    settings = SimpleReportingWorkflowSettings(
        project_root=project_root,
        session_label=args.session,
        use_rag=not args.no_rag,
        verbose=not args.quiet,
    )
    runner = SimpleAutomatedReportingRunner(settings)

    print("Starting Automated Reporting Multi-Agent System")
    print("=" * 50)

    # Show component status
    status = runner.get_status()
    print("\nComponent Status:")
    for component, available in status.items():
        icon = "[OK]" if available else "[FAIL]"
        print(f"  {icon} {component}: {'Available' if available else 'Not Available'}")

    print("\nRunning automated reporting workflow...")
    results = runner.run()

    print("\nAutomated Reporting Demo complete!")
    print(f"Results: {results}")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-reports":
        return handle_generate_reports(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())