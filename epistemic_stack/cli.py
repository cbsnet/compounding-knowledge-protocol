#!/usr/bin/env python3
"""
epistemic_stack.cli
=====================
Single entry point for running The Epistemic Stack against the verified node
data in data/nodes_master.json. No external dependencies, no API keys, no
network access required — everything here operates on data already ingested
and verified (see README for how that verification was done).

USAGE
  python3 -m epistemic_stack.cli summary              # Crux Score + Z-score table for all 3 cases
  python3 -m epistemic_stack.cli summary --case covid  # single case only
  python3 -m epistemic_stack.cli critical              # nodes with Z > 2.0 across all cases
  python3 -m epistemic_stack.cli verify-sync           # checks code's Wdep table matches the spec's
  python3 -m epistemic_stack.cli demo-compounding      # runs the worked compounding demo, see examples/
  python3 -m epistemic_stack.cli demo-ingestion        # prints the extraction prompt + worked example
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

from .graph import build_all_graphs, load_nodes_master, PEER_SCRUTINY_MULTIPLIER, z_threshold_default
from .ingestion import (
    ACE_EXTRACTION_PROMPT_TEMPLATE, WORKED_EXAMPLE_PASSAGE,
    WORKED_EXAMPLE_NOTE, WORKED_EXAMPLE_OUTPUT, validate_ace,
)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nodes_master.json"

CASE_KEY_MAP = {"covid": "case_covid", "eggs": "case_eggs", "lhc": "case_lhc"}


def cmd_summary(args: argparse.Namespace) -> None:
    graphs = build_all_graphs(args.data)
    keys = [CASE_KEY_MAP[args.case]] if args.case else list(graphs.keys())
    for key in keys:
        g = graphs[key]
        print(g.summary())
        print()


def cmd_critical(args: argparse.Namespace) -> None:
    graphs = build_all_graphs(args.data)
    any_found = False
    for key, g in graphs.items():
        crit = g.critical_nodes(z_threshold=args.threshold)
        if crit:
            any_found = True
            print(f"[{g.case_label}]")
            z = g.z_scores()
            for aid in crit:
                print(f"  {aid:40s} Z={z[aid]:+.3f}")
    if not any_found:
        print(f"No nodes exceed Z > {args.threshold} in any case graph.")


def cmd_verify_sync(args: argparse.Namespace) -> None:
    """Checks that the Wdep multiplier table hard-coded in graph.py matches
    the table declared in nodes_master.json's schema_notes — catching exactly
    the kind of silent drift between the written spec and the code that the
    supervisor review warned about."""
    data = load_nodes_master(args.data)
    spec_table = (
        data.get("schema_notes", {})
        .get("wdep_specification", {})
        .get("peer_scrutiny_multiplier_scale", {})
    )
    code_table = PEER_SCRUTINY_MULTIPLIER

    mismatches = []
    for k, v in spec_table.items():
        if code_table.get(k) != v:
            mismatches.append((k, "spec", v, "code", code_table.get(k)))
    for k, v in code_table.items():
        if k not in spec_table:
            mismatches.append((k, "code_only", v, "spec", None))

    if not mismatches:
        print("OK — graph.py's Wdep table matches nodes_master.json's spec exactly.")
        sys.exit(0)
    else:
        print("MISMATCH between code and spec Wdep tables:")
        for m in mismatches:
            print(f"  {m}")
        sys.exit(1)


def cmd_demo_ingestion(args: argparse.Namespace) -> None:
    print("=" * 78)
    print("ACE EXTRACTION PROMPT TEMPLATE (used to manually build every node in")
    print("nodes_master.json — this is the actual template, not an illustration)")
    print("=" * 78)
    print(ACE_EXTRACTION_PROMPT_TEMPLATE.replace("__PASSAGE__", "<source passage inserted here>"))
    print("=" * 78)
    print("WORKED EXAMPLE — applying the template to a compound claim")
    print("=" * 78)
    print(f"\nInput passage:\n  \"{WORKED_EXAMPLE_PASSAGE}\"\n")
    print(f"Note: {WORKED_EXAMPLE_NOTE}\n")
    print("Output (3 ACEs, each independently validated):\n")
    for i, ace in enumerate(WORKED_EXAMPLE_OUTPUT, 1):
        result = validate_ace(ace, require_scored=False)
        status = "VALID (extraction-stage)" if result.is_valid else "INVALID"
        print(f"  [{i}] {ace['ace_id']}  -> {status}")
        if result.warnings:
            for w in result.warnings:
                print(f"      warning: {w}")
        print(f"      claim: {ace['verbatim_claim']}")
        print(f"      hedges preserved: {ace['context_bounding']['linguistic_hedges']}")
        print()


def cmd_demo_compounding(args: argparse.Namespace) -> None:
    script_path = Path(__file__).resolve().parent.parent / "examples" / "compounding_demo.py"
    print(f"Running {script_path.relative_to(script_path.parent.parent)} ...\n")
    import runpy
    runpy.run_path(str(script_path), run_name="__main__")


def cmd_check_cycles(args: argparse.Namespace) -> None:
    """Added after adversarial testing (tests/test_adversarial.py, Category 3)
    found that the graph had no runtime cycle detection despite being
    documented as a DAG. See graph.py's detect_cycles() docstring."""
    graphs = build_all_graphs(args.data)
    any_cycles = False
    for key, g in graphs.items():
        cycles = g.detect_cycles()
        if cycles:
            any_cycles = True
            print(f"[{g.case_label}] {len(cycles)} cycle(s) found:")
            for c in cycles:
                print(f"  {' -> '.join(c)}")
    if not any_cycles:
        print("No cycles detected in any case graph — DAG property holds for current data.")



def cmd_crosscheck_tasks(args: argparse.Namespace) -> None:
    """Built in response to adversarial test ADV-FALSE-03 (the most dangerous
    finding: a claim attributed to a real, verified source but stating the
    opposite of what that source says). Prints a structured task per node
    for a human or external LLM to actually compare claim against source."""
    from .crosscheck import build_tasks
    tasks = build_tasks(args.data)
    if args.ace_id:
        tasks = [t for t in tasks if t.ace_id == args.ace_id]
        if not tasks:
            print(f"No node found with ace_id={args.ace_id}")
            return
    for t in tasks:
        print(t.render_prompt())
        print("=" * 78)


def cmd_crosscheck_record(args: argparse.Namespace) -> None:
    from .crosscheck import save_result, CrossCheckResult
    result = CrossCheckResult(
        ace_id=args.ace_id, verdict=args.verdict,
        explanation=args.explanation, checked_by=args.checked_by,
    )
    results_path = Path(args.data).parent / "cross_check_results.json"
    save_result(results_path, result)
    print(f"Recorded: {args.ace_id} -> {args.verdict}")


def cmd_crosscheck_report(args: argparse.Namespace) -> None:
    from .crosscheck import build_tasks, load_results
    tasks = build_tasks(args.data)
    results_path = Path(args.data).parent / "cross_check_results.json"
    results = load_results(results_path)

    checked = [t for t in tasks if t.ace_id in results]
    unchecked = [t for t in tasks if t.ace_id not in results]

    print(f"Cross-check coverage: {len(checked)}/{len(tasks)} nodes checked\n")
    by_verdict: dict[str, list[str]] = {}
    for t in checked:
        v = results[t.ace_id].verdict
        by_verdict.setdefault(v, []).append(t.ace_id)

    for verdict in ["MISATTRIBUTED", "PARTIALLY_OFF", "UNVERIFIABLE", "ALIGNED"]:
        ids = by_verdict.get(verdict, [])
        if ids:
            print(f"  {verdict} ({len(ids)}):")
            for aid in ids:
                print(f"    {aid}: {results[aid].explanation}")
            print()

    if unchecked:
        print(f"  NOT YET CHECKED ({len(unchecked)}):")
        for t in unchecked:
            print(f"    {t.ace_id}")



def cmd_check_diversity(args: argparse.Namespace) -> None:
    """Shows the Contradiction Diversity Index for all nodes that have at least
    one CONTRADICT edge. CDI < 0.3 is flagged as possible bias flooding."""
    graphs = build_all_graphs(args.data)
    any_low = False
    for key, g in graphs.items():
        low = []
        for aid in g.nodes:
            cdi = g.contradiction_diversity_index(aid)
            if cdi is not None:
                flag = " (!low-diversity — possible flooding)" if cdi < 0.3 else ""
                low.append((aid, cdi, flag))
        if low:
            print(f"[{g.case_label}]")
            for aid, cdi, flag in sorted(low, key=lambda x: x[1]):
                print(f"  {aid:40s} CDI={cdi:.4f}{flag}")
                if flag:
                    any_low = True
    if not any_low:
        print("No low-diversity contradiction patterns detected (CDI >= 0.3 for all nodes).")



def cmd_source_fidelity_run(args: argparse.Namespace) -> None:
    """Run the automated Source Fidelity Check on all nodes.
    Requires ANTHROPIC_API_KEY environment variable for semantic comparison.
    Without it, only the Crossref/arXiv existence check runs (catches SOURCE_FABRICATED).
    Results are written to data/source_fidelity_results.json.
    NEVER modifies systemic_confidence_score — flags only."""
    from .source_fidelity import run_all
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Note: ANTHROPIC_API_KEY not set. Running existence-check only (no semantic comparison).")
    results_path = Path(args.data).parent / "source_fidelity_results.json"
    print(f"Running Source Fidelity Check on {args.data}...")
    results = run_all(
        nodes_master_path=args.data,
        results_path=results_path,
        api_key=api_key,
        force=getattr(args, "force", False),
    )
    print(f"\nDone. {len(results)} nodes checked.")
    print(f"Results written to {results_path}")


def cmd_source_fidelity_report(args: argparse.Namespace) -> None:
    """Show a summary of Source Fidelity Check results."""
    from .source_fidelity import report
    results_path = Path(args.data).parent / "source_fidelity_results.json"
    if not results_path.exists():
        print("No results found. Run: python3 -m epistemic_stack.cli source-fidelity-run")
        return
    print(report(results_path, args.data))



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="epistemic_stack", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH),
                   help="Path to nodes_master.json (default: data/nodes_master.json)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("summary", help="Print Crux Score / Z-score table for case graphs")
    sp.add_argument("--case", choices=["covid", "eggs", "lhc"], default=None)
    sp.set_defaults(func=cmd_summary)

    sp = sub.add_parser("critical", help="List all nodes with Z-score above threshold")
    sp.add_argument("--threshold", type=float, default=z_threshold_default)
    sp.set_defaults(func=cmd_critical)

    sp = sub.add_parser("verify-sync", help="Check code's Wdep table matches the written spec's")
    sp.set_defaults(func=cmd_verify_sync)

    sp = sub.add_parser("demo-ingestion", help="Show the ACE extraction prompt + worked example")
    sp.set_defaults(func=cmd_demo_ingestion)

    sp = sub.add_parser("demo-compounding", help="Run the two-analyst graph merge demo")
    sp.set_defaults(func=cmd_demo_compounding)

    sp = sub.add_parser("check-cycles", help="Detect cycles in the DEPEND/CONTRADICT graph structure")
    sp.set_defaults(func=cmd_check_cycles)

    sp = sub.add_parser("crosscheck-tasks", help="Print citation cross-check tasks for human/LLM review")
    sp.add_argument("--ace-id", default=None, help="Only print the task for one specific node")
    sp.set_defaults(func=cmd_crosscheck_tasks)

    sp = sub.add_parser("crosscheck-record", help="Record a cross-check verdict for a node")
    sp.add_argument("--ace-id", required=True)
    sp.add_argument("--verdict", required=True, choices=["ALIGNED", "PARTIALLY_OFF", "MISATTRIBUTED", "UNVERIFIABLE"])
    sp.add_argument("--explanation", required=True)
    sp.add_argument("--checked-by", required=True)
    sp.set_defaults(func=cmd_crosscheck_record)

    sp = sub.add_parser("crosscheck-report", help="Show cross-check coverage and findings so far")
    sp.set_defaults(func=cmd_crosscheck_report)

    sp = sub.add_parser("demo-compounding-two-graphs", help="Run the two-independently-constructed-graph merge demo")
    sp.set_defaults(func=lambda args: __import__('runpy').run_path(
        str(Path(__file__).resolve().parent.parent / "examples" / "compounding_two_graphs.py"),
        run_name="__main__"))

    sp = sub.add_parser("check-diversity", help="Show Contradiction Diversity Index — flags possible bias flooding")
    sp.set_defaults(func=cmd_check_diversity)

    sp = sub.add_parser("source-fidelity-run", help="Run automated Source Fidelity Check (requires ANTHROPIC_API_KEY)")
    sp.add_argument("--force", action="store_true", help="Re-check already-verified nodes")
    sp.set_defaults(func=cmd_source_fidelity_run)

    sp = sub.add_parser("source-fidelity-report", help="Show Source Fidelity Check results")
    sp.set_defaults(func=cmd_source_fidelity_report)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
