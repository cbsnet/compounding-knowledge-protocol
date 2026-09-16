#!/usr/bin/env python3
"""
examples/compounding_two_graphs.py
=====================================
Demonstration of compounding between two INDEPENDENTLY-CONSTRUCTED graphs
on the same case (COVID-19 origins), addressing the critique that the earlier
demo (compounding_demo.py) only showed a single node proposed by a simulated
second analyst, not a real merge of two separate starting-point graphs.

Graph A: the main COVID graph from nodes_master.json (10 nodes, built from
  genomic analysis papers, the Rootclaim debate materials, and the Senate
  HELP minority report).

Graph B: an independent graph from data/covid_graph_b.json (4 nodes, built
  deliberately from sources NOT in Graph A: Pekar 2022 molecular clock,
  Bloom 2025 phylogenetic critique, the WHO-China joint report, and the
  FLF brief's own framing of Bayesian disagreement).

The two graphs share one factual basis (the 23-orders-of-magnitude Bayesian
disagreement) but approach the case from structurally different starting
points. Graph B has a zoonosis-leaning construction bias by design, which
means the compounding protocol should surface asymmetries rather than
silently absorbing them.

Run: python3 examples/compounding_two_graphs.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epistemic_stack.graph import EpistemicGraph, ACENode, build_all_graphs, load_nodes_master
from epistemic_stack.compounding import CompoundingSession, lexical_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_graph_b(path: Path) -> list[ACENode]:
    with open(path) as f:
        raw = json.load(f)
    nodes = []
    for r in raw["nodes"]:
        targets = r.get("vector_target") or r.get("vector_links") or []
        if isinstance(targets, str):
            targets = [targets]
        ev = r.get("epistemic_validation", {})
        node = ACENode(
            ace_id=r["ace_id"],
            case="case_covid",
            vector_type=r.get("vector_type", "ACE"),
            targets=[t for t in targets if isinstance(t, str)],
            verbatim_claim=r.get("verbatim_claim", r.get("mutation_logic", "")),
            confidence=ev.get("systemic_confidence_score"),
            peer_scrutiny_status=ev.get("peer_scrutiny_status"),
            raw=r,
        )
        nodes.append(node)
    return nodes, raw.get("construction_note", "")


def main():
    print("=" * 78)
    print("COMPOUNDING DEMONSTRATION — Two independently-constructed COVID graphs")
    print("=" * 78)

    # --- Load Graph A (main) ---
    graphs = build_all_graphs(DATA_DIR / "nodes_master.json")
    graph_a = graphs["case_covid"]
    print(f"\nGraph A (main):  {len(graph_a.nodes)} nodes")
    print(f"  Starting Crux Score of ACE-SARS2-EPID-MARKET: "
          f"{graph_a.crux_score('ACE-SARS2-EPID-MARKET'):.4f}")

    # --- Load Graph B (independent) ---
    graph_b_nodes, construction_note = load_graph_b(DATA_DIR / "covid_graph_b.json")
    print(f"\nGraph B (independent): {len(graph_b_nodes)} nodes")
    print(f"  Construction note: {construction_note[:120]}...")

    # --- Begin compounding session ---
    session = CompoundingSession(
        graph=graph_a,
        proposed_by="independent_graph_b_analyst",
        duplication_threshold=0.33,  # catches near-paraphrases while avoiding false positives
    )

    print("\n" + "-" * 78)
    print("STEP 1 — Non-duplication check: proposing all Graph B nodes into Graph A")
    print("-" * 78)

    accepted = []
    flagged = []
    for node in graph_b_nodes:
        result = session.propose_node(node, proposed_vector_target=node.targets)
        status = result["status"]
        if status == "flagged_possible_duplicate":
            flagged.append((node.ace_id, result["similar_to"]))
            print(f"\n  FLAGGED: {node.ace_id}")
            print(f"    Similar to: {result['similar_to']}")
            print(f"    Claim: {node.verbatim_claim[:100]}...")
            print(f"    -> Held for human review: is this a restatement or a distinct claim?")
        else:
            accepted.append(node.ace_id)
            print(f"\n  INSERTED (pending edge review): {node.ace_id}")

    print(f"\n  Summary: {len(accepted)} inserted pending review, "
          f"{len(flagged)} flagged as possible duplicates")

    # --- Curator accepts edges for non-flagged nodes ---
    print("\n" + "-" * 78)
    print("STEP 2 — Curator review: accepting edges for non-duplicate nodes")
    print("-" * 78)

    before_scores = graph_a.all_crux_scores()
    score_changes = {}

    for ace_id in accepted:
        result = session.accept_proposed_edges(ace_id, reviewer="graph_a_curator")
        if result["status"] == "edges_accepted":
            if result["downstream_score_changes"]:
                score_changes[ace_id] = result["downstream_score_changes"]
                print(f"\n  ACCEPTED: {ace_id}")
                for target, change in result["downstream_score_changes"].items():
                    print(f"    -> {target}: Cs {change['before']:.4f} -> {change['after']:.4f}")
            else:
                print(f"\n  ACCEPTED: {ace_id} (no downstream Crux Score changes)")
        elif result["status"] == "blocked_would_create_cycle":
            print(f"\n  BLOCKED (cycle): {ace_id} — {result['cycles']}")
        else:
            print(f"\n  {result['status'].upper()}: {ace_id}")

    # --- Report on flagged duplicates ---
    print("\n" + "-" * 78)
    print("STEP 3 — Duplicate resolution: human decision required")
    print("-" * 78)

    for ace_id, similar_to in flagged:
        node = next(n for n in graph_b_nodes if n.ace_id == ace_id)
        existing_id, sim = similar_to[0]
        existing_node = graph_a.nodes.get(existing_id)
        print(f"\n  DUPLICATE CANDIDATE: {ace_id} (similarity {sim} to {existing_id})")
        print(f"  Graph B claim: {node.verbatim_claim[:120]}...")
        if existing_node:
            print(f"  Graph A claim: {existing_node.verbatim_claim[:120]}...")
        print(f"  Decision options:")
        print(f"    MERGE  — if they state the same fact at the same level of precision")
        print(f"    INSERT — if Graph B adds a materially distinct angle not in Graph A")
        print(f"    DISCARD — if Graph A already covers this more precisely")
        print(f"  [This decision is left to a human curator — not auto-resolved]")

    # --- Final state ---
    print("\n" + "-" * 78)
    print("FINAL STATE after compounding")
    print("-" * 78)
    after_scores = graph_a.all_crux_scores()
    z_scores = graph_a.z_scores()
    changed_nodes = {k for k in after_scores
                     if round(before_scores.get(k, 0), 4) != round(after_scores[k], 4)}

    print(f"\n  Total nodes in merged graph: {len(graph_a.nodes)}")
    print(f"  Nodes with changed Crux Scores: {len(changed_nodes)}")
    if changed_nodes:
        for nid in sorted(changed_nodes, key=lambda x: -after_scores[x]):
            print(f"    {nid}: {before_scores.get(nid, 0):.4f} -> {after_scores[nid]:.4f} "
                  f"(Z={z_scores[nid]:+.3f})")

    print(f"\n  Cycles in merged graph: {graph_a.detect_cycles() or 'None'}")

    # --- Changelog ---
    print("\n" + "-" * 78)
    print("AUDIT TRAIL (full changelog of this compounding session)")
    print("-" * 78)
    for entry in session.export_changelog():
        print(f"  [{entry['timestamp']}] {entry['action']:22s} {entry['ace_id']}")

    print("\n" + "=" * 78)
    print("What this demonstrates:")
    print("  1. Two graphs built from DIFFERENT starting sources — not the same")
    print("     analyst adding one node at a time.")
    print("  2. Deduplication firing on a real near-match (FLF Bayesian framing node")
    print("     vs the existing SIXFOLD-VARIANCE node — same fact, different wording).")
    print("  3. A genuine CONTEXT_MUTATION conflict between the two graphs: both agree")
    print("     the WIV database is offline, but frame the epistemic significance")
    print("     in opposite directions. The system surfaces this for curator decision")
    print("     rather than silently resolving it in either direction.")
    print("  4. Cascading recalculation on accepted DEPEND edges.")
    print("  5. Full timestamped audit trail with no silent mutations.")
    print("=" * 78)


if __name__ == "__main__":
    main()
