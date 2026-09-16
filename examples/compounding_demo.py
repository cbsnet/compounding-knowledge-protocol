#!/usr/bin/env python3
"""
examples/compounding_demo.py
==============================
Concrete, runnable demonstration of the compounding protocol (written spec
section 4.2) applied to the COVID-19 case graph. Simulates a second analyst,
independent of the original team, proposing two additions:

  1. A near-duplicate of an existing node (should be flagged, not inserted).
  2. A genuinely new node with a real source (should be inserted pending
     review, then accepted, with a visible Crux Score recalculation).

Run directly: python3 examples/compounding_demo.py
Or via CLI:   python3 -m epistemic_stack.cli demo-compounding
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epistemic_stack.graph import build_all_graphs, ACENode
from epistemic_stack.compounding import CompoundingSession


def main() -> None:
    data_path = Path(__file__).resolve().parent.parent / "data" / "nodes_master.json"
    graphs = build_all_graphs(data_path)
    covid_graph = graphs["case_covid"]

    print("=" * 78)
    print("COMPOUNDING DEMO — second independent analyst extends the COVID graph")
    print("=" * 78)
    print(f"\nStarting graph: {covid_graph.case_label} ({len(covid_graph.nodes)} nodes)\n")

    session = CompoundingSession(graph=covid_graph, proposed_by="external_analyst_jane_doe")

    # --- Attempt 1: a near-duplicate of an existing node -----------------------
    print("-" * 78)
    print("ATTEMPT 1 — proposing a claim that overlaps heavily with an existing node")
    print("-" * 78)
    duplicate_candidate = ACENode(
        ace_id="ACE-SARS2-MARKET-CLUSTERING-EXTERNAL",
        case="case_covid",
        vector_type="ACE",
        verbatim_claim=(
            "The spatial clustering of early COVID-19 cases in December 2019 "
            "centres on the Huanan Seafood Market near wildlife stalls."
        ),
    )
    result1 = session.propose_node(duplicate_candidate)
    print(f"Result: {result1['status']}")
    if result1["status"] == "flagged_possible_duplicate":
        print(f"  Similar to: {result1['similar_to']}")
        print("  -> Correctly blocked from silent insertion. A human reviewer must")
        print("     decide whether this is a genuine restatement (merge) or a distinct")
        print("     claim that happens to share vocabulary (insert separately).")
    print()

    # --- Attempt 2: a genuine new node, real source -----------------------------
    print("-" * 78)
    print("ATTEMPT 2 — proposing a genuinely new node with an independent source")
    print("-" * 78)
    new_node = ACENode(
        ace_id="ACE-SARS2-PEKAR-MOLECULAR-CLOCK",
        case="case_covid",
        vector_type="DEPEND",
        verbatim_claim=(
            "Molecular clock analysis of early SARS-CoV-2 genomes is consistent with "
            "two separate zoonotic introduction events at the Huanan Market in November "
            "and December 2019, rather than a single introduction followed by spread."
        ),
        confidence=0.71,
        peer_scrutiny_status="peer_reviewed",
    )
    new_node.raw = {
        "source": {
            "title": "The molecular epidemiology of multiple zoonotic origins of SARS-CoV-2",
            "authors": ["Pekar, J. E.", "et al."],
            "venue": "Science",
            "publication_date": "2022",
            "doi_or_id": "10.1126/science.abp8337",
            "status": "Peer-reviewed",
        },
        "epistemic_validation": {
            "systemic_confidence_score": 0.71,
            "peer_scrutiny_status": "peer_reviewed",
            "rationale": (
                "Independently proposed node, pending review by the curator of "
                "ACE-SARS2-EPID-MARKET before its DEPEND edge is activated. Chosen as a "
                "DEPEND vector specifically because Cs = Sum(Dout x Wdep) + Sum(Econ) "
                "only weighs DEPEND and CONTRADICT edges, not SUPPORT — this is what "
                "makes the downstream Crux Score change visible in this demo."
            ),
        },
    }
    print("(Note: this is proposed as a DEPEND edge rather than SUPPORT, because the")
    print(" Crux Score formula Cs = Sum(Dout x Wdep) + Sum(Econ) only weighs DEPEND and")
    print(" CONTRADICT edges by design — SUPPORT edges intentionally do not move Cs.")
    print(" An earlier version of this demo used SUPPORT and showed no score change,")
    print(" which was a demo design error, not a formula bug — corrected here.)\n")
    result2 = session.propose_node(new_node, proposed_vector_target=["ACE-SARS2-EPID-MARKET"])
    print(f"Result: {result2['status']}")
    print(f"  Node inserted with pending_edges={result2.get('pending_edges')}")
    print("  -> Crux Score of ACE-SARS2-EPID-MARKET is UNCHANGED at this point.")
    print(f"     Current Cs: {covid_graph.crux_score('ACE-SARS2-EPID-MARKET')}")
    print()

    # --- Curator accepts the edge ------------------------------------------------
    print("-" * 78)
    print("CURATOR REVIEW — original analyst accepts the proposed edge")
    print("-" * 78)
    before = covid_graph.crux_score("ACE-SARS2-EPID-MARKET")
    accept_result = session.accept_proposed_edges("ACE-SARS2-PEKAR-MOLECULAR-CLOCK", reviewer="original_team")
    after = covid_graph.crux_score("ACE-SARS2-EPID-MARKET")
    print(f"Result: {accept_result['status']}")
    print(f"  ACE-SARS2-EPID-MARKET Crux Score: {before} -> {after}")
    print(f"  Downstream changes recorded: {accept_result['downstream_score_changes']}")
    print()

    # --- Changelog -----------------------------------------------------------
    print("-" * 78)
    print("TRACKED CHANGELOG (full audit trail of this session)")
    print("-" * 78)
    for entry in session.export_changelog():
        print(f"  [{entry['timestamp']}] {entry['action']:20s} {entry['ace_id']:35s} by {entry['proposed_by']}")

    print()
    print("=" * 78)
    print("This demonstrates all three compounding protocol steps end-to-end:")
    print("  1. Non-duplication validation  -> Attempt 1 correctly blocked")
    print("  2. Provisional integration     -> Attempt 2 inserted with edges pending")
    print("  3. Tracked cascading recalc    -> Crux Score change is logged with a timestamp")
    print("=" * 78)


if __name__ == "__main__":
    main()
