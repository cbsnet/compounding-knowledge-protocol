"""
tests/test_core.py
====================
Minimal automated test suite. Run with: python3 -m pytest tests/ -v
(or python3 tests/test_core.py if pytest is unavailable — see __main__ block).

These tests lock in behaviors that were manually verified during development
and would otherwise be easy to silently break:
  - Wdep code/spec sync (graph.py's table must match nodes_master.json's)
  - Crux Score correctly differentiates a contested node from a solid one
  - SUPPORT edges do not move Crux Score; DEPEND/CONTRADICT do (by design)
  - Compounding: duplication detection fires; accepted edges trigger a
    visible, logged downstream recalculation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epistemic_stack.graph import build_all_graphs, load_nodes_master, PEER_SCRUTINY_MULTIPLIER, ACENode
from epistemic_stack.compounding import CompoundingSession
from epistemic_stack.ingestion import validate_ace

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nodes_master.json"


def test_wdep_sync():
    data = load_nodes_master(DATA_PATH)
    spec_table = data["schema_notes"]["wdep_specification"]["peer_scrutiny_multiplier_scale"]
    assert spec_table == PEER_SCRUTINY_MULTIPLIER, (
        "graph.py's PEER_SCRUTINY_MULTIPLIER has drifted from nodes_master.json's "
        "documented Wdep specification — these must be kept in sync manually."
    )


def test_contested_node_scores_higher_than_solid_node():
    """ACE-SARS2-GENOME-FCS (contested preprint) should NOT score identically to
    ACE-SARS2-EPID-MARKET (peer-reviewed, Science) — this was a real bug found
    by running the code (unweighted Econ gave them equal scores)."""
    graphs = build_all_graphs(DATA_PATH)
    g = graphs["case_covid"]
    cs_fcs = g.crux_score("ACE-SARS2-GENOME-FCS")
    cs_market = g.crux_score("ACE-SARS2-EPID-MARKET")
    assert cs_fcs != cs_market, (
        "Contested and solid nodes received identical Crux Scores — the "
        "contradiction-weighting fix may have regressed."
    )


def test_support_edges_do_not_affect_crux_score():
    g = build_all_graphs(DATA_PATH)["case_covid"]
    target = "ACE-SARS2-EPID-MARKET"
    before = g.crux_score(target)
    support_node = ACENode(
        ace_id="TEST-SUPPORT-ONLY", case="case_covid", vector_type="SUPPORT",
        targets=[target], confidence=0.9, peer_scrutiny_status="peer_reviewed",
        verbatim_claim="test support claim unrelated to existing wording",
    )
    g.add_node(support_node)
    after = g.crux_score(target)
    assert before == after, "SUPPORT edges should not move Crux Score by formula design."


def test_dependency_edge_triggers_visible_recalculation():
    g = build_all_graphs(DATA_PATH)["case_covid"]
    session = CompoundingSession(graph=g, proposed_by="test_analyst")
    target = "ACE-SARS2-EPID-MARKET"
    before = g.crux_score(target)

    new_node = ACENode(
        ace_id="TEST-DEPEND-NODE", case="case_covid", vector_type="DEPEND",
        confidence=0.8, peer_scrutiny_status="peer_reviewed",
        verbatim_claim="a genuinely distinct claim about an unrelated genomic dataset entirely",
    )
    result = session.propose_node(new_node, proposed_vector_target=[target])
    assert result["status"] == "inserted_pending_review"
    assert g.crux_score(target) == before, "Score must not change before curator acceptance."

    accept = session.accept_proposed_edges("TEST-DEPEND-NODE", reviewer="test_curator")
    after = g.crux_score(target)
    assert after != before, "Score must change after acceptance — cascading recalculation failed."
    assert target in accept["downstream_score_changes"], "Change must be logged for the affected node."


def test_cdi_same_source_returns_zero():
    """CDI=0 when all CONTRADICT edges come from the same analyst source.
    With a single-team dataset, this is expected — not a false positive."""
    g = build_all_graphs(DATA_PATH)["case_covid"]
    cdi = g.contradiction_diversity_index("ACE-SARS2-EPID-MARKET")
    # Two CONTRADICT edges but both from original_team → CDI=0
    assert cdi == 0.0, f"Expected 0.0 for same-source contradictions, got {cdi}"


def test_cdi_independent_analyst_raises_diversity():
    """CDI rises toward 1.0 when a genuinely independent analyst adds a CONTRADICT edge.
    This is the mechanism the compounding protocol enables."""
    g = build_all_graphs(DATA_PATH)["case_covid"]
    session = CompoundingSession(graph=g, proposed_by="independent_analyst_x")
    node = ACENode(
        ace_id="TEST-CDI-IND", case="case_covid", vector_type="CONTRADICT",
        targets=["ACE-SARS2-EPID-MARKET"],
        verbatim_claim="independent critique from a distinct analyst with entirely different framing of the market evidence",
        confidence=0.6, peer_scrutiny_status="peer_reviewed",
    )
    session.propose_node(node, proposed_vector_target=["ACE-SARS2-EPID-MARKET"])
    session.accept_proposed_edges("TEST-CDI-IND", reviewer="test_curator")
    cdi = g.contradiction_diversity_index("ACE-SARS2-EPID-MARKET")
    assert cdi > 0.5, f"Expected CDI > 0.5 with independent analyst, got {cdi}"



    """Regression test for the cycle found by tests/test_adversarial.py
    Category 3: two nodes proposed as mutual DEPEND targets must not be
    silently accepted — the cycle must be detected and the edges rolled back."""
    g = build_all_graphs(DATA_PATH)["case_covid"]
    session = CompoundingSession(graph=g, proposed_by="adversarial_test")

    node_a = ACENode(
        ace_id="TEST-CIRC-A", case="case_covid", vector_type="DEPEND",
        confidence=0.7, peer_scrutiny_status="peer_reviewed",
        verbatim_claim="genomic sequencing anomaly indicator requires upstream lineage verification first",
    )
    node_b = ACENode(
        ace_id="TEST-CIRC-B", case="case_covid", vector_type="DEPEND",
        confidence=0.7, peer_scrutiny_status="peer_reviewed",
        verbatim_claim="epidemiological cluster timing assessment needs prior genomic confirmation step",
    )
    session.propose_node(node_a, proposed_vector_target=["TEST-CIRC-B"])
    session.propose_node(node_b, proposed_vector_target=["TEST-CIRC-A"])

    session.accept_proposed_edges("TEST-CIRC-A", reviewer="test_curator")
    result_b = session.accept_proposed_edges("TEST-CIRC-B", reviewer="test_curator")

    assert result_b["status"] == "blocked_would_create_cycle", (
        f"Expected the second acceptance to be blocked by cycle detection, got: {result_b['status']}"
    )
    cycles = g.detect_cycles()
    assert not cycles, f"Graph should have zero active cycles after the block+rollback, found: {cycles}"


def test_duplication_detection_fires_on_near_identical_claim():
    g = build_all_graphs(DATA_PATH)["case_covid"]
    session = CompoundingSession(graph=g, proposed_by="test_analyst")
    near_dup = ACENode(
        ace_id="TEST-DUP", case="case_covid", vector_type="ACE",
        verbatim_claim="The spatial clustering of early COVID-19 cases in December 2019 centres on the Huanan Seafood Market near wildlife stalls.",
    )
    result = session.propose_node(near_dup)
    assert result["status"] == "flagged_possible_duplicate"
    assert len(result["similar_to"]) > 0


def test_ace_validator_rejects_missing_status():
    bad = {"ace_id": "X", "verbatim_claim": "y", "source": {"title": "t"}}
    result = validate_ace(bad, require_scored=False)
    assert not result.is_valid
    assert any("status" in e for e in result.errors)


def test_ace_validator_requires_scoring_block_when_require_scored():
    candidate = {
        "ace_id": "X", "verbatim_claim": "y",
        "source": {"title": "t", "status": "Peer-reviewed"},
        "epistemic_validation": None,
    }
    result = validate_ace(candidate, require_scored=True)
    assert not result.is_valid


def test_source_fidelity_fabricated_doi_detection():
    """SOURCE_FABRICATED is returned when the DOI existence check fails.
    Uses a mock to avoid network dependency in the test suite."""
    from epistemic_stack.source_fidelity import run_fidelity_check, SOURCE_FABRICATED

    fake_node = {
        "ace_id": "ADV-FALSE-01",
        "verbatim_claim": "A fabricated claim attributed to a non-existent paper.",
        "source": {
            "doi_or_id": "10.1016/j.jve.2023.00412",
            "status": "Peer-reviewed",
        }
    }

    import epistemic_stack.source_fidelity as sf
    original = sf._check_doi_exists
    sf._check_doi_exists = lambda doi: False

    try:
        result = run_fidelity_check(fake_node, api_key=None)
        assert result.status == SOURCE_FABRICATED, \
            f"Expected SOURCE_FABRICATED, got {result.status}"
        assert "confirmed absent" in result.detail.lower() or "fabricated" in result.status.lower()
    finally:
        sf._check_doi_exists = original


def test_source_fidelity_real_doi_passes_existence():
    """A real DOI that passes existence check proceeds to fetch stage."""
    from epistemic_stack.source_fidelity import run_fidelity_check, SOURCE_FABRICATED

    real_node = {
        "ace_id": "ACE-TEST-REAL-DOI",
        "verbatim_claim": "A claim attributed to a real, existing paper.",
        "source": {
            "doi_or_id": "10.1126/science.abp8715",
            "status": "Peer-reviewed",
        }
    }

    import epistemic_stack.source_fidelity as sf
    original_doi = sf._check_doi_exists
    original_fetch = sf._fetch_text
    sf._check_doi_exists = lambda doi: True
    sf._fetch_text = lambda url, **kw: (None, False)

    try:
        result = run_fidelity_check(real_node, api_key=None)
        assert result.status != SOURCE_FABRICATED, \
            "Real DOI should not be flagged as fabricated"
    finally:
        sf._check_doi_exists = original_doi
        sf._fetch_text = original_fetch


def test_source_fidelity_network_failure_is_not_fabricated():
    """Regression test for a real bug found on 2026-07-13: a network error,
    timeout, or blocked endpoint when checking DOI/arXiv existence must be
    reported as UNCERTAIN (inconclusive), never as SOURCE_FABRICATED. An
    inconclusive check is not evidence that a source does not exist. This
    bug caused two well-known, real arXiv papers (Plaga 2008, Giddings &
    Mangano 2008) to be misreported as fabricated when export.arxiv.org
    was unreachable from the network, while the papers were confirmed to
    exist by direct browser access to arxiv.org."""
    from epistemic_stack.source_fidelity import run_fidelity_check, SOURCE_FABRICATED, UNCERTAIN

    node = {
        "ace_id": "ACE-TEST-NETWORK-FAILURE",
        "verbatim_claim": "A claim attributed to a real paper that cannot be verified due to network issues.",
        "source": {
            "doi_or_id": "arXiv:0808.1415",
            "status": "Peer-reviewed",
        }
    }

    import epistemic_stack.source_fidelity as sf
    original = sf._check_arxiv_exists
    # Simulate a network failure (timeout, blocked endpoint) — returns None,
    # not False, per the fixed function signature.
    sf._check_arxiv_exists = lambda arxiv_id: None

    try:
        result = run_fidelity_check(node, api_key=None)
        assert result.status != SOURCE_FABRICATED, \
            f"A network failure must never be reported as SOURCE_FABRICATED, got {result.status}"
        assert result.status == UNCERTAIN, \
            f"Expected UNCERTAIN for an inconclusive check, got {result.status}"
    finally:
        sf._check_arxiv_exists = original


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
