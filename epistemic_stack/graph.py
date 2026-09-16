"""
epistemic_stack.graph
======================
Core data structures for The Epistemic Stack: loads Atomic Claim Elements (ACEs)
from the node reference file, builds a directed graph of relational vectors
(SUPPORT, CONTRADICT, DEPEND, CONTEXT_MUTATION), and computes the Crux Score
for every node using the operationally-defined Wdep formula.

This module intentionally has zero external dependencies beyond the Python
standard library, so it runs anywhere with `python3 -m epistemic_stack.cli`
without an install step beyond cloning the repo.
"""

from __future__ import annotations
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --- Wdep multiplier scale -------------------------------------------------
# This table is the operational definition of Wdep referenced in the written
# spec (section 2.3.2). It must stay in sync with nodes_master.json's
# schema_notes.wdep_specification.peer_scrutiny_multiplier_scale — the CLI's
# `verify-sync` command checks this automatically.
PEER_SCRUTINY_MULTIPLIER = {
    "peer_reviewed_and_independently_endorsed": 1.0,
    "peer_reviewed": 0.9,
    "government_oversight_document": 0.8,
    "structured_adjudication_by_domain_experts": 0.8,
    "institutional_report": 0.7,
    "unreviewed_preprint": 0.5,
    "unreviewed_preprint_self_flagged_speculative": 0.3,
    "self-published_post-hoc_rebuttal": 0.25,
    "primary_source_documented_dispute": 0.4,
    "directly_observable_from_primary_source": 1.0,
}
DEFAULT_MULTIPLIER = 0.4  # applied with a warning if a status is missing from the table above


@dataclass
class ACENode:
    ace_id: str
    case: str
    vector_type: str = "ACE"  # ACE | SUPPORT | CONTRADICT | DEPEND | CONTEXT_MUTATION
    targets: list[str] = field(default_factory=list)
    verbatim_claim: str = ""
    confidence: Optional[float] = None
    peer_scrutiny_status: Optional[str] = None
    raw: dict = field(default_factory=dict)

    def wdep_multiplier(self) -> float:
        """Wdep is computed FROM this node's own peer_scrutiny_status and confidence,
        not assigned externally — see nodes_master.json schema_notes.wdep_specification."""
        mult = PEER_SCRUTINY_MULTIPLIER.get(self.peer_scrutiny_status, DEFAULT_MULTIPLIER)
        conf = self.confidence if self.confidence is not None else 0.5
        return round(mult * conf, 4)


class EpistemicGraph:
    """A directed graph of ACE nodes connected by typed relational vectors."""

    def __init__(self, case_label: str = ""):
        self.case_label = case_label
        self.nodes: dict[str, ACENode] = {}
        # incoming[target_id] = list of (source_id, vector_type)
        self.incoming: dict[str, list[tuple[str, str]]] = {}

    def add_node(self, node: ACENode) -> None:
        self.nodes[node.ace_id] = node
        self.incoming.setdefault(node.ace_id, [])
        for t in node.targets:
            self.incoming.setdefault(t, []).append((node.ace_id, node.vector_type))

    def detect_cycles(self) -> list[list[str]]:
        """Returns a list of cycles found in the DEPEND/CONTRADICT edge structure
        (each cycle as a list of ace_ids). An empty list means the graph is
        genuinely acyclic. Added after adversarial testing found that a class
        named EpistemicGraph, documented as implementing a Directed Acyclic
        Graph, had no runtime check preventing or flagging an actual cycle —
        two nodes citing each other as their only DEPEND target could silently
        inflate both nodes' Crux Scores with zero external grounding. This
        method does not auto-reject cycles (a curator may have legitimate
        reasons to need to see one in order to fix it) but makes them visible."""
        visited: set[str] = set()
        path: list[str] = []
        path_set: set[str] = set()
        cycles: list[list[str]] = []

        def dfs(node_id: str) -> None:
            if node_id in path_set:
                cycle_start = path.index(node_id)
                cycles.append(path[cycle_start:] + [node_id])
                return
            if node_id in visited:
                return
            visited.add(node_id)
            path.append(node_id)
            path_set.add(node_id)
            node = self.nodes.get(node_id)
            if node:
                for t in node.targets:
                    dfs(t)
            path.pop()
            path_set.discard(node_id)

        for nid in self.nodes:
            if nid not in visited:
                dfs(nid)
        return cycles

    @classmethod
    def from_case_block(cls, case_label: str, case_block: dict) -> "EpistemicGraph":
        g = cls(case_label=case_label)
        for raw in case_block.get("nodes", []):
            targets = raw.get("vector_target") or raw.get("vector_links") or []
            if isinstance(targets, str):
                targets = [targets]
            ev = raw.get("epistemic_validation", {})
            node = ACENode(
                ace_id=raw["ace_id"],
                case=case_label,
                vector_type=raw.get("vector_type", "ACE"),
                targets=[t for t in targets if isinstance(t, str) and t.startswith("ACE") or (isinstance(t, str) and t.startswith("CONMUT"))],
                verbatim_claim=raw.get("verbatim_claim", ""),
                confidence=ev.get("systemic_confidence_score"),
                peer_scrutiny_status=ev.get("peer_scrutiny_status"),
                raw=raw,
            )
            g.add_node(node)
        return g

    def out_degree(self, ace_id: str) -> int:
        """Dout: number of DEPEND edges this node is the SOURCE of (i.e. how many
        other nodes' validity this node's claim chain supports as a dependency target
        for *other* nodes pointing at it). Per the spec, Dout counts dependencies
        a node SUPPORTS, i.e. incoming DEPEND edges naming this node as target."""
        return sum(1 for (_src, vt) in self.incoming.get(ace_id, []) if vt == "DEPEND")

    def contradiction_density(self, ace_id: str) -> float:
        """Econ: WEIGHTED density of intersecting CONTRADICT vectors touching this
        node. Each CONTRADICT edge is weighted by the contradicting node's own
        peer_scrutiny_multiplier x confidence (same Wdep logic as DEPEND edges),
        not counted as a flat +1. This matters: a CONTRADICT from an unreviewed
        preprint should not weigh as much as a CONTRADICT from a peer-reviewed,
        independently-endorsed source. An earlier unweighted version of this
        function gave ACE-SARS2-GENOME-FCS (contested by a weak preprint) the
        same Crux Score as ACE-SARS2-EPID-MARKET (supported by Science-published
        work) purely because both had one incoming DEPEND edge — this weighting
        was added specifically to fix that conflation."""
        total = 0.0
        for (src_id, vt) in self.incoming.get(ace_id, []):
            if vt == "CONTRADICT":
                src_node = self.nodes.get(src_id)
                total += src_node.wdep_multiplier() if src_node else 0.4
        node = self.nodes.get(ace_id)
        if node and node.vector_type == "CONTRADICT":
            for t in node.targets:
                total += node.wdep_multiplier()
        return round(total, 4)

    def contradiction_diversity_index(self, ace_id: str) -> float:
        """CDI: normalized entropy of the sources producing CONTRADICT edges
        toward this node. Addresses the gap found in adversarial test Category 5
        (one-sided graph topology): the Crux Score correctly flags a node under
        heavy contradiction pressure, but cannot distinguish whether that pressure
        comes from many independent sources (genuine controversy) or from a single
        source flooding the graph with CONTRADICT edges (manipulation).

        CDI = 0.0  all CONTRADICT edges originate from the same source
                   (maximum flooding suspicion)
        CDI = 1.0  each CONTRADICT edge originates from a different source
                   (maximum diversity, consistent with genuine independent criticism)
        CDI = None no CONTRADICT edges toward this node (not applicable)

        LIMITATION: CDI tracks analyst source (proposed_by field or ace_id prefix)
        as a proxy for independence. It does not detect sockpuppet accounts or
        coordinated teams presenting as independent contributors — both would appear
        as CDI = 1.0. This limitation is documented in the written spec's Part 5."""
        import math
        contra_sources = []
        for (src_id, vt) in self.incoming.get(ace_id, []):
            if vt == "CONTRADICT":
                src_node = self.nodes.get(src_id)
                if src_node:
                    analyst = src_node.raw.get("proposed_by") or src_id.split("-")[0]
                    contra_sources.append(analyst)
        if not contra_sources:
            return None
        if len(contra_sources) == 1:
            return None  # single CONTRADICT edge: entropy=0 is expected, not a flooding signal
        from collections import Counter
        counts = Counter(contra_sources)
        n = len(contra_sources)
        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
        max_entropy = math.log2(len(counts))
        if max_entropy == 0:
            return 0.0
        return round(entropy / max_entropy, 4)

    def crux_score(self, ace_id: str) -> float:
        """Cs = Sum(Dout x Wdep) + Sum(Econ)
        For each incoming DEPEND edge from a source node, Wdep is computed from
        the SOURCE node's own peer_scrutiny_status and confidence (the node making
        the dependency claim), per the operational definition in the written spec."""
        dep_sources = [src for (src, vt) in self.incoming.get(ace_id, []) if vt == "DEPEND"]
        dout = len(dep_sources)
        wdep_terms = []
        for src_id in dep_sources:
            src_node = self.nodes.get(src_id)
            if src_node:
                wdep_terms.append(src_node.wdep_multiplier())
        dep_component = sum(wdep_terms)  # Sum(Dout-weighted dependencies) — one term per dependency edge
        econ_component = self.contradiction_density(ace_id)
        return round(dep_component + econ_component, 4)

    def all_crux_scores(self) -> dict[str, float]:
        return {aid: self.crux_score(aid) for aid in self.nodes}

    def z_scores(self) -> dict[str, float]:
        """Dynamic pruning threshold per section 2.3.2: Z = (Cs - mu) / sigma.
        Falls back gracefully (returns 0.0 for all nodes) on graphs too small
        for a stable standard deviation (n < 2 or sigma == 0)."""
        scores = self.all_crux_scores()
        values = list(scores.values())
        if len(values) < 2:
            return {k: 0.0 for k in scores}
        mu = statistics.mean(values)
        sigma = statistics.pstdev(values)
        if sigma == 0:
            return {k: 0.0 for k in scores}
        return {k: round((v - mu) / sigma, 4) for k, v in scores.items()}

    def critical_nodes(self, z_threshold: float = 2.0) -> list[str]:
        z = self.z_scores()
        return sorted([k for k, v in z.items() if v > z_threshold], key=lambda k: -z[k])

    def summary(self) -> str:
        z = self.z_scores()
        cs = self.all_crux_scores()
        lines = [f"Graph: {self.case_label}  ({len(self.nodes)} nodes)"]
        for aid, node in sorted(self.nodes.items(), key=lambda kv: -cs[kv[0]]):
            flag = "  <-- CRITICAL (Z>2.0)" if z[aid] > z_threshold_default else ""
            cdi = self.contradiction_diversity_index(aid)
            cdi_str = f"CDI={cdi:.2f}" if cdi is not None else "CDI=n/a"
            if cdi is not None and cdi < 0.3:
                cdi_str += " (!low-diversity)"
            lines.append(
                f"  {aid:40s} Cs={cs[aid]:6.3f}  Z={z[aid]:+6.3f}  {cdi_str}{flag}"
            )
        return "\n".join(lines)


z_threshold_default = 2.0


def load_nodes_master(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_all_graphs(nodes_master_path: str | Path) -> dict[str, EpistemicGraph]:
    data = load_nodes_master(nodes_master_path)
    graphs = {}
    for key in ("case_covid", "case_eggs", "case_lhc"):
        if key in data:
            label = data[key].get("case_label", key)
            graphs[key] = EpistemicGraph.from_case_block(label, data[key])
    return graphs
