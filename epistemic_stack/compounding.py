"""
epistemic_stack.compounding
=============================
Implements the three-step compounding protocol described in the written spec
(section 4.2), so that "two independent analysts extend the same graph" is a
runnable demonstration rather than only a paragraph of prose.

Step 1: Non-duplication validation (lightweight lexical overlap check; see
        note below on why this is a deliberately simple stand-in).
Step 2: Provisional relational integration (new external nodes/edges enter
        flagged, not auto-merged into Crux Score-affecting positions).
Step 3: Tracked cascading recalculation with a timestamped changelog.

HONEST SCOPE NOTE: Step 1's duplication check uses word-overlap similarity,
not semantic embedding similarity. This is intentionally simple and is
flagged as a known limitation in the written spec — a production version
would use embedding similarity, which this repo does not include to avoid
an unverified claim of having tested it.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .graph import EpistemicGraph, ACENode


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_similarity(a: str, b: str) -> float:
    """Jaccard similarity over tokenized words. Simple, transparent, and
    explicitly NOT presented as semantic similarity — see module docstring."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class ChangelogEntry:
    timestamp: str
    action: str
    ace_id: str
    detail: str
    proposed_by: str


@dataclass
class CompoundingSession:
    """Tracks a single external analyst's proposed additions to an existing graph."""
    graph: EpistemicGraph
    proposed_by: str
    changelog: list[ChangelogEntry] = field(default_factory=list)
    duplication_threshold: float = 0.35

    def _log(self, action: str, ace_id: str, detail: str) -> None:
        self.changelog.append(ChangelogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            ace_id=ace_id,
            detail=detail,
            proposed_by=self.proposed_by,
        ))

    # --- Step 1: non-duplication validation ---------------------------------
    def check_duplication(self, candidate_claim: str) -> list[tuple[str, float]]:
        """Returns [(existing_ace_id, similarity)] for any existing node whose
        verbatim_claim exceeds duplication_threshold similarity to the candidate."""
        hits = []
        for aid, node in self.graph.nodes.items():
            sim = lexical_similarity(candidate_claim, node.verbatim_claim)
            if sim >= self.duplication_threshold:
                hits.append((aid, round(sim, 3)))
        return sorted(hits, key=lambda x: -x[1])

    # --- Step 2: provisional relational integration -------------------------
    def propose_node(self, node: ACENode, proposed_vector_target: Optional[list[str]] = None) -> dict:
        """Adds a node flagged proposed_by=external. Any DEPEND/CONTRADICT/SUPPORT
        edge it proposes toward an EXISTING node does NOT get folded into that
        target's incoming list until accept_proposed_edges() is called — this is
        what prevents a single external contributor from silently shifting another
        curator's Crux Score, per the written spec's stated safeguard."""
        dup_hits = self.check_duplication(node.verbatim_claim)
        if dup_hits:
            self._log("DUPLICATE_FLAGGED", node.ace_id,
                       f"Similarity to existing nodes: {dup_hits}")
            return {
                "status": "flagged_possible_duplicate",
                "ace_id": node.ace_id,
                "similar_to": dup_hits,
                "action_required": "human review before insertion",
            }

        node.raw["proposed_by"] = "external"
        node.raw["pending_edges"] = proposed_vector_target or []
        self.graph.nodes[node.ace_id] = node
        self.graph.incoming.setdefault(node.ace_id, [])
        self._log("NODE_PROPOSED", node.ace_id,
                   f"Inserted with no active edges yet. Proposed targets pending review: {proposed_vector_target}")
        return {
            "status": "inserted_pending_review",
            "ace_id": node.ace_id,
            "pending_edges": proposed_vector_target or [],
        }

    def accept_proposed_edges(self, ace_id: str, reviewer: str) -> dict:
        """Curator-side acceptance: activates the pending edges, which is the
        moment Crux Score for downstream nodes actually changes.

        Checks for cycles BEFORE activating edges (added after adversarial
        testing found the graph had no cycle protection at all — see
        graph.py detect_cycles() and tests/test_adversarial.py Category 3).
        A detected cycle does not silently block the merge; it is returned
        to the reviewer as a required decision point, since rejecting all
        cycles outright could also hide legitimate cases the reviewer wants
        to inspect and resolve manually."""
        node = self.graph.nodes.get(ace_id)
        if node is None:
            return {"status": "error", "detail": f"No such node: {ace_id}"}
        pending = node.raw.get("pending_edges", [])
        if not pending:
            return {"status": "no_pending_edges", "ace_id": ace_id}

        before_scores = self.graph.all_crux_scores()  # MUST be captured before any edge activation

        # Tentatively activate to check for cycles, roll back if needed
        node.targets = pending
        for t in pending:
            self.graph.incoming.setdefault(t, []).append((node.ace_id, node.vector_type))
        cycles = self.graph.detect_cycles()
        relevant_cycles = [c for c in cycles if ace_id in c]
        if relevant_cycles:
            # Roll back
            node.targets = []
            for t in pending:
                self.graph.incoming[t] = [
                    (s, vt) for (s, vt) in self.graph.incoming[t] if s != ace_id
                ]
            self._log("CYCLE_BLOCKED", ace_id, f"Proposed edges would create cycle(s): {relevant_cycles}")
            return {
                "status": "blocked_would_create_cycle",
                "ace_id": ace_id,
                "cycles": relevant_cycles,
                "action_required": "human review — cycle must be resolved before these edges can be accepted",
            }

        node.raw["pending_edges"] = []
        node.raw["accepted_by"] = reviewer
        after_scores = self.graph.all_crux_scores()

        # --- Step 3: tracked cascading recalculation ---
        changed = {
            aid: {"before": before_scores.get(aid, 0.0), "after": after_scores.get(aid, 0.0)}
            for aid in after_scores
            if round(before_scores.get(aid, 0.0), 4) != round(after_scores.get(aid, 0.0), 4)
        }
        self._log("EDGES_ACCEPTED", ace_id,
                   f"Reviewer={reviewer}. Downstream Crux Score changes: {changed}")
        return {
            "status": "edges_accepted",
            "ace_id": ace_id,
            "downstream_score_changes": changed,
        }

    def export_changelog(self) -> list[dict]:
        return [
            {"timestamp": c.timestamp, "action": c.action, "ace_id": c.ace_id,
             "detail": c.detail, "proposed_by": c.proposed_by}
            for c in self.changelog
        ]
