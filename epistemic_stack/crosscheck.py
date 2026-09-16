"""
epistemic_stack.crosscheck
=============================
Citation cross-checking: for each ACE node with a verifiable identifier
(DOI or arXiv ID), resolve a URL to the actual source and structure a
claim-vs-source comparison task.

HONEST SCOPE NOTE: this module does NOT itself run an LLM to judge
semantic alignment between a claim and its source — doing so inside a
zero-dependency, offline-runnable repository would require either an API
key (breaking the "no setup, no network access required" promise the
rest of this repo makes) or a bundled model (impractical). What this
module DOES do, honestly:

  1. Resolves a fetchable URL for each node's identifier (DOI -> doi.org
     redirect, arXiv ID -> abstract page), so a human reviewer or an
     external LLM call can retrieve the actual source text.
  2. Produces a structured "cross-check task" per node: the claim as
     written, the resolved URL, and an explicit judgment rubric — so that
     whoever (human or LLM) performs the actual comparison does so against
     a consistent standard, not an ad hoc one.
  3. Provides a place to RECORD cross-check results once performed
     (cross_check_results.json), with a CLI command to report on them.

This was built in direct response to adversarial test Category 1's most
dangerous finding (tests/test_adversarial.py, ADV-FALSE-03): a claim
attributed to a REAL, already-verified source, but stating the opposite
of what that source actually reports. No identifier-existence check
catches this. Only re-reading the source against the claim does.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


CROSSCHECK_RUBRIC = """
CITATION CROSS-CHECK TASK
==========================
For the claim and source below, determine whether the claim accurately
represents what the source actually says. This is NOT a check of whether
the claim is true in some absolute sense — it is a check of whether the
ATTRIBUTION is honest.

Score on this scale:
  ALIGNED        — the claim is a fair, non-misleading representation of
                    what the source says, even if paraphrased or
                    summarized. Minor loss of nuance is acceptable.
  PARTIALLY_OFF  — the claim captures something real in the source but
                    overstates, understates, or drops a material
                    qualifier (e.g. dropping "in a subgroup of patients
                    with X" or upgrading "suggests" to "proves").
  MISATTRIBUTED  — the claim states something the source does not say,
                    or states the OPPOSITE of what the source concludes.
                    This is the most serious finding — flag it loudly.
  UNVERIFIABLE   — the source could not be retrieved or read well enough
                    to judge (paywalled abstract only, fetch failed, etc).
                    This is a logistics failure, not a finding about the
                    claim — do not default to ALIGNED in this case.

CLAIM TO CHECK:
__CLAIM__

SOURCE TO CHECK AGAINST (resolved URL — fetch and read before judging):
__URL__

Return: {"verdict": "<one of the four above>", "explanation": "<1-3 sentences citing the specific text that supports your verdict>"}
"""


@dataclass
class CrossCheckTask:
    ace_id: str
    claim: str
    resolved_url: Optional[str]
    identifier: str
    identifier_type: str  # "doi" | "arxiv" | "unresolvable"

    def render_prompt(self) -> str:
        return (CROSSCHECK_RUBRIC
                .replace("__CLAIM__", self.claim)
                .replace("__URL__", self.resolved_url or "UNRESOLVED — no fetchable URL for this identifier"))


@dataclass
class CrossCheckResult:
    ace_id: str
    verdict: str  # ALIGNED | PARTIALLY_OFF | MISATTRIBUTED | UNVERIFIABLE
    explanation: str
    checked_by: str  # e.g. "claude-sonnet-4-6-manual-session-2026-06-21"


def resolve_url(source: dict) -> tuple[Optional[str], str]:
    """Returns (url, identifier_type). Priority: explicit url field (for
    sources like web articles or government reports that have a real,
    citable URL but no DOI) > arXiv > DOI > PMC > unresolvable."""
    if source.get("url"):
        return source["url"], "direct_url"

    arxiv_id = source.get("arxiv_id") or source.get("arxiv")
    if arxiv_id:
        clean = arxiv_id.replace("arXiv:", "").strip()
        return f"https://arxiv.org/abs/{clean}", "arxiv"

    doi = source.get("doi_or_id", "")
    if doi.startswith("arXiv:"):
        clean = doi.replace("arXiv:", "").strip()
        return f"https://arxiv.org/abs/{clean}", "arxiv"
    if doi.startswith("10."):
        return f"https://doi.org/{doi}", "doi"
    if doi.startswith("PMC"):
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{doi}/", "pmc"

    # institutional/government documents without a DOI or known URL: no reliable auto-resolve
    return None, "unresolvable"


def build_tasks(nodes_master_path: str | Path) -> list[CrossCheckTask]:
    with open(nodes_master_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks: list[CrossCheckTask] = []
    for case_key in ("case_covid", "case_eggs", "case_lhc"):
        for raw in data.get(case_key, {}).get("nodes", []):
            claim = raw.get("verbatim_claim") or raw.get("mutation_logic")
            if not claim:
                continue
            source = raw.get("source", {})
            url, id_type = resolve_url(source)
            tasks.append(CrossCheckTask(
                ace_id=raw["ace_id"],
                claim=claim,
                resolved_url=url,
                identifier=source.get("doi_or_id") or source.get("arxiv_id") or "none",
                identifier_type=id_type,
            ))
    return tasks


def load_results(results_path: str | Path) -> dict[str, CrossCheckResult]:
    path = Path(results_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        k: CrossCheckResult(ace_id=k, **v) for k, v in raw.items()
    }


def save_result(results_path: str | Path, result: CrossCheckResult) -> None:
    path = Path(results_path)
    existing = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing[result.ace_id] = {
        "verdict": result.verdict,
        "explanation": result.explanation,
        "checked_by": result.checked_by,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
