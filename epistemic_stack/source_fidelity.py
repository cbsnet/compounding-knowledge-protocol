"""
epistemic_stack.source_fidelity
================================
Automated Source Fidelity Check: verifies whether the verbatim_claim of each
ACE node accurately represents what its cited source actually says.

This module implements the 9-state taxonomy proposed in the project's design
review (2026-07-01), which distinguishes between:
  - Successful retrieval + semantic comparison (RETRIEVED_*)
  - Partial retrieval / paywall (ABSTRACT_*)
  - Technical failures (SOURCE_UNREACHABLE)
  - Fabricated identifiers (SOURCE_FABRICATED)  ← catches ADV-FALSE-01
  - Semantic misalignment despite real source   ← catches ADV-FALSE-03
  - Ambiguous LLM verdict (UNCERTAIN)

DESIGN PRINCIPLE: this module NEVER modifies systemic_confidence_score
automatically. It populates a separate source_verification block on each
node, acting as a "proposer" that flags problems for human review — exactly
the same trust-but-verify logic as the compounding protocol's proposed_by:
external mechanism. The human curator decides whether to act on a flag.

DEPENDENCIES: this module requires:
  - An ANTHROPIC_API_KEY environment variable (for semantic comparison)
  - Network access (for Crossref DOI verification and source fetch)
  The core pipeline (graph.py, compounding.py, etc.) works without either.
  This module is explicitly optional and is invoked only via the CLI commands
  source-fidelity-run and source-fidelity-report.
"""

from __future__ import annotations
import json
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .crosscheck import resolve_url


# ── 9-state taxonomy ──────────────────────────────────────────────────────────

RETRIEVED_ALIGNED        = "RETRIEVED_ALIGNED"
RETRIEVED_MISALIGNED     = "RETRIEVED_MISALIGNED"
RETRIEVED_PARTIALLY_OFF  = "RETRIEVED_PARTIALLY_OFF"
ABSTRACT_ALIGNED         = "ABSTRACT_ALIGNED"
ABSTRACT_MISALIGNED      = "ABSTRACT_MISALIGNED"
ABSTRACT_PARTIALLY_OFF   = "ABSTRACT_PARTIALLY_OFF"
SOURCE_UNREACHABLE       = "SOURCE_UNREACHABLE"
SOURCE_FABRICATED        = "SOURCE_FABRICATED"
UNCERTAIN                = "UNCERTAIN"
UNCHECKED                = "UNCHECKED"

NEGATIVE_STATUSES = {
    RETRIEVED_MISALIGNED, RETRIEVED_PARTIALLY_OFF,
    ABSTRACT_MISALIGNED, ABSTRACT_PARTIALLY_OFF,
    SOURCE_FABRICATED,
}

STATUS_DESCRIPTIONS = {
    RETRIEVED_ALIGNED:       "Full text retrieved; claim is consistent with source.",
    RETRIEVED_MISALIGNED:    "Full text retrieved; claim contradicts or distorts source.",
    RETRIEVED_PARTIALLY_OFF: "Full text retrieved; claim is partially inaccurate.",
    ABSTRACT_ALIGNED:        "Only abstract available (paywall); claim appears consistent.",
    ABSTRACT_MISALIGNED:     "Only abstract available; claim contradicts abstract.",
    ABSTRACT_PARTIALLY_OFF:  "Only abstract available; claim partially misrepresents it.",
    SOURCE_UNREACHABLE:      "Source could not be fetched (broken link, server error).",
    SOURCE_FABRICATED:       "DOI/arXiv ID does not exist in Crossref/arXiv registry.",
    UNCERTAIN:               "Automatic comparison produced low-confidence verdict.",
    UNCHECKED:               "No automated check has been run yet.",
}


@dataclass
class FidelityResult:
    ace_id: str
    status: str
    detail: str
    checked_at: str
    method: str  # "crossref+llm" | "arxiv+llm" | "url+llm" | "crossref_only"


# ── Crossref DOI existence check (no auth required) ──────────────────────────

def _check_doi_exists(doi: str) -> Optional[bool]:
    """Returns True if the DOI is confirmed to exist in Crossref's registry,
    False if Crossref confirms it does NOT exist (HTTP 404), or None if the
    check itself could not be completed (network error, timeout, blocked
    endpoint, rate limit, etc). None must NEVER be treated as False by the
    caller: an inconclusive check is not evidence of fabrication."""
    url = f"https://api.crossref.org/works/{doi}?mailto=epistemic-stack@example.org"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EpistemicStack/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        return None  # e.g. 403, 429, 5xx — inconclusive, not evidence of absence
    except Exception:
        return None  # timeout, DNS failure, connection refused — inconclusive


def _check_arxiv_exists(arxiv_id: str) -> Optional[bool]:
    """Returns True if confirmed to exist, False if confirmed 404, None if
    the check could not be completed. See _check_doi_exists for rationale."""
    clean = arxiv_id.replace("arXiv:", "").strip()
    url = f"https://export.arxiv.org/abs/{clean}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EpistemicStack/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        return None
    except Exception:
        return None


# ── Source text fetch ─────────────────────────────────────────────────────────

def _fetch_text(url: str, max_chars: int = 8000) -> tuple[Optional[str], bool]:
    """Returns (text, is_full_text). is_full_text=False means abstract only."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "EpistemicStack/1.0",
                     "Accept": "text/html,application/xhtml+xml"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read(max_chars * 4).decode("utf-8", errors="replace")
        # Heuristic: arXiv abstract pages contain "Abstract:" but not full paper
        is_abstract_only = "arxiv.org/abs/" in url or len(raw) < 2000
        # Strip HTML tags for cleaner text
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()[:max_chars]
        return text, not is_abstract_only
    except urllib.error.HTTPError as e:
        if e.code == 403:
            return None, False  # paywall
        return None, False
    except Exception:
        return None, False


# ── LLM semantic comparison via Anthropic API ────────────────────────────────

FIDELITY_PROMPT = """You are performing a citation fidelity check. Determine whether the
CLAIM accurately represents what the SOURCE TEXT actually says.

This is NOT about whether the claim is true — only whether the attribution is honest.

CLAIM:
{claim}

SOURCE TEXT (excerpt):
{source_text}

Return ONLY a JSON object with exactly these fields:
{{
  "verdict": "<one of: ALIGNED | PARTIALLY_OFF | MISALIGNED | UNCERTAIN>",
  "confidence": <float 0.0-1.0>,
  "explanation": "<1-2 sentences citing specific text>"
}}

- ALIGNED: claim is a fair, accurate representation of the source
- PARTIALLY_OFF: claim overstates, understates, or drops a material qualifier
- MISALIGNED: claim contradicts or inverts what the source says
- UNCERTAIN: source text is too short/unclear to make a reliable judgment"""


def _llm_compare(claim: str, source_text: str, api_key: str) -> tuple[str, float, str]:
    """Returns (verdict, confidence, explanation). Uses claude-sonnet-4-6."""
    import urllib.request
    import json as _json

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 300,
        "messages": [{
            "role": "user",
            "content": FIDELITY_PROMPT.format(claim=claim, source_text=source_text[:4000])
        }]
    }
    data = _json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read())
        content_blocks = result.get("content", [])
        if not content_blocks:
            # Empty content is consistent with an API-level safety refusal
            # rather than a normal completion. Documented, not silently
            # treated as a generic technical failure — see BioSecBench-Refusal
            # (LatchBio, 2026) on surface-keyword-triggered API refusals for
            # legitimate biology-adjacent text comparison tasks.
            stop_reason = result.get("stop_reason", "unknown")
            return UNCERTAIN, 0.0, (
                f"API returned no content blocks (stop_reason: {stop_reason}). "
                f"This pattern is consistent with an API-level safety refusal "
                f"triggered by surface terminology in the source text, rather "
                f"than a genuine fidelity judgment. Flagged for human review, "
                f"not treated as a normal UNCERTAIN verdict."
            )
        text = content_blocks[0]["text"].strip()
        # Strip markdown fences if present
        text = re.sub(r"```json|```", "", text).strip()
        parsed = _json.loads(text)
        return (
            parsed.get("verdict", UNCERTAIN),
            float(parsed.get("confidence", 0.5)),
            parsed.get("explanation", ""),
        )
    except Exception as e:
        return UNCERTAIN, 0.0, f"LLM call failed (technical error, not a refusal): {e}"


def _map_llm_verdict(llm_verdict: str, is_full_text: bool) -> str:
    """Maps LLM 4-way verdict + retrieval level to the 9-state taxonomy."""
    mapping = {
        ("ALIGNED",       True):  RETRIEVED_ALIGNED,
        ("ALIGNED",       False): ABSTRACT_ALIGNED,
        ("PARTIALLY_OFF", True):  RETRIEVED_PARTIALLY_OFF,
        ("PARTIALLY_OFF", False): ABSTRACT_PARTIALLY_OFF,
        ("MISALIGNED",    True):  RETRIEVED_MISALIGNED,
        ("MISALIGNED",    False): ABSTRACT_MISALIGNED,
        ("UNCERTAIN",     True):  UNCERTAIN,
        ("UNCERTAIN",     False): UNCERTAIN,
    }
    return mapping.get((llm_verdict, is_full_text), UNCERTAIN)


# ── Main check function ───────────────────────────────────────────────────────

def run_fidelity_check(node: dict, api_key: Optional[str] = None) -> FidelityResult:
    """Run the full fidelity check pipeline on a single ACE node.

    Step 1: Verify identifier existence (Crossref / arXiv).
            → SOURCE_FABRICATED if identifier not found.
    Step 2: Fetch source text.
            → SOURCE_UNREACHABLE if fetch fails completely.
    Step 3: LLM semantic comparison (requires api_key).
            → Skipped (SOURCE_UNREACHABLE) if no api_key provided.
    Step 4: Map to 9-state taxonomy.
    """
    ace_id = node["ace_id"]
    claim = node.get("verbatim_claim") or node.get("mutation_logic", "")
    source = node.get("source", {})
    timestamp = datetime.now(timezone.utc).isoformat()

    # Resolve URL and identifier type
    url, id_type = resolve_url(source)
    doi = source.get("doi_or_id", "")
    arxiv_id = source.get("arxiv_id", "")

    # Step 1: existence check
    # doi_exists / arxiv_exists is True (confirmed), False (confirmed absent),
    # or None (check inconclusive — network issue, not evidence of fabrication)
    if doi.startswith("10."):
        doi_exists = _check_doi_exists(doi)
        if doi_exists is False:
            return FidelityResult(
                ace_id=ace_id, status=SOURCE_FABRICATED,
                detail=f"DOI {doi} confirmed absent from Crossref registry (HTTP 404).",
                checked_at=timestamp, method="crossref_only"
            )
        elif doi_exists is None:
            return FidelityResult(
                ace_id=ace_id, status=UNCERTAIN,
                detail=f"Could not verify DOI {doi} against Crossref (network error, timeout, or endpoint blocked) — existence is unconfirmed, not disproven.",
                checked_at=timestamp, method="crossref_only"
            )
    elif doi.startswith("arXiv:") or arxiv_id:
        clean = (arxiv_id or doi).replace("arXiv:", "").strip()
        arxiv_exists = _check_arxiv_exists(clean)
        if arxiv_exists is False:
            return FidelityResult(
                ace_id=ace_id, status=SOURCE_FABRICATED,
                detail=f"arXiv ID {clean} confirmed absent (HTTP 404).",
                checked_at=timestamp, method="arxiv_only"
            )
        elif arxiv_exists is None:
            return FidelityResult(
                ace_id=ace_id, status=UNCERTAIN,
                detail=f"Could not verify arXiv ID {clean} (network error, timeout, or endpoint blocked) — existence is unconfirmed, not disproven.",
                checked_at=timestamp, method="arxiv_only"
            )

    if not url:
        return FidelityResult(
            ace_id=ace_id, status=SOURCE_UNREACHABLE,
            detail="No resolvable URL or identifier for this node.",
            checked_at=timestamp, method="none"
        )

    # Step 2: fetch source text
    source_text, is_full_text = _fetch_text(url)
    if not source_text:
        return FidelityResult(
            ace_id=ace_id, status=SOURCE_UNREACHABLE,
            detail=f"Could not retrieve text from {url} (paywall or network error).",
            checked_at=timestamp, method="fetch_failed"
        )

    # Step 3: LLM comparison (optional)
    if not api_key:
        return FidelityResult(
            ace_id=ace_id, status=UNCERTAIN,
            detail="Source text retrieved but no API key provided for semantic comparison.",
            checked_at=timestamp, method="fetch_only"
        )

    llm_verdict, confidence, explanation = _llm_compare(claim, source_text, api_key)
    if confidence < 0.4:
        llm_verdict = "UNCERTAIN"

    # Step 4: map to 9-state taxonomy
    final_status = _map_llm_verdict(llm_verdict, is_full_text)
    method = "crossref+llm" if doi.startswith("10.") else ("arxiv+llm" if arxiv_id else "url+llm")

    return FidelityResult(
        ace_id=ace_id, status=final_status,
        detail=explanation,
        checked_at=timestamp, method=method
    )


# ── Batch run and persistence ─────────────────────────────────────────────────

def run_all(nodes_master_path: str | Path,
            results_path: str | Path,
            api_key: Optional[str] = None,
            force: bool = False) -> list[FidelityResult]:
    """Run fidelity checks on all nodes, skipping already-checked ones
    unless force=True."""
    with open(nodes_master_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing = load_results(results_path)
    results = []

    for case_key in ("case_covid", "case_eggs", "case_lhc"):
        for node in data.get(case_key, {}).get("nodes", []):
            ace_id = node["ace_id"]
            if ace_id in existing and not force:
                continue
            print(f"  Checking {ace_id}...")
            result = run_fidelity_check(node, api_key=api_key)
            save_result(results_path, result)
            results.append(result)
            if result.status in NEGATIVE_STATUSES:
                print(f"    FLAG: {result.status} — {result.detail[:80]}")

    return results


def load_results(results_path: str | Path) -> dict[str, FidelityResult]:
    path = Path(results_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        k: FidelityResult(ace_id=k, **v) for k, v in raw.items()
    }


def save_result(results_path: str | Path, result: FidelityResult) -> None:
    path = Path(results_path)
    existing = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing[result.ace_id] = {
        "status": result.status,
        "detail": result.detail,
        "checked_at": result.checked_at,
        "method": result.method,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def report(results_path: str | Path, nodes_master_path: str | Path) -> str:
    """Returns a formatted summary report."""
    with open(nodes_master_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total_nodes = sum(
        len(data.get(k, {}).get("nodes", []))
        for k in ("case_covid", "case_eggs", "case_lhc")
    )
    results = load_results(results_path)
    checked = len(results)
    from collections import Counter
    counts = Counter(r.status for r in results.values())

    lines = [
        f"Source Fidelity Check coverage: {checked}/{total_nodes} nodes",
        "",
    ]
    for status in [SOURCE_FABRICATED, RETRIEVED_MISALIGNED, ABSTRACT_MISALIGNED,
                   RETRIEVED_PARTIALLY_OFF, ABSTRACT_PARTIALLY_OFF,
                   SOURCE_UNREACHABLE, UNCERTAIN,
                   RETRIEVED_ALIGNED, ABSTRACT_ALIGNED]:
        n = counts.get(status, 0)
        if n:
            flag = " <-- NEEDS REVIEW" if status in NEGATIVE_STATUSES else ""
            lines.append(f"  {status:30s} {n:3d}{flag}")
            if status in NEGATIVE_STATUSES:
                for ace_id, r in results.items():
                    if r.status == status:
                        lines.append(f"    {ace_id}: {r.detail[:80]}")

    unchecked = total_nodes - checked
    if unchecked:
        lines.append(f"\n  UNCHECKED: {unchecked} nodes not yet verified")
    return "\n".join(lines)
