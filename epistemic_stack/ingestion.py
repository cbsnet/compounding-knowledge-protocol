"""
epistemic_stack.ingestion
==========================
The Ingestion layer's job is to turn unstructured prose into Atomic Claim
Elements (ACEs). This module is deliberately honest about its current scope:
it does NOT ship a trained or benchmarked extraction model. What it provides:

  1. The actual prompt template used to manually construct every ACE node in
     nodes_master.json (this is not aspirational — it is what was used).
  2. A structural validator that checks any candidate ACE dict against the
     schema before it can enter the graph, catching the most common failure
     mode flagged in review: missing epistemic_validation blocks, missing
     source identifiers, or confidence scores asserted without a rationale.
  3. A worked, end-to-end example showing the prompt applied to one real
     sentence from the COVID case material, with the actual model output
     and a note on what a human reviewer should check.

What this module explicitly does NOT claim: precision/recall benchmarks,
inter-model comparison, or inter-annotator agreement statistics. Building
those requires a labeled evaluation set that does not yet exist. This is
listed in the written spec's limitations section rather than implied here.
"""

from __future__ import annotations
from dataclasses import dataclass


ACE_EXTRACTION_PROMPT_TEMPLATE = """You are extracting Atomic Claim Elements (ACEs) from a source passage for an
epistemic knowledge graph. Follow these rules exactly.

RULES
1. Extract one ACE per distinct, independently-checkable factual or
   interpretive claim. Do not merge two claims with different evidential
   bases into one ACE, even if they appear in the same sentence.
2. Preserve the author's original hedging language verbatim in a
   `linguistic_hedges` list. Do not upgrade "suggests" to "demonstrates" or
   downgrade "demonstrates" to "suggests" — this is the single most common
   and most damaging extraction error.
3. If the passage compares or weighs multiple claims against each other
   (e.g. "X is strong, Y is weak"), extract each claim as a SEPARATE ACE and
   record the comparison itself as a separate relational note, not folded
   into either claim's confidence score.
4. Do not assign a systemic_confidence_score yourself. Leave it null. Scoring
   is a separate, accountable step performed by a human reviewer or a
   distinct assessment pass — conflating extraction and scoring hides where
   judgment was applied.
5. If the source's claim status is ambiguous (preprint vs. peer-reviewed,
   primary vs. secondary), do not guess. Set source.status to
   "UNVERIFIED — requires manual check" rather than asserting a status.

OUTPUT FORMAT
Return a JSON list of ACE objects, each with exactly these fields:
{
  "ace_id": "<short, unique, descriptive slug>",
  "verbatim_claim": "<the claim, paraphrased only enough to stand alone>",
  "source": {"title": null, "authors": null, "publication_date": null,
             "status": "UNVERIFIED — requires manual check"},
  "context_bounding": {
      "epistemic_modality": "<one of: empirical_inductive_argument |
        contested_empirical_hypothesis | dose_response_association |
        speculative_counter_hypothesis | structured_adjudicated_judgment |
        meta_analytical_observation | other>",
      "linguistic_hedges": ["<verbatim hedge words/phrases from source>"]
  },
  "epistemic_validation": null
}

SOURCE PASSAGE TO EXTRACT FROM:
__PASSAGE__
"""


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]


REQUIRED_TOP_LEVEL = ["ace_id", "verbatim_claim", "source"]
REQUIRED_SOURCE_FIELDS = ["title", "status"]


def validate_ace(candidate: dict, *, require_scored: bool = False) -> ValidationResult:
    """Structural validator for a candidate ACE before it is allowed into the graph.

    require_scored=True enforces that epistemic_validation is present and
    populated — use this gate for nodes about to enter nodes_master.json (the
    submission's source of truth). Use require_scored=False for raw extractor
    output, which is expected to have epistemic_validation=null per the
    prompt template above (scoring is a deliberately separate step).
    """
    errors: list[str] = []
    warnings: list[str] = []

    for field_name in REQUIRED_TOP_LEVEL:
        if field_name not in candidate or candidate[field_name] in (None, ""):
            errors.append(f"Missing required field: {field_name}")

    source = candidate.get("source", {})
    if isinstance(source, dict):
        if "status" not in source or not source["status"]:
            errors.append("Missing required source field: source.status")
        if require_scored and not source.get("title"):
            errors.append("Missing required source field: source.title (required once a node is scored)")
        elif not source.get("title"):
            warnings.append(
                f"{candidate.get('ace_id', '<unknown>')}: source.title is empty — "
                "acceptable at extraction stage if status is 'UNVERIFIED', but must "
                "be filled in before this node can be scored."
            )
        if source.get("status") == "UNVERIFIED — requires manual check":
            warnings.append(
                f"{candidate.get('ace_id', '<unknown>')}: source status is unverified — "
                "this node must not be added to nodes_master.json until a human "
                "confirms a real DOI/arXiv ID/institutional document identifier."
            )
    else:
        errors.append("source field must be an object")

    cb = candidate.get("context_bounding", {})
    if not isinstance(cb, dict) or not cb.get("epistemic_modality"):
        warnings.append(
            f"{candidate.get('ace_id', '<unknown>')}: missing epistemic_modality — "
            "extraction may have lost hedging information."
        )

    ev = candidate.get("epistemic_validation")
    if require_scored:
        if not ev or "systemic_confidence_score" not in ev or "peer_scrutiny_status" not in ev:
            errors.append(
                "epistemic_validation with systemic_confidence_score and "
                "peer_scrutiny_status is required for nodes entering nodes_master.json "
                "(see review finding 3.6 / 3.4 — scores must be present and have a stated rationale)."
            )
        elif "rationale" not in ev:
            warnings.append(
                f"{candidate.get('ace_id', '<unknown>')}: confidence score present but no "
                "rationale field — scores without rationale are flagged by review as "
                "non-auditable."
            )

    return ValidationResult(is_valid=(len(errors) == 0), errors=errors, warnings=warnings)


# --- Worked example: prompt applied to one real sentence from the case material ---
# Source: Scott Alexander, "Practically-A-Book Review: Rootclaim $100,000 Lab Leak
# Debate" (astralcodexten.com, 2024-03-28), reporting Peter Miller's debate argument.
WORKED_EXAMPLE_PASSAGE = (
    "The market clustering evidence is strong, but the intermediate host evidence "
    "is weak, and the lab leak evidence is circumstantial but not impossible."
)

# This is a CONSTRUCTED illustrative sentence (used in the supervisor review to
# probe extraction of compound/comparative claims) rather than a verbatim quote
# from the source — flagged here explicitly so it is never mistaken for a citation.
WORKED_EXAMPLE_NOTE = (
    "This sentence is a compound, comparative claim: it asserts THREE separate "
    "evidential weightings (market clustering=strong, intermediate host=weak, "
    "lab leak=circumstantial-not-impossible) in one breath. Rule 3 of the prompt "
    "template requires splitting this into three ACEs rather than one node with "
    "an averaged or blended confidence. The correct extraction is shown below — "
    "the comparison ITSELF (these three are being weighed against each other by "
    "the same speaker) is preserved as a structural note, not lost."
)

WORKED_EXAMPLE_OUTPUT = [
    {
        "ace_id": "ACE-EXAMPLE-MARKET-CLUSTERING-STRENGTH",
        "verbatim_claim": "Market clustering evidence for COVID-19 origins is strong.",
        "source": {"title": None, "authors": None, "publication_date": None,
                    "status": "UNVERIFIED — requires manual check"},
        "context_bounding": {
            "epistemic_modality": "contested_empirical_hypothesis",
            "linguistic_hedges": ["strong"]
        },
        "epistemic_validation": None
    },
    {
        "ace_id": "ACE-EXAMPLE-INTERMEDIATE-HOST-WEAKNESS",
        "verbatim_claim": "Intermediate host evidence for COVID-19 zoonotic origin is weak.",
        "source": {"title": None, "authors": None, "publication_date": None,
                    "status": "UNVERIFIED — requires manual check"},
        "context_bounding": {
            "epistemic_modality": "contested_empirical_hypothesis",
            "linguistic_hedges": ["weak"]
        },
        "epistemic_validation": None
    },
    {
        "ace_id": "ACE-EXAMPLE-LAB-LEAK-CIRCUMSTANTIAL",
        "verbatim_claim": "Lab leak evidence for COVID-19 origins is circumstantial but not impossible.",
        "source": {"title": None, "authors": None, "publication_date": None,
                    "status": "UNVERIFIED — requires manual check"},
        "context_bounding": {
            "epistemic_modality": "speculative_counter_hypothesis",
            "linguistic_hedges": ["circumstantial", "not impossible"]
        },
        "epistemic_validation": None
    },
]
