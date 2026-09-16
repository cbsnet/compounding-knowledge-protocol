# The Epistemic Stack — Reference Implementation

This repository implements the Structure and Assessment layers 
described in the written spec, plus a transparency
artifact for the Ingestion layer (see "Honest scope" below).

## Requirements

Python 3.10+. No external dependencies — the entire core (`epistemic_stack/`)
uses only the standard library, specifically so this runs anywhere without
an install step.

## Run it (one command, no setup)

```bash
git clone <repo-url>
cd epistemic-stack
python3 -m epistemic_stack.cli summary
```

That's it. No API keys, no network calls, no virtual environment required to
see the core pipeline run against the real, source-verified case data in
`data/nodes_master.json`.

## What each command does

```bash
python3 -m epistemic_stack.cli summary                 # Crux Score + Z-score table, all 3 cases
python3 -m epistemic_stack.cli summary --case covid     # one case only
python3 -m epistemic_stack.cli critical                 # nodes with Z > 2.0, across all cases
python3 -m epistemic_stack.cli check-cycles               # verify the DAG property actually holds
python3 -m epistemic_stack.cli verify-sync               # checks code's Wdep table == spec's Wdep table
python3 -m epistemic_stack.cli demo-ingestion             # prints the real ACE extraction prompt + a worked example
python3 -m epistemic_stack.cli demo-compounding            # runs the two-analyst graph merge demo end-to-end
```

Run the test suite:

```bash
python3 tests/test_core.py          # 8 tests — locks in fixed bugs, prevents regressions
python3 tests/test_adversarial.py    # red-team suite — see "Adversarial testing results" below
```

## Repository structure

```
epistemic_stack/
  graph.py         Core DAG, Crux Score (Cs) and Z-score computation, cycle detection
  ingestion.py      ACE extraction prompt template + structural validator
  compounding.py    Three-step multi-analyst protocol (spec section 4.2)
  cli.py             Command-line entry point
data/
  nodes_master.json  The 25-node, source-verified reference dataset (same
                      file the written submission's worked examples derive
                      from — single source of truth, see its document_meta)
examples/
  compounding_demo.py  Runnable demonstration of two independent analysts
                        extending the COVID graph
tests/
  test_core.py         Automated tests, see "Bugs found by running this code" below
  test_adversarial.py  Red-team suite — see "Adversarial testing results" below
```

## Adversarial testing results

Added in response to supervisor review finding 2.5 ("Assenza di Testing
Avversariale"). Run with `python3 tests/test_adversarial.py`. This suite
is deliberately bounded (3-4 cases per category against the reviewer's full
proposal of 10 false claims / 5 COI sources / 3 biased-analyst graphs, which
would need a labeled adversarial benchmark this submission does not have
time to build and validate) — the goal is honest characterization of current
defenses, not a comprehensive red-team score.

**Result: 1 of 5 adversarial cases caught.**

| Category | Attack | Caught? |
|---|---|---|
| 1. Fabricated source | A plausible-sounding DOI/journal/author for a paper that does not exist | No |
| 1. Fabricated institutional claim | An "internal CERN memo" contradicting the real published LSAG report | No |
| 1. Misattributed real source | A real source (Scott Alexander's writeup) cited for a claim that is the OPPOSITE of what it actually says | No |
| 2. Hidden conflict of interest | A study with omitted industry funding | No |
| 3. Circular reasoning | Two nodes proposed as each other's sole DEPEND target | **Yes** (after a fix — see below) |

**What this means concretely:** the validator and ELM check structural
completeness and source-status presence, not factual accuracy. Nothing in
this pipeline currently re-fetches a cited source to verify it says what is
attributed to it, cross-references a funding-disclosure database, or
existed prior to running this test. These are named, scoped gaps, not a
vague aspiration — Category 1 and 2 defenses do not exist yet and are
listed as follow-up work in the written submission's limitations section.

**Category 3 was found broken, then fixed, by this red-team pass.**
`EpistemicGraph` is documented and named as a Directed Acyclic Graph, but
had no runtime check preventing an actual cycle — two nodes citing each
other as their only DEPEND target successfully inflated both nodes' Crux
Scores from pure mutual reference, with zero external grounding. This was
caught BY running the adversarial test, not theorized about it. Fixed by
adding `EpistemicGraph.detect_cycles()` and wiring it into
`CompoundingSession.accept_proposed_edges()`, so a proposed edge that would
create a cycle is now detected and rolled back rather than silently
accepted (`python3 -m epistemic_stack.cli check-cycles` runs this against
all three case graphs; `tests/test_core.py`'s
`test_circular_dependency_is_blocked_on_acceptance` locks the fix in).

## Honest scope — what this repository does and does not claim

This section exists because an earlier version of this submission described
components in prose without having run them, and a structured review
correctly identified that as the submission's most serious weakness. Running
the code surfaced real bugs (documented below) that the prose description
had not — which is itself the argument for why this section needs to be
direct about current limits rather than restating the architecture as if it
were already fully proven.

**Implemented and tested:**
- Crux Score / Z-score computation over the full 18-node dataset across all
  three case studies, with an operationally defined Wdep (see
  `data/nodes_master.json` → `schema_notes.wdep_specification`, and
  `graph.py`'s `verify-sync` command, which checks the two stay aligned).
- The three-step compounding protocol, runnable end-to-end against real
  graph data, including a visible, logged downstream Crux Score
  recalculation — not only specified in prose.
- Structural ACE validation (schema completeness, required source/status
  fields) with two distinct severity gates: lenient for raw extraction
  output, strict for anything entering `nodes_master.json`.

**Provided but explicitly not benchmarked:**
- The ACE extraction prompt template (`ingestion.py`) is the real template
  used to manually construct the 18 nodes in `nodes_master.json` — it is not
  illustrative. What it does NOT include: precision/recall numbers,
  inter-model comparison, or inter-annotator agreement statistics. Producing
  those requires a labeled evaluation set that does not yet exist. Building
  one is listed as follow-up work in the written spec's limitations section,
  not implied to already exist here.

**Not yet implemented:**
- Semantic (embedding-based) duplicate detection. `compounding.py`'s
  duplication check uses plain word-overlap (Jaccard similarity) as a
  transparent, dependency-free stand-in — flagged explicitly in that
  module's docstring rather than presented as the production approach.
- Automated, large-scale Authority Shielding classification (the Empirical
  Lineage Metric is specified and was applied manually to the worked
  examples in the written submission; this repo does not include a
  classifier trained or tested against a held-out set).

## Bugs found by running this code (kept here deliberately)

Two real defects were caught only because this code was actually executed,
not just described — concrete evidence for why an executable artifact
matters more than a correct-sounding specification:

1. **Unweighted contradiction density.** The first working version of
   `crux_score()` gave `ACE-SARS2-GENOME-FCS` (contested by a single
   unreviewed preprint) the exact same Crux Score as
   `ACE-SARS2-EPID-MARKET` (supported by a Science-published study) —
   because the formula counted CONTRADICT edges by raw count, not weighted
   by the contradicting source's own credibility. Fixed by routing
   `Econ` through the same Wdep-style weighting already used for `DEPEND`
   edges (see `contradiction_density()` in `graph.py`). `tests/test_core.py`
   now locks this in.

2. **Demo using the wrong vector type.** The first version of
   `examples/compounding_demo.py` proposed a `SUPPORT` edge to demonstrate
   cascading recalculation, and showed no score change — which is *correct
   behavior per the formula* (`Cs = Sum(Dout x Wdep) + Sum(Econ)` only
   weighs `DEPEND` and `CONTRADICT`), but was the wrong choice for a demo
   meant to show a visible recalculation. Fixed by using a `DEPEND` edge
   instead. This was a demo design error, not a formula bug, and is called
   out explicitly in the demo's own console output so it doesn't read as a
   silently corrected mistake.

3. **Z > 2.0 threshold does not fire on graphs this small.** Running
   `python3 -m epistemic_stack.cli critical` against all three case graphs
   (4, 5, and 9 nodes respectively) returns zero critical nodes anywhere.
   This is a real, load-bearing limitation, not a display bug: with so few
   nodes, the standard deviation of Crux Scores is large relative to any
   single node's deviation from the mean, so Z rarely clears 2.0 even for
   the node a human reader would clearly judge as the most contested
   (`ACE-SARS2-EPID-MARKET`, Z=+1.77, the highest in the dataset but still
   under threshold). The written submission's qualitative claim that the
   COVID crux node "remains high-tension" is about its *relative* position
   (highest Z-score in its graph) rather than its absolute Z-score clearing
   2.0 — this repo's `critical` command makes that distinction visible
   rather than papering over it. The honest fix is more nodes per graph
   (15-20 for COVID, per the written spec's known-gaps section), which
   would be expected to produce a more differentiated distribution; a
   percentile-based fallback (already specified in the written spec, section
   2.3.2) is also available for exactly this small-graph case via
   `EpistemicGraph.critical_nodes()`'s threshold parameter, though the CLI
   does not yet expose a percentile mode — listed here as follow-up work
   rather than silently worked around.

## Relationship to the written submission

`data/nodes_master.json` in this repository is the same file referenced as
the single source of truth in the written spec (`document_meta.purpose`).
The written submission's Part 3 worked examples, the cross-case comparison
table, and this code's `summary` command output are derived from the same
18 nodes — running `python3 -m epistemic_stack.cli summary` should produce
numbers consistent with what the written document describes qualitatively.
