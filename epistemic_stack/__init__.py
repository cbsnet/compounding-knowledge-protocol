"""The Epistemic Stack — reference implementation.

Modules:
  graph        Core DAG structure, Crux Score / Z-score computation.
  ingestion    ACE extraction prompt template and structural validator.
  compounding  Multi-analyst graph extension protocol (non-duplication,
               provisional integration, tracked cascading recalculation).
"""

from .graph import EpistemicGraph, ACENode, build_all_graphs, load_nodes_master
from .ingestion import validate_ace, ACE_EXTRACTION_PROMPT_TEMPLATE
from .compounding import CompoundingSession

__all__ = [
    "EpistemicGraph", "ACENode", "build_all_graphs", "load_nodes_master",
    "validate_ace", "ACE_EXTRACTION_PROMPT_TEMPLATE", "CompoundingSession",
]
