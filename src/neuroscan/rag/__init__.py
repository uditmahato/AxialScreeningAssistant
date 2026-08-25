"""Retrieval-Augmented Generation over the curated medical corpus.

Requirement: a RAG model over a carefully selected set of medical data
comprising a minimum of 50 medical documents, so the system provides accurate
advice for any abnormality classification.

The design principle throughout is that the model is a *summariser of retrieved
text*, never a source of medical knowledge in its own right. When retrieval
returns nothing above the relevance threshold, the system returns a fixed safe
response rather than letting the model answer from its parameters - the failure
mode that produces confident, fabricated medical advice.
"""

from neuroscan.rag.advisory import AdvisoryEngine, AdvisoryResult
from neuroscan.rag.corpus import CorpusDocument, load_corpus
from neuroscan.rag.llm_provider import LLMProvider, build_llm_provider
from neuroscan.rag.vectorstore import VectorStore, build_index, load_index

__all__ = [
    "AdvisoryEngine",
    "AdvisoryResult",
    "CorpusDocument",
    "LLMProvider",
    "VectorStore",
    "build_index",
    "build_llm_provider",
    "load_corpus",
    "load_index",
]
