"""FAISS vector store construction and retrieval.

The embedding model is multilingual by deliberate choice. A Nepali-speaking
health worker asking "मस्तिष्कमा गाँठो भएमा के गर्ने?" must retrieve from an
English clinical corpus, because maintaining a fully parallel Nepali corpus at
this quality is not achievable within the project. A multilingual
sentence-transformer maps both languages into a shared space, so cross-lingual
retrieval works without translating the corpus.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neuroscan.utils import get_logger, read_json, write_json

if TYPE_CHECKING:  # pragma: no cover
    from neuroscan.config import Config

log = get_logger("rag.vectorstore")

INDEX_METADATA_FILE = "index_metadata.json"


class VectorStoreError(RuntimeError):
    """Raised when the index cannot be built, loaded or queried."""


@dataclass
class RetrievedChunk:
    """One retrieved chunk with its relevance score and provenance."""

    content: str
    score: float
    title: str
    doc_id: str
    category: str
    severity: str
    sources: str
    last_reviewed: str

    def citation(self) -> str:
        return f"{self.title} (reviewed {self.last_reviewed})"


def _make_e5_embeddings(base_cls):
    """Build an E5 embedding class that applies the required prefixes.

    The E5 family is trained with an asymmetric objective: queries are encoded
    with a ``query: `` prefix and documents with ``passage: ``. The prefixes
    are not decoration - they are how the model distinguishes the two roles,
    and omitting them costs a large fraction of its retrieval quality.
    """

    class E5Embeddings(base_cls):  # type: ignore[misc, valid-type]
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return super().embed_documents([f"passage: {t}" for t in texts])

        def embed_query(self, text: str) -> list[float]:
            return super().embed_query(f"query: {text}")

    return E5Embeddings


def build_embeddings(cfg: Config):
    """Instantiate the sentence-transformer embedding model.

    **Why an E5 model rather than a paraphrase model.** The obvious choice -
    ``paraphrase-multilingual-MiniLM`` - is trained for *symmetric* similarity:
    judging whether two sentences of comparable length mean the same thing.
    Retrieval here is *asymmetric*: a four-word question is matched against a
    several-hundred-word clinical passage.

    Measured on this corpus, the paraphrase model failed the cases that matter
    most. "Could it be an infection rather than cancer?" returned a list of
    tumour types with the ring-enhancing differential document absent from the
    top ten entirely - the single worst possible answer for a Nepali user,
    since that document exists specifically to stop infection being mistaken
    for cancer. The E5 family is trained for question-to-passage retrieval and
    handles it correctly.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    device = cfg.rag.embedding_device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            device = "cpu"

    model_name = cfg.rag.embedding_model
    log.info("Loading embedding model %s on %s", model_name, device)

    kwargs = {
        "model_name": model_name,
        "model_kwargs": {"device": device},
        # Normalised embeddings make inner product equivalent to cosine
        # similarity, which is what the relevance thresholds assume.
        "encode_kwargs": {"normalize_embeddings": True},
    }

    if "e5" in model_name.lower():
        return _make_e5_embeddings(HuggingFaceEmbeddings)(**kwargs)
    return HuggingFaceEmbeddings(**kwargs)


class VectorStore:
    """Thin wrapper over a LangChain FAISS store.

    Adds two things the raw store does not provide: score-thresholded retrieval
    that can legitimately return nothing, and typed results carrying the
    provenance needed for attribution.
    """

    def __init__(self, store: Any, cfg: Config, metadata: dict[str, Any] | None = None) -> None:
        self.store = store
        self.cfg = cfg
        self.metadata = metadata or {}

    @property
    def size(self) -> int:
        try:
            return int(self.store.index.ntotal)
        except AttributeError:  # pragma: no cover
            return 0

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        category_filter: str | None = None,
        min_score: float | None = None,
        search_type: str | None = None,
        relative_floor: float = 0.5,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Natural-language query, in English or Nepali.
            k: Number of chunks to return.
            category_filter: Restrict to one corpus category.
            min_score: Override the absolute relevance floor.
            relative_floor: Keep only chunks scoring at least this fraction of
                the best hit for this query. Set to 0 to disable.
            search_type: ``'mmr'`` or ``'similarity'``, overriding the config.
                The two serve genuinely different purposes here. The advisory
                wants **mmr**: it must present a differential, so five chunks
                from the single closest document would be a worse answer than
                five chunks spanning infection, tumour and next steps. A direct
                chatbot question wants **similarity**: the user asked about one
                thing, and diversity just dilutes the answer with adjacent
                topics.

        Returns an empty list when nothing clears the relevance threshold. That
        is a meaningful outcome, not a failure: the advisory layer converts it
        into a fixed safe response rather than allowing the model to answer
        from memory.
        """
        if not query.strip():
            return []

        top_k = k if k is not None else self.cfg.rag.top_k
        threshold = min_score if min_score is not None else self.cfg.rag.score_threshold
        mode = search_type or self.cfg.rag.search_type

        filter_dict = {"category": category_filter} if category_filter else None

        try:
            if mode == "mmr":
                # Maximal Marginal Relevance trades a little relevance for
                # diversity. It matters here because several corpus documents
                # cover overlapping ground; without it, all five results can
                # come from one document and the differential is lost.
                documents = self.store.max_marginal_relevance_search(
                    query,
                    k=top_k,
                    fetch_k=self.cfg.rag.fetch_k,
                    lambda_mult=self.cfg.rag.mmr_lambda,
                    filter=filter_dict,
                )
                scored = [(doc, None) for doc in documents]
            else:
                scored = self.store.similarity_search_with_relevance_scores(
                    query, k=top_k, filter=filter_dict
                )
        except Exception as exc:
            log.error("Vector search failed: %s", exc)
            return []

        candidates: list[tuple[Any, float]] = []
        for document, score in scored:
            # MMR does not return scores; those results are taken on trust
            # because MMR already selected them from a relevance-ranked pool.
            relevance = 1.0 if score is None else max(0.0, float(score))
            if score is not None and relevance < threshold:
                continue
            candidates.append((document, relevance))

        # Relative floor, applied on top of the absolute one.
        #
        # An absolute threshold alone cannot work across query lengths.
        # Cosine similarity against a long document chunk is systematically
        # lower for a four-word question ("Where do I go?") than for a
        # fifteen-word one, so any fixed cut-off either discards every short
        # question or admits noise on every long one. Keeping results within a
        # fraction of the best hit for *this* query adapts automatically:
        # it trims the long tail without assuming an absolute scale.
        if candidates and relative_floor > 0:
            best_score = max(score for _, score in candidates)
            cutoff = best_score * relative_floor
            candidates = [(d, s) for d, s in candidates if s >= cutoff]

        results: list[RetrievedChunk] = []
        for document, relevance in candidates:
            meta = document.metadata
            results.append(
                RetrievedChunk(
                    content=document.page_content,
                    score=relevance,
                    title=str(meta.get("title", "Untitled")),
                    doc_id=str(meta.get("doc_id", "")),
                    category=str(meta.get("category", "")),
                    severity=str(meta.get("severity", "informational")),
                    sources=str(meta.get("sources", "")),
                    last_reviewed=str(meta.get("last_reviewed", "unknown")),
                )
            )

        log.debug("Retrieved %d chunk(s) for query: %.60s", len(results), query)
        return results

    def search_multi(
        self,
        queries: list[str],
        *,
        k: int | None = None,
        min_score: float | None = None,
        search_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve against several phrasings of one information need, merged.

        Short, anaphoric questions embed poorly. "Could it be an infection
        rather than cancer?" is dominated by the word *cancer*, and retrieves
        a list of tumour types while the ring-enhancing differential - the
        document that actually answers it - does not appear at all. That is the
        exact failure this project cannot afford, because it is the question a
        Nepali health worker most needs answered correctly.

        Issuing the raw question alongside a context-expanded rewrite and
        merging the results recovers the missing documents, at the cost of one
        extra embedding lookup. Chunks are deduplicated on content and ranked
        by their best score across queries.

        The first query is authoritative. Later queries only *fill remaining
        slots* with chunks the first did not already find - they never
        outrank it.

        That ordering is the whole design. An earlier version merged all
        queries by best score, and the longer, keyword-stuffed expansion
        reliably outscored the user's actual question: every query, including
        "How much will it cost?", came back with the same generic set of
        tumour documents. A verbose expansion embeds as "generic brain MRI
        text" and matches every long clinical document equally well, which
        destroys precisely the signal the user supplied.

        Args:
            queries: Alternative phrasings. The first must be the user's own
                wording.
            k: Number of chunks to return after merging.
            min_score: Absolute relevance floor applied per query.
            search_type: Passed through to :meth:`search`.
        """
        top_k = k if k is not None else self.cfg.rag.top_k
        merged: list[RetrievedChunk] = []
        seen: set[str] = set()

        for position, query in enumerate(queries):
            if not query or not query.strip():
                continue
            if len(merged) >= top_k:
                break

            # The primary query keeps the normal relative floor. Supplementary
            # queries are only topping up, so they are filtered harder to
            # avoid dragging in weakly-related material.
            chunks = self.search(
                query,
                k=top_k,
                min_score=min_score,
                search_type=search_type,
                relative_floor=0.5 if position == 0 else 0.7,
            )
            for chunk in chunks:
                key = chunk.content[:200]
                if key in seen:
                    continue
                seen.add(key)
                merged.append(chunk)
                if len(merged) >= top_k:
                    break

        log.debug("Multi-query retrieval: %d quer(ies) -> %d chunk(s)", len(queries), len(merged))
        return merged

    def save(self, index_dir: Path) -> Path:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(index_dir))
        write_json(index_dir / INDEX_METADATA_FILE, self.metadata)
        log.info("Index saved to %s (%d vectors)", index_dir, self.size)
        return index_dir


def build_index(cfg: Config, *, rebuild: bool = False) -> VectorStore:
    """Build the FAISS index from the knowledge base.

    Raises:
        VectorStoreError: If the corpus does not meet the corpus minimum,
            or if index construction fails.
    """
    from langchain_community.vectorstores import FAISS

    from neuroscan.rag.corpus import chunk_documents, corpus_statistics, load_corpus

    index_dir = cfg.paths.index_dir
    if index_dir.exists() and not rebuild:
        log.info("Index already exists at %s - pass rebuild=True to regenerate", index_dir)
        return load_index(cfg)

    documents = load_corpus(cfg.paths.knowledge_base_dir)
    stats = corpus_statistics(documents)

    if stats["medical_documents"] < cfg.rag.min_corpus_documents:
        raise VectorStoreError(
            f"Corpus has only {stats['medical_documents']} medical documents, but "
            f"The corpus requires at least {cfg.rag.min_corpus_documents}. "
            f"Add documents under {cfg.paths.knowledge_base_dir / 'medical'}."
        )

    chunks = chunk_documents(documents, cfg)
    embeddings = build_embeddings(cfg)

    log.info("Embedding %d chunk(s) - this may take a minute on first run", len(chunks))
    try:
        store = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        raise VectorStoreError(f"Failed to build the FAISS index: {exc}") from exc

    metadata = {
        "embedding_model": cfg.rag.embedding_model,
        "chunk_size": cfg.rag.chunk_size,
        "chunk_overlap": cfg.rag.chunk_overlap,
        "n_chunks": len(chunks),
        "corpus_statistics": stats,
    }

    if index_dir.exists() and rebuild:
        shutil.rmtree(index_dir)

    vector_store = VectorStore(store, cfg, metadata)
    vector_store.save(index_dir)

    log.info(
        "Index built: %d document(s) -> %d chunk(s) -> %d vector(s)",
        stats["total_documents"], len(chunks), vector_store.size,
    )
    return vector_store


def load_index(cfg: Config) -> VectorStore:
    """Load a previously built index.

    Raises:
        VectorStoreError: If the index is missing or was built with a different
            embedding model - vectors from two different models share a space
            only by coincidence, and mixing them produces silently meaningless
            retrieval.
    """
    from langchain_community.vectorstores import FAISS

    index_dir = cfg.paths.index_dir
    if not index_dir.exists() or not any(index_dir.iterdir()):
        raise VectorStoreError(
            f"No FAISS index at {index_dir}. Build it first:\n"
            f"  python scripts/build_index.py --rebuild"
        )

    metadata: dict[str, Any] = {}
    metadata_path = index_dir / INDEX_METADATA_FILE
    if metadata_path.exists():
        try:
            metadata = read_json(metadata_path)
        except Exception as exc:
            log.warning("Could not read index metadata: %s", exc)

    stored_model = metadata.get("embedding_model")
    if stored_model and stored_model != cfg.rag.embedding_model:
        raise VectorStoreError(
            f"Index was built with embedding model {stored_model!r} but the configuration "
            f"specifies {cfg.rag.embedding_model!r}. Vectors from different models are not "
            f"comparable. Rebuild with:\n  python scripts/build_index.py --rebuild"
        )

    embeddings = build_embeddings(cfg)
    try:
        # allow_dangerous_deserialization is required because FAISS stores the
        # docstore as a pickle. Safe here: the index is built locally by this
        # project from its own repository-controlled corpus.
        store = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
    except Exception as exc:
        raise VectorStoreError(f"Failed to load the FAISS index from {index_dir}: {exc}") from exc

    vector_store = VectorStore(store, cfg, metadata)
    log.info("Loaded index from %s (%d vectors)", index_dir, vector_store.size)
    return vector_store


__all__ = [
    "INDEX_METADATA_FILE",
    "RetrievedChunk",
    "VectorStore",
    "VectorStoreError",
    "build_embeddings",
    "build_index",
    "load_index",
]
