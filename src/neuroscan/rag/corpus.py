"""Loading and chunking the knowledge base.

Frontmatter is parsed into per-chunk metadata so that retrieved text can be
attributed to a named source with a review date. Attribution is not cosmetic
here: a clinical answer whose provenance cannot be shown is an answer a
clinician has no way to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from neuroscan.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from neuroscan.config import Config

log = get_logger("rag.corpus")

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class CorpusError(RuntimeError):
    """Raised when the knowledge base is missing or malformed."""


@dataclass
class CorpusDocument:
    """One knowledge base document with its parsed frontmatter."""

    path: Path
    doc_id: str
    title: str
    category: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> str:
        return str(self.metadata.get("severity", "informational"))

    @property
    def maps_to_class(self) -> list[str]:
        value = self.metadata.get("maps_to_class", [])
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value]

    @property
    def sources(self) -> list[str]:
        value = self.metadata.get("sources", [])
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value]

    @property
    def last_reviewed(self) -> str:
        return str(self.metadata.get("last_reviewed", "unknown"))

    def citation(self) -> str:
        """Short attribution string shown alongside a generated answer."""
        return f"{self.title} (reviewed {self.last_reviewed})"


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from body content.

    A malformed frontmatter block is logged and skipped rather than raised: one
    bad document must not prevent the entire index from building.
    """
    match = FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}, text

    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        log.warning("Malformed frontmatter, treating document as unstructured: %s", exc)
        return {}, text[match.end() :]

    if not isinstance(metadata, dict):
        return {}, text[match.end() :]

    return metadata, text[match.end() :]


def load_corpus(
    knowledge_base_dir: Path,
    *,
    include_nepal: bool = True,
) -> list[CorpusDocument]:
    """Load every markdown document from the knowledge base.

    Args:
        knowledge_base_dir: Root of the knowledge base.
        include_nepal: Include the Nepal-specific healthcare database.

    Raises:
        CorpusError: If the directory is missing or contains no documents.
    """
    root = Path(knowledge_base_dir)
    if not root.exists():
        raise CorpusError(
            f"Knowledge base directory not found: {root}\n"
            f"The corpus ships with the repository - check the path in configs/default.yaml."
        )

    documents: list[CorpusDocument] = []
    skipped = 0

    for path in sorted(root.rglob("*.md")):
        # README and VERIFICATION are project documentation, not clinical content.
        if path.name in {"README.md", "SOURCES.md"}:
            continue
        if not include_nepal and "nepal" in path.parts:
            continue

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("Could not read %s: %s", path, exc)
            skipped += 1
            continue

        metadata, body = parse_frontmatter(raw)
        if not body.strip():
            log.warning("Document has no body content, skipping: %s", path)
            skipped += 1
            continue

        doc_id = str(metadata.get("id") or path.stem)
        title = str(metadata.get("title") or path.stem.replace("-", " ").title())
        category = str(metadata.get("category") or path.parent.name)

        documents.append(
            CorpusDocument(
                path=path,
                doc_id=doc_id,
                title=title,
                category=category,
                content=body.strip(),
                metadata=metadata,
            )
        )

    if not documents:
        raise CorpusError(f"No usable documents found under {root}")

    duplicates = _find_duplicate_ids(documents)
    if duplicates:
        log.warning(
            "Duplicate document ids found: %s. Citations may be ambiguous.", duplicates
        )

    log.info(
        "Loaded %d corpus document(s)%s",
        len(documents),
        f" ({skipped} skipped)" if skipped else "",
    )
    return documents


def _find_duplicate_ids(documents: list[CorpusDocument]) -> list[str]:
    seen: dict[str, int] = {}
    for document in documents:
        seen[document.doc_id] = seen.get(document.doc_id, 0) + 1
    return sorted(doc_id for doc_id, count in seen.items() if count > 1)


def chunk_documents(
    documents: list[CorpusDocument],
    cfg: Config,
) -> list[Any]:
    """Split documents into overlapping chunks as LangChain ``Document`` objects.

    Splitting is markdown-aware and prefers heading and paragraph boundaries.
    That matters for this corpus specifically: several documents open with an
    emergency instruction, and a splitter that cut mid-section could return a
    chunk carrying the caveat without the finding it qualifies, or the reverse.
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.rag.chunk_size,
        chunk_overlap=cfg.rag.chunk_overlap,
        length_function=len,
        # Ordered most- to least-preferred break point.
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for document in documents:
        # Prepending the title to each chunk means an isolated chunk retrieved
        # from the middle of a document still identifies what it is about,
        # which materially improves embedding relevance.
        header = f"# {document.title}\n\n"
        pieces = splitter.split_text(document.content)

        for index, piece in enumerate(pieces):
            chunks.append(
                Document(
                    page_content=header + piece if index > 0 else piece,
                    metadata={
                        "doc_id": document.doc_id,
                        "title": document.title,
                        "category": document.category,
                        "subcategory": str(document.metadata.get("subcategory", "")),
                        "severity": document.severity,
                        "maps_to_class": ",".join(document.maps_to_class),
                        "audience": ",".join(
                            str(a) for a in document.metadata.get("audience", [])
                        ),
                        "last_reviewed": document.last_reviewed,
                        "sources": " | ".join(document.sources),
                        "source_path": str(document.path),
                        "chunk_index": index,
                        "chunk_count": len(pieces),
                    },
                )
            )

    log.info(
        "Split %d document(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(documents), len(chunks), cfg.rag.chunk_size, cfg.rag.chunk_overlap,
    )
    return chunks


def corpus_statistics(documents: list[CorpusDocument]) -> dict[str, Any]:
    """Summary used to evidence the 50-document corpus requirement."""
    from collections import Counter

    categories = Counter(d.category for d in documents)
    severities = Counter(d.severity for d in documents)
    medical = [d for d in documents if d.category != "nepal"]

    return {
        "total_documents": len(documents),
        "medical_documents": len(medical),
        "nepal_documents": len(documents) - len(medical),
        "by_category": dict(categories),
        "by_severity": dict(severities),
        "total_words": sum(len(d.content.split()) for d in documents),
        "unique_sources": len({s for d in documents for s in d.sources}),
        "meets_corpus_minimum": len(medical) >= 50,
    }


__all__ = [
    "CorpusDocument",
    "CorpusError",
    "chunk_documents",
    "corpus_statistics",
    "load_corpus",
    "parse_frontmatter",
]
