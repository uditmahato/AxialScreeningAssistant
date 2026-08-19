#!/usr/bin/env python
"""Build the FAISS retrieval index over the knowledge base.

Usage:
    python scripts/build_index.py --rebuild
    python scripts/build_index.py --stats
    python scripts/build_index.py --query "ring enhancing lesion in Nepal"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroscan.config import load_config
from neuroscan.rag.corpus import corpus_statistics, load_corpus
from neuroscan.rag.vectorstore import VectorStoreError, build_index, load_index
from neuroscan.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Axial Screening Assistant FAISS index")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even if an index exists")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics and exit")
    parser.add_argument("--query", default=None, help="Run a test query against the index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def show_stats(cfg) -> int:
    documents = load_corpus(cfg.paths.knowledge_base_dir)
    stats = corpus_statistics(documents)

    print("\n" + "=" * 78)
    print("KNOWLEDGE BASE STATISTICS")
    print("=" * 78)
    print(f"  Total documents    : {stats['total_documents']}")
    print(f"  Medical documents  : {stats['medical_documents']}")
    print(f"  Nepal documents    : {stats['nepal_documents']}")
    print(f"  Total words        : {stats['total_words']:,}")
    print(f"  Unique sources     : {stats['unique_sources']}")
    print("\n  By category:")
    for category, count in sorted(stats["by_category"].items(), key=lambda kv: -kv[1]):
        print(f"    {category:20s} {count:>3}")
    print("\n  By severity:")
    for severity, count in sorted(stats["by_severity"].items(), key=lambda kv: -kv[1]):
        print(f"    {severity:20s} {count:>3}")

    target = cfg.rag.min_corpus_documents
    met = stats["meets_corpus_minimum"]
    print(f"\n  Corpus size (>= {target} medical documents): "
          f"{'MET' if met else 'NOT MET'} ({stats['medical_documents']})")
    print()
    return 0 if met else 2


def run_query(cfg, query: str, top_k: int) -> int:
    store = load_index(cfg)
    results = store.search(query, k=top_k)

    print("\n" + "=" * 78)
    print(f"QUERY: {query}")
    print("=" * 78)

    if not results:
        print("\n  No passages retrieved above the relevance threshold "
              f"({cfg.rag.score_threshold}).\n")
        return 0

    for i, chunk in enumerate(results, start=1):
        print(f"\n  [{i}] {chunk.title}")
        print(f"      category={chunk.category} severity={chunk.severity} "
              f"reviewed={chunk.last_reviewed}")
        preview = " ".join(chunk.content.split())[:280]
        print(f"      {preview}...")
    print()
    return 0


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    cfg = load_config(args.config)
    cfg.paths.ensure_all()

    if args.stats:
        return show_stats(cfg)

    if args.query:
        return run_query(cfg, args.query, args.top_k)

    store = build_index(cfg, rebuild=args.rebuild)

    print("\n" + "=" * 78)
    print("INDEX BUILT")
    print("=" * 78)
    print(f"  Location        : {cfg.paths.index_dir}")
    print(f"  Vectors         : {store.size:,}")
    print(f"  Embedding model : {cfg.rag.embedding_model}")

    stats = store.metadata.get("corpus_statistics", {})
    if stats:
        print(f"  Documents       : {stats.get('total_documents')} "
              f"({stats.get('medical_documents')} medical, "
              f"{stats.get('nepal_documents')} Nepal)")
        print(f"  Corpus size     : {'MET' if stats.get('meets_corpus_minimum') else 'NOT MET'}")

    print("\n  Test it:  python scripts/build_index.py --query \"what is a tuberculoma\"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VectorStoreError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None
