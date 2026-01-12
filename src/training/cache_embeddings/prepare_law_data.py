#!/usr/bin/env python3
"""
Prepare law domain dataset for cache embedding training.

Downloads and formats legal queries from CaseHOLD dataset to create
~60k legal queries for training domain-specific cache embeddings.

Usage:
    python3 prepare_law_data.py --output data/cache_embeddings/law/unlabeled_queries.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.cache_embeddings.common_utils import (
    setup_logging,
    save_jsonl,
)

logger = logging.getLogger(__name__)

# Optional dependency
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.error("datasets library not available. Install: pip install datasets")


def download_casehold() -> List[Dict[str, str]]:
    """
    Download CaseHOLD dataset from HuggingFace.

    CaseHOLD contains legal case holdings with multiple choice questions
    about legal reasoning. We extract the prompts as legal queries.

    Returns:
        List of dictionaries with 'query' and 'source' fields
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required. Install: pip install datasets")

    logger.info("Downloading CaseHOLD dataset from HuggingFace...")
    logger.info("This may take 5-10 minutes for the first download...")

    # Load CaseHOLD dataset - use reglab/casehold instead
    # Dataset structure: {'citing_prompt', 'holding_0', 'holding_1', ...}
    dataset = load_dataset("reglab/casehold", split="train")

    logger.info(f"Downloaded {len(dataset)} examples from CaseHOLD")

    queries = []
    for idx, item in enumerate(dataset):
        # Extract the citing prompt (legal question/context)
        if 'citing_prompt' in item and item['citing_prompt']:
            prompt = item['citing_prompt'].strip()

            # Clean up the prompt
            # CaseHOLD prompts are legal case excerpts - extract the key question
            if len(prompt) > 50:  # Filter out very short fragments
                queries.append({
                    "query": prompt,
                    "source": "casehold",
                    "index": idx
                })

        if (idx + 1) % 10000 == 0:
            logger.info(f"Processed {idx + 1} examples...")

    logger.info(f"Extracted {len(queries)} legal queries from CaseHOLD")
    return queries


def clean_and_deduplicate(queries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Clean and deduplicate queries.

    Args:
        queries: List of query dictionaries

    Returns:
        Cleaned and deduplicated list
    """
    logger.info(f"Cleaning {len(queries)} queries...")

    # Deduplicate by query text
    seen = set()
    cleaned = []

    for item in queries:
        query = item['query']

        # Skip duplicates
        if query in seen:
            continue

        # Skip very short queries (likely fragments)
        if len(query.split()) < 8:
            continue

        # Skip very long queries (likely full case excerpts)
        if len(query) > 500:
            # Truncate to first 500 chars as a legal question
            query = query[:500].rsplit('.', 1)[0] + '.'
            item['query'] = query

        seen.add(query)
        cleaned.append(item)

    logger.info(f"After cleaning: {len(cleaned)} unique queries")
    return cleaned


def sample_queries(queries: List[Dict[str, str]], target_count: int) -> List[Dict[str, str]]:
    """
    Sample queries to reach target count.

    Args:
        queries: List of query dictionaries
        target_count: Target number of queries

    Returns:
        Sampled list (or full list if smaller than target)
    """
    if len(queries) <= target_count:
        logger.info(f"Dataset has {len(queries)} queries (target: {target_count})")
        return queries

    logger.info(f"Sampling {target_count} queries from {len(queries)} total")

    # Use every Nth query for even distribution
    step = len(queries) / target_count
    sampled = [queries[int(i * step)] for i in range(target_count)]

    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Prepare law domain dataset for cache embedding training"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path (e.g., data/cache_embeddings/law/unlabeled_queries.jsonl)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=60000,
        help="Target number of queries (default: 60000, similar to medical domain)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)

    logger.info("=" * 80)
    logger.info("Law Domain Data Preparation")
    logger.info("=" * 80)
    logger.info(f"Output file: {args.output}")
    logger.info(f"Target count: {args.target_count:,} queries")
    logger.info("")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Download CaseHOLD
    logger.info("Step 1: Downloading CaseHOLD dataset...")
    queries = download_casehold()

    # Step 2: Clean and deduplicate
    logger.info("\nStep 2: Cleaning and deduplicating queries...")
    queries = clean_and_deduplicate(queries)

    # Step 3: Sample to target count
    logger.info("\nStep 3: Sampling queries...")
    queries = sample_queries(queries, args.target_count)

    # Step 4: Save to JSONL
    logger.info(f"\nStep 4: Saving to {args.output}...")

    # Convert to simple format (just query text)
    output_data = [{"query": q["query"]} for q in queries]
    save_jsonl(output_data, args.output)

    # Print statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total queries: {len(queries):,}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info("")
    logger.info("Sample queries:")
    for i, q in enumerate(queries[:5], 1):
        preview = q["query"][:100] + "..." if len(q["query"]) > 100 else q["query"]
        logger.info(f"  {i}. {preview}")
    logger.info("")
    logger.info("✅ Dataset preparation complete!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify queries: head -5 data/cache_embeddings/law/unlabeled_queries.jsonl | jq .")
    logger.info("  2. Test with 50 queries: Use generate_training_data.py with --max-queries 50")


if __name__ == "__main__":
    main()
