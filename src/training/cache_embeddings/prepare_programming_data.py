#!/usr/bin/env python3
"""
Prepare programming domain dataset from CoNaLa (Code/Natural Language Challenge).

Dataset: neulab/conala (mined config)
Size: 594,000 examples
License: MIT (commercial use allowed)
Quality: HIGH - Human-refined natural language intents from Stack Overflow

Why better than raw Stack Overflow:
- Questions are rewritten and refined by humans for clarity
- "Crowdsourced revised intents that better reflect the full meaning of the code"
- Clean, well-formatted natural language queries
- Focused on actual programming tasks users would search for

Source: https://huggingface.co/datasets/neulab/conala
Paper: https://arxiv.org/abs/1805.08949
"""

from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path


def prepare_conala_dataset(
    output_path: str = "data/cache_embeddings/programming/unlabeled_queries.jsonl",
    max_queries: int = None,
    min_length: int = 10,
    max_length: int = 200,
):
    """
    Prepare CoNaLa mined dataset for cache embedding training.

    Args:
        output_path: Where to save the queries
        max_queries: Limit number of queries (None = all)
        min_length: Minimum query length in characters
        max_length: Maximum query length in characters
    """
    print("=" * 80)
    print("Preparing CoNaLa Programming Dataset")
    print("=" * 80)
    print(f"\nDataset: neulab/conala (mined config)")
    print(f"License: MIT")
    print(f"Quality: Human-refined Stack Overflow intents\n")

    # Load dataset
    print("Loading CoNaLa mined dataset...")
    # Using codeparrot/conala-mined-curated instead of neulab/conala
    # This is the same dataset but in modern Parquet format
    dataset = load_dataset("codeparrot/conala-mined-curated", split="train")
    print(f"✓ Loaded {len(dataset):,} entries\n")

    # Extract queries
    print("Extracting and filtering queries...")
    queries = []

    for item in tqdm(dataset, desc="Processing"):
        # Use rewritten_intent if available (human-refined), else intent
        query = item.get("rewritten_intent") or item.get("intent", "")
        query = query.strip()

        # Filter by length
        if min_length <= len(query) <= max_length:
            queries.append(query)

    print(f"\n✓ Extracted {len(queries):,} queries after filtering")

    # Remove duplicates
    print(f"\nRemoving duplicates...")
    unique_queries = list(set(queries))
    duplicates_removed = len(queries) - len(unique_queries)
    print(
        f"✓ {len(unique_queries):,} unique queries ({duplicates_removed:,} duplicates removed)"
    )

    # Limit if requested
    if max_queries and len(unique_queries) > max_queries:
        unique_queries = unique_queries[:max_queries]
        print(f"✓ Limited to {max_queries:,} queries")

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    with open(output_file, "w") as f:
        for query in unique_queries:
            f.write(json.dumps({"query": query}) + "\n")

    print(f"✓ Saved {len(unique_queries):,} queries")

    # Statistics
    avg_length = sum(len(q) for q in unique_queries) / len(unique_queries)

    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    print(f"Total unique queries: {len(unique_queries):,}")
    print(f"Average length: {avg_length:.1f} characters")
    print(f"Size on disk: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    print("\nSample queries:")
    for i, query in enumerate(unique_queries[:10], 1):
        print(f"  {i}. {query}")

    print("\n" + "=" * 80)
    print("✓ Dataset preparation complete!")
    print("=" * 80)

    return len(unique_queries)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare CoNaLa programming dataset for cache embedding training"
    )
    parser.add_argument(
        "--output",
        default="data/cache_embeddings/programming/unlabeled_queries.jsonl",
        help="Output path for queries JSONL file",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to extract (default: all)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum query length in characters (default: 10)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum query length in characters (default: 200)",
    )

    args = parser.parse_args()

    prepare_conala_dataset(
        output_path=args.output,
        max_queries=args.max_queries,
        min_length=args.min_length,
        max_length=args.max_length,
    )
