"""
Analyze hard examples being mined across iterations.

This script re-runs a small test to examine:
1. What hard examples are being mined each iteration
2. Whether we're running out of hard examples (plateau explanation)
3. Query coverage and diversity
"""

import pickle
import os
from pathlib import Path
from typing import List, Dict
from collections import Counter, defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from robust_iterative_miner import RobustIterativeHardNegativeMiner


def analyze_iteration_data(checkpoint_dir: str, iteration: int):
    """Analyze checkpoint data for a specific iteration."""

    results = {
        'iteration': iteration,
        'hard_pos_count': 0,
        'hard_neg_count': 0,
        'triplet_count': 0,
        'queries_with_hard_examples': 0,
        'hard_example_distribution': {},
        'score_distribution': Counter()
    }

    # Load scored pairs checkpoint
    scored_pairs_path = f"{checkpoint_dir}/iter_{iteration}_scored_pairs.pkl"
    if not os.path.exists(scored_pairs_path):
        print(f"No scored pairs found for iteration {iteration}")
        return None

    with open(scored_pairs_path, 'rb') as f:
        checkpoint = pickle.load(f)
        scored_pairs = checkpoint['data']['scored_pairs']

    print(f"\n{'='*60}")
    print(f"Iteration {iteration} Analysis")
    print(f"{'='*60}")
    print(f"Total scored pairs: {len(scored_pairs)}")

    # Analyze score distribution
    for pair in scored_pairs:
        results['score_distribution'][pair['llm_score']] += 1

    print(f"\nLLM Score Distribution:")
    for score in sorted(results['score_distribution'].keys()):
        count = results['score_distribution'][score]
        pct = 100 * count / len(scored_pairs)
        print(f"  Score {score}: {count:4d} ({pct:5.1f}%)")

    # Analyze hard examples
    hard_pos = []
    hard_neg = []
    query_hard_examples = defaultdict(lambda: {'pos': 0, 'neg': 0})

    for pair in scored_pairs:
        # Hard positive: LLM says relevant but model ranked low
        if pair['llm_score'] >= 3 and pair['model_rank'] > 10:
            hard_pos.append(pair)
            query_hard_examples[pair['query']]['pos'] += 1

        # Hard negative: LLM says not relevant but model ranked high
        elif pair['llm_score'] <= 2 and pair['model_rank'] <= 5:
            hard_neg.append(pair)
            query_hard_examples[pair['query']]['neg'] += 1

    results['hard_pos_count'] = len(hard_pos)
    results['hard_neg_count'] = len(hard_neg)
    results['queries_with_hard_examples'] = len(query_hard_examples)

    print(f"\nHard Examples Mined:")
    print(f"  Hard Positives: {len(hard_pos)} (LLM≥3, rank>10)")
    print(f"  Hard Negatives: {len(hard_neg)} (LLM≤2, rank≤5)")
    print(f"  Queries with hard examples: {len(query_hard_examples)}")

    # Count triplets (queries with BOTH pos and neg)
    queries_with_triplets = 0
    total_triplets = 0

    for query, counts in query_hard_examples.items():
        if counts['pos'] > 0 and counts['neg'] > 0:
            queries_with_triplets += 1
            # Each positive pairs with one negative
            total_triplets += counts['pos']

    results['triplet_count'] = total_triplets

    print(f"\nTriplet Formation:")
    print(f"  Queries with BOTH pos+neg: {queries_with_triplets}")
    print(f"  Total triplets formed: {total_triplets}")

    # Show example hard positives
    if hard_pos:
        print(f"\nExample Hard Positives (first 2):")
        for i, pair in enumerate(hard_pos[:2]):
            print(f"\n  [{i+1}] Rank: {pair['model_rank']}, LLM Score: {pair['llm_score']}")
            print(f"      Query: {pair['query'][:80]}...")
            print(f"      Chunk: {pair['chunk']['text'][:80]}...")

    # Show example hard negatives
    if hard_neg:
        print(f"\nExample Hard Negatives (first 2):")
        for i, pair in enumerate(hard_neg[:2]):
            print(f"\n  [{i+1}] Rank: {pair['model_rank']}, LLM Score: {pair['llm_score']}")
            print(f"      Query: {pair['query'][:80]}...")
            print(f"      Chunk: {pair['chunk']['text'][:80]}...")

    return results


def main():
    """Run analysis on recent test data."""

    import argparse
    parser = argparse.ArgumentParser(description="Analyze hard examples from iterative mining")
    parser.add_argument("--checkpoint-dir", default="checkpoints/analysis", help="Checkpoint directory")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of queries to test")
    parser.add_argument("--num-iterations", type=int, default=3, help="Number of iterations")
    parser.add_argument("--llm-endpoint", default="http://localhost:8000/v1", help="vLLM endpoint")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing checkpoints")
    args = parser.parse_args()

    if args.analyze_only:
        # Just analyze existing checkpoints
        print("Analyzing existing checkpoints...")
        all_results = []
        for iteration in range(args.num_iterations):
            result = analyze_iteration_data(args.checkpoint_dir, iteration)
            if result:
                all_results.append(result)

        # Summary comparison
        if len(all_results) > 1:
            print(f"\n{'='*60}")
            print("CROSS-ITERATION COMPARISON")
            print(f"{'='*60}")
            print(f"\n{'Iteration':<12} {'Hard Pos':<12} {'Hard Neg':<12} {'Triplets':<12}")
            print("-" * 60)
            for r in all_results:
                print(f"{r['iteration']:<12} {r['hard_pos_count']:<12} {r['hard_neg_count']:<12} {r['triplet_count']:<12}")

            # Check for plateau
            if len(all_results) >= 2:
                triplet_change = all_results[-1]['triplet_count'] - all_results[-2]['triplet_count']
                pct_change = 100 * triplet_change / max(all_results[-2]['triplet_count'], 1)
                print(f"\nTriplet change (iter {len(all_results)-2} → {len(all_results)-1}): {triplet_change:+d} ({pct_change:+.1f}%)")

                if abs(pct_change) < 10:
                    print("⚠️  WARNING: Plateau detected - triplet count not changing significantly")

    else:
        # Run new test
        print("Loading data...")
        data_dir = "data"

        with open(f"{data_dir}/corpus_chunks.pkl", "rb") as f:
            corpus_chunks = pickle.load(f)

        with open(f"{data_dir}/train_queries.pkl", "rb") as f:
            all_queries = pickle.load(f)

        print(f"Loaded {len(corpus_chunks)} chunks, {len(all_queries)} queries")
        print(f"Using {args.num_queries} queries for analysis")

        # Initialize miner
        llm_client = OpenAI(base_url=args.llm_endpoint, api_key="EMPTY")

        miner = RobustIterativeHardNegativeMiner(
            base_model_name="sentence-transformers/all-MiniLM-L12-v2",
            llm_client=llm_client,
            llm_model="Qwen/Qwen2.5-7B-Instruct",
            corpus_chunks=corpus_chunks,
            output_dir=f"models/analysis",
            checkpoint_dir=args.checkpoint_dir,
            device="cpu"
        )

        # Run iterations with analysis
        import random
        all_results = []

        for iteration in range(args.num_iterations):
            print(f"\n{'#'*60}")
            print(f"# Iteration {iteration}/{args.num_iterations - 1}")
            print(f"{'#'*60}")

            # Sample queries
            random.seed(42 + iteration)
            queries = random.sample(all_queries, min(args.num_queries, len(all_queries)))

            # Run iteration
            miner.run_iteration(queries, iteration=iteration)

            # Analyze results
            result = analyze_iteration_data(args.checkpoint_dir, iteration)
            if result:
                all_results.append(result)

        # Summary
        if len(all_results) > 1:
            print(f"\n{'='*60}")
            print("FINAL SUMMARY")
            print(f"{'='*60}")
            print(f"\n{'Iteration':<12} {'Hard Pos':<12} {'Hard Neg':<12} {'Triplets':<12}")
            print("-" * 60)
            for r in all_results:
                print(f"{r['iteration']:<12} {r['hard_pos_count']:<12} {r['hard_neg_count']:<12} {r['triplet_count']:<12}")


if __name__ == "__main__":
    main()
