"""
Test iterative mining with a SMALL SAMPLE (30-50 queries).

This script validates the pipeline before running full training.
Use this to verify prompts, LLM judging, and data format are correct.
"""

import pickle
import json
from iterative_miner import IterativeHardNegativeMiner
from openai import OpenAI
import argparse


def main():
    parser = argparse.ArgumentParser(description="Test iterative mining with small sample")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--num-queries", type=int, default=30, help="Number of queries to test (default: 30)")
    parser.add_argument("--num-candidates", type=int, default=10, help="Number of candidates per query (default: 10)")
    parser.add_argument("--llm-endpoint", default=None, help="vLLM endpoint (e.g., http://localhost:8000/v1)")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct", help="LLM model name")
    parser.add_argument("--output-file", default="small_sample_results.json", help="Output file for inspection")
    args = parser.parse_args()

    print("="*60)
    print("SMALL SAMPLE TEST - Iterative Hard-Negative Mining")
    print("="*60)
    print(f"Queries: {args.num_queries}")
    print(f"Candidates per query: {args.num_candidates}")
    print(f"LLM model: {args.llm_model}")
    print(f"LLM endpoint: {args.llm_endpoint or 'OpenAI API'}")
    print("="*60)

    # Load data
    print("\n[1/5] Loading data...")
    with open(f"{args.data_dir}/corpus_chunks.pkl", "rb") as f:
        corpus_chunks = pickle.load(f)

    with open(f"{args.data_dir}/train_queries.pkl", "rb") as f:
        train_queries = pickle.load(f)

    # Take small sample
    train_queries = train_queries[:args.num_queries]
    print(f"Loaded {len(corpus_chunks)} chunks, using {len(train_queries)} queries")

    # Initialize LLM client
    if args.llm_endpoint:
        llm_client = OpenAI(base_url=args.llm_endpoint, api_key="EMPTY")
        print(f"Using vLLM endpoint: {args.llm_endpoint}")
    else:
        llm_client = OpenAI()  # Uses OPENAI_API_KEY
        print("Using OpenAI API (set OPENAI_API_KEY)")

    # Initialize miner
    print("\n[2/5] Initializing miner...")
    miner = IterativeHardNegativeMiner(
        base_model_name="sentence-transformers/all-MiniLM-L12-v2",
        llm_client=llm_client,
        llm_model=args.llm_model,
        corpus_chunks=corpus_chunks,
        output_dir="models/test-small-sample"
    )

    # Embed corpus
    print("\n[3/5] Embedding corpus...")
    miner.chunk_embeddings = miner.model.encode(
        [c['text'] for c in miner.corpus_chunks],
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Retrieve candidates (use fewer candidates for faster testing)
    print(f"\n[4/5] Retrieving top-{args.num_candidates} candidates...")
    candidates = miner.retrieve_candidates(train_queries, k=args.num_candidates)

    total_pairs = sum(len(cands) for cands in candidates)
    print(f"Total (query, chunk) pairs to judge: {total_pairs}")

    # LLM judging
    print(f"\n[5/5] LLM judging {total_pairs} pairs...")
    print("This will take ~1-2 minutes with vLLM or ~5-10 minutes with OpenAI API")
    scored_pairs = miner.llm_judge_relevance(candidates)

    # Analyze results
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)

    score_distribution = {1: 0, 2: 0, 3: 0, 4: 0}
    for pair in scored_pairs:
        score_distribution[pair['llm_score']] += 1

    print("\nLLM Score Distribution:")
    print(f"  Score 1 (Not relevant): {score_distribution[1]} ({score_distribution[1]/total_pairs*100:.1f}%)")
    print(f"  Score 2 (Slightly relevant): {score_distribution[2]} ({score_distribution[2]/total_pairs*100:.1f}%)")
    print(f"  Score 3 (Moderately relevant): {score_distribution[3]} ({score_distribution[3]/total_pairs*100:.1f}%)")
    print(f"  Score 4 (Highly relevant): {score_distribution[4]} ({score_distribution[4]/total_pairs*100:.1f}%)")

    # Mine hard examples
    hard_pos, hard_neg = miner.mine_hard_examples(scored_pairs)
    print(f"\nHard Examples Mined:")
    print(f"  Hard positives (LLM score >=3, model rank >10): {len(hard_pos)}")
    print(f"  Hard negatives (LLM score <=2, model rank <=5): {len(hard_neg)}")

    # Save results for inspection
    results = {
        "config": {
            "num_queries": args.num_queries,
            "num_candidates_per_query": args.num_candidates,
            "total_pairs_judged": total_pairs,
            "llm_model": args.llm_model
        },
        "score_distribution": score_distribution,
        "hard_examples": {
            "hard_positives": len(hard_pos),
            "hard_negatives": len(hard_neg)
        },
        "sample_judgments": []
    }

    # Save 10 sample judgments for inspection
    for i, pair in enumerate(scored_pairs[:10]):
        results["sample_judgments"].append({
            "query": pair['query'],
            "chunk_preview": pair['chunk']['text'][:200] + "...",
            "llm_score": pair['llm_score'],
            "model_rank": pair['model_rank'],
            "is_ground_truth": pair['chunk']['chunk_id'] in pair['ground_truth_chunk_ids']
        })

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_file}")
    print("\nInspect the file to verify:")
    print("  1. LLM scores make sense (relevant chunks get 3-4, irrelevant get 1-2)")
    print("  2. Hard examples are being mined (should have some of each)")
    print("  3. Sample judgments look correct")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("If results look good:")
    print("  1. Run full training: python iterative_miner.py --num-queries 1000 --llm-endpoint <endpoint>")
    print("  2. Evaluate: python evaluate.py")
    print("\nIf results look wrong:")
    print("  1. Check LLM prompt in iterative_miner.py:llm_judge_relevance()")
    print("  2. Adjust hard-negative mining thresholds in iterative_miner.py:mine_hard_examples()")
    print("  3. Re-run this test")


if __name__ == "__main__":
    main()
