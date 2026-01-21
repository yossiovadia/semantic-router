"""
Evaluation script for embedding models.

Compares baseline vs specialized model using retrieval metrics:
- MRR@5 (Mean Reciprocal Rank)
- DCG@5 (Discounted Cumulative Gain)
- Precision@5
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm


def evaluate_retrieval(
    model: SentenceTransformer,
    test_queries: List[Dict],
    corpus_chunks: List[Dict],
    k: int = 5
) -> Dict[str, float]:
    """
    Evaluate retrieval quality with MRR@k, DCG@k, Precision@k.

    Args:
        model: SentenceTransformer model to evaluate
        test_queries: List of test queries with ground_truth_chunk_ids
        corpus_chunks: List of corpus chunks
        k: Top-k to evaluate (default: 5)

    Returns:
        Dict with metrics: MRR@k, DCG@k, Precision@k
    """
    print(f"Evaluating model: {model}")

    # Embed corpus
    print("Embedding corpus...")
    chunk_embeddings = model.encode(
        [c['text'] for c in corpus_chunks],
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Embed queries
    print("Embedding queries...")
    query_embeddings = model.encode(
        [q['query'] for q in test_queries],
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Calculate similarities (queries x chunks)
    print("Calculating similarities...")
    similarities = np.dot(query_embeddings, chunk_embeddings.T)

    # Metrics
    mrr_scores = []
    dcg_scores = []
    precision_scores = []
    recall_scores = []

    print(f"Evaluating on {len(test_queries)} queries...")
    for i, query in enumerate(tqdm(test_queries)):
        # Get rankings (indices sorted by similarity, descending)
        ranked_indices = np.argsort(similarities[i])[::-1]

        # Ground truth chunk IDs
        gt_chunk_ids = set(query['ground_truth_chunk_ids'])

        # Check top-k
        top_k_indices = ranked_indices[:k]
        top_k_chunk_ids = [corpus_chunks[idx]['chunk_id'] for idx in top_k_indices]

        # MRR@k: Reciprocal rank of first relevant result
        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(top_k_chunk_ids, start=1):
            if chunk_id in gt_chunk_ids:
                reciprocal_rank = 1.0 / rank
                break
        mrr_scores.append(reciprocal_rank)

        # DCG@k: Discounted Cumulative Gain
        dcg = 0.0
        for rank, chunk_id in enumerate(top_k_chunk_ids, start=1):
            relevance = 1.0 if chunk_id in gt_chunk_ids else 0.0
            dcg += relevance / np.log2(rank + 1)  # +1 because rank starts at 1
        dcg_scores.append(dcg)

        # Precision@k: Fraction of top-k that are relevant
        num_relevant = sum(1 for chunk_id in top_k_chunk_ids if chunk_id in gt_chunk_ids)
        precision = num_relevant / k
        precision_scores.append(precision)

        # Recall@k: Fraction of relevant docs found in top-k
        recall = num_relevant / len(gt_chunk_ids) if gt_chunk_ids else 0.0
        recall_scores.append(recall)

    return {
        f'MRR@{k}': np.mean(mrr_scores),
        f'DCG@{k}': np.mean(dcg_scores),
        f'Precision@{k}': np.mean(precision_scores),
        f'Recall@{k}': np.mean(recall_scores)
    }


def main():
    """Main evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate embedding models")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--baseline-model", default="sentence-transformers/all-MiniLM-L12-v2", help="Baseline model")
    parser.add_argument("--specialized-model", default="models/medical-specialized/final", help="Specialized model path")
    parser.add_argument("--k", type=int, default=5, help="Top-k for metrics")
    args = parser.parse_args()

    print("="*60)
    print("EVALUATION: Baseline vs Specialized Model")
    print("="*60)

    # Load data
    print("\nLoading data...")
    with open(f"{args.data_dir}/corpus_chunks.pkl", "rb") as f:
        corpus_chunks = pickle.load(f)

    with open(f"{args.data_dir}/test_queries.pkl", "rb") as f:
        test_queries = pickle.load(f)

    print(f"Loaded {len(corpus_chunks)} corpus chunks")
    print(f"Loaded {len(test_queries)} test queries")

    # Load models
    print("\nLoading models...")
    print(f"Baseline: {args.baseline_model}")
    baseline_model = SentenceTransformer(args.baseline_model)

    print(f"Specialized: {args.specialized_model}")
    try:
        specialized_model = SentenceTransformer(args.specialized_model)
    except Exception as e:
        print(f"Error loading specialized model: {e}")
        print("Make sure you've trained the model first with iterative_miner.py")
        return

    # Evaluate baseline
    print("\n" + "="*60)
    print("BASELINE MODEL EVALUATION")
    print("="*60)
    baseline_metrics = evaluate_retrieval(baseline_model, test_queries, corpus_chunks, k=args.k)

    # Evaluate specialized
    print("\n" + "="*60)
    print("SPECIALIZED MODEL EVALUATION")
    print("="*60)
    specialized_metrics = evaluate_retrieval(specialized_model, test_queries, corpus_chunks, k=args.k)

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\nBaseline ({args.baseline_model}):")
    for metric, value in baseline_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nSpecialized ({args.specialized_model}):")
    for metric, value in specialized_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nImprovement:")
    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        specialized_val = specialized_metrics[metric]

        if baseline_val > 0:
            improvement = ((specialized_val - baseline_val) / baseline_val) * 100
            print(f"  {metric}: {improvement:+.1f}%")
        else:
            print(f"  {metric}: N/A (baseline is 0)")

    # Save results
    results = {
        'baseline': {
            'model': args.baseline_model,
            'metrics': baseline_metrics
        },
        'specialized': {
            'model': args.specialized_model,
            'metrics': specialized_metrics
        }
    }

    results_path = "evaluation_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {results_path}")

    # Success criteria check
    print("\n" + "="*60)
    print("SUCCESS CRITERIA CHECK")
    print("="*60)

    mrr_key = f'MRR@{args.k}'
    dcg_key = f'DCG@{args.k}'

    mrr_improvement = ((specialized_metrics[mrr_key] - baseline_metrics[mrr_key]) / baseline_metrics[mrr_key]) * 100 if baseline_metrics[mrr_key] > 0 else 0
    dcg_improvement = ((specialized_metrics[dcg_key] - baseline_metrics[dcg_key]) / baseline_metrics[dcg_key]) * 100 if baseline_metrics[dcg_key] > 0 else 0

    target_improvement = 15.0  # 15% target

    print(f"Target: +{target_improvement}% improvement in MRR@{args.k} and DCG@{args.k}")
    print(f"Achieved:")
    print(f"  MRR@{args.k}: {mrr_improvement:+.1f}% {'✓' if mrr_improvement >= target_improvement else '✗'}")
    print(f"  DCG@{args.k}: {dcg_improvement:+.1f}% {'✓' if dcg_improvement >= target_improvement else '✗'}")

    if mrr_improvement >= target_improvement and dcg_improvement >= target_improvement:
        print(f"\n🎉 SUCCESS! Specialized model meets success criteria!")
    else:
        print(f"\n⚠️  Specialized model did not meet success criteria.")
        print(f"Consider running more iterations or tuning hyperparameters.")


if __name__ == "__main__":
    main()
