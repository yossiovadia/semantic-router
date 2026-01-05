#!/usr/bin/env python3
"""
Cache Embedding Model Evaluation Script

Evaluates domain-specific cache embedding models against baseline models
using cache-specific metrics: Precision@K, Recall@K, MRR, and cache hit rate.

Based on arXiv:2504.02268v1 evaluation methodology.

Usage:
    # Evaluate LoRA model vs baseline
    python -m src.training.cache_embeddings.evaluate_model \
        --model models/programming-cache-lora \
        --baseline sentence-transformers/all-MiniLM-L12-v2 \
        --test-data data/cache_embeddings/programming/real_data_test.jsonl \
        --output results/programming-evaluation.json

    # Quick evaluation (top-3 only)
    python -m src.training.cache_embeddings.evaluate_model \
        --model models/programming-cache-lora \
        --baseline sentence-transformers/all-MiniLM-L12-v2 \
        --test-data data/cache_embeddings/programming/real_data_test.jsonl \
        --k-values 1,3,5

    # No baseline comparison
    python -m src.training.cache_embeddings.evaluate_model \
        --model models/programming-cache-lora \
        --test-data data/cache_embeddings/programming/real_data_test.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .common_utils import (
    calculate_cache_metrics,
    load_jsonl,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between vectors using numpy.

    Args:
        a: Single embedding or batch of embeddings (shape: [D] or [N, D])
        b: Single embedding or batch of embeddings (shape: [D] or [M, D])

    Returns:
        Cosine similarity scores (scalar or array)
    """
    # Ensure 2D
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity
    similarities = np.dot(a_norm, b_norm.T)

    # If single query, return 1D array
    if similarities.shape[0] == 1:
        return similarities[0]

    return similarities


class CacheEvaluator:
    """Evaluator for cache embedding models."""

    def __init__(
        self,
        model_path: str,
        baseline_path: Optional[str] = None,
        device: str = "cpu",
        is_lora: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to fine-tuned model (LoRA or full)
            baseline_path: Path to baseline model for comparison
            device: Device to run on (cpu/cuda)
            is_lora: Whether model_path is a LoRA adapter
        """
        self.device = device
        logger.info(f"Loading model from: {model_path}")

        # Load fine-tuned model
        if is_lora:
            # Load base model and LoRA adapter
            base_model_path = self._get_base_model_path(model_path)
            logger.info(f"Loading base model: {base_model_path}")
            base_model = SentenceTransformer(base_model_path, device=device)

            # Load LoRA adapter
            logger.info(f"Loading LoRA adapter from: {model_path}")
            self.model = PeftModel.from_pretrained(
                base_model[0].auto_model, model_path
            )
            self.model.eval()

            # Wrap back into SentenceTransformer
            base_model[0].auto_model = self.model
            self.model = base_model
        else:
            self.model = SentenceTransformer(model_path, device=device)

        logger.info("Fine-tuned model loaded successfully")

        # Load baseline model if provided
        self.baseline = None
        if baseline_path:
            logger.info(f"Loading baseline model: {baseline_path}")
            self.baseline = SentenceTransformer(baseline_path, device=device)
            logger.info("Baseline model loaded successfully")

    def _get_base_model_path(self, lora_path: str) -> str:
        """Extract base model path from LoRA adapter config."""
        config_path = Path(lora_path) / "adapter_config.json"
        with open(config_path) as f:
            config = json.load(f)
        return config["base_model_name_or_path"]

    def encode(self, texts: List[str], model: Optional[object] = None) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            model: Model to use (defaults to self.model)

        Returns:
            Embeddings as numpy array
        """
        if model is None:
            model = self.model

        with torch.no_grad():
            embeddings = model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
        return embeddings

    def evaluate_cache_retrieval(
        self,
        test_data: List[Dict],
        k_values: List[int] = [1, 3, 5, 10],
        use_baseline: bool = False,
    ) -> Dict:
        """
        Evaluate cache retrieval performance.

        For each test query (anchor), we:
        1. Build a cache of all positive examples
        2. Compute similarity between query and all cache entries
        3. Check if the correct positive is in top-K results

        Args:
            test_data: List of test triplets
            k_values: K values for Precision@K and Recall@K
            use_baseline: Whether to use baseline model

        Returns:
            Dictionary of evaluation metrics
        """
        model = self.baseline if use_baseline else self.model
        model_name = "baseline" if use_baseline else "fine-tuned"

        logger.info(f"Evaluating {model_name} model on {len(test_data)} samples")

        # Extract queries and cache entries
        queries = [item["anchor"] for item in test_data]
        positives = [item["positive"] for item in test_data]
        negatives = [
            item.get("negative") or item.get("hard_negative") for item in test_data
        ]

        # Build cache: all positives + all negatives
        cache_entries = list(set(positives + negatives))
        logger.info(f"Cache size: {len(cache_entries)} entries")

        # Encode
        logger.info("Encoding queries...")
        query_embeddings = self.encode(queries, model)

        logger.info("Encoding cache entries...")
        cache_embeddings = self.encode(cache_entries, model)

        # Create mapping from cache entry to index
        cache_to_idx = {entry: idx for idx, entry in enumerate(cache_entries)}

        # Evaluate
        metrics = {
            "total_queries": len(queries),
            "cache_size": len(cache_entries),
        }

        for k in k_values:
            hits = 0
            reciprocal_ranks = []

            for query_idx, (query_emb, positive) in enumerate(
                zip(query_embeddings, positives)
            ):
                # Compute similarities
                similarities = cosine_similarity_numpy(query_emb, cache_embeddings)

                # Get top-K indices
                top_k_indices = np.argsort(similarities)[::-1][:k]
                top_k_entries = [cache_entries[idx] for idx in top_k_indices]

                # Check if positive is in top-K
                if positive in top_k_entries:
                    hits += 1

                    # Calculate reciprocal rank
                    rank = top_k_entries.index(positive) + 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)

            # Calculate metrics
            precision_at_k = hits / len(queries)
            mrr = np.mean(reciprocal_ranks)

            metrics[f"precision@{k}"] = precision_at_k
            metrics[f"mrr@{k}"] = mrr
            metrics[f"cache_hits@{k}"] = hits

            logger.info(
                f"{model_name} - P@{k}: {precision_at_k:.4f}, MRR@{k}: {mrr:.4f}"
            )

        return metrics

    def evaluate_embedding_quality(
        self, test_data: List[Dict], use_baseline: bool = False
    ) -> Dict:
        """
        Evaluate embedding quality using cosine similarity.

        Measures:
        - Average similarity between anchor and positive
        - Average similarity between anchor and negative
        - Margin (positive_sim - negative_sim)

        Args:
            test_data: List of test triplets
            use_baseline: Whether to use baseline model

        Returns:
            Dictionary of quality metrics
        """
        model = self.baseline if use_baseline else self.model
        model_name = "baseline" if use_baseline else "fine-tuned"

        logger.info(f"Evaluating embedding quality for {model_name} model")

        anchors = [item["anchor"] for item in test_data]
        positives = [item["positive"] for item in test_data]
        negatives = [
            item.get("negative") or item.get("hard_negative") for item in test_data
        ]

        # Encode
        anchor_embeddings = self.encode(anchors, model)
        positive_embeddings = self.encode(positives, model)
        negative_embeddings = self.encode(negatives, model)

        # Calculate similarities
        positive_sims = []
        negative_sims = []

        for anchor_emb, pos_emb, neg_emb in zip(
            anchor_embeddings, positive_embeddings, negative_embeddings
        ):
            pos_sim = cosine_similarity_numpy(anchor_emb, pos_emb)
            neg_sim = cosine_similarity_numpy(anchor_emb, neg_emb)

            # Handle scalar or array results
            if isinstance(pos_sim, np.ndarray):
                pos_sim = pos_sim.item()
            if isinstance(neg_sim, np.ndarray):
                neg_sim = neg_sim.item()

            positive_sims.append(pos_sim)
            negative_sims.append(neg_sim)

        # Calculate metrics
        avg_positive_sim = np.mean(positive_sims)
        avg_negative_sim = np.mean(negative_sims)
        margin = avg_positive_sim - avg_negative_sim

        metrics = {
            "avg_positive_similarity": float(avg_positive_sim),
            "avg_negative_similarity": float(avg_negative_sim),
            "margin": float(margin),
            "std_positive_similarity": float(np.std(positive_sims)),
            "std_negative_similarity": float(np.std(negative_sims)),
        }

        logger.info(f"{model_name} - Avg Positive Sim: {avg_positive_sim:.4f}")
        logger.info(f"{model_name} - Avg Negative Sim: {avg_negative_sim:.4f}")
        logger.info(f"{model_name} - Margin: {margin:.4f}")

        return metrics

    def compare_models(self, test_data: List[Dict], k_values: List[int]) -> Dict:
        """
        Compare fine-tuned model against baseline.

        Args:
            test_data: List of test triplets
            k_values: K values for evaluation

        Returns:
            Dictionary with comparison results
        """
        if self.baseline is None:
            raise ValueError("Baseline model not provided")

        logger.info(
            "=" * 80 + "\n" + "Comparing Fine-tuned vs Baseline" + "\n" + "=" * 80
        )

        # Evaluate both models
        finetuned_cache = self.evaluate_cache_retrieval(
            test_data, k_values, use_baseline=False
        )
        baseline_cache = self.evaluate_cache_retrieval(
            test_data, k_values, use_baseline=True
        )

        finetuned_quality = self.evaluate_embedding_quality(
            test_data, use_baseline=False
        )
        baseline_quality = self.evaluate_embedding_quality(test_data, use_baseline=True)

        # Calculate improvements
        improvements = {}
        for k in k_values:
            ft_precision = finetuned_cache[f"precision@{k}"]
            bl_precision = baseline_cache[f"precision@{k}"]

            if bl_precision > 0:
                improvement = ((ft_precision - bl_precision) / bl_precision) * 100
            else:
                improvement = 0.0

            improvements[f"precision@{k}_improvement_pct"] = improvement

        # Margin improvement
        ft_margin = finetuned_quality["margin"]
        bl_margin = baseline_quality["margin"]
        if bl_margin > 0:
            margin_improvement = ((ft_margin - bl_margin) / bl_margin) * 100
        else:
            margin_improvement = 0.0

        results = {
            "fine_tuned": {
                "cache_retrieval": finetuned_cache,
                "embedding_quality": finetuned_quality,
            },
            "baseline": {
                "cache_retrieval": baseline_cache,
                "embedding_quality": baseline_quality,
            },
            "improvements": {
                **improvements,
                "margin_improvement_pct": margin_improvement,
            },
        }

        # Print comparison summary
        self._print_comparison_summary(results, k_values)

        return results

    def _print_comparison_summary(self, results: Dict, k_values: List[int]):
        """Print human-readable comparison summary."""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)

        logger.info("\n📊 Cache Retrieval Performance:")
        logger.info("-" * 80)
        logger.info(
            f"{'Metric':<20} {'Fine-tuned':<15} {'Baseline':<15} {'Improvement':<15}"
        )
        logger.info("-" * 80)

        for k in k_values:
            ft_p = results["fine_tuned"]["cache_retrieval"][f"precision@{k}"]
            bl_p = results["baseline"]["cache_retrieval"][f"precision@{k}"]
            imp = results["improvements"][f"precision@{k}_improvement_pct"]

            logger.info(
                f"Precision@{k:<2}        {ft_p:<15.4f} {bl_p:<15.4f} {imp:>+14.2f}%"
            )

        logger.info("\n📈 Embedding Quality:")
        logger.info("-" * 80)

        ft_margin = results["fine_tuned"]["embedding_quality"]["margin"]
        bl_margin = results["baseline"]["embedding_quality"]["margin"]
        margin_imp = results["improvements"]["margin_improvement_pct"]

        logger.info(f"{'Metric':<20} {'Fine-tuned':<15} {'Baseline':<15} {'Improvement'}")
        logger.info("-" * 80)
        logger.info(
            f"Margin              {ft_margin:<15.4f} {bl_margin:<15.4f} {margin_imp:>+14.2f}%"
        )

        # Success indicator
        logger.info("\n" + "=" * 80)
        best_improvement = max(
            results["improvements"][f"precision@{k}_improvement_pct"] for k in k_values
        )
        if best_improvement >= 10.0:
            logger.info("✅ SUCCESS: Achieved >10% precision improvement!")
        elif best_improvement > 0:
            logger.info(
                f"⚠️  PARTIAL: {best_improvement:.2f}% improvement (target: >10%)"
            )
        else:
            logger.info("❌ FAILED: No improvement over baseline")
        logger.info("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cache embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to fine-tuned model (LoRA adapter or full model)",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        help="Path to baseline model for comparison",
    )
    parser.add_argument(
        "--test-data",
        "-t",
        required=True,
        help="Path to test data (JSONL with triplets)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save evaluation results (JSON)",
    )
    parser.add_argument(
        "--k-values",
        default="1,3,5,10",
        help="Comma-separated K values for Precision@K (default: 1,3,5,10)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Model is a full fine-tuned model, not LoRA",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Setup
    setup_logging()
    set_seed(args.seed)

    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    logger.info(f"Evaluating with K values: {k_values}")

    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    test_data = load_jsonl(args.test_data)
    logger.info(f"Loaded {len(test_data)} test samples")

    # Initialize evaluator
    evaluator = CacheEvaluator(
        model_path=args.model,
        baseline_path=args.baseline,
        device=args.device,
        is_lora=not args.no_lora,
    )

    # Run evaluation
    if args.baseline:
        results = evaluator.compare_models(test_data, k_values)
    else:
        # Evaluate only fine-tuned model
        logger.info("Evaluating fine-tuned model only (no baseline comparison)")
        cache_metrics = evaluator.evaluate_cache_retrieval(test_data, k_values)
        quality_metrics = evaluator.evaluate_embedding_quality(test_data)

        results = {
            "fine_tuned": {
                "cache_retrieval": cache_metrics,
                "embedding_quality": quality_metrics,
            }
        }

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
