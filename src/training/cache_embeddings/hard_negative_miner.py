#!/usr/bin/env python3
"""
Hard Negative Mining for Cache Embeddings

Mines semantically similar but incorrect negatives to make training more challenging.
Uses embedding similarity to find hard negatives within the same domain/category.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """Mine hard negatives using embedding similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        device: str = "cpu"
    ):
        """Initialize miner with embedding model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info("Model loaded successfully")

    def mine_hard_negatives(
        self,
        data: List[Dict],
        num_negatives: int = 5,
        strategy: str = "similarity",
        min_similarity: float = 0.3,
        max_similarity: float = 0.7
    ) -> List[Dict]:
        """
        Mine hard negatives for each sample.

        Strategy options:
        - "similarity": Find negatives with medium similarity (not too easy, not duplicate)
        - "same_category": Negatives from same category/domain
        - "mixed": Combination of both

        Args:
            data: List of samples with anchor/positive
            num_negatives: Number of hard negatives per sample
            strategy: Mining strategy
            min_similarity: Minimum similarity threshold
            max_similarity: Maximum similarity threshold (avoid duplicates)

        Returns:
            Augmented data with hard negatives
        """
        logger.info(f"Mining hard negatives for {len(data)} samples")
        logger.info(f"Strategy: {strategy}, Negatives per sample: {num_negatives}")

        # Extract all unique texts and categories
        all_samples = []
        for item in data:
            all_samples.append({
                "text": item["positive"],
                "category": item.get("category", "unknown"),
                "original_item": item
            })

        # Encode all texts
        logger.info("Encoding all samples...")
        texts = [s["text"] for s in all_samples]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # Build category index
        category_index = {}
        for idx, sample in enumerate(all_samples):
            category = sample["category"]
            if category not in category_index:
                category_index[category] = []
            category_index[category].append(idx)

        logger.info(f"Categories found: {list(category_index.keys())}")

        # Mine hard negatives
        augmented_data = []
        for idx, item in enumerate(tqdm(data, desc="Mining hard negatives")):
            anchor_emb = embeddings[idx]
            category = item.get("category", "unknown")

            # Compute similarities to all other samples
            similarities = np.dot(embeddings, anchor_emb)

            # Find candidates based on strategy
            if strategy == "same_category":
                # Only consider samples from same category
                candidate_indices = [i for i in category_index[category] if i != idx]
            elif strategy == "similarity":
                # Find samples in the similarity sweet spot
                candidate_indices = []
                for i, sim in enumerate(similarities):
                    if i != idx and min_similarity <= sim <= max_similarity:
                        candidate_indices.append(i)
            elif strategy == "mixed":
                # Prefer same category, but use similarity as tiebreaker
                same_category = [i for i in category_index[category] if i != idx]
                # Filter by similarity
                candidate_indices = []
                for i in same_category:
                    sim = similarities[i]
                    if min_similarity <= sim <= max_similarity:
                        candidate_indices.append(i)
                # If not enough, relax to just same category
                if len(candidate_indices) < num_negatives:
                    candidate_indices = same_category
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Sample hard negatives
            if len(candidate_indices) >= num_negatives:
                # Sort by similarity (descending) and pick top ones
                if strategy in ["similarity", "mixed"]:
                    candidate_sims = [(i, similarities[i]) for i in candidate_indices]
                    candidate_sims.sort(key=lambda x: x[1], reverse=True)
                    selected_indices = [i for i, _ in candidate_sims[:num_negatives]]
                else:
                    selected_indices = random.sample(candidate_indices, num_negatives)
            else:
                # Not enough candidates, use all available
                selected_indices = candidate_indices

            # Create augmented samples
            hard_negatives = [texts[i] for i in selected_indices]

            # Create one sample per hard negative
            for hard_neg in hard_negatives:
                augmented_item = {
                    "anchor": item["anchor"],
                    "positive": item["positive"],
                    "hard_negative": hard_neg,
                    "domain": item.get("domain", "unknown"),
                    "category": category,
                    "source": "hard_negative_mined"
                }
                augmented_data.append(augmented_item)

        logger.info(f"Generated {len(augmented_data)} samples with hard negatives")
        return augmented_data


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives for cache embeddings")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input JSONL file with anchor/positive pairs"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSONL file with hard negatives"
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L12-v2",
        help="Embedding model for similarity computation"
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=5,
        help="Number of hard negatives per sample (default: 5)"
    )
    parser.add_argument(
        "--strategy",
        choices=["similarity", "same_category", "mixed"],
        default="mixed",
        help="Hard negative mining strategy (default: mixed)"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.4,
        help="Minimum similarity threshold (default: 0.4, optimized for hard negatives)"
    )
    parser.add_argument(
        "--max-similarity",
        type=float,
        default=0.75,
        help="Maximum similarity threshold (default: 0.75, optimized for hard negatives)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load input data
    logger.info(f"Loading data from: {args.input}")
    data = []
    with open(args.input) as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} samples")

    # Mine hard negatives
    miner = HardNegativeMiner(model_name=args.model)
    augmented_data = miner.mine_hard_negatives(
        data=data,
        num_negatives=args.num_negatives,
        strategy=args.strategy,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity
    )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for item in augmented_data:
            f.write(json.dumps(item) + '\n')

    logger.info(f"Saved {len(augmented_data)} samples to: {args.output}")

    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("Hard Negative Mining Complete")
    logger.info("=" * 80)
    logger.info(f"Original samples: {len(data)}")
    logger.info(f"Augmented samples: {len(augmented_data)}")
    logger.info(f"Augmentation factor: {len(augmented_data) / len(data):.1f}x")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Similarity range: {args.min_similarity:.2f} - {args.max_similarity:.2f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
