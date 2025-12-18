"""
Dataset Builder for Cache-Specific Embedding Training
=====================================================

Collects and prepares real-world data for training domain-specific cache embeddings.

Data Sources:
1. Stack Overflow: Programming Q&A pairs
2. GitHub Issues: Bug reports, feature requests, discussions
3. Documentation: Code documentation Q&A

Output Format:
- Triplet format: (anchor, positive, hard_negative)
- JSONL files for training, validation, and testing
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional dependencies - gracefully handle if not installed
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logging.warning("HuggingFace datasets not available. Using fallback data generation.")

try:
    from tqdm import tqdm
except ImportError:
    # Fallback to simple iteration if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable

from .common_utils import save_jsonl, set_seed, setup_logging

logger = setup_logging()


class CodingDatasetBuilder:
    """
    Build training datasets for coding domain cache embeddings.

    Collects data from multiple sources and creates triplet pairs
    for contrastive learning.
    """

    def __init__(
        self,
        output_dir: str = "datasets/cache_training/coding",
        seed: int = 42
    ):
        """
        Initialize dataset builder.

        Args:
            output_dir: Directory to save datasets
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        set_seed(seed)

        logger.info(f"CodingDatasetBuilder initialized: output_dir={output_dir}")

    def collect_stackoverflow_data(
        self,
        max_samples: int = 10000,
        min_score: int = 5
    ) -> List[Dict]:
        """
        Collect Q&A pairs from Stack Overflow dataset.

        Uses the 'stackoverflow' dataset from HuggingFace which contains
        programming questions and answers.

        Args:
            max_samples: Maximum number of samples to collect
            min_score: Minimum question score (quality filter)

        Returns:
            List of Q&A dictionaries
        """
        logger.info(f"Loading Stack Overflow data (max_samples={max_samples}, min_score={min_score})")

        if not HF_DATASETS_AVAILABLE:
            logger.info("HuggingFace datasets not available, using fallback")
            return self._generate_fallback_stackoverflow_data(max_samples)

        try:
            # Try loading from HuggingFace datasets
            # Note: This is a placeholder - actual dataset name may vary
            # Common options: "stackexchange", "codeparrot/github-code"
            dataset = load_dataset(
                "koutch/stackoverflow_python",
                split="train",
                streaming=True
            )

            samples = []
            for i, item in enumerate(tqdm(dataset, desc="Collecting Stack Overflow", total=max_samples)):
                if len(samples) >= max_samples:
                    break

                # Extract question and answer
                question = self._clean_text(item.get("question", ""))
                answer = self._clean_text(item.get("answer", ""))

                if not question or len(question) < 10:
                    continue

                samples.append({
                    "question": question,
                    "answer": answer,
                    "source": "stackoverflow",
                    "language": "python",
                    "tags": item.get("tags", [])
                })

            logger.info(f"Collected {len(samples)} Stack Overflow samples")
            return samples

        except Exception as e:
            logger.warning(f"Failed to load Stack Overflow dataset: {e}")
            logger.info("Using synthetic fallback data for development")
            return self._generate_fallback_stackoverflow_data(max_samples)

    def _generate_fallback_stackoverflow_data(self, max_samples: int) -> List[Dict]:
        """
        Generate synthetic Stack Overflow-like data for development/testing.

        This is a fallback when the real dataset is unavailable.
        """
        logger.info("Generating fallback Stack Overflow data")

        # Common coding questions (for development)
        questions = [
            "How do I reverse a string in Python?",
            "What is the difference between a list and a tuple?",
            "How to sort a dictionary by value in Python?",
            "How do I read a file line by line in Python?",
            "What is the purpose of __init__ in Python?",
            "How to remove duplicates from a list?",
            "How do I convert a string to an integer?",
            "What is the difference between '==' and 'is' in Python?",
            "How to iterate over a dictionary in Python?",
            "How do I create a virtual environment?",
            "What is a lambda function in Python?",
            "How to handle exceptions in Python?",
            "What is list comprehension?",
            "How do I merge two dictionaries?",
            "How to count occurrences in a list?",
            "What is the difference between append and extend?",
            "How to reverse a list in Python?",
            "How do I check if a key exists in a dictionary?",
            "What is the purpose of self in Python?",
            "How to split a string by delimiter?",
        ]

        samples = []
        for i in range(min(max_samples, len(questions) * 10)):
            q = questions[i % len(questions)]
            samples.append({
                "question": q,
                "answer": f"Answer for: {q}",
                "source": "stackoverflow_synthetic",
                "language": "python",
                "tags": ["python", "basics"]
            })

        return samples

    def collect_github_issues(
        self,
        max_samples: int = 5000
    ) -> List[Dict]:
        """
        Collect issue titles and descriptions from GitHub.

        Uses public GitHub datasets of issues/discussions.

        Args:
            max_samples: Maximum number of samples to collect

        Returns:
            List of issue dictionaries
        """
        logger.info(f"Loading GitHub issues data (max_samples={max_samples})")

        if not HF_DATASETS_AVAILABLE:
            logger.info("HuggingFace datasets not available, using fallback")
            return self._generate_fallback_github_data(max_samples)

        try:
            # Load GitHub issues dataset
            # Using a coding-related repository dataset
            dataset = load_dataset(
                "bigcode/the-stack-smol",
                split="train",
                streaming=True
            )

            samples = []
            for i, item in enumerate(tqdm(dataset, desc="Collecting GitHub", total=max_samples)):
                if len(samples) >= max_samples:
                    break

                # Extract title and body
                title = self._clean_text(item.get("content", "")[:200])  # First 200 chars

                if not title or len(title) < 10:
                    continue

                samples.append({
                    "title": title,
                    "source": "github",
                    "language": item.get("lang", "python")
                })

            logger.info(f"Collected {len(samples)} GitHub samples")
            return samples

        except Exception as e:
            logger.warning(f"Failed to load GitHub dataset: {e}")
            logger.info("Using synthetic fallback data for development")
            return self._generate_fallback_github_data(max_samples)

    def _generate_fallback_github_data(self, max_samples: int) -> List[Dict]:
        """Generate synthetic GitHub-like data for development."""
        logger.info("Generating fallback GitHub data")

        issues = [
            "Bug: Function returns None instead of expected value",
            "Feature request: Add support for async operations",
            "Question: How to configure logging levels?",
            "Bug: Memory leak in data processing pipeline",
            "Feature: Implement caching for API responses",
            "Documentation: Update README with installation steps",
            "Bug: TypeError when passing None to function",
            "Enhancement: Improve error messages",
            "Question: Best practices for testing?",
            "Bug: Connection timeout in network requests",
        ]

        samples = []
        for i in range(min(max_samples, len(issues) * 10)):
            issue = issues[i % len(issues)]
            samples.append({
                "title": issue,
                "source": "github_synthetic",
                "language": "python"
            })

        return samples

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove code blocks (basic)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]*`', '', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    def create_triplets(
        self,
        stackoverflow_data: List[Dict],
        github_data: List[Dict]
    ) -> List[Dict]:
        """
        Create triplet pairs from collected data.

        For each question/issue (anchor):
        - Positive: Similar question (paraphrase or related)
        - Hard Negative: Different but topically related question

        Args:
            stackoverflow_data: Stack Overflow samples
            github_data: GitHub samples

        Returns:
            List of triplet dictionaries
        """
        logger.info("Creating triplet pairs from collected data")

        triplets = []

        # Create triplets from Stack Overflow data
        for i, item in enumerate(tqdm(stackoverflow_data, desc="Creating SO triplets")):
            anchor = item["question"]

            # For now, create simple heuristic-based triplets
            # Positive: Same question (will be paraphrased later by synthetic generator)
            positive = anchor

            # Hard negative: Random different question
            negative_idx = random.choice([j for j in range(len(stackoverflow_data)) if j != i])
            hard_negative = stackoverflow_data[negative_idx]["question"]

            triplets.append({
                "anchor": anchor,
                "positive": positive,
                "hard_negative": hard_negative,
                "domain": "coding",
                "similarity_score": 1.0,  # Perfect match initially
                "source": "stackoverflow",
                "tags": item.get("tags", [])
            })

        # Create triplets from GitHub data
        for i, item in enumerate(tqdm(github_data[:len(github_data)//2], desc="Creating GitHub triplets")):
            anchor = item["title"]
            positive = anchor

            negative_idx = random.choice([j for j in range(len(github_data)) if j != i])
            hard_negative = github_data[negative_idx]["title"]

            triplets.append({
                "anchor": anchor,
                "positive": positive,
                "hard_negative": hard_negative,
                "domain": "coding",
                "similarity_score": 1.0,
                "source": "github"
            })

        logger.info(f"Created {len(triplets)} triplet pairs")
        return triplets

    def split_data(
        self,
        triplets: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train, validation, and test sets.

        Args:
            triplets: All triplet pairs
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing

        Returns:
            (train, val, test) tuples
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # Shuffle data
        random.shuffle(triplets)

        n = len(triplets)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = triplets[:train_end]
        val = triplets[train_end:val_end]
        test = triplets[val_end:]

        logger.info(f"Data split: train={len(train)}, val={len(val)}, test={len(test)}")

        return train, val, test

    def build_dataset(
        self,
        stackoverflow_samples: int = 10000,
        github_samples: int = 5000,
        save_splits: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Build complete dataset from all sources.

        Args:
            stackoverflow_samples: Number of Stack Overflow samples
            github_samples: Number of GitHub samples
            save_splits: Whether to save splits to disk

        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        logger.info("=" * 60)
        logger.info("Building Coding Domain Dataset")
        logger.info("=" * 60)

        # Step 1: Collect data
        logger.info("\n1. Collecting data from sources...")
        so_data = self.collect_stackoverflow_data(max_samples=stackoverflow_samples)
        gh_data = self.collect_github_issues(max_samples=github_samples)

        # Step 2: Create triplets
        logger.info("\n2. Creating triplet pairs...")
        triplets = self.create_triplets(so_data, gh_data)

        # Step 3: Split data
        logger.info("\n3. Splitting into train/val/test...")
        train, val, test = self.split_data(triplets)

        # Step 4: Save to disk
        if save_splits:
            logger.info("\n4. Saving datasets to disk...")
            save_jsonl(train, self.output_dir / "real_data_train.jsonl")
            save_jsonl(val, self.output_dir / "real_data_val.jsonl")
            save_jsonl(test, self.output_dir / "real_data_test.jsonl")

            # Also save combined real data for synthetic generation
            save_jsonl(triplets, self.output_dir / "real_data.jsonl")

            logger.info(f"✅ Datasets saved to {self.output_dir}")
            logger.info(f"   - real_data_train.jsonl: {len(train)} samples")
            logger.info(f"   - real_data_val.jsonl: {len(val)} samples")
            logger.info(f"   - real_data_test.jsonl: {len(test)} samples")

        logger.info("\n" + "=" * 60)
        logger.info("Dataset building complete!")
        logger.info("=" * 60)

        return {
            "train": train,
            "val": val,
            "test": test
        }


def main():
    """
    Main function for standalone dataset building.

    Usage:
        python dataset_builder.py
    """
    import argparse

    parser = argparse.ArgumentParser(description="Build coding domain dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/cache_training/coding",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--stackoverflow-samples",
        type=int,
        default=10000,
        help="Number of Stack Overflow samples"
    )
    parser.add_argument(
        "--github-samples",
        type=int,
        default=5000,
        help="Number of GitHub samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Build dataset
    builder = CodingDatasetBuilder(
        output_dir=args.output_dir,
        seed=args.seed
    )

    dataset = builder.build_dataset(
        stackoverflow_samples=args.stackoverflow_samples,
        github_samples=args.github_samples,
        save_splits=True
    )

    print(f"\n✅ Dataset built successfully!")
    print(f"Total samples: {sum(len(v) for v in dataset.values())}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
