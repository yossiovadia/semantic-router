"""
Robust Iterative Hard-Negative Mining Pipeline with Checkpointing

Features:
- Automatic checkpointing at every step
- Resume from any checkpoint
- Graceful error handling
- Progress tracking
- Disk-based intermediate results
"""

import os
import pickle
import json
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
from openai import OpenAI


class RobustIterativeHardNegativeMiner:
    """
    Production-ready iterative hard-negative mining with checkpointing.

    Saves state after every step. Can resume from any point if interrupted.
    """

    def __init__(
        self,
        base_model_name: str,
        llm_client: OpenAI,
        llm_model: str,
        corpus_chunks: List[Dict],
        output_dir: str = "models",
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda"
    ):
        self.base_model_name = base_model_name
        self.model = SentenceTransformer(base_model_name, device=device)
        self.llm = llm_client
        self.llm_model = llm_model
        self.corpus_chunks = corpus_chunks
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.chunk_embeddings = None

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def get_checkpoint_path(self, iteration: int, step: str) -> str:
        """Get path for checkpoint file."""
        return f"{self.checkpoint_dir}/iter_{iteration}_{step}.pkl"

    def save_checkpoint(self, iteration: int, step: str, data: Dict):
        """Save checkpoint to disk."""
        checkpoint_path = self.get_checkpoint_path(iteration, step)
        checkpoint = {
            'iteration': iteration,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, iteration: int, step: str) -> Optional[Dict]:
        """Load checkpoint from disk if exists."""
        checkpoint_path = self.get_checkpoint_path(iteration, step)
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
            return checkpoint['data']
        return None

    def checkpoint_exists(self, iteration: int, step: str) -> bool:
        """Check if checkpoint exists."""
        return os.path.exists(self.get_checkpoint_path(iteration, step))

    def run_iteration(self, queries: List[Dict], iteration: int = 0, force_restart: bool = False) -> SentenceTransformer:
        """
        Run single iteration with checkpointing at every step.

        Can resume from any checkpoint if interrupted.

        Args:
            queries: List of query dicts
            iteration: Iteration number
            force_restart: If True, ignore checkpoints and restart from scratch

        Returns:
            Updated model
        """
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")

        # Step 1: Embed corpus
        if not force_restart and self.checkpoint_exists(iteration, 'corpus_embeddings'):
            print("\n[Step 1/5] Loading corpus embeddings from checkpoint...")
            data = self.load_checkpoint(iteration, 'corpus_embeddings')
            self.chunk_embeddings = data['chunk_embeddings']
        else:
            print("\n[Step 1/5] Embedding corpus...")
            self.chunk_embeddings = self.model.encode(
                [c['text'] for c in self.corpus_chunks],
                show_progress_bar=True,
                convert_to_numpy=True
            )
            print(f"Embedded {len(self.corpus_chunks)} chunks")

            # Save checkpoint
            self.save_checkpoint(iteration, 'corpus_embeddings', {
                'chunk_embeddings': self.chunk_embeddings
            })

        # Step 2: Retrieve candidates
        if not force_restart and self.checkpoint_exists(iteration, 'candidates'):
            print("\n[Step 2/5] Loading candidates from checkpoint...")
            data = self.load_checkpoint(iteration, 'candidates')
            candidates = data['candidates']
        else:
            print("\n[Step 2/5] Retrieving candidates...")
            candidates = self.retrieve_candidates(queries, k=50)
            print(f"Retrieved candidates for {len(candidates)} queries")

            # Save checkpoint
            self.save_checkpoint(iteration, 'candidates', {
                'candidates': candidates
            })

        # Step 3: LLM judging (MOST EXPENSIVE - checkpoint aggressively)
        if not force_restart and self.checkpoint_exists(iteration, 'scored_pairs'):
            print("\n[Step 3/5] Loading LLM judgments from checkpoint...")
            data = self.load_checkpoint(iteration, 'scored_pairs')
            scored_pairs = data['scored_pairs']
        else:
            print("\n[Step 3/5] LLM judging relevance...")
            scored_pairs = self.llm_judge_relevance_robust(candidates, iteration)
            print(f"LLM judged {len(scored_pairs)} (query, chunk) pairs")

            # Save checkpoint
            self.save_checkpoint(iteration, 'scored_pairs', {
                'scored_pairs': scored_pairs
            })

        # Step 4: Mine hard examples
        if not force_restart and self.checkpoint_exists(iteration, 'hard_examples'):
            print("\n[Step 4/5] Loading hard examples from checkpoint...")
            data = self.load_checkpoint(iteration, 'hard_examples')
            hard_pos = data['hard_pos']
            hard_neg = data['hard_neg']
        else:
            print("\n[Step 4/5] Mining hard examples...")
            hard_pos, hard_neg = self.mine_hard_examples(scored_pairs)
            print(f"Mined {len(hard_pos)} hard positives, {len(hard_neg)} hard negatives")

            # Save checkpoint
            self.save_checkpoint(iteration, 'hard_examples', {
                'hard_pos': hard_pos,
                'hard_neg': hard_neg
            })

        if len(hard_pos) == 0 and len(hard_neg) == 0:
            print("WARNING: No hard examples found. Skipping fine-tuning.")
            return self.model

        # Step 5: Fine-tune
        if not force_restart and self.checkpoint_exists(iteration, 'fine_tuned_model'):
            print("\n[Step 5/5] Loading fine-tuned model from checkpoint...")
            model_path = f"{self.output_dir}/iteration_{iteration}"
            self.model = SentenceTransformer(model_path, device=self.device)
        else:
            print(f"\n[Step 5/5] Fine-tuning model...")
            self.fine_tune(hard_pos, hard_neg, iteration)

            # Save model
            model_path = f"{self.output_dir}/iteration_{iteration}"
            self.model.save(model_path)
            print(f"Model saved to: {model_path}")

            # Mark completion
            self.save_checkpoint(iteration, 'fine_tuned_model', {
                'model_path': model_path,
                'completed': True
            })

        print(f"\n✓ Iteration {iteration} complete!")
        return self.model

    def llm_judge_relevance_robust(self, candidates: List[List[Dict]], iteration: int) -> List[Dict]:
        """
        LLM judge with incremental checkpointing (saves every 100 judgments).

        Can resume from partial progress if interrupted.
        """
        # Check for partial checkpoint
        partial_checkpoint_path = f"{self.checkpoint_dir}/iter_{iteration}_llm_partial.pkl"

        if os.path.exists(partial_checkpoint_path):
            print(f"Found partial LLM checkpoint, resuming...")
            with open(partial_checkpoint_path, 'rb') as f:
                partial_data = pickle.load(f)
            scored_pairs = partial_data['scored_pairs']
            start_idx = partial_data['next_idx']
            print(f"Resuming from judgment {start_idx}/{len(candidates)}")
        else:
            scored_pairs = []
            start_idx = 0

        # Flatten candidates
        all_pairs = []
        for query_candidates in candidates:
            all_pairs.extend(query_candidates)

        total_pairs = len(all_pairs)
        print(f"Judging {total_pairs - start_idx} candidates...")

        # Process in batches with checkpointing
        checkpoint_interval = 100  # Save every 100 judgments

        for i, pair in enumerate(tqdm(all_pairs[start_idx:], desc="LLM judging"), start=start_idx):
            try:
                score = self._judge_single_pair(pair['query'], pair['chunk']['text'])

                scored_pairs.append({
                    'query': pair['query'],
                    'chunk': pair['chunk'],
                    'model_rank': pair['rank'],
                    'llm_score': score,
                    'ground_truth_chunk_ids': pair['ground_truth_chunk_ids']
                })

                # Periodic checkpoint
                if (i + 1) % checkpoint_interval == 0:
                    with open(partial_checkpoint_path, 'wb') as f:
                        pickle.dump({
                            'scored_pairs': scored_pairs,
                            'next_idx': i + 1,
                            'timestamp': datetime.now().isoformat()
                        }, f)
                    print(f"\n✓ Partial checkpoint saved: {i + 1}/{total_pairs}")

            except Exception as e:
                print(f"\nERROR judging pair {i}: {e}")
                # Save emergency checkpoint
                with open(partial_checkpoint_path, 'wb') as f:
                    pickle.dump({
                        'scored_pairs': scored_pairs,
                        'next_idx': i,
                        'timestamp': datetime.now().isoformat(),
                        'error': str(e)
                    }, f)
                print(f"Emergency checkpoint saved at index {i}")
                raise

        # Clean up partial checkpoint on completion
        if os.path.exists(partial_checkpoint_path):
            os.remove(partial_checkpoint_path)

        return scored_pairs

    def _judge_single_pair(self, query: str, chunk: str) -> int:
        """Judge single (query, chunk) pair."""
        prompt = f"""Rate how relevant this passage is to the question.

Question: {query}

Passage: {chunk}

Rate from 1-4:
1 = Not relevant at all
2 = Slightly relevant
3 = Moderately relevant
4 = Highly relevant

Output only the number (1, 2, 3, or 4)."""

        try:
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )

            score_text = response.choices[0].message.content.strip()

            # Extract number
            for char in score_text:
                if char.isdigit():
                    score = int(char)
                    if 1 <= score <= 4:
                        return score

            # Default to 2 if parsing fails
            print(f"Warning: Could not parse score '{score_text}', defaulting to 2")
            return 2

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return 2  # Default on error

    def retrieve_candidates(self, queries: List[Dict], k: int = 50) -> List[List[Dict]]:
        """Retrieve top-k candidates for each query."""
        query_embeddings = self.model.encode(
            [q['query'] for q in queries],
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Cosine similarity
        similarities = np.dot(query_embeddings, self.chunk_embeddings.T)

        # Get top-k for each query
        candidates = []
        for i, query in enumerate(queries):
            top_k_indices = np.argsort(similarities[i])[-k:][::-1]

            query_candidates = []
            for rank, chunk_idx in enumerate(top_k_indices):
                query_candidates.append({
                    'query': query['query'],
                    'chunk': self.corpus_chunks[chunk_idx],
                    'rank': rank + 1,
                    'similarity': float(similarities[i][chunk_idx]),
                    'ground_truth_chunk_ids': query['ground_truth_chunk_ids']
                })

            candidates.append(query_candidates)

        return candidates

    def mine_hard_examples(self, scored_pairs: List[Dict]) -> Tuple[List[Tuple], List[Tuple]]:
        """Mine hard positives and hard negatives."""
        hard_pos = []
        hard_neg = []

        for pair in scored_pairs:
            # Hard positive: LLM says relevant but model ranked low
            if pair['llm_score'] >= 3 and pair['model_rank'] > 10:
                hard_pos.append((pair['query'], pair['chunk']['text']))

            # Hard negative: LLM says not relevant but model ranked high
            elif pair['llm_score'] <= 2 and pair['model_rank'] <= 5:
                hard_neg.append((pair['query'], pair['chunk']['text']))

        return hard_pos, hard_neg

    def fine_tune(self, hard_pos: List[Tuple], hard_neg: List[Tuple], iteration: int):
        """
        Fine-tune model on hard examples using TripletLoss (per paper).

        Creates (query, positive, negative) triplets as per the paper's approach.

        ANTI-CATASTROPHIC-FORGETTING MEASURES:
        1. Very low learning rate (1e-8) - 50x lower than paper to preserve pre-trained knowledge
        2. Single epoch - less aggressive updates
        3. Larger margin (0.2) - only update for clear violations
        4. Weight decay (0.01) - L2 regularization to prevent overfitting
        5. Triplet subsampling - cap at 500 triplets per iteration to prevent overfitting
        """
        train_examples = []

        # Create query -> positive/negative mapping
        query_to_pos = {}
        query_to_neg = {}

        for query, pos_chunk in hard_pos:
            if query not in query_to_pos:
                query_to_pos[query] = []
            query_to_pos[query].append(pos_chunk)

        for query, neg_chunk in hard_neg:
            if query not in query_to_neg:
                query_to_neg[query] = []
            query_to_neg[query].append(neg_chunk)

        # Create triplets: (query, positive, negative)
        # Only for queries that have BOTH positives AND negatives
        import random
        random.seed(42 + iteration)  # Different seed per iteration for diversity

        for query in query_to_pos.keys():
            if query in query_to_neg:
                # Get all positives and negatives for this query
                positives = query_to_pos[query]
                negatives = query_to_neg[query]

                # Create triplets by pairing each positive with a random negative
                for pos in positives:
                    neg = random.choice(negatives)
                    train_examples.append(InputExample(texts=[query, pos, neg]))

        if len(train_examples) == 0:
            print("WARNING: No triplets formed (queries need both positives AND negatives)")
            return

        # ANTI-FORGETTING FIX #5: Cap triplets to prevent overfitting
        MAX_TRIPLETS = 500
        if len(train_examples) > MAX_TRIPLETS:
            print(f"Subsampling {len(train_examples)} triplets down to {MAX_TRIPLETS} to prevent overfitting")
            random.shuffle(train_examples)
            train_examples = train_examples[:MAX_TRIPLETS]

        print(f"Training on {len(train_examples)} triplets from {len(hard_pos)} pos, {len(hard_neg)} neg")

        # Paper uses batch_size=128 (16 per GPU × 8 GPUs), but we use CPU
        # Reduce to 16 for CPU training
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

        # ANTI-FORGETTING FIX #3: Larger margin (0.2 vs 0.1) - only update for clear violations
        train_loss = losses.TripletLoss(
            model=self.model,
            distance_metric=losses.TripletDistanceMetric.EUCLIDEAN,
            triplet_margin=0.2  # Increased from 0.1 to reduce unnecessary updates
        )

        # Calculate warmup steps: 10% of total steps (paper doesn't specify, this is standard)
        num_training_steps = len(train_dataloader)
        warmup_steps = max(1, int(0.1 * num_training_steps))

        print(f"Training steps: {num_training_steps}, Warmup steps: {warmup_steps}")

        # Force CPU to avoid OOM with vLLM on GPU
        old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        try:
            # ANTI-FORGETTING FIXES:
            # - lr=1e-8 (50x lower than paper's 5e-7) to preserve pre-trained knowledge
            # - epochs=1 (vs 2) to reduce overfitting
            # - weight_decay=0.01 for L2 regularization
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,  # REDUCED from 2 to prevent overfitting
                warmup_steps=warmup_steps,
                optimizer_params={
                    'lr': 1e-8,  # 50x lower than paper to prevent catastrophic forgetting
                },
                weight_decay=0.01,  # L2 regularization to stay close to pre-trained weights
                show_progress_bar=True,
                output_path=f"{self.output_dir}/iteration_{iteration}_checkpoint",
                use_amp=False
            )
        finally:
            # Restore CUDA visibility
            if old_cuda_visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)


def main():
    """Example usage with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Robust iterative hard-negative mining")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of queries per iteration")
    parser.add_argument("--num-iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--llm-endpoint", required=True, help="vLLM endpoint URL")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct", help="LLM model name")
    parser.add_argument("--output-dir", default="models/robust", help="Output directory")
    parser.add_argument("--checkpoint-dir", default="checkpoints/robust", help="Checkpoint directory")
    parser.add_argument("--start-iteration", type=int, default=0, help="Start from iteration N")
    parser.add_argument("--force-restart", action="store_true", help="Ignore checkpoints, restart from scratch")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    with open(f"{args.data_dir}/corpus_chunks.pkl", "rb") as f:
        corpus_chunks = pickle.load(f)

    with open(f"{args.data_dir}/train_queries.pkl", "rb") as f:
        all_queries = pickle.load(f)

    print(f"Loaded {len(corpus_chunks)} chunks, {len(all_queries)} total queries")
    print(f"Using {args.num_queries} queries per iteration")

    # Initialize miner
    llm_client = OpenAI(base_url=args.llm_endpoint, api_key="EMPTY")

    miner = RobustIterativeHardNegativeMiner(
        base_model_name="sentence-transformers/all-MiniLM-L12-v2",
        llm_client=llm_client,
        llm_model=args.llm_model,
        corpus_chunks=corpus_chunks,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        device="cpu"  # Use CPU to avoid GPU OOM with vLLM
    )

    # Run iterations
    for iteration in range(args.start_iteration, args.num_iterations):
        # Sample queries
        import random
        random.seed(42 + iteration)  # Different sample each iteration
        queries = random.sample(all_queries, min(args.num_queries, len(all_queries)))

        print(f"\n{'#'*60}")
        print(f"# Starting Iteration {iteration}/{args.num_iterations - 1}")
        print(f"{'#'*60}")

        try:
            miner.run_iteration(queries, iteration=iteration, force_restart=args.force_restart)
        except KeyboardInterrupt:
            print("\n\n✋ Training interrupted by user")
            print(f"You can resume from iteration {iteration} by running:")
            print(f"python robust_iterative_miner.py --start-iteration {iteration} [other args]")
            break
        except Exception as e:
            print(f"\n\n❌ ERROR in iteration {iteration}: {e}")
            print(f"Checkpoints saved. You can resume from iteration {iteration} by running:")
            print(f"python robust_iterative_miner.py --start-iteration {iteration} [other args]")
            raise

    print("\n" + "="*60)
    print("✓ Training complete!")
    print(f"Final model saved to: {args.output_dir}/iteration_{args.num_iterations - 1}")
    print("="*60)


if __name__ == "__main__":
    main()
