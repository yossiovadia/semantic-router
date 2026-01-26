"""
Iterative Hard-Negative Mining Pipeline

Implements the paper's method: iterative hard-negative mining with LLM-as-judge
"""

import os
import pickle
from typing import List, Dict, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
from openai import OpenAI


class IterativeHardNegativeMiner:
    """
    Iterative hard-negative mining for domain-aware embeddings.

    Args:
        base_model_name: Name of base SentenceTransformer model
        llm_client: OpenAI-compatible client for LLM-as-judge
        llm_model: Model name for LLM judging
        corpus_chunks: List of corpus chunks
        output_dir: Directory to save models
    """

    def __init__(
        self,
        base_model_name: str,
        llm_client: OpenAI,
        llm_model: str,
        corpus_chunks: List[Dict],
        output_dir: str = "models",
        device: str = "cuda"
    ):
        self.model = SentenceTransformer(base_model_name, device=device)
        self.llm = llm_client
        self.llm_model = llm_model
        self.corpus_chunks = corpus_chunks
        self.output_dir = output_dir
        self.chunk_embeddings = None

        os.makedirs(output_dir, exist_ok=True)

    def run_iteration(self, queries: List[Dict], iteration: int = 0) -> SentenceTransformer:
        """
        Run single iteration of hard-negative mining.

        Args:
            queries: List of query dicts with 'query' and 'ground_truth_chunk_ids'
            iteration: Iteration number

        Returns:
            Updated model
        """
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")

        # Step 1: Embed corpus with current model
        print("\n[Step 1/5] Embedding corpus...")
        self.chunk_embeddings = self.model.encode(
            [c['text'] for c in self.corpus_chunks],
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"Embedded {len(self.corpus_chunks)} chunks")

        # Step 2: Retrieve candidates for each query
        print("\n[Step 2/5] Retrieving candidates...")
        candidates = self.retrieve_candidates(queries, k=50)
        print(f"Retrieved candidates for {len(candidates)} queries")

        # Step 3: LLM judges relevance
        print("\n[Step 3/5] LLM judging relevance...")
        scored_pairs = self.llm_judge_relevance(candidates)
        print(f"LLM judged {len(scored_pairs)} (query, chunk) pairs")

        # Step 4: Mine hard examples
        print("\n[Step 4/5] Mining hard examples...")
        hard_pos, hard_neg = self.mine_hard_examples(scored_pairs)
        print(f"Mined {len(hard_pos)} hard positives, {len(hard_neg)} hard negatives")

        if len(hard_pos) == 0 and len(hard_neg) == 0:
            print("WARNING: No hard examples found. Skipping fine-tuning.")
            return self.model

        # Step 5: Fine-tune model
        print(f"\n[Step 5/5] Fine-tuning model...")
        self.fine_tune(hard_pos, hard_neg, iteration)

        # Save model after iteration
        model_path = f"{self.output_dir}/iteration_{iteration}"
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")

        return self.model

    def retrieve_candidates(self, queries: List[Dict], k: int = 50) -> List[List[Dict]]:
        """
        Retrieve top-k chunks for each query.

        Args:
            queries: List of query dicts
            k: Number of top candidates to retrieve

        Returns:
            List of candidate lists (one per query)
        """
        query_embeddings = self.model.encode(
            [q['query'] for q in queries],
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Cosine similarity: (queries, chunks)
        similarities = np.dot(query_embeddings, self.chunk_embeddings.T)

        # Get top-k indices for each query
        top_k_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]  # Descending order

        candidates = []
        for i, query in enumerate(tqdm(queries, desc="Preparing candidates")):
            query_candidates = []
            for rank, chunk_idx in enumerate(top_k_indices[i]):
                query_candidates.append({
                    'query': query['query'],
                    'chunk': self.corpus_chunks[chunk_idx],
                    'model_rank': rank + 1,
                    'similarity': float(similarities[i][chunk_idx]),
                    'ground_truth_chunk_ids': query.get('ground_truth_chunk_ids', [])
                })
            candidates.append(query_candidates)

        return candidates

    def llm_judge_relevance(self, candidates: List[List[Dict]]) -> List[Dict]:
        """
        LLM judges (query, chunk) relevance with scores 1-4.

        Args:
            candidates: List of candidate lists

        Returns:
            List of scored (query, chunk) pairs
        """
        scored_pairs = []

        # Flatten candidates for batch processing
        flat_candidates = [cand for query_cands in candidates for cand in query_cands]

        print(f"Judging {len(flat_candidates)} candidates...")

        for candidate in tqdm(flat_candidates, desc="LLM judging"):
            # Improved prompt based on paper's criteria (Table II)
            prompt = f"""You are a medical information relevance judge. Rate how well this passage answers the medical query.

**Scoring Criteria:**
- Score 4: The passage EXPLICITLY and COMPLETELY answers the query with all necessary details
- Score 3: The passage PARTIALLY answers the query - relevant and clear connection, but missing some details
- Score 2: The passage is SOMEWHAT relevant to the query topic, but does NOT answer it
- Score 1: The passage is NOT relevant - completely different topic from the query

**Query (Medical Question):**
{candidate['query']}

**Passage (Medical Information):**
{candidate['chunk']['text']}

**Instructions:**
- Consider: Does this passage actually ANSWER the question being asked?
- A passage can mention related topics but still not answer the specific question (score 2 or 1)
- Only give score 4 if someone reading ONLY this passage would have their question fully answered

Respond with ONLY the number 1, 2, 3, or 4:"""

            try:
                response = self.llm.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1
                )

                llm_score_str = response.choices[0].message.content.strip()
                llm_score = int(llm_score_str)

                # Validate score
                if llm_score not in [1, 2, 3, 4]:
                    llm_score = 2  # Default

            except Exception as e:
                print(f"Error judging relevance: {e}")
                llm_score = 2  # Default to slightly relevant

            scored_pairs.append({
                'query': candidate['query'],
                'chunk': candidate['chunk'],
                'llm_score': llm_score,
                'model_rank': candidate['model_rank'],
                'ground_truth_chunk_ids': candidate['ground_truth_chunk_ids']
            })

        return scored_pairs

    def mine_hard_examples(self, scored_pairs: List[Dict]) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Mine hard positives and negatives.

        Hard positive: LLM says relevant (score >= 3) but model ranked low (rank > 10)
        Hard negative: LLM says not relevant (score <= 2) but model ranked high (rank <= 5)

        Args:
            scored_pairs: List of scored (query, chunk) pairs

        Returns:
            (hard_positives, hard_negatives) as lists of (query, chunk_text) tuples
        """
        hard_pos = []
        hard_neg = []

        for pair in scored_pairs:
            chunk_id = pair['chunk']['chunk_id']
            is_ground_truth = chunk_id in pair['ground_truth_chunk_ids']

            # Hard positive: LLM says relevant but model ranked low
            if pair['llm_score'] >= 3 and pair['model_rank'] > 10:
                hard_pos.append((pair['query'], pair['chunk']['text']))

            # Hard negative: LLM says not relevant but model ranked high
            elif pair['llm_score'] <= 2 and pair['model_rank'] <= 5:
                hard_neg.append((pair['query'], pair['chunk']['text']))

        return hard_pos, hard_neg

    def fine_tune(self, hard_pos: List[Tuple], hard_neg: List[Tuple], iteration: int):
        """
        Fine-tune model on hard examples using contrastive loss.

        Args:
            hard_pos: List of (query, positive_chunk) tuples
            hard_neg: List of (query, negative_chunk) tuples
            iteration: Current iteration number
        """
        # Create training examples
        train_examples = []

        # Hard positives (label=1.0)
        for query, pos_chunk in hard_pos:
            train_examples.append(InputExample(texts=[query, pos_chunk], label=1.0))

        # Hard negatives (label=0.0)
        for query, neg_chunk in hard_neg:
            train_examples.append(InputExample(texts=[query, neg_chunk], label=0.0))

        if len(train_examples) == 0:
            print("No training examples to fine-tune on")
            return

        print(f"Training on {len(train_examples)} examples...")

        # DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

        # Contrastive loss (CosineSimilarityLoss)
        train_loss = losses.CosineSimilarityLoss(self.model)

        # Fine-tune (force CPU to avoid OOM with vLLM on GPU)
        import os
        old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPUs from PyTorch

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
            show_progress_bar=True,
            output_path=f"{self.output_dir}/iteration_{iteration}_checkpoint",
            use_amp=False  # Disable automatic mixed precision (GPU feature)
        )

        # Restore CUDA visibility
        if old_cuda_visible is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)

        print(f"Fine-tuning complete for iteration {iteration}")


def main():
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Iterative Hard-Negative Mining")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="models/medical-specialized", help="Output directory")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L12-v2", help="Base model")
    parser.add_argument("--llm-model", default="gpt-3.5-turbo", help="LLM model for judging")
    parser.add_argument("--llm-endpoint", default=None, help="LLM endpoint (if using vLLM)")
    parser.add_argument("--iterations", type=int, default=2, help="Number of iterations")
    parser.add_argument("--num-queries", type=int, default=1000, help="Number of training queries to use")
    args = parser.parse_args()

    print("Loading data...")
    with open(f"{args.data_dir}/corpus_chunks.pkl", "rb") as f:
        corpus_chunks = pickle.load(f)

    with open(f"{args.data_dir}/train_queries.pkl", "rb") as f:
        train_queries = pickle.load(f)

    # Use subset of queries for faster iteration
    if args.num_queries < len(train_queries):
        train_queries = train_queries[:args.num_queries]

    print(f"Loaded {len(corpus_chunks)} chunks, {len(train_queries)} queries")

    # Initialize LLM client
    if args.llm_endpoint:
        # vLLM endpoint
        llm_client = OpenAI(base_url=args.llm_endpoint, api_key="EMPTY")
    else:
        # OpenAI API
        llm_client = OpenAI()  # Uses OPENAI_API_KEY env var

    # Initialize miner
    miner = IterativeHardNegativeMiner(
        base_model_name=args.base_model,
        llm_client=llm_client,
        llm_model=args.llm_model,
        corpus_chunks=corpus_chunks,
        output_dir=args.output_dir,
        device="cpu"  # Use CPU since vLLM is using all GPUs
    )

    # Run iterations
    for i in range(args.iterations):
        miner.run_iteration(train_queries, iteration=i)

    # Save final model
    final_model_path = f"{args.output_dir}/final"
    miner.model.save(final_model_path)
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
