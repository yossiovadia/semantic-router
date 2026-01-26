"""
Offline analysis of dataset characteristics to understand:
1. Query diversity and coverage
2. Theoretical hard example potential
3. Why we might be plateauing
"""

import pickle
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def analyze_query_diversity(queries):
    """Analyze diversity of queries."""
    print("\n" + "="*60)
    print("QUERY DIVERSITY ANALYSIS")
    print("="*60)

    # Length distribution
    lengths = [len(q['query'].split()) for q in queries]
    print(f"\nQuery lengths:")
    print(f"  Mean: {np.mean(lengths):.1f} words")
    print(f"  Min:  {np.min(lengths)} words")
    print(f"  Max:  {np.max(lengths)} words")
    print(f"  Std:  {np.std(lengths):.1f} words")

    # Look for patterns
    query_starts = defaultdict(int)
    for q in queries:
        words = q['query'].split()
        if words:
            first_word = words[0].lower()
            query_starts[first_word] += 1

    print(f"\nMost common starting words:")
    for word, count in sorted(query_starts.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = 100 * count / len(queries)
        print(f"  '{word}': {count} ({pct:.1f}%)")

    # Check if queries are very similar
    print(f"\nTotal unique queries: {len(queries)}")
    unique_queries = len(set(q['query'] for q in queries))
    print(f"Unique query texts: {unique_queries}")
    if unique_queries < len(queries):
        print(f"⚠️  WARNING: {len(queries) - unique_queries} duplicate queries found")


def analyze_corpus_coverage(queries, corpus_chunks):
    """Analyze how well queries cover the corpus."""
    print("\n" + "="*60)
    print("CORPUS COVERAGE ANALYSIS")
    print("="*60)

    # Ground truth distribution
    all_gt_chunks = []
    for q in queries:
        all_gt_chunks.extend(q['ground_truth_chunk_ids'])

    unique_gt_chunks = len(set(all_gt_chunks))
    corpus_size = len(corpus_chunks)

    print(f"\nCorpus statistics:")
    print(f"  Total chunks: {corpus_size}")
    print(f"  Chunks referenced by queries: {unique_gt_chunks}")
    print(f"  Coverage: {100 * unique_gt_chunks / corpus_size:.1f}%")
    print(f"  Unreferenced chunks: {corpus_size - unique_gt_chunks}")

    # Chunks per query
    chunks_per_query = [len(q['ground_truth_chunk_ids']) for q in queries]
    print(f"\nGround truth chunks per query:")
    print(f"  Mean: {np.mean(chunks_per_query):.1f}")
    print(f"  Min:  {np.min(chunks_per_query)}")
    print(f"  Max:  {np.max(chunks_per_query)}")

    if np.mean(chunks_per_query) < 2:
        print(f"  ℹ️  Note: Low chunks/query ratio limits hard positive mining")


def estimate_hard_example_potential(queries, corpus_chunks, model_name="sentence-transformers/all-MiniLM-L12-v2"):
    """Estimate potential for hard example mining."""
    print("\n" + "="*60)
    print("HARD EXAMPLE POTENTIAL ESTIMATION")
    print("="*60)
    print(f"Loading model: {model_name}")

    model = SentenceTransformer(model_name, device="cpu")

    # Sample subset for speed
    sample_size = min(50, len(queries))
    import random
    random.seed(42)
    sampled_queries = random.sample(queries, sample_size)

    print(f"\nEmbedding {sample_size} queries and corpus...")

    # Embed
    query_embeddings = model.encode(
        [q['query'] for q in sampled_queries],
        show_progress_bar=True,
        convert_to_numpy=True
    )

    corpus_embeddings = model.encode(
        [c['text'] for c in corpus_chunks],
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Calculate similarities
    similarities = np.dot(query_embeddings, corpus_embeddings.T)

    # For each query, analyze ranking
    hard_pos_potential = 0
    hard_neg_potential = 0

    for i, query in enumerate(sampled_queries):
        gt_chunk_ids = set(query['ground_truth_chunk_ids'])
        query_sims = similarities[i]

        # Get rankings
        ranked_indices = np.argsort(query_sims)[::-1]

        # Find where ground truth chunks are ranked
        gt_ranks = []
        for chunk_id in gt_chunk_ids:
            rank = np.where(ranked_indices == chunk_id)[0]
            if len(rank) > 0:
                gt_ranks.append(rank[0] + 1)  # 1-indexed

        # Hard positive potential: ground truth ranked > 10
        hard_pos_potential += sum(1 for r in gt_ranks if r > 10)

        # Hard negative potential: non-ground-truth in top 5
        top_5_indices = set(ranked_indices[:5])
        non_gt_in_top5 = len(top_5_indices - gt_chunk_ids)
        hard_neg_potential += non_gt_in_top5

    print(f"\nEstimated hard example potential (from {sample_size} queries):")
    print(f"  Hard positive candidates: {hard_pos_potential} (GT chunks ranked >10)")
    print(f"  Hard negative candidates: {hard_neg_potential} (non-GT in top-5)")

    avg_hard_pos = hard_pos_potential / sample_size
    avg_hard_neg = hard_neg_potential / sample_size

    print(f"\nPer query averages:")
    print(f"  Hard positives: {avg_hard_pos:.1f}")
    print(f"  Hard negatives: {avg_hard_neg:.1f}")

    # Scale to 100 queries
    scaled_hard_pos = avg_hard_pos * 100
    scaled_hard_neg = avg_hard_neg * 100

    print(f"\nProjection for 100 queries:")
    print(f"  Expected hard positives: ~{scaled_hard_pos:.0f}")
    print(f"  Expected hard negatives: ~{scaled_hard_neg:.0f}")

    # Triplet potential (need both)
    triplet_potential = min(scaled_hard_pos, scaled_hard_neg)
    print(f"  Expected triplets: ~{triplet_potential:.0f}")

    if triplet_potential < 50:
        print(f"\n⚠️  WARNING: Low triplet potential (<50)")
        print(f"     This may explain limited improvement and early plateau")
        print(f"     Consider:")
        print(f"     • Using more queries (500-1000)")
        print(f"     • Relaxing hard example criteria")
        print(f"     • Checking if model is already well-suited to this domain")

    return {
        'hard_pos_potential': hard_pos_potential,
        'hard_neg_potential': hard_neg_potential,
        'avg_hard_pos': avg_hard_pos,
        'avg_hard_neg': avg_hard_neg
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze dataset characteristics")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of queries to analyze")
    args = parser.parse_args()

    print("="*60)
    print("DATASET ANALYSIS FOR ITERATIVE MINING")
    print("="*60)

    # Load data
    print("\nLoading data...")
    with open(f"{args.data_dir}/corpus_chunks.pkl", "rb") as f:
        corpus_chunks = pickle.load(f)

    with open(f"{args.data_dir}/train_queries.pkl", "rb") as f:
        all_queries = pickle.load(f)

    print(f"Loaded {len(corpus_chunks)} corpus chunks")
    print(f"Loaded {len(all_queries)} total queries")

    # Sample queries
    import random
    random.seed(42)
    queries = random.sample(all_queries, min(args.num_queries, len(all_queries)))
    print(f"Analyzing {len(queries)} queries")

    # Run analyses
    analyze_query_diversity(queries)
    analyze_corpus_coverage(queries, corpus_chunks)
    potential = estimate_hard_example_potential(queries, corpus_chunks)

    # Final recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if potential['avg_hard_pos'] < 1 or potential['avg_hard_neg'] < 1:
        print("\n⚠️  LOW HARD EXAMPLE POTENTIAL DETECTED")
        print("\nPossible reasons:")
        print("1. Model already well-suited to this domain")
        print("2. Queries too easy (model ranks ground truth highly)")
        print("3. Hard example criteria too strict")
        print("\nSuggestions:")
        print("• Try different base model (smaller or larger)")
        print("• Relax mining criteria (rank>10 → rank>5 for hard pos)")
        print("• Use harder queries if available")
    else:
        print("\n✓ Adequate hard example potential")
        print(f"  ~{potential['avg_hard_pos']:.1f} hard pos / query")
        print(f"  ~{potential['avg_hard_neg']:.1f} hard neg / query")

    if args.num_queries < 500:
        print(f"\nℹ️  Currently using {args.num_queries} queries")
        print(f"   Consider scaling to 500-1000 for better coverage")


if __name__ == "__main__":
    main()
