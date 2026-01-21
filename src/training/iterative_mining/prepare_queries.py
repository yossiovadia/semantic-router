"""
Prepare queries from MedQuAD Q&A pairs.

Uses existing MedQuAD questions (unlike the paper which generates queries).
"""

import json
import pickle
from typing import List, Dict
from tqdm import tqdm


def prepare_queries(data_dir: str = "data"):
    """
    Prepare queries from MedQuAD Q&A pairs.

    Args:
        data_dir: Directory containing processed corpus data
    """
    print("Loading Q&A pairs...")

    # Load train/val/test splits
    with open(f"{data_dir}/train_qas.pkl", "rb") as f:
        train_qas = pickle.load(f)

    with open(f"{data_dir}/val_qas.pkl", "rb") as f:
        val_qas = pickle.load(f)

    with open(f"{data_dir}/test_qas.pkl", "rb") as f:
        test_qas = pickle.load(f)

    # Load corpus chunks to map Q&A IDs to chunk IDs
    with open(f"{data_dir}/corpus_chunks.pkl", "rb") as f:
        corpus_chunks = pickle.load(f)

    # Create mapping from QA ID to chunk IDs
    print("Creating QA ID to chunk ID mapping...")
    qa_id_to_chunk_ids = {}
    for chunk in corpus_chunks:
        qa_id = chunk['qa_id']
        if qa_id not in qa_id_to_chunk_ids:
            qa_id_to_chunk_ids[qa_id] = []
        qa_id_to_chunk_ids[qa_id].append(chunk['chunk_id'])

    # Prepare train queries
    print("\nPreparing training queries...")
    train_queries = []
    for qa in tqdm(train_qas):
        chunk_ids = qa_id_to_chunk_ids.get(qa['id'], [])
        if chunk_ids:  # Only include queries with corresponding chunks
            train_queries.append({
                'query': qa['question'],
                'ground_truth_qa_id': qa['id'],
                'ground_truth_chunk_ids': chunk_ids,
                'qtype': qa.get('qtype', 'information')
            })

    # Prepare validation queries
    print("Preparing validation queries...")
    val_queries = []
    for qa in tqdm(val_qas):
        chunk_ids = qa_id_to_chunk_ids.get(qa['id'], [])
        if chunk_ids:
            val_queries.append({
                'query': qa['question'],
                'ground_truth_qa_id': qa['id'],
                'ground_truth_chunk_ids': chunk_ids,
                'qtype': qa.get('qtype', 'information')
            })

    # Prepare test queries
    print("Preparing test queries...")
    test_queries = []
    for qa in tqdm(test_qas):
        chunk_ids = qa_id_to_chunk_ids.get(qa['id'], [])
        if chunk_ids:
            test_queries.append({
                'query': qa['question'],
                'ground_truth_qa_id': qa['id'],
                'ground_truth_chunk_ids': chunk_ids,
                'qtype': qa.get('qtype', 'information')
            })

    print(f"\nQuery statistics:")
    print(f"  Training queries: {len(train_queries)}")
    print(f"  Validation queries: {len(val_queries)}")
    print(f"  Test queries: {len(test_queries)}")

    # Save queries
    print("\nSaving queries...")

    with open(f"{data_dir}/train_queries.pkl", "wb") as f:
        pickle.dump(train_queries, f)

    with open(f"{data_dir}/val_queries.pkl", "wb") as f:
        pickle.dump(val_queries, f)

    with open(f"{data_dir}/test_queries.pkl", "wb") as f:
        pickle.dump(test_queries, f)

    # Save first 20 as JSON for inspection
    with open(f"{data_dir}/train_queries_sample.json", "w") as f:
        json.dump(train_queries[:20], f, indent=2)

    print(f"\nQueries saved to {data_dir}/")
    print(f"  - train_queries.pkl ({len(train_queries)} queries)")
    print(f"  - val_queries.pkl ({len(val_queries)} queries)")
    print(f"  - test_queries.pkl ({len(test_queries)} queries)")
    print(f"  - train_queries_sample.json (first 20 for inspection)")

    # Print sample query
    if train_queries:
        print("\nSample training query:")
        sample = train_queries[0]
        print(f"  Query: {sample['query']}")
        print(f"  Ground truth QA ID: {sample['ground_truth_qa_id']}")
        print(f"  Ground truth chunk IDs: {sample['ground_truth_chunk_ids']}")
        print(f"  Question type: {sample['qtype']}")


if __name__ == "__main__":
    prepare_queries()
