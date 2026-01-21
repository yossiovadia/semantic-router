"""
Prepare MedQuAD corpus for training.

Downloads MedQuAD dataset and prepares corpus chunks from Q&A pairs.
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def download_medquad():
    """
    Download MedQuAD dataset from HuggingFace datasets.

    Note: MedQuAD is available on HuggingFace as 'medical_questions_pairs'
    or we can use the datasets library to load it directly.
    """
    try:
        from datasets import load_dataset

        print("Loading MedQuAD dataset from HuggingFace...")
        # Load the medical_questions_pairs dataset which contains MedQuAD
        dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")

        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting alternative method...")

        # Fallback: Create a minimal synthetic dataset for testing
        print("Creating minimal synthetic medical Q&A dataset for testing...")
        return create_synthetic_medical_qa()


def create_synthetic_medical_qa():
    """Create a small synthetic medical Q&A dataset for testing."""
    synthetic_data = {
        'train': [
            {
                'qtype': 'information',
                'Question': 'What are the symptoms of diabetes?',
                'Answer': 'Diabetes symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, irritability, blurred vision, slow-healing sores, and frequent infections.'
            },
            {
                'qtype': 'information',
                'Question': 'What causes high blood pressure?',
                'Answer': 'High blood pressure can be caused by various factors including genetics, age, diet (especially high sodium intake), lack of physical activity, obesity, excessive alcohol consumption, stress, and certain chronic conditions.'
            },
            # Add more synthetic examples here for a real implementation
        ]
    }

    print(f"Created synthetic dataset with {len(synthetic_data['train'])} examples")
    print("Note: For production use, download the full MedQuAD dataset")

    return synthetic_data


def chunk_text(text: str, max_length: int = 1000) -> List[str]:
    """
    Split long text into chunks respecting sentence boundaries.

    Args:
        text: Text to chunk
        max_length: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    # Simple sentence splitting (for better results, use nltk or spacy)
    sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def prepare_corpus(output_dir: str = "data"):
    """
    Prepare corpus from MedQuAD dataset.

    Args:
        output_dir: Directory to save processed data
    """
    os.makedirs(output_dir, exist_ok=True)

    # Download dataset
    dataset = download_medquad()

    # Extract Q&A pairs
    print("\nProcessing Q&A pairs...")
    medquad_data = []

    # Handle different dataset formats
    if isinstance(dataset, dict) and 'train' in dataset:
        data_split = dataset['train']
    else:
        # If it's a HuggingFace dataset object
        try:
            data_split = list(dataset['train'])
        except:
            data_split = dataset

    for idx, item in enumerate(tqdm(data_split)):
        # Handle different key names in dataset
        question = item.get('Question', item.get('question', ''))
        answer = item.get('Answer', item.get('answer', ''))
        qtype = item.get('qtype', item.get('question_type', 'information'))

        if question and answer:
            medquad_data.append({
                'id': idx,
                'question': question,
                'answer': answer,
                'qtype': qtype
            })

    print(f"Loaded {len(medquad_data)} Q&A pairs")

    # Create corpus chunks from answers
    print("\nCreating corpus chunks...")
    corpus_chunks = []

    for qa_pair in tqdm(medquad_data):
        answer = qa_pair['answer']

        # Chunk long answers if needed (keep 500-1000 chars)
        if len(answer) > 1000:
            chunks = chunk_text(answer, max_length=1000)
        else:
            chunks = [answer]

        for chunk_text_str in chunks:
            corpus_chunks.append({
                'chunk_id': len(corpus_chunks),
                'text': chunk_text_str,
                'source_question': qa_pair['question'],
                'qa_id': qa_pair['id'],
                'qtype': qa_pair['qtype']
            })

    print(f"Created {len(corpus_chunks)} corpus chunks")

    # Split Q&A pairs into train/val/test (70/15/15)
    print("\nSplitting data...")
    train_qas, temp_qas = train_test_split(medquad_data, test_size=0.3, random_state=42)
    val_qas, test_qas = train_test_split(temp_qas, test_size=0.5, random_state=42)

    print(f"Train: {len(train_qas)} Q&A pairs")
    print(f"Val: {len(val_qas)} Q&A pairs")
    print(f"Test: {len(test_qas)} Q&A pairs")

    # Save data
    print("\nSaving processed data...")

    with open(f"{output_dir}/corpus_chunks.pkl", "wb") as f:
        pickle.dump(corpus_chunks, f)

    with open(f"{output_dir}/train_qas.pkl", "wb") as f:
        pickle.dump(train_qas, f)

    with open(f"{output_dir}/val_qas.pkl", "wb") as f:
        pickle.dump(val_qas, f)

    with open(f"{output_dir}/test_qas.pkl", "wb") as f:
        pickle.dump(test_qas, f)

    # Also save as JSON for inspection
    with open(f"{output_dir}/corpus_chunks.json", "w") as f:
        json.dump(corpus_chunks[:100], f, indent=2)  # Save first 100 for inspection

    print(f"\nData saved to {output_dir}/")
    print(f"  - corpus_chunks.pkl ({len(corpus_chunks)} chunks)")
    print(f"  - train_qas.pkl ({len(train_qas)} pairs)")
    print(f"  - val_qas.pkl ({len(val_qas)} pairs)")
    print(f"  - test_qas.pkl ({len(test_qas)} pairs)")
    print(f"  - corpus_chunks.json (first 100 chunks for inspection)")


if __name__ == "__main__":
    prepare_corpus()
