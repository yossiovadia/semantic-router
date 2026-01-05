#!/usr/bin/env python3
"""
LLM-based augmentation for MedQuAD unlabeled queries.
Following the paper's approach (arXiv:2504.02268v1).
"""

import json
import requests
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def extract_text(value):
    """Extract text from either string or {"text": "..."} format."""
    if isinstance(value, dict):
        return value.get('text', '')
    return str(value) if value else ''

def generate_paraphrases_ollama(query: str, model: str = "qwen2.5:1.5b", num_paraphrases: int = 3) -> list:
    """Generate diverse paraphrases using Ollama (following paper's Listing 1)."""
    
    prompt = f"""You are a helpful medical expert. Generate {num_paraphrases} unique paraphrases of the given query.

Original Query: '{query}'

Each paraphrase should:
1. Preserve the original meaning but use different wording or sentence structure.
2. Avoid changing medical intent or introducing new information.
3. Be professionally written and clear.

Example:
Original Query: "What are the best ways to reduce stress?"
Paraphrased Queries:
1. "How can a person effectively manage stress?"
2. "What strategies help in reducing stress levels?"
3. "What methods are effective for stress reduction?"

Return ONLY a JSON object with a key 'paraphrases' containing a list of exactly {num_paraphrases} paraphrased strings (not objects).
Format: {{"paraphrases": ["string1", "string2", "string3"]}}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": -1  # Keep model loaded indefinitely (clears context each request)
            },
            timeout=60
        )
        
        result = response.json()
        paraphrases_data = json.loads(result['response'])
        
        # Extract and clean paraphrases
        paraphrases = []
        for p in paraphrases_data.get('paraphrases', []):
            text = extract_text(p)
            if text:
                paraphrases.append(text)
        
        return paraphrases
    
    except Exception as e:
        print(f"Error generating paraphrases for '{query[:50]}...': {e}")
        return []


def generate_hard_negatives_ollama(query: str, model: str = "qwen2.5:1.5b", num_negatives: int = 2) -> list:
    """Generate hard negatives using Ollama (following paper's Listing 2)."""
    
    prompt = f"""You are a helpful medical expert. Given a medical query, generate {num_negatives} distinct but related queries that explore different aspects.

Guidelines:
1. The new queries should be related to the original but focus on different subtopics, perspectives, or medical contexts.
2. They should not be simple rewordings or slight variations of the original.
3. Consider different patient populations, alternative diagnostic methods, treatments, or physiological explanations.

Examples:
Original Query: "How to reduce stress?"
Distinct Queries:
1. "How can athletes manage stress during high-pressure competitions?"
2. "What are effective stress management strategies for children with ADHD?"

Original Query: "What is diabetes?"
Distinct Queries:
1. "How does gestational diabetes differ from type 2 diabetes?"
2. "What are the long-term complications of uncontrolled diabetes?"

Now, generate {num_negatives} distinct queries for this input:
Original Query: {query}

Return ONLY a JSON object with 'negatives' containing a list of exactly {num_negatives} strings (not objects).
Format: {{"negatives": ["string1", "string2"]}}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": -1  # Keep model loaded indefinitely (clears context each request)
            },
            timeout=60
        )
        
        result = response.json()
        negatives_data = json.loads(result['response'])
        
        # Extract and clean negatives
        negatives = []
        for n in negatives_data.get('negatives', []):
            text = extract_text(n)
            if text:
                negatives.append(text)
        
        return negatives
    
    except Exception as e:
        print(f"Error generating negatives for '{query[:50]}...': {e}")
        return []


def process_query(query: str, model: str, num_paraphrases: int, num_negatives: int):
    """Process a single query to generate training samples."""
    
    # Generate paraphrases (positive samples)
    paraphrases = generate_paraphrases_ollama(query, model, num_paraphrases)
    
    # Generate hard negatives
    hard_negatives = generate_hard_negatives_ollama(query, model, num_negatives)
    
    # Create training samples
    samples = []
    
    # Positive pairs (paraphrases)
    for paraphrase in paraphrases:
        samples.append({
            "anchor": paraphrase,
            "positive": query,
            "is_duplicate": 1
        })
    
    # Negative pairs
    for hard_negative in hard_negatives:
        samples.append({
            "anchor": query,
            "hard_negative": hard_negative,
            "is_duplicate": 0
        })
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Augment MedQuAD queries with LLM")
    parser.add_argument("--input", required=True, help="Input JSONL with unlabeled queries")
    parser.add_argument("--output", required=True, help="Output JSONL for training")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Ollama model to use")
    parser.add_argument("--paraphrases", type=int, default=3, help="Paraphrases per query")
    parser.add_argument("--negatives", type=int, default=2, help="Hard negatives per query")
    parser.add_argument("--max-queries", type=int, help="Max queries to process (for testing)")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    
    args = parser.parse_args()
    
    # Load queries
    print(f"Loading queries from {args.input}...")
    queries = []
    with open(args.input) as f:
        for line in f:
            data = json.loads(line)
            queries.append(data['query'])
    
    if args.max_queries:
        queries = queries[:args.max_queries]
        print(f"Limited to {args.max_queries} queries for testing")
    
    print(f"Loaded {len(queries)} queries")
    print(f"Using model: {args.model}")
    print(f"Paraphrases per query: {args.paraphrases}")
    print(f"Hard negatives per query: {args.negatives}")
    print(f"Expected output: ~{len(queries) * (args.paraphrases + args.negatives)} samples")
    print()
    
    # Process queries in parallel
    all_samples = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_query, query, args.model, args.paraphrases, args.negatives): query
            for query in queries
        }
        
        for future in tqdm(as_completed(futures), total=len(queries), desc="Augmenting"):
            try:
                samples = future.result()
                all_samples.extend(samples)
            except Exception as e:
                print(f"Error processing query: {e}")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\n✓ Generated {len(all_samples)} training samples")
    print(f"✓ Saved to {args.output}")
    print(f"\nAugmentation factor: {len(all_samples) / len(queries):.1f}x")
    
    # Show sample
    if all_samples:
        print("\nSample training data:")
        for sample in all_samples[:3]:
            print(f"  {json.dumps(sample, indent=2)}")


if __name__ == "__main__":
    main()
