#!/usr/bin/env python3.11
"""VSR Domain Classifier Benchmark

Fast, standalone accuracy test for the domain classification model.
No Ollama, no PII — just domain classification on labeled test data.

Test queries are in data/benchmark_domain_queries.json (924 hand-written)
plus MMLU-Pro validation split (70 unseen academic questions).

Usage:
    python3.11 src/training/model_eval/benchmark_domain.py
    python3.11 src/training/model_eval/benchmark_domain.py --style academic
    python3.11 src/training/model_eval/benchmark_domain.py --style real
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths relative to repo root (script lives in src/training/model_eval/)
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = REPO_ROOT / "results"


def load_model(model_path=None):
    if model_path is None:
        model_path = MODELS_DIR / "mom-domain-classifier"
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        raise SystemExit(1)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    model.to("cpu")
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return tokenizer, model, id2label


def load_test_data(style="all"):
    queries = []

    # MMLU-Pro validation split (academic, model has NOT seen these)
    if style in ("all", "academic"):
        try:
            from datasets import load_dataset

            ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
            for i in range(len(ds)):
                queries.append(
                    {
                        "query": ds[i]["question"],
                        "expected": ds[i]["category"].lower(),
                        "style": "academic",
                    }
                )
        except Exception as e:
            print(f"[WARN] Could not load MMLU-Pro validation: {e}")

    # Static benchmark (diverse real-world styles)
    if style in ("all", "real"):
        static_file = DATA_DIR / "benchmark_domain_queries.json"
        if static_file.exists():
            with open(static_file) as f:
                for item in json.load(f):
                    queries.append(
                        {
                            "query": item["query"],
                            "expected": item["expected"].lower(),
                            "style": item.get("style", "real"),
                        }
                    )
        else:
            print(f"[WARN] No static benchmark at {static_file}")

    return queries


def classify(tokenizer, model, id2label, query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = probs.argmax(dim=-1).item()
        confidence = probs[0][pred_id].item()
    return id2label.get(pred_id, str(pred_id)).lower(), confidence


def run_benchmark(style="all", model_path=None):
    print(f"Loading model...")
    tokenizer, model, id2label = load_model(model_path)

    print(f"Loading test data (style={style})...")
    queries = load_test_data(style)
    if not queries:
        print("[ERROR] No test data loaded")
        return

    print(f"Classifying {len(queries)} queries...\n")
    start = time.time()

    # Per-domain stats
    stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "confused_with": defaultdict(int),
            "high_conf_wrong": 0,
            "conf_correct": [],
            "conf_wrong": [],
            "failures": [],
        }
    )
    # Per-style stats
    style_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for item in queries:
        query = item["query"]
        expected = item["expected"]
        qstyle = item.get("style", "unknown")

        predicted, confidence = classify(tokenizer, model, id2label, query)
        correct = predicted == expected

        s = stats[expected]
        s["total"] += 1
        style_stats[qstyle]["total"] += 1

        if correct:
            s["correct"] += 1
            s["conf_correct"].append(confidence)
            style_stats[qstyle]["correct"] += 1
        else:
            s["confused_with"][predicted] += 1
            s["conf_wrong"].append(confidence)
            if confidence > 0.8:
                s["high_conf_wrong"] += 1
            if len(s["failures"]) < 3:  # Keep up to 3 example failures
                s["failures"].append(
                    {
                        "query": query[:80],
                        "predicted": predicted,
                        "confidence": round(confidence, 3),
                    }
                )

    elapsed = time.time() - start
    total_correct = sum(s["correct"] for s in stats.values())
    total_count = sum(s["total"] for s in stats.values())
    overall_acc = 100 * total_correct / total_count

    # ── Print results ──
    print(f"{'='*80}")
    print(f"  DOMAIN CLASSIFIER BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"  Queries tested:   {total_count}")
    print(
        f"  Time:             {elapsed:.1f}s ({elapsed/total_count*1000:.0f}ms/query)"
    )
    print(f"  Overall accuracy: {total_correct}/{total_count} ({overall_acc:.1f}%)")
    print(f"{'='*80}")

    # Column legend
    print(
        f"""
  Columns:
    Pass/Total  — correct predictions / total queries for that domain
    Acc         — accuracy percentage
    AvgConf     — average model confidence on CORRECT predictions (0-1)
    HiErr       — high-confidence errors: model was >80% sure but WRONG
    Top confusion — which domain steals the most queries (and how many)
"""
    )

    # Per-domain table
    print(
        f"{'Domain':<18} {'Pass':>6} {'Total':>6} {'Acc':>7}  {'AvgConf':>8} {'HiErr':>6}  Top confusion"
    )
    print("-" * 85)

    for d in sorted(stats):
        s = stats[d]
        rate = 100 * s["correct"] / s["total"] if s["total"] else 0
        avg_conf = (
            (sum(s["conf_correct"]) / len(s["conf_correct"]))
            if s["conf_correct"]
            else 0
        )
        confused = sorted(s["confused_with"].items(), key=lambda x: -x[1])
        top = f"-> {confused[0][0]} ({confused[0][1]}x)" if confused else ""
        flag = " !!" if rate < 70 else ""
        print(
            f"{d:<18} {s['correct']:>6} {s['total']:>6} {rate:>6.1f}%{flag} {avg_conf:>7.2f}  {s['high_conf_wrong']:>5}  {top}"
        )

    # Per-style breakdown
    print(
        f"""
  Query styles:
    academic       — textbook/exam-style questions (MMLU-Pro format)
    conversational — how real users type queries (lowercase, casual)
    short          — 1-5 word fragments or keyword queries
    boundary       — cross-domain edge cases (could belong to multiple domains)
    gibberish      — random text, typos, nonsense (should classify as "other")
"""
    )

    print(f"{'Style':<20} {'Pass':>6} {'Total':>6} {'Acc':>7}")
    print("-" * 42)
    for st in sorted(style_stats):
        s = style_stats[st]
        rate = 100 * s["correct"] / s["total"] if s["total"] else 0
        print(f"{st:<20} {s['correct']:>6} {s['total']:>6} {rate:>6.1f}%")

    # Example failures
    print(f"\nSample failures (up to 3 per domain):")
    print("-" * 85)
    for d in sorted(stats):
        for f in stats[d]["failures"]:
            print(f"  [{d:>16}] \"{f['query']}\"")
            print(f"  {'':>18} predicted: {f['predicted']} (conf: {f['confidence']})")

    # Summary
    print(f"\n{'='*80}")
    strong = sorted(
        [
            d
            for d, s in stats.items()
            if s["total"] >= 3 and 100 * s["correct"] / s["total"] >= 80
        ]
    )
    weak = sorted(
        [
            d
            for d, s in stats.items()
            if s["total"] >= 3 and 100 * s["correct"] / s["total"] < 70
        ]
    )
    borderline = sorted(
        [
            d
            for d, s in stats.items()
            if s["total"] >= 3 and 70 <= 100 * s["correct"] / s["total"] < 80
        ]
    )
    hc_wrong = sum(s["high_conf_wrong"] for s in stats.values())

    print(f"  Strong (>=80%):      {', '.join(strong) if strong else 'none'}")
    print(f"  Borderline (70-80%): {', '.join(borderline) if borderline else 'none'}")
    print(f"  Weak (<70%):         {', '.join(weak) if weak else 'none'}")
    print(f"  High-confidence errors: {hc_wrong}")
    print(f"{'='*80}\n")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total_count,
        "overall_accuracy": round(overall_acc, 1),
        "elapsed_seconds": round(elapsed, 1),
        "per_domain": {
            d: {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": (
                    round(100 * s["correct"] / s["total"], 1) if s["total"] else 0
                ),
                "confused_with": dict(s["confused_with"]),
                "high_conf_wrong": s["high_conf_wrong"],
                "failures": s["failures"],
            }
            for d, s in stats.items()
        },
        "per_style": {
            st: {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": round(100 * s["correct"] / s["total"], 1),
            }
            for st, s in style_stats.items()
        },
    }

    outfile = RESULTS_DIR / f"domain_benchmark_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(report, f, indent=2)

    latest = RESULTS_DIR / "domain_benchmark_latest.json"
    with open(latest, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Results saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSR Domain Classifier Benchmark")
    parser.add_argument(
        "--style",
        choices=["all", "academic", "real"],
        default="all",
        help="Query style to test: all, academic (MMLU-Pro), real (diverse)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory (default: models/mom-domain-classifier)",
    )
    args = parser.parse_args()
    model_path = Path(args.model_path) if args.model_path else None
    run_benchmark(args.style, model_path)
