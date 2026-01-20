#!/usr/bin/env python3
"""Benchmark embedding computation time for different model formats on CPU."""

import time
import argparse
import numpy as np
from pathlib import Path

# Force CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def benchmark_pytorch(model_path: str, test_names: list, truncate_dim: int, warmup: int = 10, iterations: int = 100):
    """Benchmark PyTorch SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"PyTorch FP32: {model_path}")
    print(f"{'='*60}")

    start_load = time.time()
    model = SentenceTransformer(model_path, device="cpu")
    load_time = time.time() - start_load
    print(f"  Model load time: {load_time:.2f}s")

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        emb = model.encode(test_names[0], convert_to_numpy=True, normalize_embeddings=True)
        if truncate_dim:
            emb = emb[:truncate_dim]

    # Benchmark single queries
    print(f"  Benchmarking single queries ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        name = test_names[i % len(test_names)]
        start = time.time()
        emb = model.encode(name, convert_to_numpy=True, normalize_embeddings=True)
        if truncate_dim:
            emb = emb[:truncate_dim]
        times.append((time.time() - start) * 1000)

    return {
        "format": "PyTorch FP32",
        "load_time_s": load_time,
        "avg_ms": np.mean(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def benchmark_onnx(model_path: str, test_names: list, truncate_dim: int, warmup: int = 10, iterations: int = 100, label: str = "ONNX"):
    """Benchmark ONNX model (works for both FP16/INT8)."""
    import sys
    sys.path.insert(0, '/home/roncewind/roncewind.git/SzEmbeddings')
    from onnx_sentence_transformer import load_onnx_model

    print(f"\n{'='*60}")
    print(f"{label}: {model_path}")
    print(f"{'='*60}")

    start_load = time.time()
    model = load_onnx_model(model_path, providers=['CPUExecutionProvider'])
    load_time = time.time() - start_load
    print(f"  Model load time: {load_time:.2f}s")

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        emb = model.encode([test_names[0]])

    # Benchmark single queries
    print(f"  Benchmarking single queries ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        name = test_names[i % len(test_names)]
        start = time.time()
        emb = model.encode([name])
        times.append((time.time() - start) * 1000)

    return {
        "format": label,
        "load_time_s": load_time,
        "avg_ms": np.mean(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding computation time")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--truncate_dim", type=int, default=512, help="Matryoshka truncation dimension")
    args = parser.parse_args()

    # Test names (mix of personal and business)
    test_names = [
        "John Smith",
        "Toyota Motor Corporation",
        "María García López",
        "Microsoft Corporation",
        "Александр Петрович Иванов",
        "Samsung Electronics Co., Ltd.",
        "محمد علي",
        "Apple Inc.",
        "Jean-Pierre Dubois",
        "Volkswagen Aktiengesellschaft",
        "김철수",
        "Amazon.com, Inc.",
        "Müller GmbH & Co. KG",
        "Robert James Williams III",
        "ГОСУДАРСТВЕННОЕ ПРЕДПРИЯТИЕ",
    ]

    print("="*60)
    print("EMBEDDING COMPUTATION BENCHMARK (CPU)")
    print("="*60)
    print(f"Test names: {len(test_names)}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print(f"Truncate dim: {args.truncate_dim}")

    results = []

    # Model paths
    models = {
        "PyTorch FP32 (Personal)": {
            "path": "/home/roncewind/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model",
            "type": "pytorch"
        },
        "PyTorch FP32 (Business)": {
            "path": "/home/roncewind/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model",
            "type": "pytorch"
        },
        "ORT FP16 (Personal)": {
            "path": "/home/roncewind/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-ort-fp16",
            "type": "onnx",
            "label": "ORT FP16"
        },
        "ORT FP16 (Business)": {
            "path": "/home/roncewind/roncewind.git/BizNames/output/phase10_quantization/ort_fp16",
            "type": "onnx",
            "label": "ORT FP16"
        },
        "ONNX INT8 (Personal)": {
            "path": "/home/roncewind/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-onnx-int8-v3",
            "type": "onnx",
            "label": "ONNX INT8"
        },
        "ONNX INT8 (Business)": {
            "path": "/home/roncewind/roncewind.git/BizNames/output/phase10_quantization/onnx_int8",
            "type": "onnx",
            "label": "ONNX INT8"
        },
    }

    for name, config in models.items():
        path = config["path"]
        if not Path(path).exists():
            print(f"\n⚠️  Skipping {name}: path not found")
            continue

        try:
            if config["type"] == "pytorch":
                result = benchmark_pytorch(path, test_names, args.truncate_dim, args.warmup, args.iterations)
            else:
                result = benchmark_onnx(path, test_names, args.truncate_dim, args.warmup, args.iterations, config.get("label", "ONNX"))
            result["model"] = name
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error benchmarking {name}: {e}")

    # Summary
    print("\n")
    print("="*80)
    print("SUMMARY: Embedding Computation Time (CPU)")
    print("="*80)
    print(f"{'Model':<30} {'Avg (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} {'Load (s)':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['model']:<30} {r['avg_ms']:<12.1f} {r['p50_ms']:<12.1f} {r['p95_ms']:<12.1f} {r['load_time_s']:<10.2f}")

    # Aggregate by format
    print("\n")
    print("="*80)
    print("AGGREGATED BY FORMAT (Average of Personal + Business models)")
    print("="*80)

    formats = {}
    for r in results:
        fmt = r["format"]
        if fmt not in formats:
            formats[fmt] = []
        formats[fmt].append(r)

    print(f"{'Format':<20} {'Avg (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12}")
    print("-"*60)

    format_avgs = {}
    for fmt, fmt_results in formats.items():
        avg = np.mean([r["avg_ms"] for r in fmt_results])
        p50 = np.mean([r["p50_ms"] for r in fmt_results])
        p95 = np.mean([r["p95_ms"] for r in fmt_results])
        format_avgs[fmt] = avg
        print(f"{fmt:<20} {avg:<12.1f} {p50:<12.1f} {p95:<12.1f}")

    # Impact on total query time
    print("\n")
    print("="*80)
    print("ESTIMATED TOTAL QUERY TIME (Embedding + DB Search)")
    print("="*80)

    # DB search times from comparison document
    db_times = {
        "PyTorch FP32": 227,
        "ORT FP16": 270,
        "ONNX INT8": 353,
    }

    print(f"{'Format':<20} {'Embedding':<12} {'DB Search':<12} {'Total':<12} {'vs PyTorch':<12}")
    print("-"*72)

    baseline_total = None
    for fmt in ["PyTorch FP32", "ORT FP16", "ONNX INT8"]:
        if fmt in format_avgs and fmt in db_times:
            emb_time = format_avgs[fmt]
            db_time = db_times[fmt]
            total = emb_time + db_time

            if baseline_total is None:
                baseline_total = total
                diff = "—"
            else:
                diff = f"{((total - baseline_total) / baseline_total) * 100:+.1f}%"

            print(f"{fmt:<20} {emb_time:<12.1f} {db_time:<12.1f} {total:<12.1f} {diff:<12}")


if __name__ == "__main__":
    main()
