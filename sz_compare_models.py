#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Compare Senzing embedding model evaluation results.
#
# Reads JSON result files from sz_evaluate_model.py and generates
# side-by-side comparison reports.
#
# Usage:
#   python sz_compare_models.py results/*.json
#   python sz_compare_models.py --output comparison_report.txt results/*.json
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_results(file_paths: list[str]) -> list[dict[str, Any]]:
    """Load evaluation results from JSON files."""
    results = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}", file=sys.stderr)
    return results


def print_comparison_report(results: list[dict[str, Any]], output_file: str | None = None) -> None:
    """Print formatted comparison report."""
    if not results:
        print("No results to compare")
        return

    f = open(output_file, 'w') if output_file else sys.stdout

    # Group by test set and model type
    by_test_set: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in results:
        test_set = r.get('test_set', 'unknown')
        model_type = r.get('model_type', 'unknown')

        if test_set not in by_test_set:
            by_test_set[test_set] = {}
        if model_type not in by_test_set[test_set]:
            by_test_set[test_set][model_type] = []

        by_test_set[test_set][model_type].append(r)

    # Sort results within each group by embedding accuracy (descending)
    for test_set in by_test_set:
        for model_type in by_test_set[test_set]:
            by_test_set[test_set][model_type].sort(
                key=lambda x: x.get('metrics', {}).get('emb_accuracy', 0),
                reverse=True
            )

    # Print report
    f.write("=" * 120 + "\n")
    f.write("Senzing Embedding Model Comparison Report\n")
    f.write("=" * 120 + "\n\n")

    for test_set, model_types in sorted(by_test_set.items()):
        f.write(f"\n{test_set.upper()} Test Set\n")
        f.write("-" * 120 + "\n\n")

        for model_type, test_results in sorted(model_types.items()):
            f.write(f"\n{model_type.capitalize()} Names\n")
            f.write("-" * 120 + "\n\n")

            # Header - Updated to include Pos>Neg (primary metric)
            f.write(f"{'Model':<40s} {'Queries':>9s} {'GNR Pos>Neg':>11s} {'GNR Acc':>8s} "
                   f"{'EMB Pos>Neg':>11s} {'EMB Acc':>8s} {'Δ Pos>Neg':>10s} {'Δ Acc':>7s}\n")
            f.write("-" * 120 + "\n")

            # Results
            for r in test_results:
                model_path = r.get('model_path', 'Unknown')
                model_name = Path(model_path).parent.name  # Get the directory name
                metrics = r.get('metrics', {})

                triplets = metrics.get('total_triplets', 0)

                # NEW: Relative ranking (primary metric)
                gnr_pos_gt_neg = metrics.get('gnr_relative_ranking', {}).get('positive_gt_negative_rate', 0.0)
                emb_pos_gt_neg = metrics.get('emb_relative_ranking', {}).get('positive_gt_negative_rate', 0.0)
                delta_pos_gt_neg = emb_pos_gt_neg - gnr_pos_gt_neg

                # Legacy: Top-1 accuracy
                gnr_acc = metrics.get('gnr_accuracy', 0.0)
                emb_acc = metrics.get('emb_accuracy', 0.0)
                delta_acc = emb_acc - gnr_acc

                f.write(f"{model_name:<40s} {triplets:>9,} {gnr_pos_gt_neg:>10.1f}% {gnr_acc:>7.2f}% "
                       f"{emb_pos_gt_neg:>10.1f}% {emb_acc:>7.2f}% {delta_pos_gt_neg:>+9.1f}% {delta_acc:>+6.2f}%\n")

            # Detailed metrics for top model
            if test_results:
                top_model = test_results[0]
                metrics = top_model.get('metrics', {})

                f.write(f"\n\nTop Model: {Path(top_model['model_path']).parent.name}\n")
                f.write("-" * 120 + "\n")

                # NEW: Relative Ranking (PRIMARY METRIC)
                gnr_rr = metrics.get('gnr_relative_ranking', {})
                emb_rr = metrics.get('emb_relative_ranking', {})

                f.write("\nRELATIVE RANKING (PRIMARY METRIC):\n")
                f.write(f"  Positive > Negative Rate:\n")
                f.write(f"    Name-Only:           {gnr_rr.get('positive_gt_negative_rate', 0.0):.1f}%\n")
                f.write(f"    Embeddings:          {emb_rr.get('positive_gt_negative_rate', 0.0):.1f}%\n")
                f.write(f"    Delta:               {emb_rr.get('positive_gt_negative_rate', 0.0) - gnr_rr.get('positive_gt_negative_rate', 0.0):+.1f}%\n")

                f.write("\n  Scenarios (Embeddings):\n")
                f.write(f"    Both found, correct order:   {emb_rr.get('both_found_correct_order', 0.0):.1f}%\n")
                f.write(f"    Both found, wrong order:     {emb_rr.get('both_found_wrong_order', 0.0):.1f}%\n")
                f.write(f"    Only positive found:         {emb_rr.get('only_positive_found', 0.0):.1f}%\n")
                f.write(f"    Only negative found:         {emb_rr.get('only_negative_found', 0.0):.1f}%\n")

                # NEW: NDCG@K
                gnr_ndcg = metrics.get('gnr_ndcg', {})
                emb_ndcg = metrics.get('emb_ndcg', {})

                f.write("\nRANKING QUALITY (NDCG@K):\n")
                for k in [5, 10, 20, 100]:
                    gnr_val = gnr_ndcg.get(k, 0.0)
                    emb_val = emb_ndcg.get(k, 0.0)
                    delta = emb_val - gnr_val
                    f.write(f"  NDCG@{k:<3}:  Name-Only={gnr_val:.4f}, Embeddings={emb_val:.4f}, Delta={delta:+.4f}\n")

                # NEW: Threshold Analysis (if available)
                emb_ta = metrics.get('emb_threshold_analysis', {})
                if emb_ta:
                    balanced = emb_ta.get('recommended_balanced', {})
                    if balanced:
                        f.write("\nRECOMMENDED THRESHOLDS (Balanced):\n")
                        f.write(f"  sameScore:        {balanced.get('sameScore', 0)}\n")
                        f.write(f"  closeScore:       {balanced.get('closeScore', 0)}  (cutoff)\n")
                        f.write(f"  likelyScore:      {balanced.get('likelyScore', 0)}\n")
                        f.write(f"  plausibleScore:   {balanced.get('plausibleScore', 0)}\n")
                        f.write(f"  unlikelyScore:    {balanced.get('unlikelyScore', 0)}\n")
                        f.write(f"  Est. pos recall:  {balanced.get('estimated_pos_recall', 0):.0f}%\n")

                # NEW: Variant Type Breakdown (if available)
                variant_metrics = metrics.get('variant_metrics', {})
                if variant_metrics:
                    f.write("\nVARIANT TYPE BREAKDOWN (Top 5 by Pos>Neg rate):\n")
                    sorted_variants = sorted(
                        variant_metrics.items(),
                        key=lambda x: x[1]['relative_ranking']['positive_gt_negative_rate'],
                        reverse=True
                    )[:5]

                    for vtype, vm in sorted_variants:
                        pos_gt_neg = vm['relative_ranking']['positive_gt_negative_rate']
                        top1 = vm['accuracy_top1']
                        count = vm['count']
                        f.write(f"  {vtype:<30s}  Pos>Neg={pos_gt_neg:>5.1f}%, Top-1={top1:>5.1f}%, Count={count:>5,}\n")

                # NEW: Rescue Rate
                rescue = metrics.get('rescue_rates', {})
                f.write("\nRESCUE ANALYSIS:\n")
                f.write(f"  Rescue rate:          {rescue.get('rescue_rate', 0.0):.1f}% (EMB found when name-only failed)\n")
                f.write(f"  Both correct:         {rescue.get('both_correct', 0.0):.1f}%\n")
                f.write(f"  Both wrong:           {rescue.get('both_wrong', 0.0):.1f}%\n")

                # LEGACY METRICS
                f.write("\nLEGACY METRICS (Top-1, MRR, Recall@K):\n")
                f.write(f"  Name-Only Accuracy:    {metrics.get('gnr_accuracy', 0.0):.2f}%\n")
                f.write(f"  Embeddings Accuracy:   {metrics.get('emb_accuracy', 0.0):.2f}%\n")
                f.write(f"  Delta:                 {metrics.get('emb_accuracy', 0.0) - metrics.get('gnr_accuracy', 0.0):+.2f}%\n")
                f.write(f"  Name-Only MRR:         {metrics.get('gnr_mrr', 0.0):.4f}\n")
                f.write(f"  Embeddings MRR:        {metrics.get('emb_mrr', 0.0):.4f}\n")

                # Evaluation metadata
                f.write("\nEVALUATION METADATA:\n")
                f.write(f"  Total queries:         {top_model.get('total_triplets', 0):,}\n")
                f.write(f"  Evaluated queries:     {top_model.get('evaluated_triplets', 0):,}\n")
                f.write(f"  Skipped queries:       {top_model.get('skipped_triplets', 0):,}\n")
                f.write(f"  Evaluation time:       {top_model.get('evaluation_time_seconds', 0.0):.1f} seconds\n")
                f.write(f"  Data source:           {top_model.get('data_source', 'N/A')}\n")

            f.write("\n")

    f.write("\n" + "=" * 120 + "\n")
    f.write("Comparison Key:\n")
    f.write("  GNR = Name-only search (traditional Senzing name matching - baseline)\n")
    f.write("  EMB = GNR + Embedding combined search (name matching + semantic similarity)\n")
    f.write("  Pos>Neg = PRIMARY METRIC: % where positive entity ranks above negative (relative ranking)\n")
    f.write("  Acc = Legacy top-1 accuracy metric (% where positive ranks #1)\n")
    f.write("  Δ = Delta (EMB - GNR); positive = embeddings help, negative = embeddings hurt\n")
    f.write("  Rescue Rate = % where embeddings found correct match that name-only missed\n")
    f.write("  NDCG@K = Normalized Discounted Cumulative Gain (ranking quality, 0.0-1.0)\n")
    f.write("=" * 120 + "\n")

    if output_file:
        f.close()
        print(f"✅ Comparison report saved to {output_file}")


def print_summary_table(results: list[dict[str, Any]]) -> None:
    """Print quick summary table to stdout."""
    if not results:
        print("No results to summarize")
        return

    print("\n" + "=" * 120)
    print("Quick Summary (sorted by EMB Pos>Neg rate)")
    print("=" * 120)
    print(f"{'Model':<35s} {'Test Set':<15s} {'Type':<10s} {'GNR Pos>Neg':>11s} {'EMB Pos>Neg':>11s} "
          f"{'GNR Acc':>8s} {'EMB Acc':>8s} {'Δ':>7s}")
    print("-" * 120)

    # Sort by embedding Pos>Neg rate (primary metric)
    for r in sorted(results, key=lambda x: (x.get('test_set', ''), x.get('model_type', ''),
                                            -x.get('metrics', {}).get('emb_relative_ranking', {}).get('positive_gt_negative_rate', 0))):
        model_name = Path(r.get('model_path', 'Unknown')).parent.name
        test_set = r.get('test_set', 'unknown')
        model_type = r.get('model_type', 'unknown')
        metrics = r.get('metrics', {})

        gnr_pos_gt_neg = metrics.get('gnr_relative_ranking', {}).get('positive_gt_negative_rate', 0.0)
        emb_pos_gt_neg = metrics.get('emb_relative_ranking', {}).get('positive_gt_negative_rate', 0.0)
        gnr_acc = metrics.get('gnr_accuracy', 0.0)
        emb_acc = metrics.get('emb_accuracy', 0.0)
        delta = emb_acc - gnr_acc

        print(f"{model_name:<35s} {test_set:<15s} {model_type:<10s} {gnr_pos_gt_neg:>10.1f}% {emb_pos_gt_neg:>10.1f}% "
              f"{gnr_acc:>7.2f}% {emb_acc:>7.2f}% {delta:>+6.2f}%")

    print("=" * 120 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sz_compare_models",
        description="Compare Senzing embedding model evaluation results"
    )

    parser.add_argument("result_files", nargs="+", help="JSON result files from sz_evaluate_model.py")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file for detailed report (default: stdout)")
    parser.add_argument("--summary-only", action="store_true",
                       help="Print quick summary table only")

    args = parser.parse_args()

    # Load results
    results = load_results(args.result_files)

    if not results:
        print("❌ No valid result files found", file=sys.stderr)
        sys.exit(1)

    print(f"📊 Loaded {len(results)} result file(s)")

    # Print reports
    if args.summary_only:
        print_summary_table(results)
    else:
        print_summary_table(results)
        print_comparison_report(results, args.output)
