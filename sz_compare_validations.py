#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Compare Senzing production validation results.
#
# Reads JSON result files from sz_validate_production.py and generates
# side-by-side comparison reports.
#
# Usage:
#   python sz_compare_validations.py validation_results/*.json
#   python sz_compare_validations.py --output comparison.txt validation_results/*.json
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_results(file_paths: list[str]) -> list[dict[str, Any]]:
    """Load validation results from JSON files."""
    results = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Add source filename for reference
                data['_source_file'] = Path(file_path).name
                results.append(data)
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}", file=sys.stderr)
    return results


def extract_test_set_name(result: dict[str, Any]) -> str:
    """Extract test set name from result metadata."""
    metadata = result.get('metadata', {})
    input_file = metadata.get('input_file', 'unknown')
    return Path(input_file).stem


def extract_model_identifier(result: dict[str, Any]) -> str:
    """Extract model identifier from result metadata."""
    metadata = result.get('metadata', {})

    # Try to get business model path (most results will have this)
    biz_model = metadata.get('biz_model_path', '')
    if biz_model:
        # Extract model directory name (e.g., "phase9b_labse")
        return Path(biz_model).parent.name

    # Fallback to name model
    name_model = metadata.get('name_model_path', '')
    if name_model:
        return Path(name_model).parent.name

    return 'unknown'


def print_comparison_report(results: list[dict[str, Any]], output_file: str | None = None) -> None:
    """Print formatted comparison report."""
    if not results:
        print("No results to compare")
        return

    f = open(output_file, 'w') if output_file else sys.stdout

    # Group by test set
    by_test_set: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        test_set = extract_test_set_name(r)
        if test_set not in by_test_set:
            by_test_set[test_set] = []
        by_test_set[test_set].append(r)

    # Print report header
    f.write("=" * 120 + "\n")
    f.write("Production Validation Comparison Report\n")
    f.write("=" * 120 + "\n\n")

    for test_set, test_results in sorted(by_test_set.items()):
        f.write(f"\n{test_set.upper()} Test Set\n")
        f.write("-" * 120 + "\n\n")

        # Sort by embedding Recall@10 (descending)
        test_results.sort(
            key=lambda x: x.get('metrics', {}).get('embedding', {}).get('recall_at_k', {}).get('10', 0),
            reverse=True
        )

        # Comparison table
        f.write(f"{'Model':<40s} {'Cases':>8s} {'Name R@1':>10s} {'Emb R@1':>10s} "
               f"{'Emb R@10':>10s} {'Not Found':>11s} {'MRR':>8s}\n")
        f.write("-" * 120 + "\n")

        for r in test_results:
            model_id = extract_model_identifier(r)
            source_file = r.get('_source_file', 'unknown')
            metadata = r.get('metadata', {})
            metrics = r.get('metrics', {})

            total_cases = metadata.get('total_test_cases', 0)

            name_metrics = metrics.get('name_only', {})
            emb_metrics = metrics.get('embedding', {})

            name_r1 = name_metrics.get('recall_at_k', {}).get('1', 0.0)
            emb_r1 = emb_metrics.get('recall_at_k', {}).get('1', 0.0)
            emb_r10 = emb_metrics.get('recall_at_k', {}).get('10', 0.0)
            emb_not_found = emb_metrics.get('not_found_rate', 0.0)
            emb_mrr = emb_metrics.get('mrr', 0.0)

            # Use source filename if model_id is unknown
            display_name = model_id if model_id != 'unknown' else source_file[:40]

            f.write(f"{display_name:<40s} {total_cases:>8,} {name_r1:>9.1f}% {emb_r1:>9.1f}% "
                   f"{emb_r10:>9.1f}% {emb_not_found:>10.1f}% {emb_mrr:>8.4f}\n")

        # Detailed comparison for top result
        if test_results:
            f.write("\n\nDetailed Metrics (Top Result):\n")
            f.write("-" * 120 + "\n")

            top = test_results[0]
            metrics = top.get('metrics', {})
            impact = metrics.get('impact', {})

            f.write(f"\nModel: {extract_model_identifier(top)}\n")
            f.write(f"Source: {top.get('_source_file', 'unknown')}\n")

            f.write("\nOverall Metrics (Embedding Mode):\n")
            emb = metrics.get('embedding', {})
            f.write(f"  Recall@1:       {emb.get('recall_at_k', {}).get('1', 0):.1f}%\n")
            f.write(f"  Recall@5:       {emb.get('recall_at_k', {}).get('5', 0):.1f}%\n")
            f.write(f"  Recall@10:      {emb.get('recall_at_k', {}).get('10', 0):.1f}%\n")
            f.write(f"  Recall@100:     {emb.get('recall_at_k', {}).get('100', 0):.1f}%\n")
            f.write(f"  Not Found Rate: {emb.get('not_found_rate', 0):.1f}%\n")
            f.write(f"  MRR:            {emb.get('mrr', 0):.4f}\n")
            f.write(f"  Mean Rank:      {emb.get('mean_rank', 0):.1f}\n")

            f.write("\nEmbedding Impact:\n")
            f.write(f"  Improvement Rate: {impact.get('improvement_rate', 0):.1f}%  "
                   f"({impact.get('improvement_count', 0):,} cases)\n")
            f.write(f"  Degradation Rate: {impact.get('degradation_rate', 0):.1f}%  "
                   f"({impact.get('degradation_count', 0):,} cases)\n")

            # Threshold recommendations
            if 'threshold_recommendations' in top and top['threshold_recommendations']:
                f.write("\nRecommended Thresholds (Balanced):\n")
                balanced = top['threshold_recommendations'].get('balanced', {})
                cfg = balanced.get('cfg_cfrtn', {})
                bal_metrics = balanced.get('metrics', {})

                f.write(f"  closeScore:       {cfg.get('closeScore', 0):>3}\n")
                f.write(f"  Expected Recall:  {bal_metrics.get('recall', 0):.1f}%\n")
                f.write(f"  Expected Precision: {bal_metrics.get('precision', 0):.1f}%\n")
                f.write(f"  F1 Score:         {bal_metrics.get('f1', 0):.2f}\n")

            # PostgreSQL metrics if available
            if 'postgresql_metrics' in top and top['postgresql_metrics']:
                f.write("\nPostgreSQL Validation:\n")
                pg = top['postgresql_metrics']
                f.write(f"  Total Queries:          {pg.get('total_queries', 0):,}\n")
                f.write(f"  Perfect Sibling Recall: {pg.get('perfect_sibling_recall_rate', 0):.1f}%\n")
                f.write(f"  Avg Sibling Recall:     {pg.get('avg_sibling_recall', 0):.1f}%\n")
                f.write(f"  Avg Query Time:         {pg.get('avg_query_time_ms', 0):.2f} ms\n")

            # Breakdowns by record type
            if 'breakdowns' in top and 'by_record_type' in top['breakdowns']:
                f.write("\nBreakdown by Record Type:\n")
                for rec_type, rec_metrics in sorted(top['breakdowns']['by_record_type'].items()):
                    f.write(f"  {rec_type:<20s}: "
                           f"R@1={rec_metrics.get('recall_at_k', {}).get('1', 0):.1f}%, "
                           f"R@10={rec_metrics.get('recall_at_k', {}).get('10', 0):.1f}%, "
                           f"Not Found={rec_metrics.get('not_found_rate', 0):.1f}%\n")

            f.write("\n")

    f.write("\n" + "=" * 120 + "\n")
    f.write("Comparison Key:\n")
    f.write("  Name R@1 = Name-only search Recall@1 (baseline)\n")
    f.write("  Emb R@1 = Embedding search Recall@1\n")
    f.write("  Emb R@10 = Embedding search Recall@10 (PRIMARY METRIC)\n")
    f.write("  Not Found = % where expected entity not in results\n")
    f.write("  MRR = Mean Reciprocal Rank (higher is better)\n")
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
    print("Quick Summary (sorted by Embedding Recall@10)")
    print("=" * 120)
    print(f"{'Model/File':<40s} {'Test Set':<25s} {'Cases':>8s} {'Emb R@10':>10s} "
          f"{'Not Found':>11s} {'MRR':>8s}")
    print("-" * 120)

    # Sort by embedding Recall@10 (descending)
    for r in sorted(results, key=lambda x: (
        extract_test_set_name(x),
        -x.get('metrics', {}).get('embedding', {}).get('recall_at_k', {}).get('10', 0)
    )):
        model_id = extract_model_identifier(r)
        test_set = extract_test_set_name(r)
        source_file = r.get('_source_file', 'unknown')

        display_name = model_id if model_id != 'unknown' else source_file[:40]

        metadata = r.get('metadata', {})
        metrics = r.get('metrics', {})
        emb = metrics.get('embedding', {})

        total_cases = metadata.get('total_test_cases', 0)
        emb_r10 = emb.get('recall_at_k', {}).get('10', 0.0)
        emb_not_found = emb.get('not_found_rate', 0.0)
        emb_mrr = emb.get('mrr', 0.0)

        print(f"{display_name:<40s} {test_set:<25s} {total_cases:>8,} {emb_r10:>9.1f}% "
              f"{emb_not_found:>10.1f}% {emb_mrr:>8.4f}")

    print("=" * 120 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sz_compare_validations",
        description="Compare Senzing production validation results"
    )

    parser.add_argument("result_files", nargs="+",
                       help="JSON result files from sz_validate_production.py")
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
