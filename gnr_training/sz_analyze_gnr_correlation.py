#!/usr/bin/env python3
"""
sz_analyze_gnr_correlation.py - Analyze cosine vs GNR score correlation.

Computes:
- Pearson/Spearman correlation between cosine and GNR scores
- Score distributions by GNR bucket (CLOSE, LIKELY, PLAUSIBLE, etc.)
- ROC-AUC: Can cosine separate same-entity vs different-entity?
- Problem cases: high cosine + low GNR (false positives)

Usage:
    python sz_analyze_gnr_correlation.py \
        --input data/gnr_alignment/pairs_with_gnr_scores.jsonl \
        --output results/gnr_correlation_baseline.json

    # Compare with baseline
    python sz_analyze_gnr_correlation.py \
        --input data/gnr_alignment/pairs_with_gnr_scores_v2.jsonl \
        --baseline results/gnr_correlation_baseline.json \
        --output results/gnr_correlation_after_training.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pairs(filepath: str) -> list[dict]:
    """Load scored pairs from JSONL file."""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def compute_roc_auc(y_true: list[int], y_scores: list[float]) -> float:
    """Compute ROC AUC score manually (no sklearn dependency)."""
    if len(set(y_true)) < 2:
        return 0.5  # Can't compute AUC with single class

    # Sort by scores descending
    sorted_indices = np.argsort(-np.array(y_scores))
    y_true_sorted = np.array(y_true)[sorted_indices]

    # Count positives and negatives
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Compute AUC using trapezoidal rule
    tpr_prev = 0
    fpr_prev = 0
    auc = 0
    tp = 0
    fp = 0

    prev_score = None
    for i, (label, score) in enumerate(zip(y_true_sorted, np.array(y_scores)[sorted_indices])):
        if prev_score is not None and score != prev_score:
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev = tpr
            fpr_prev = fpr

        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    # Final point
    tpr = tp / n_pos
    fpr = fp / n_neg
    auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2

    return auc


def analyze_correlation(pairs: list[dict]) -> dict:
    """Analyze correlation between cosine and GNR scores."""

    # Filter pairs with both scores
    valid_pairs = [
        p for p in pairs
        if p.get('gnr_score') is not None and p.get('cosine_sim') is not None
    ]

    if not valid_pairs:
        logger.error("No pairs with both GNR and cosine scores")
        return {}

    logger.info(f"Analyzing {len(valid_pairs):,} pairs with both scores")

    gnr_scores = np.array([p['gnr_score'] for p in valid_pairs])
    cosine_scores = np.array([p['cosine_sim'] for p in valid_pairs])
    same_entity = np.array([1 if p['same_entity'] else 0 for p in valid_pairs])

    results = {
        'total_pairs': len(pairs),
        'valid_pairs': len(valid_pairs),
        'pairs_with_gnr': sum(1 for p in pairs if p.get('gnr_score') is not None),
        'pairs_with_cosine': sum(1 for p in pairs if p.get('cosine_sim') is not None),
    }

    # Correlation metrics
    pearson_r, pearson_p = stats.pearsonr(cosine_scores, gnr_scores)
    spearman_r, spearman_p = stats.spearmanr(cosine_scores, gnr_scores)

    results['correlation'] = {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
    }

    logger.info(f"Pearson correlation: {pearson_r:.4f} (p={pearson_p:.2e})")
    logger.info(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})")

    # Score distributions
    results['gnr_distribution'] = {
        'min': float(gnr_scores.min()),
        'max': float(gnr_scores.max()),
        'mean': float(gnr_scores.mean()),
        'std': float(gnr_scores.std()),
        'median': float(np.median(gnr_scores)),
    }

    results['cosine_distribution'] = {
        'min': float(cosine_scores.min()),
        'max': float(cosine_scores.max()),
        'mean': float(cosine_scores.mean()),
        'std': float(cosine_scores.std()),
        'median': float(np.median(cosine_scores)),
    }

    # Distributions by GNR bucket
    bucket_stats = defaultdict(lambda: {'count': 0, 'cosine_values': []})
    for p in valid_pairs:
        bucket = p.get('gnr_bucket', 'UNKNOWN')
        bucket_stats[bucket]['count'] += 1
        bucket_stats[bucket]['cosine_values'].append(p['cosine_sim'])

    results['by_gnr_bucket'] = {}
    for bucket, data in bucket_stats.items():
        cosines = data['cosine_values']
        results['by_gnr_bucket'][bucket] = {
            'count': data['count'],
            'cosine_mean': float(np.mean(cosines)),
            'cosine_std': float(np.std(cosines)),
            'cosine_min': float(min(cosines)),
            'cosine_max': float(max(cosines)),
        }
        logger.info(f"Bucket {bucket}: {data['count']:,} pairs, cosine mean={np.mean(cosines):.3f}")

    # Distributions by same_entity
    same_entity_pairs = [p for p in valid_pairs if p['same_entity']]
    diff_entity_pairs = [p for p in valid_pairs if not p['same_entity']]

    results['by_entity_relationship'] = {
        'same_entity': {
            'count': len(same_entity_pairs),
            'gnr_mean': float(np.mean([p['gnr_score'] for p in same_entity_pairs])) if same_entity_pairs else None,
            'cosine_mean': float(np.mean([p['cosine_sim'] for p in same_entity_pairs])) if same_entity_pairs else None,
        },
        'different_entity': {
            'count': len(diff_entity_pairs),
            'gnr_mean': float(np.mean([p['gnr_score'] for p in diff_entity_pairs])) if diff_entity_pairs else None,
            'cosine_mean': float(np.mean([p['cosine_sim'] for p in diff_entity_pairs])) if diff_entity_pairs else None,
        }
    }

    if same_entity_pairs:
        logger.info(f"Same-entity: {len(same_entity_pairs):,} pairs, "
                   f"GNR={np.mean([p['gnr_score'] for p in same_entity_pairs]):.1f}, "
                   f"cosine={np.mean([p['cosine_sim'] for p in same_entity_pairs]):.3f}")

    if diff_entity_pairs:
        logger.info(f"Diff-entity: {len(diff_entity_pairs):,} pairs, "
                   f"GNR={np.mean([p['gnr_score'] for p in diff_entity_pairs]):.1f}, "
                   f"cosine={np.mean([p['cosine_sim'] for p in diff_entity_pairs]):.3f}")

    # ROC-AUC: Can cosine separate same-entity from different-entity?
    if same_entity_pairs and diff_entity_pairs:
        roc_auc = compute_roc_auc(same_entity.tolist(), cosine_scores.tolist())
        results['classification'] = {
            'roc_auc_cosine': float(roc_auc),
        }
        logger.info(f"ROC-AUC (cosine separates same vs diff entity): {roc_auc:.4f}")

    # Problem cases analysis
    # High cosine but low GNR (false positives)
    high_cosine_low_gnr = [
        p for p in valid_pairs
        if p['cosine_sim'] >= 0.70 and p['gnr_score'] < 50
    ]

    # Low cosine but high GNR (missed matches)
    low_cosine_high_gnr = [
        p for p in valid_pairs
        if p['cosine_sim'] < 0.50 and p['gnr_score'] >= 85
    ]

    results['problem_cases'] = {
        'high_cosine_low_gnr': {
            'count': len(high_cosine_low_gnr),
            'percentage': 100 * len(high_cosine_low_gnr) / len(valid_pairs),
            'examples': [
                {
                    'name_a': p['name_a'],
                    'name_b': p['name_b'],
                    'cosine': p['cosine_sim'],
                    'gnr': p['gnr_score'],
                    'same_entity': p['same_entity'],
                }
                for p in high_cosine_low_gnr[:10]  # Top 10 examples
            ]
        },
        'low_cosine_high_gnr': {
            'count': len(low_cosine_high_gnr),
            'percentage': 100 * len(low_cosine_high_gnr) / len(valid_pairs),
            'examples': [
                {
                    'name_a': p['name_a'],
                    'name_b': p['name_b'],
                    'cosine': p['cosine_sim'],
                    'gnr': p['gnr_score'],
                    'same_entity': p['same_entity'],
                }
                for p in low_cosine_high_gnr[:10]  # Top 10 examples
            ]
        }
    }

    logger.info(f"High cosine (≥0.70) + low GNR (<50): {len(high_cosine_low_gnr):,} "
               f"({100*len(high_cosine_low_gnr)/len(valid_pairs):.1f}%)")
    logger.info(f"Low cosine (<0.50) + high GNR (≥85): {len(low_cosine_high_gnr):,} "
               f"({100*len(low_cosine_high_gnr)/len(valid_pairs):.1f}%)")

    # Breakdown by pair_type
    pair_type_stats = defaultdict(lambda: {'count': 0, 'gnr': [], 'cosine': []})
    for p in valid_pairs:
        pt = p.get('pair_type', 'unknown')
        pair_type_stats[pt]['count'] += 1
        pair_type_stats[pt]['gnr'].append(p['gnr_score'])
        pair_type_stats[pt]['cosine'].append(p['cosine_sim'])

    results['by_pair_type'] = {}
    for pt, data in pair_type_stats.items():
        results['by_pair_type'][pt] = {
            'count': data['count'],
            'gnr_mean': float(np.mean(data['gnr'])),
            'cosine_mean': float(np.mean(data['cosine'])),
        }
        logger.info(f"Pair type '{pt}': {data['count']:,} pairs, "
                   f"GNR={np.mean(data['gnr']):.1f}, cosine={np.mean(data['cosine']):.3f}")

    return results


def compare_with_baseline(current: dict, baseline: dict) -> dict:
    """Compare current results with baseline."""
    comparison = {}

    # Correlation improvement
    if 'correlation' in current and 'correlation' in baseline:
        curr_spearman = current['correlation']['spearman_r']
        base_spearman = baseline['correlation']['spearman_r']
        comparison['spearman_improvement'] = {
            'baseline': base_spearman,
            'current': curr_spearman,
            'change': curr_spearman - base_spearman,
            'percent_change': 100 * (curr_spearman - base_spearman) / abs(base_spearman) if base_spearman != 0 else 0,
        }
        logger.info(f"Spearman: {base_spearman:.4f} → {curr_spearman:.4f} "
                   f"({comparison['spearman_improvement']['percent_change']:+.1f}%)")

    # ROC-AUC improvement
    if 'classification' in current and 'classification' in baseline:
        curr_auc = current['classification']['roc_auc_cosine']
        base_auc = baseline['classification']['roc_auc_cosine']
        comparison['roc_auc_improvement'] = {
            'baseline': base_auc,
            'current': curr_auc,
            'change': curr_auc - base_auc,
        }
        logger.info(f"ROC-AUC: {base_auc:.4f} → {curr_auc:.4f} ({curr_auc - base_auc:+.4f})")

    # Problem cases improvement
    if 'problem_cases' in current and 'problem_cases' in baseline:
        curr_fp = current['problem_cases']['high_cosine_low_gnr']['percentage']
        base_fp = baseline['problem_cases']['high_cosine_low_gnr']['percentage']
        curr_fn = current['problem_cases']['low_cosine_high_gnr']['percentage']
        base_fn = baseline['problem_cases']['low_cosine_high_gnr']['percentage']

        comparison['false_positive_improvement'] = {
            'baseline': base_fp,
            'current': curr_fp,
            'change': curr_fp - base_fp,
        }
        comparison['missed_match_improvement'] = {
            'baseline': base_fn,
            'current': curr_fn,
            'change': curr_fn - base_fn,
        }
        logger.info(f"False positives: {base_fp:.1f}% → {curr_fp:.1f}% ({curr_fp - base_fp:+.1f}%)")
        logger.info(f"Missed matches: {base_fn:.1f}% → {curr_fn:.1f}% ({curr_fn - base_fn:+.1f}%)")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        prog='sz_analyze_gnr_correlation',
        description='Analyze cosine vs GNR score correlation'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to scored pairs JSONL file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output analysis JSON file path'
    )
    parser.add_argument(
        '--baseline', '-b',
        type=str,
        default=None,
        help='Path to baseline analysis JSON for comparison'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Load pairs
    logger.info(f"Loading pairs from {args.input}")
    pairs = load_pairs(args.input)
    logger.info(f"Loaded {len(pairs):,} pairs")

    # Analyze
    results = analyze_correlation(pairs)

    # Compare with baseline if provided
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            logger.info(f"Comparing with baseline: {args.baseline}")
            with open(baseline_path, 'r') as f:
                baseline = json.load(f)
            results['comparison'] = compare_with_baseline(results, baseline)
        else:
            logger.warning(f"Baseline file not found: {args.baseline}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing to {args.output}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logger.info("Done!")


if __name__ == '__main__':
    main()
