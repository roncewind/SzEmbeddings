#!/usr/bin/env python3
"""
sz_validate_with_gnr_comparison.py - Validate embedding search with GNR score comparison.

For each query:
1. Run embedding search to get candidates
2. For each candidate, compute cosine similarity AND get GNR score via why_search
3. Compare rankings: does higher cosine correlate with higher GNR?

This measures how well our embedding model acts as a proxy for GNR scoring.
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_cases(filepath: str) -> list[dict]:
    """Load test cases from JSONL file."""
    cases = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_validation_with_gnr(
    test_cases: list[dict],
    engine,
    name_model,
    biz_model,
    max_candidates: int = 10,
    sample_size: int = None
):
    """
    Run validation comparing embedding cosine with GNR scores.

    Returns dict with correlation metrics and detailed results.
    """
    from senzing import SzEngineFlags

    if sample_size and sample_size < len(test_cases):
        import random
        random.seed(42)
        test_cases = random.sample(test_cases, sample_size)

    results = {
        'total_queries': len(test_cases),
        'queries_with_results': 0,
        'queries_with_multiple_results': 0,
        'all_cosines': [],
        'all_gnr_scores': [],
        'per_query_correlations': [],
        'by_record_type': defaultdict(lambda: {'cosines': [], 'gnr_scores': [], 'correlations': []}),
        'detailed_results': []
    }

    for i, case in enumerate(test_cases):
        # Determine query and model
        query_name = case.get('variant_query') or case.get('query_name') or case.get('name')
        record_type = case.get('record_type', 'PERSON')
        expected_record_id = case.get('expected_record_id') or case.get('record_id')

        if not query_name:
            continue

        model = name_model if record_type == 'PERSON' else biz_model

        # Get query embedding
        query_emb = model.encode([query_name], normalize_embeddings=True)[0]

        # Build search attributes
        if record_type == 'PERSON':
            search_attrs = {"NAME_FULL": query_name}
        else:
            search_attrs = {"NAME_ORG": query_name}

        try:
            # Search with feature scores to get GNR
            search_result = engine.search_by_attributes(
                json.dumps(search_attrs),
                SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
            )
            search_data = json.loads(search_result)

            entities = search_data.get('RESOLVED_ENTITIES', [])

            if not entities:
                continue

            results['queries_with_results'] += 1

            if len(entities) >= 2:
                results['queries_with_multiple_results'] += 1

            query_cosines = []
            query_gnr_scores = []
            query_details = []

            for entity in entities[:max_candidates]:
                entity_id = entity.get('ENTITY', {}).get('RESOLVED_ENTITY', {}).get('ENTITY_ID')

                # Get GNR score from match info
                match_info = entity.get('MATCH_INFO', {})
                feature_scores = match_info.get('FEATURE_SCORES', {})
                name_scores = feature_scores.get('NAME', [])
                gnr_score = name_scores[0].get('SCORE', 0) if name_scores else 0
                gnr_bucket = name_scores[0].get('SCORE_BUCKET', 'UNKNOWN') if name_scores else 'UNKNOWN'

                # Get entity names to compute cosine - names are in FEATURES.NAME
                entity_names = []
                resolved_entity = entity.get('ENTITY', {}).get('RESOLVED_ENTITY', {})
                features = resolved_entity.get('FEATURES', {})

                # Get names from NAME feature
                for name_feat in features.get('NAME', []):
                    n = name_feat.get('FEAT_DESC')
                    if n:
                        entity_names.append(n)
                    # Also check FEAT_DESC_VALUES for additional name variants
                    for val in name_feat.get('FEAT_DESC_VALUES', []):
                        v = val.get('FEAT_DESC')
                        if v and v not in entity_names:
                            entity_names.append(v)

                if entity_names:
                    # Compute max cosine with any of the entity's names
                    entity_embs = model.encode(entity_names[:10], normalize_embeddings=True)  # Limit for speed
                    cosine_scores = [float(np.dot(query_emb, e)) for e in entity_embs]
                    max_cosine = max(cosine_scores)
                    best_match_name = entity_names[cosine_scores.index(max_cosine)]

                    query_cosines.append(max_cosine)
                    query_gnr_scores.append(gnr_score)

                    query_details.append({
                        'entity_id': entity_id,
                        'cosine': max_cosine,
                        'gnr_score': gnr_score,
                        'gnr_bucket': gnr_bucket,
                        'matched_name': best_match_name
                    })

            if len(query_cosines) >= 2:
                # Compute per-query correlation
                corr, _ = spearmanr(query_cosines, query_gnr_scores)
                if not np.isnan(corr):
                    results['per_query_correlations'].append(corr)
                    results['by_record_type'][record_type]['correlations'].append(corr)

            # Accumulate all scores
            results['all_cosines'].extend(query_cosines)
            results['all_gnr_scores'].extend(query_gnr_scores)
            results['by_record_type'][record_type]['cosines'].extend(query_cosines)
            results['by_record_type'][record_type]['gnr_scores'].extend(query_gnr_scores)

            # Store detailed result
            results['detailed_results'].append({
                'query': query_name,
                'record_type': record_type,
                'expected_record_id': expected_record_id,
                'num_results': len(entities),
                'candidates': query_details
            })

        except Exception as e:
            logger.debug(f"Error processing query '{query_name}': {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(test_cases)}")

    return results


def compute_correlation_metrics(results: dict) -> dict:
    """Compute overall correlation metrics from results."""
    metrics = {}

    cosines = results['all_cosines']
    gnr_scores = results['all_gnr_scores']

    if len(cosines) >= 2:
        # Overall correlations
        spearman_r, spearman_p = spearmanr(cosines, gnr_scores)
        kendall_t, kendall_p = kendalltau(cosines, gnr_scores)

        metrics['overall'] = {
            'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else 0,
            'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else 1,
            'kendall_tau': float(kendall_t) if not np.isnan(kendall_t) else 0,
            'kendall_p': float(kendall_p) if not np.isnan(kendall_p) else 1,
            'total_pairs': len(cosines),
        }

        # Score distributions
        metrics['cosine_distribution'] = {
            'min': float(min(cosines)),
            'max': float(max(cosines)),
            'mean': float(np.mean(cosines)),
            'std': float(np.std(cosines)),
        }
        metrics['gnr_distribution'] = {
            'min': float(min(gnr_scores)),
            'max': float(max(gnr_scores)),
            'mean': float(np.mean(gnr_scores)),
            'std': float(np.std(gnr_scores)),
        }

    # Per-query correlation stats
    if results['per_query_correlations']:
        corrs = results['per_query_correlations']
        metrics['per_query'] = {
            'mean_correlation': float(np.mean(corrs)),
            'median_correlation': float(np.median(corrs)),
            'std_correlation': float(np.std(corrs)),
            'positive_correlation_rate': float(sum(1 for c in corrs if c > 0) / len(corrs)),
            'num_queries': len(corrs),
        }

    # By record type
    metrics['by_record_type'] = {}
    for rtype, data in results['by_record_type'].items():
        if len(data['cosines']) >= 2:
            sr, _ = spearmanr(data['cosines'], data['gnr_scores'])
            metrics['by_record_type'][rtype] = {
                'spearman_r': float(sr) if not np.isnan(sr) else 0,
                'num_pairs': len(data['cosines']),
                'mean_correlation': float(np.mean(data['correlations'])) if data['correlations'] else 0,
            }

    return metrics


def analyze_ranking_agreement(results: dict) -> dict:
    """Analyze how often cosine and GNR agree on rankings."""
    agreement_stats = {
        'total_comparisons': 0,
        'ranking_agreements': 0,
        'ranking_disagreements': 0,
        'ties': 0,
    }

    for detail in results['detailed_results']:
        candidates = detail.get('candidates', [])
        if len(candidates) < 2:
            continue

        # Compare all pairs of candidates
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1, c2 = candidates[i], candidates[j]

                cosine_order = np.sign(c1['cosine'] - c2['cosine'])
                gnr_order = np.sign(c1['gnr_score'] - c2['gnr_score'])

                agreement_stats['total_comparisons'] += 1

                if cosine_order == gnr_order:
                    if cosine_order == 0:
                        agreement_stats['ties'] += 1
                    else:
                        agreement_stats['ranking_agreements'] += 1
                else:
                    agreement_stats['ranking_disagreements'] += 1

    if agreement_stats['total_comparisons'] > 0:
        non_ties = agreement_stats['total_comparisons'] - agreement_stats['ties']
        if non_ties > 0:
            agreement_stats['agreement_rate'] = agreement_stats['ranking_agreements'] / non_ties
        else:
            agreement_stats['agreement_rate'] = 0

    return agreement_stats


def main():
    parser = argparse.ArgumentParser(
        description='Validate embedding search with GNR score comparison'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input test cases JSONL file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output results JSON file'
    )
    parser.add_argument(
        '--name_model_path',
        type=str,
        required=True,
        help='Path to personal names ONNX model'
    )
    parser.add_argument(
        '--biz_model_path',
        type=str,
        required=True,
        help='Path to business names ONNX model'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample size (for faster testing)'
    )
    parser.add_argument(
        '--max_candidates',
        type=int,
        default=10,
        help='Max candidates to score per query (default: 10)'
    )

    args = parser.parse_args()

    # Load models
    logger.info("Loading models...")
    sys.path.insert(0, str(Path(__file__).parent))
    from onnx_sentence_transformer import load_onnx_model

    name_model = load_onnx_model(args.name_model_path)
    biz_model = load_onnx_model(args.biz_model_path)
    logger.info(f"  Personal names: {name_model.embedding_dimension}d")
    logger.info(f"  Business names: {biz_model.embedding_dimension}d")

    # Initialize Senzing
    logger.info("Initializing Senzing...")
    from sz_utils import get_senzing_config
    from senzing_core import SzAbstractFactoryCore

    settings = get_senzing_config()
    sz_factory = SzAbstractFactoryCore("GNRValidation", settings, verbose_logging=0)
    engine = sz_factory.create_engine()

    # Load test cases
    logger.info(f"Loading test cases from {args.input}...")
    test_cases = load_test_cases(args.input)
    logger.info(f"  Loaded {len(test_cases)} test cases")

    # Run validation
    logger.info("Running validation with GNR comparison...")
    start_time = time.time()

    results = run_validation_with_gnr(
        test_cases,
        engine,
        name_model,
        biz_model,
        max_candidates=args.max_candidates,
        sample_size=args.sample
    )

    elapsed = time.time() - start_time
    logger.info(f"  Completed in {elapsed:.1f}s")

    # Compute metrics
    logger.info("Computing correlation metrics...")
    metrics = compute_correlation_metrics(results)
    ranking_agreement = analyze_ranking_agreement(results)

    # Print summary
    print("\n" + "=" * 70)
    print("EMBEDDING vs GNR RANKING CORRELATION")
    print("=" * 70)

    print(f"\nQueries processed: {results['total_queries']}")
    print(f"Queries with results: {results['queries_with_results']}")
    print(f"Queries with 2+ results: {results['queries_with_multiple_results']}")

    if 'overall' in metrics:
        print(f"\nOverall Correlation (across {metrics['overall']['total_pairs']} candidate pairs):")
        print(f"  Spearman:    {metrics['overall']['spearman_r']:+.4f} (p={metrics['overall']['spearman_p']:.2e})")
        print(f"  Kendall tau: {metrics['overall']['kendall_tau']:+.4f} (p={metrics['overall']['kendall_p']:.2e})")

    if 'per_query' in metrics:
        print(f"\nPer-Query Correlation ({metrics['per_query']['num_queries']} queries with 2+ results):")
        print(f"  Mean:   {metrics['per_query']['mean_correlation']:+.4f}")
        print(f"  Median: {metrics['per_query']['median_correlation']:+.4f}")
        print(f"  Positive correlation rate: {100*metrics['per_query']['positive_correlation_rate']:.1f}%")

    if ranking_agreement['total_comparisons'] > 0:
        print(f"\nRanking Agreement (pairwise comparisons):")
        print(f"  Total comparisons: {ranking_agreement['total_comparisons']}")
        print(f"  Agreements: {ranking_agreement['ranking_agreements']} ({100*ranking_agreement.get('agreement_rate', 0):.1f}%)")
        print(f"  Disagreements: {ranking_agreement['ranking_disagreements']}")
        print(f"  Ties: {ranking_agreement['ties']}")

    if 'by_record_type' in metrics:
        print(f"\nBy Record Type:")
        for rtype, data in metrics['by_record_type'].items():
            print(f"  {rtype}: Spearman={data['spearman_r']:+.4f} ({data['num_pairs']} pairs)")

    if 'cosine_distribution' in metrics:
        cd = metrics['cosine_distribution']
        gd = metrics['gnr_distribution']
        print(f"\nScore Distributions:")
        print(f"  Cosine: {cd['min']:.3f} - {cd['max']:.3f} (mean: {cd['mean']:.3f})")
        print(f"  GNR:    {gd['min']:.0f} - {gd['max']:.0f} (mean: {gd['mean']:.1f})")

    print("=" * 70)

    # Save results
    output_data = {
        'metadata': {
            'input_file': args.input,
            'name_model': args.name_model_path,
            'biz_model': args.biz_model_path,
            'sample_size': args.sample,
            'max_candidates': args.max_candidates,
            'elapsed_seconds': elapsed,
        },
        'summary': {
            'total_queries': results['total_queries'],
            'queries_with_results': results['queries_with_results'],
            'queries_with_multiple_results': results['queries_with_multiple_results'],
        },
        'correlation_metrics': metrics,
        'ranking_agreement': ranking_agreement,
        # Don't save detailed results to keep file size manageable
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
