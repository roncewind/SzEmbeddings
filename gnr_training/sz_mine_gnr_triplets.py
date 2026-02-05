#!/usr/bin/env python3
"""
sz_mine_gnr_triplets.py - Mine triplets from scored pairs for training.

Mining logic:
For each anchor name:
  - Positive: name from SAME entity with GNR ≥ positive_gnr_min
  - Negative: name from DIFFERENT entity where:
    - GNR < negative_gnr_max (confirmed different)
    - cosine > negative_cosine_min (embedding was fooled - hard negative)

Usage:
    python sz_mine_gnr_triplets.py \
        --input data/gnr_alignment/pairs_with_gnr_scores.jsonl \
        --output data/gnr_alignment/triplets_gnr_aligned.jsonl \
        --positive_gnr_min 85 \
        --negative_gnr_max 50 \
        --negative_cosine_min 0.50 \
        --max_triplets_per_anchor 5
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

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


def mine_triplets(
    pairs: list[dict],
    positive_gnr_min: int = 85,
    negative_gnr_max: int = 50,
    negative_cosine_min: float = 0.50,
    max_triplets_per_anchor: int = 5,
    seed: int = 42,
    include_null_gnr: bool = False
) -> list[dict]:
    """Mine triplets from scored pairs.

    Args:
        pairs: List of scored pairs with gnr_score, cosine_sim, same_entity
        positive_gnr_min: Minimum GNR score for positive pairs
        negative_gnr_max: Maximum GNR score for negative pairs
        negative_cosine_min: Minimum cosine for hard negatives (model was fooled)
        max_triplets_per_anchor: Maximum triplets per anchor name
        seed: Random seed for reproducibility
        include_null_gnr: If True, include pairs with null GNR:
            - Within-entity null GNR → positives (GNR couldn't score, but same entity)
            - Cross-entity null GNR → negatives (GNR confirms no match)

    Returns:
        List of triplets {anchor, positive, negative}
    """
    random.seed(seed)

    # Separate pairs by GNR availability
    pairs_with_gnr = [p for p in pairs if p.get('gnr_score') is not None and p.get('cosine_sim') is not None]
    pairs_null_gnr = [p for p in pairs if p.get('gnr_score') is None and p.get('cosine_sim') is not None]

    logger.info(f"Pairs with GNR score: {len(pairs_with_gnr):,}")
    logger.info(f"Pairs with NULL GNR: {len(pairs_null_gnr):,}")

    if include_null_gnr:
        null_same_entity = [p for p in pairs_null_gnr if p.get('same_entity')]
        null_diff_entity = [p for p in pairs_null_gnr if not p.get('same_entity')]
        logger.info(f"  NULL GNR same-entity (will be positives): {len(null_same_entity):,}")
        logger.info(f"  NULL GNR diff-entity (will be negatives): {len(null_diff_entity):,}")

    # Build name -> entity_id mapping from all pairs
    name_to_entity = {}
    for p in pairs:
        name_a = p['name_a']
        name_b = p['name_b']
        entity_a = p.get('entity_a')
        entity_b = p.get('entity_b')
        if entity_a:
            name_to_entity[name_a] = entity_a
        if entity_b:
            name_to_entity[name_b] = entity_b

    logger.info(f"Built name->entity mapping for {len(name_to_entity):,} names")

    # Build indexes
    # For each name, track positive partners (same entity, high GNR or null GNR)
    # and hard negative partners (different entity, high cosine but low/null GNR)

    positive_partners = defaultdict(list)  # name -> [(partner_name, gnr_score, cosine)]
    negative_partners = defaultdict(list)  # name -> [(partner_name, gnr_score, cosine)]

    # Process pairs with GNR scores
    for p in pairs_with_gnr:
        name_a = p['name_a']
        name_b = p['name_b']
        gnr = p['gnr_score']
        cosine = p['cosine_sim']
        same_entity = p['same_entity']

        if same_entity and gnr >= positive_gnr_min:
            # Good positive pair (both directions)
            positive_partners[name_a].append((name_b, gnr, cosine))
            positive_partners[name_b].append((name_a, gnr, cosine))

        elif not same_entity and gnr < negative_gnr_max and cosine >= negative_cosine_min:
            # Hard negative (both directions)
            negative_partners[name_a].append((name_b, gnr, cosine))
            negative_partners[name_b].append((name_a, gnr, cosine))

    # Process pairs with NULL GNR if enabled
    if include_null_gnr:
        null_positive_count = 0
        null_negative_count = 0

        for p in pairs_null_gnr:
            name_a = p['name_a']
            name_b = p['name_b']
            cosine = p['cosine_sim']
            same_entity = p['same_entity']

            if same_entity:
                # Within-entity null GNR → treat as positive (GNR couldn't score cross-script)
                # Use GNR=100 as placeholder since they're same entity
                positive_partners[name_a].append((name_b, 100, cosine))
                positive_partners[name_b].append((name_a, 100, cosine))
                null_positive_count += 1

            elif cosine >= negative_cosine_min:
                # Cross-entity null GNR → treat as negative (GNR confirms no match)
                # Use GNR=0 as placeholder since they don't match
                negative_partners[name_a].append((name_b, 0, cosine))
                negative_partners[name_b].append((name_a, 0, cosine))
                null_negative_count += 1

        logger.info(f"Added from NULL GNR: {null_positive_count:,} positives, {null_negative_count:,} negatives")

    logger.info(f"Names with positive partners: {len(positive_partners):,}")
    logger.info(f"Names with hard negative partners: {len(negative_partners):,}")

    # Count potential positives and negatives
    total_positives = sum(len(v) for v in positive_partners.values())
    total_negatives = sum(len(v) for v in negative_partners.values())
    logger.info(f"Total positive partnerships: {total_positives:,}")
    logger.info(f"Total hard negative partnerships: {total_negatives:,}")

    # Mine triplets
    triplets = []
    names_with_both = set(positive_partners.keys()) & set(negative_partners.keys())
    logger.info(f"Names with both positive and negative partners: {len(names_with_both):,}")

    for anchor in names_with_both:
        positives = positive_partners[anchor]
        negatives = negative_partners[anchor]

        # Sort positives by GNR descending (prefer strongest matches)
        positives.sort(key=lambda x: -x[1])

        # Sort negatives by cosine descending (prefer hardest negatives)
        negatives.sort(key=lambda x: -x[2])

        # Generate triplets (up to max_triplets_per_anchor)
        triplet_count = 0
        for pos_name, pos_gnr, pos_cosine in positives:
            if triplet_count >= max_triplets_per_anchor:
                break

            for neg_name, neg_gnr, neg_cosine in negatives:
                if triplet_count >= max_triplets_per_anchor:
                    break

                triplet = {
                    'anchor': anchor,
                    'positive': pos_name,
                    'negative': neg_name,
                    'anchor_group': name_to_entity.get(anchor, ''),
                    'positive_gnr': pos_gnr,
                    'positive_cosine': pos_cosine,
                    'negative_gnr': neg_gnr,
                    'negative_cosine': neg_cosine,
                }
                triplets.append(triplet)
                triplet_count += 1

    logger.info(f"Generated {len(triplets):,} triplets")

    # Shuffle triplets
    random.shuffle(triplets)

    return triplets


def analyze_triplets(triplets: list[dict]) -> dict:
    """Analyze triplet statistics."""
    if not triplets:
        return {}

    stats = {
        'total_triplets': len(triplets),
        'unique_anchors': len(set(t['anchor'] for t in triplets)),
        'unique_positives': len(set(t['positive'] for t in triplets)),
        'unique_negatives': len(set(t['negative'] for t in triplets)),
    }

    # GNR score distributions
    pos_gnr = [t['positive_gnr'] for t in triplets]
    neg_gnr = [t['negative_gnr'] for t in triplets]
    pos_cosine = [t['positive_cosine'] for t in triplets]
    neg_cosine = [t['negative_cosine'] for t in triplets]

    import numpy as np

    stats['positive_gnr'] = {
        'min': min(pos_gnr),
        'max': max(pos_gnr),
        'mean': float(np.mean(pos_gnr)),
    }
    stats['negative_gnr'] = {
        'min': min(neg_gnr),
        'max': max(neg_gnr),
        'mean': float(np.mean(neg_gnr)),
    }
    stats['positive_cosine'] = {
        'min': float(min(pos_cosine)),
        'max': float(max(pos_cosine)),
        'mean': float(np.mean(pos_cosine)),
    }
    stats['negative_cosine'] = {
        'min': float(min(neg_cosine)),
        'max': float(max(neg_cosine)),
        'mean': float(np.mean(neg_cosine)),
    }

    # Margin analysis (positive cosine - negative cosine)
    margins = [p - n for p, n in zip(pos_cosine, neg_cosine)]
    stats['cosine_margin'] = {
        'min': float(min(margins)),
        'max': float(max(margins)),
        'mean': float(np.mean(margins)),
        'negative_margin_count': sum(1 for m in margins if m < 0),
        'negative_margin_pct': 100 * sum(1 for m in margins if m < 0) / len(margins),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        prog='sz_mine_gnr_triplets',
        description='Mine triplets from scored pairs for training'
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
        help='Output triplets JSONL file path'
    )
    parser.add_argument(
        '--positive_gnr_min',
        type=int,
        default=85,
        help='Minimum GNR score for positives (default: 85)'
    )
    parser.add_argument(
        '--negative_gnr_max',
        type=int,
        default=50,
        help='Maximum GNR score for negatives (default: 50)'
    )
    parser.add_argument(
        '--negative_cosine_min',
        type=float,
        default=0.50,
        help='Minimum cosine for hard negatives (default: 0.50)'
    )
    parser.add_argument(
        '--max_triplets_per_anchor',
        type=int,
        default=5,
        help='Max triplets per anchor name (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--stats_output',
        type=str,
        default=None,
        help='Output stats JSON file (optional)'
    )
    parser.add_argument(
        '--include_null_gnr',
        action='store_true',
        help='Include pairs with null GNR: within-entity as positives, cross-entity as negatives'
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

    # Mine triplets
    logger.info("Mining triplets")
    logger.info(f"  Positive GNR min: {args.positive_gnr_min}")
    logger.info(f"  Negative GNR max: {args.negative_gnr_max}")
    logger.info(f"  Negative cosine min: {args.negative_cosine_min}")
    logger.info(f"  Max triplets per anchor: {args.max_triplets_per_anchor}")
    logger.info(f"  Include null GNR pairs: {args.include_null_gnr}")

    triplets = mine_triplets(
        pairs,
        positive_gnr_min=args.positive_gnr_min,
        negative_gnr_max=args.negative_gnr_max,
        negative_cosine_min=args.negative_cosine_min,
        max_triplets_per_anchor=args.max_triplets_per_anchor,
        seed=args.seed,
        include_null_gnr=args.include_null_gnr
    )

    # Analyze triplets
    stats = analyze_triplets(triplets)

    logger.info("Triplet statistics:")
    logger.info(f"  Total triplets: {stats.get('total_triplets', 0):,}")
    logger.info(f"  Unique anchors: {stats.get('unique_anchors', 0):,}")
    logger.info(f"  Unique positives: {stats.get('unique_positives', 0):,}")
    logger.info(f"  Unique negatives: {stats.get('unique_negatives', 0):,}")

    if 'positive_gnr' in stats:
        logger.info(f"  Positive GNR: {stats['positive_gnr']['mean']:.1f} "
                   f"(range: {stats['positive_gnr']['min']}-{stats['positive_gnr']['max']})")
    if 'negative_gnr' in stats:
        logger.info(f"  Negative GNR: {stats['negative_gnr']['mean']:.1f} "
                   f"(range: {stats['negative_gnr']['min']}-{stats['negative_gnr']['max']})")
    if 'positive_cosine' in stats:
        logger.info(f"  Positive cosine: {stats['positive_cosine']['mean']:.3f} "
                   f"(range: {stats['positive_cosine']['min']:.3f}-{stats['positive_cosine']['max']:.3f})")
    if 'negative_cosine' in stats:
        logger.info(f"  Negative cosine: {stats['negative_cosine']['mean']:.3f} "
                   f"(range: {stats['negative_cosine']['min']:.3f}-{stats['negative_cosine']['max']:.3f})")
    if 'cosine_margin' in stats:
        logger.info(f"  Cosine margin (pos-neg): {stats['cosine_margin']['mean']:.3f} "
                   f"(range: {stats['cosine_margin']['min']:.3f}-{stats['cosine_margin']['max']:.3f})")
        logger.info(f"  Triplets with negative margin: {stats['cosine_margin']['negative_margin_count']:,} "
                   f"({stats['cosine_margin']['negative_margin_pct']:.1f}%)")

    # Write triplets (training format: anchor, positive, negative, anchor_group)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing triplets to {args.output}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            # Training format (compatible with train_sbert_contrastive.py)
            training_triplet = {
                'anchor': triplet['anchor'],
                'positive': triplet['positive'],
                'negative': triplet['negative'],
                'anchor_group': triplet.get('anchor_group', ''),
            }
            f.write(json.dumps(training_triplet, ensure_ascii=False) + '\n')

    # Write stats if requested
    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing stats to {args.stats_output}")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

    logger.info("Done!")


if __name__ == '__main__':
    main()
