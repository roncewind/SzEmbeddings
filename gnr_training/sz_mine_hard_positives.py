#!/usr/bin/env python3
"""
sz_mine_hard_positives.py - Mine hard positive triplets for cross-script training.

Hard positives are pairs where:
- Same entity (ground truth match)
- High GNR score (≥85) OR null GNR with same_entity=True
- Low cosine similarity (<threshold) - model is failing on these

These are typically cross-script pairs (Latin↔CJK, Latin↔Cyrillic, etc.) that
the embedding model struggles with.

Output format matches training_triplets.jsonl for easy mixing:
{
    "anchor": "Nintendo",
    "positive": "任天堂",
    "negative": "Nintex",
    "anchor_group": "Q8093"
}

Usage:
    python sz_mine_hard_positives.py \
        --input data/gnr_alignment/pairs_with_gnr_scores.jsonl \
        --output data/gnr_alignment/hard_positive_triplets.jsonl \
        --cosine_max 0.60 \
        --gnr_min 85 \
        --negatives_per_positive 3
"""

import argparse
import json
import logging
import random
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_script(text: str) -> str:
    """Detect the primary script of a text string."""
    script_counts = defaultdict(int)

    for char in text:
        if char.isspace() or char in '.,;:!?-()[]{}\"\'':
            continue

        # Check common script ranges
        code = ord(char)

        # CJK
        if (0x4E00 <= code <= 0x9FFF or      # CJK Unified Ideographs
            0x3400 <= code <= 0x4DBF or      # CJK Extension A
            0x3000 <= code <= 0x303F or      # CJK Symbols
            0x30A0 <= code <= 0x30FF or      # Katakana
            0x3040 <= code <= 0x309F):       # Hiragana
            script_counts['CJK'] += 1
        # Korean
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            script_counts['KOREAN'] += 1
        # Cyrillic
        elif 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F:
            script_counts['CYRILLIC'] += 1
        # Arabic
        elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
            script_counts['ARABIC'] += 1
        # Hebrew
        elif 0x0590 <= code <= 0x05FF:
            script_counts['HEBREW'] += 1
        # Devanagari
        elif 0x0900 <= code <= 0x097F:
            script_counts['DEVANAGARI'] += 1
        # Tamil
        elif 0x0B80 <= code <= 0x0BFF:
            script_counts['TAMIL'] += 1
        # Thai
        elif 0x0E00 <= code <= 0x0E7F:
            script_counts['THAI'] += 1
        # Greek
        elif 0x0370 <= code <= 0x03FF:
            script_counts['GREEK'] += 1
        # Latin (basic + extended)
        elif (0x0041 <= code <= 0x005A or    # A-Z
              0x0061 <= code <= 0x007A or    # a-z
              0x00C0 <= code <= 0x024F):     # Latin Extended
            script_counts['LATIN'] += 1
        else:
            script_counts['OTHER'] += 1

    if not script_counts:
        return 'UNKNOWN'

    return max(script_counts, key=script_counts.get)


def is_cross_script(name_a: str, name_b: str) -> bool:
    """Check if two names are in different scripts."""
    script_a = detect_script(name_a)
    script_b = detect_script(name_b)
    return script_a != script_b and script_a != 'UNKNOWN' and script_b != 'UNKNOWN'


def load_pairs(filepath: str) -> list[dict]:
    """Load scored pairs from JSONL file."""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def mine_hard_positives(
    pairs: list[dict],
    cosine_max: float = 0.60,
    gnr_min: int = 85,
    negatives_per_positive: int = 3,
    include_null_gnr: bool = True,
    seed: int = 42
) -> list[dict]:
    """Mine hard positive triplets.

    Args:
        pairs: List of scored pairs
        cosine_max: Maximum cosine for hard positives (model struggling)
        gnr_min: Minimum GNR for positives (confirmed matches)
        negatives_per_positive: Number of negatives to sample per positive pair
        include_null_gnr: Include same-entity pairs with null GNR
        seed: Random seed

    Returns:
        List of triplets in training format
    """
    random.seed(seed)

    # Separate hard positives and potential negatives
    hard_positives = []
    negative_pool = defaultdict(list)  # entity_id -> [names]
    name_to_entity = {}

    for p in pairs:
        name_a = p['name_a']
        name_b = p['name_b']
        entity_a = p.get('entity_a')
        entity_b = p.get('entity_b')
        gnr = p.get('gnr_score')
        cosine = p.get('cosine_sim')
        same_entity = p.get('same_entity', False)

        # Track name->entity mapping
        if entity_a:
            name_to_entity[name_a] = entity_a
        if entity_b:
            name_to_entity[name_b] = entity_b

        # Build negative pool from different-entity pairs
        if not same_entity and entity_b:
            negative_pool[entity_b].append(name_b)

        # Skip if no cosine
        if cosine is None:
            continue

        # Hard positive criteria:
        # 1. Same entity
        # 2. Low cosine (model struggling)
        # 3. High GNR OR null GNR (confirmed match)
        if same_entity and cosine < cosine_max:
            if gnr is not None and gnr >= gnr_min:
                hard_positives.append(p)
            elif gnr is None and include_null_gnr:
                hard_positives.append(p)

    logger.info(f"Found {len(hard_positives):,} hard positive pairs (cosine < {cosine_max}, same entity)")

    # Analyze script distribution
    cross_script_count = 0
    script_pairs = defaultdict(int)
    for p in hard_positives:
        script_a = detect_script(p['name_a'])
        script_b = detect_script(p['name_b'])
        if script_a != script_b:
            cross_script_count += 1
            pair_key = tuple(sorted([script_a, script_b]))
            script_pairs[pair_key] += 1

    logger.info(f"Cross-script pairs: {cross_script_count:,} ({100*cross_script_count/len(hard_positives):.1f}%)")
    logger.info("Script pair distribution:")
    for pair, count in sorted(script_pairs.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {pair[0]} ↔ {pair[1]}: {count:,}")

    # Build flat negative pool
    all_negatives = []
    negative_entities = set()
    for entity_id, names in negative_pool.items():
        for name in names:
            all_negatives.append((name, entity_id))
            negative_entities.add(entity_id)

    logger.info(f"Negative pool: {len(all_negatives):,} names from {len(negative_entities):,} entities")

    # Generate triplets
    triplets = []

    # More efficient: sample and reject instead of filtering entire pool
    for p in hard_positives:
        anchor = p['name_a']
        positive = p['name_b']
        anchor_entity = p.get('entity_a', '')

        # Sample negatives with rejection sampling (much faster)
        sampled = []
        attempts = 0
        max_attempts = negatives_per_positive * 10

        while len(sampled) < negatives_per_positive and attempts < max_attempts:
            neg_name, neg_entity = random.choice(all_negatives)
            if neg_entity != anchor_entity:
                sampled.append((neg_name, neg_entity))
            attempts += 1

        if not sampled:
            continue

        for neg_name, neg_entity in sampled:
            triplet = {
                'anchor': anchor,
                'positive': positive,
                'negative': neg_name,
                'anchor_group': anchor_entity,
            }
            triplets.append(triplet)

        # Also create reverse triplet (positive as anchor)
        for neg_name, neg_entity in sampled:
            triplet = {
                'anchor': positive,
                'positive': anchor,
                'negative': neg_name,
                'anchor_group': anchor_entity,
            }
            triplets.append(triplet)

    logger.info(f"Generated {len(triplets):,} triplets from hard positives")

    # Shuffle
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

    # Cross-script analysis
    cross_script = 0
    script_pairs = defaultdict(int)

    for t in triplets:
        anchor_script = detect_script(t['anchor'])
        positive_script = detect_script(t['positive'])

        if anchor_script != positive_script:
            cross_script += 1
            pair_key = tuple(sorted([anchor_script, positive_script]))
            script_pairs[pair_key] += 1

    stats['cross_script_triplets'] = cross_script
    stats['cross_script_percentage'] = 100 * cross_script / len(triplets)
    stats['script_pair_distribution'] = dict(
        (f"{k[0]}_{k[1]}", v) for k, v in
        sorted(script_pairs.items(), key=lambda x: -x[1])
    )

    return stats


def main():
    parser = argparse.ArgumentParser(
        prog='sz_mine_hard_positives',
        description='Mine hard positive triplets for cross-script training'
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
        '--cosine_max',
        type=float,
        default=0.60,
        help='Maximum cosine for hard positives (default: 0.60)'
    )
    parser.add_argument(
        '--gnr_min',
        type=int,
        default=85,
        help='Minimum GNR score for positives (default: 85)'
    )
    parser.add_argument(
        '--negatives_per_positive',
        type=int,
        default=3,
        help='Number of negatives per positive pair (default: 3)'
    )
    parser.add_argument(
        '--include_null_gnr',
        action='store_true',
        default=True,
        help='Include same-entity pairs with null GNR (default: True)'
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

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Load pairs
    logger.info(f"Loading pairs from {args.input}")
    pairs = load_pairs(args.input)
    logger.info(f"Loaded {len(pairs):,} pairs")

    # Mine hard positives
    logger.info("Mining hard positive triplets")
    logger.info(f"  Cosine max: {args.cosine_max}")
    logger.info(f"  GNR min: {args.gnr_min}")
    logger.info(f"  Negatives per positive: {args.negatives_per_positive}")
    logger.info(f"  Include null GNR: {args.include_null_gnr}")

    triplets = mine_hard_positives(
        pairs,
        cosine_max=args.cosine_max,
        gnr_min=args.gnr_min,
        negatives_per_positive=args.negatives_per_positive,
        include_null_gnr=args.include_null_gnr,
        seed=args.seed
    )

    # Analyze
    stats = analyze_triplets(triplets)

    logger.info("Triplet statistics:")
    logger.info(f"  Total triplets: {stats.get('total_triplets', 0):,}")
    logger.info(f"  Unique anchors: {stats.get('unique_anchors', 0):,}")
    logger.info(f"  Unique positives: {stats.get('unique_positives', 0):,}")
    logger.info(f"  Unique negatives: {stats.get('unique_negatives', 0):,}")
    logger.info(f"  Cross-script triplets: {stats.get('cross_script_triplets', 0):,} "
               f"({stats.get('cross_script_percentage', 0):.1f}%)")

    if 'script_pair_distribution' in stats:
        logger.info("  Top script pairs:")
        for pair, count in list(stats['script_pair_distribution'].items())[:5]:
            logger.info(f"    {pair}: {count:,}")

    # Write triplets
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing triplets to {args.output}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')

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
