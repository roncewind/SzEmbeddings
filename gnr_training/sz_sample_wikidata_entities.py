#!/usr/bin/env python3
"""
sz_sample_wikidata_entities.py - Sample entities with good alias diversity from Wikidata CSV.

Samples entities prioritizing:
- Entities with 3+ aliases (enough for within-entity pairs)
- Entities with cross-script aliases (Latin + CJK/Cyrillic/Arabic)
- Avoids entities with only 1-2 aliases (less training value)

Usage:
    python sz_sample_wikidata_entities.py \
        --input ~/roncewind.git/BizNames/data/20250901_biznames_wikidata.csv \
        --output data/gnr_alignment/wikidata_entities_20k.jsonl \
        --sample 20000 \
        --min_aliases 3 \
        --prefer_cross_script \
        --seed 42
"""

import argparse
import csv
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


def detect_script(text: str) -> set[str]:
    """Detect Unicode scripts present in text."""
    scripts = set()
    for char in text:
        if char.isalpha():
            try:
                name = unicodedata.name(char, '')
                if 'CJK' in name or 'HIRAGANA' in name or 'KATAKANA' in name:
                    scripts.add('CJK')
                elif 'HANGUL' in name:
                    scripts.add('Korean')
                elif 'CYRILLIC' in name:
                    scripts.add('Cyrillic')
                elif 'ARABIC' in name:
                    scripts.add('Arabic')
                elif 'HEBREW' in name:
                    scripts.add('Hebrew')
                elif 'THAI' in name:
                    scripts.add('Thai')
                elif 'DEVANAGARI' in name or 'BENGALI' in name or 'TAMIL' in name or 'TELUGU' in name:
                    scripts.add('Indic')
                elif 'GREEK' in name:
                    scripts.add('Greek')
                elif 'LATIN' in name or char.isascii():
                    scripts.add('Latin')
                else:
                    # Group other scripts
                    scripts.add('Other')
            except ValueError:
                pass
    return scripts


def has_cross_script_aliases(aliases: list[dict]) -> bool:
    """Check if aliases span multiple scripts."""
    all_scripts = set()
    for alias in aliases:
        scripts = detect_script(alias['name'])
        all_scripts.update(scripts)

    # Must have Latin plus at least one non-Latin script
    non_latin = all_scripts - {'Latin', 'Other'}
    return 'Latin' in all_scripts and len(non_latin) > 0


def count_cross_script_pairs(aliases: list[dict]) -> int:
    """Count potential cross-script pairs (Latin ↔ non-Latin)."""
    latin_aliases = []
    non_latin_aliases = []

    for alias in aliases:
        scripts = detect_script(alias['name'])
        if 'Latin' in scripts and len(scripts) == 1:
            latin_aliases.append(alias)
        elif 'Latin' not in scripts:
            non_latin_aliases.append(alias)

    return len(latin_aliases) * len(non_latin_aliases)


def main():
    parser = argparse.ArgumentParser(
        prog='sz_sample_wikidata_entities',
        description='Sample entities with good alias diversity from Wikidata CSV'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to Wikidata CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=20000,
        help='Number of entities to sample (default: 20000)'
    )
    parser.add_argument(
        '--min_aliases',
        type=int,
        default=3,
        help='Minimum aliases per entity (default: 3)'
    )
    parser.add_argument(
        '--prefer_cross_script',
        action='store_true',
        help='Prioritize entities with cross-script aliases'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Read and group by entity
    logger.info(f"Reading entities from {args.input}")
    entities = defaultdict(lambda: {'canonical': None, 'aliases': []})

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    row_count = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            entity_id = row['id']
            entities[entity_id]['canonical'] = row['canonical']
            entities[entity_id]['aliases'].append({
                'name': row['name'],
                'language': row['language']
            })

            if row_count % 100000 == 0:
                logger.info(f"Processed {row_count:,} rows, {len(entities):,} entities so far")

    logger.info(f"Total: {row_count:,} rows, {len(entities):,} unique entities")

    # Filter entities with enough aliases
    logger.info(f"Filtering entities with at least {args.min_aliases} aliases")
    eligible_entities = []

    for entity_id, data in entities.items():
        num_aliases = len(data['aliases'])
        if num_aliases >= args.min_aliases:
            # Deduplicate aliases by name
            seen_names = set()
            unique_aliases = []
            for alias in data['aliases']:
                if alias['name'] not in seen_names:
                    seen_names.add(alias['name'])
                    unique_aliases.append(alias)

            if len(unique_aliases) >= args.min_aliases:
                cross_script = has_cross_script_aliases(unique_aliases)
                cross_script_count = count_cross_script_pairs(unique_aliases)
                eligible_entities.append({
                    'entity_id': entity_id,
                    'canonical': data['canonical'],
                    'aliases': unique_aliases,
                    'has_cross_script': cross_script,
                    'cross_script_pairs': cross_script_count
                })

    logger.info(f"Eligible entities (≥{args.min_aliases} aliases): {len(eligible_entities):,}")

    # Count cross-script entities
    cross_script_entities = [e for e in eligible_entities if e['has_cross_script']]
    logger.info(f"Entities with cross-script aliases: {len(cross_script_entities):,}")

    # Sample strategy
    if args.prefer_cross_script and len(cross_script_entities) > 0:
        # Prioritize cross-script entities
        logger.info("Sampling with cross-script preference")

        # Sort by cross-script pair count (descending) for deterministic selection
        cross_script_entities.sort(key=lambda x: -x['cross_script_pairs'])
        non_cross_script = [e for e in eligible_entities if not e['has_cross_script']]
        random.shuffle(non_cross_script)

        # Take as many cross-script as possible (up to 80% of sample)
        max_cross_script = min(len(cross_script_entities), int(args.sample * 0.8))

        # Shuffle cross-script entities but keep top ones
        top_cross_script = cross_script_entities[:max_cross_script]
        random.shuffle(top_cross_script)

        # Fill remainder with non-cross-script
        remainder_needed = args.sample - len(top_cross_script)
        sampled = top_cross_script + non_cross_script[:remainder_needed]

        # Final shuffle
        random.shuffle(sampled)
    else:
        # Random sampling
        logger.info("Random sampling")
        random.shuffle(eligible_entities)
        sampled = eligible_entities[:args.sample]

    # Truncate to requested sample size
    sampled = sampled[:args.sample]

    logger.info(f"Sampled {len(sampled):,} entities")

    # Compute statistics
    total_aliases = sum(len(e['aliases']) for e in sampled)
    cross_script_sampled = sum(1 for e in sampled if e['has_cross_script'])
    alias_counts = [len(e['aliases']) for e in sampled]

    logger.info(f"Statistics:")
    logger.info(f"  Total aliases: {total_aliases:,}")
    logger.info(f"  Average aliases per entity: {total_aliases / len(sampled):.1f}")
    logger.info(f"  Min aliases: {min(alias_counts)}")
    logger.info(f"  Max aliases: {max(alias_counts)}")
    logger.info(f"  Cross-script entities: {cross_script_sampled:,} ({100*cross_script_sampled/len(sampled):.1f}%)")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing to {args.output}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entity in sampled:
            # Clean output format (remove internal fields)
            output_record = {
                'entity_id': entity['entity_id'],
                'canonical': entity['canonical'],
                'aliases': entity['aliases']
            }
            f.write(json.dumps(output_record, ensure_ascii=False) + '\n')

    logger.info("Done!")


if __name__ == '__main__':
    main()
