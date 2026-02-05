#!/usr/bin/env python3
"""
sz_generate_gnr_pairs.py - Generate within-entity and cross-entity name pairs.

Generates pairs for GNR scoring:
1. Within-entity pairs: All pairs of aliases from the same entity
2. Cross-entity pairs: Embedding search to find similar names from different entities

Usage:
    python sz_generate_gnr_pairs.py \
        --entities data/gnr_alignment/wikidata_entities_20k.jsonl \
        --output data/gnr_alignment/name_pairs_for_gnr.jsonl \
        --within_entity_max_pairs 6 \
        --cross_entity_candidates 5 \
        --model_path ~/999gz.git/name_model/biznames_model/e5_onnx_fp32 \
        --threshold 0.43 \
        --prioritize_cross_script
"""

import argparse
import json
import logging
import sys
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

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
                    scripts.add('Other')
            except ValueError:
                pass
    return scripts


def is_cross_script_pair(name_a: str, name_b: str) -> bool:
    """Check if two names are in different scripts."""
    scripts_a = detect_script(name_a)
    scripts_b = detect_script(name_b)

    # Cross-script if they have no major scripts in common
    major_scripts = {'Latin', 'CJK', 'Korean', 'Cyrillic', 'Arabic', 'Hebrew', 'Greek', 'Indic', 'Thai'}
    common_major = (scripts_a & scripts_b) & major_scripts
    return len(common_major) == 0


def load_onnx_model(model_path: str, providers: list[str] | None = None):
    """Load ONNX model."""
    sys.path.insert(0, str(Path(__file__).parent))
    from onnx_sentence_transformer import load_onnx_model as _load_onnx_model
    return _load_onnx_model(model_path, providers=providers)


def generate_within_entity_pairs(
    entities: list[dict],
    max_pairs_per_entity: int,
    prioritize_cross_script: bool
) -> list[dict]:
    """Generate within-entity pairs from aliases."""
    pairs = []

    for entity in entities:
        entity_id = entity['entity_id']
        aliases = entity['aliases']

        if len(aliases) < 2:
            continue

        # Generate all possible pairs
        alias_names = [a['name'] for a in aliases]
        all_pairs = list(combinations(range(len(alias_names)), 2))

        if prioritize_cross_script:
            # Separate cross-script pairs from same-script pairs
            cross_script_pairs = []
            same_script_pairs = []

            for i, j in all_pairs:
                if is_cross_script_pair(alias_names[i], alias_names[j]):
                    cross_script_pairs.append((i, j))
                else:
                    same_script_pairs.append((i, j))

            # Prioritize cross-script pairs
            selected_pairs = cross_script_pairs[:max_pairs_per_entity]
            if len(selected_pairs) < max_pairs_per_entity:
                remaining = max_pairs_per_entity - len(selected_pairs)
                selected_pairs.extend(same_script_pairs[:remaining])
        else:
            selected_pairs = all_pairs[:max_pairs_per_entity]

        for i, j in selected_pairs:
            pair_type = 'within_entity_cross_script' if is_cross_script_pair(alias_names[i], alias_names[j]) else 'within_entity_same_script'
            pairs.append({
                'name_a': alias_names[i],
                'name_b': alias_names[j],
                'entity_a': entity_id,
                'entity_b': entity_id,
                'same_entity': True,
                'pair_type': pair_type
            })

    return pairs


def generate_cross_entity_pairs(
    entities: list[dict],
    model,
    candidates_per_entity: int,
    threshold: float
) -> list[dict]:
    """Generate cross-entity pairs using embedding search."""
    pairs = []

    # Build index of all names to entity IDs
    name_to_entity = {}
    all_names = []

    for entity in entities:
        entity_id = entity['entity_id']
        for alias in entity['aliases']:
            name = alias['name']
            if name not in name_to_entity:
                name_to_entity[name] = entity_id
                all_names.append(name)

    logger.info(f"Computing embeddings for {len(all_names):,} unique names")

    # Compute embeddings in batches
    batch_size = 128
    all_embeddings = []

    for i in range(0, len(all_names), batch_size):
        batch = all_names[i:i+batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True)
        all_embeddings.append(embeddings)

        if (i + batch_size) % 10000 < batch_size:
            logger.info(f"Computed embeddings for {min(i + batch_size, len(all_names)):,} names")

    all_embeddings = np.vstack(all_embeddings)
    logger.info(f"Total embeddings: {all_embeddings.shape}")

    # For each entity, find similar names from other entities
    logger.info("Finding cross-entity candidates")
    processed_entities = 0

    for entity in entities:
        entity_id = entity['entity_id']

        # Get embeddings for this entity's aliases
        entity_names = [a['name'] for a in entity['aliases']]

        for query_name in entity_names[:3]:  # Limit queries per entity
            query_idx = all_names.index(query_name)
            query_embedding = all_embeddings[query_idx:query_idx+1]

            # Compute cosine similarities
            similarities = np.dot(all_embeddings, query_embedding.T).flatten()

            # Find top candidates above threshold
            candidates_found = 0
            sorted_indices = np.argsort(-similarities)

            for idx in sorted_indices:
                if candidates_found >= candidates_per_entity:
                    break

                candidate_name = all_names[idx]
                candidate_entity = name_to_entity[candidate_name]
                similarity = float(similarities[idx])

                # Skip same entity or below threshold
                if candidate_entity == entity_id:
                    continue
                if similarity < threshold:
                    break

                # Create pair (avoid duplicates by canonical ordering)
                if query_name < candidate_name:
                    pair_key = (query_name, candidate_name)
                else:
                    pair_key = (candidate_name, query_name)

                pairs.append({
                    'name_a': query_name,
                    'name_b': candidate_name,
                    'entity_a': entity_id,
                    'entity_b': candidate_entity,
                    'same_entity': False,
                    'pair_type': 'cross_entity_embedding_match',
                    'initial_cosine': similarity
                })
                candidates_found += 1

        processed_entities += 1
        if processed_entities % 1000 == 0:
            logger.info(f"Processed {processed_entities:,} entities, generated {len(pairs):,} cross-entity pairs")

    return pairs


def deduplicate_pairs(pairs: list[dict]) -> list[dict]:
    """Remove duplicate pairs (order-independent)."""
    seen = set()
    unique_pairs = []

    for pair in pairs:
        # Create canonical key (sorted names + same_entity flag)
        key = tuple(sorted([pair['name_a'], pair['name_b']])) + (pair['same_entity'],)
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    return unique_pairs


def main():
    parser = argparse.ArgumentParser(
        prog='sz_generate_gnr_pairs',
        description='Generate within-entity and cross-entity name pairs for GNR scoring'
    )
    parser.add_argument(
        '--entities', '-e',
        type=str,
        required=True,
        help='Path to sampled entities JSONL file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output pairs JSONL file path'
    )
    parser.add_argument(
        '--within_entity_max_pairs',
        type=int,
        default=6,
        help='Max within-entity pairs per entity (default: 6)'
    )
    parser.add_argument(
        '--cross_entity_candidates',
        type=int,
        default=5,
        help='Cross-entity candidates per entity name (default: 5)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to ONNX model for embedding search'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.43,
        help='Cosine similarity threshold for cross-entity (default: 0.43)'
    )
    parser.add_argument(
        '--prioritize_cross_script',
        action='store_true',
        help='Prioritize cross-script within-entity pairs'
    )
    parser.add_argument(
        '--skip_cross_entity',
        action='store_true',
        help='Skip cross-entity pair generation (within-entity only)'
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Use CUDA GPU acceleration for embedding computation'
    )

    args = parser.parse_args()

    input_path = Path(args.entities)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.entities}")
        sys.exit(1)

    # Load entities
    logger.info(f"Loading entities from {args.entities}")
    entities = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entities.append(json.loads(line))

    logger.info(f"Loaded {len(entities):,} entities")

    # Generate within-entity pairs
    logger.info("Generating within-entity pairs")
    within_pairs = generate_within_entity_pairs(
        entities,
        args.within_entity_max_pairs,
        args.prioritize_cross_script
    )
    logger.info(f"Generated {len(within_pairs):,} within-entity pairs")

    # Count cross-script within-entity pairs
    cross_script_within = sum(1 for p in within_pairs if p['pair_type'] == 'within_entity_cross_script')
    logger.info(f"  Cross-script: {cross_script_within:,}")
    logger.info(f"  Same-script: {len(within_pairs) - cross_script_within:,}")

    # Generate cross-entity pairs
    cross_pairs = []
    if not args.skip_cross_entity:
        logger.info("Loading embedding model")
        if args.cuda:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("Using CUDA GPU acceleration")
        else:
            providers = None
        model = load_onnx_model(args.model_path, providers=providers)

        logger.info("Generating cross-entity pairs")
        cross_pairs = generate_cross_entity_pairs(
            entities,
            model,
            args.cross_entity_candidates,
            args.threshold
        )
        logger.info(f"Generated {len(cross_pairs):,} cross-entity pairs")

    # Combine and deduplicate
    all_pairs = within_pairs + cross_pairs
    all_pairs = deduplicate_pairs(all_pairs)
    logger.info(f"Total unique pairs: {len(all_pairs):,}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing to {args.output}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    # Summary statistics
    within_count = sum(1 for p in all_pairs if p['same_entity'])
    cross_count = len(all_pairs) - within_count

    logger.info("Summary:")
    logger.info(f"  Within-entity pairs: {within_count:,}")
    logger.info(f"  Cross-entity pairs: {cross_count:,}")
    logger.info(f"  Total pairs: {len(all_pairs):,}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
