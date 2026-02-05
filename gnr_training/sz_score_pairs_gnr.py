#!/usr/bin/env python3
"""
sz_score_pairs_gnr.py - Score name pairs with GNR (via why_search) and compute cosine similarity.

For each (name_a, name_b) pair:
1. Get GNR score by calling why_search with name_a against the entity containing name_b
2. Compute cosine similarity between embeddings of name_a and name_b

Requires: Senzing loaded with the Wikidata entities (run sz_load_embeddings_onnx.py first)

Usage:
    python sz_score_pairs_gnr.py \
        --pairs data/gnr_alignment/name_pairs_for_gnr.jsonl \
        --model_path ~/999gz.git/name_model/biznames_model/onnx_int8 \
        --output data/gnr_alignment/pairs_with_gnr_scores.jsonl \
        --threads 8
"""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress transformers warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def load_model(model_path: str):
    """Load model - tries ONNX first, then PyTorch/SentenceTransformer."""
    model_path = Path(model_path)

    # Check if it's an ONNX model
    onnx_files = ['model.onnx', 'model.ort', 'transformer_fp16.onnx']
    is_onnx = any((model_path / f).exists() for f in onnx_files)

    if is_onnx:
        logger.info("Loading ONNX model")
        sys.path.insert(0, str(Path(__file__).parent))
        from onnx_sentence_transformer import load_onnx_model as _load_onnx_model
        return _load_onnx_model(str(model_path), intra_op_num_threads=1)
    else:
        logger.info("Loading PyTorch/SentenceTransformer model")
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(str(model_path))


def get_entity_id_mapping(engine, data_source: str) -> dict[str, int]:
    """Build mapping from RECORD_ID to Senzing ENTITY_ID."""
    from senzing import SzEngineFlags

    logger.info("Building RECORD_ID to ENTITY_ID mapping")

    # Get all entities for the data source
    # We'll query entity by record ID one at a time, caching results
    mapping = {}

    # Export all entities with record data
    try:
        export_flags = (
            SzEngineFlags.SZ_EXPORT_INCLUDE_ALL_ENTITIES |
            SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_SUMMARY |
            SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_DATA
        )
        export_handle = engine.export_json_entity_report(export_flags)

        count = 0
        while True:
            entity_json = engine.fetch_next(export_handle)
            if not entity_json:
                break

            entity = json.loads(entity_json)
            entity_id = entity.get('RESOLVED_ENTITY', {}).get('ENTITY_ID')

            # Get record IDs from this entity's RECORDS list
            record_data_list = entity.get('RESOLVED_ENTITY', {}).get('RECORDS', [])
            for record_data in record_data_list:
                if record_data.get('DATA_SOURCE') == data_source:
                    record_id = record_data.get('RECORD_ID')
                    if record_id:
                        mapping[record_id] = entity_id

            count += 1
            if count % 5000 == 0:
                logger.info(f"Processed {count:,} entities, found {len(mapping):,} record mappings")

        # Note: In Senzing v4, export handles are auto-closed when iteration completes

    except Exception as e:
        logger.error(f"Error exporting entities: {e}")
        raise

    logger.info(f"Built mapping for {len(mapping):,} records")
    return mapping


def get_gnr_score_for_pair(
    engine,
    name_a: str,
    entity_id: int,
    record_type: str = "ORGANIZATION",
    data_source: str = "WIKIDATA"
) -> tuple[int | None, str | None]:
    """Get GNR score by calling why_search."""
    from senzing import SzEngineFlags

    if record_type == "PERSON":
        search_attrs = {"NAME_FULL": name_a}
    else:
        search_attrs = {"NAME_ORG": name_a}

    try:
        why_result = engine.why_search(
            json.dumps(search_attrs),
            entity_id,
            SzEngineFlags.SZ_WHY_SEARCH_DEFAULT_FLAGS | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
        )
        why_data = json.loads(why_result)

        if why_data.get('WHY_RESULTS'):
            match_info = why_data['WHY_RESULTS'][0].get('MATCH_INFO', {})
            feature_scores = match_info.get('FEATURE_SCORES', {})

            if 'NAME' in feature_scores and feature_scores['NAME']:
                gnr_score = feature_scores['NAME'][0].get('SCORE')
                gnr_bucket = feature_scores['NAME'][0].get('SCORE_BUCKET')
                return gnr_score, gnr_bucket

    except Exception as e:
        logger.debug(f"why_search error for '{name_a}' vs entity {entity_id}: {e}")

    return None, None


def compute_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    # Embeddings should already be normalized
    return float(np.dot(emb_a, emb_b))


class PairScorer:
    """Thread-safe pair scorer."""

    def __init__(
        self,
        engine,
        model,
        record_id_to_entity: dict[str, int],
        data_source: str = "WIKIDATA",
        record_type: str = "ORGANIZATION",
        skip_gnr: bool = False
    ):
        self.engine = engine
        self.model = model
        self.record_id_to_entity = record_id_to_entity
        self.data_source = data_source
        self.record_type = record_type
        self.skip_gnr = skip_gnr
        self.lock = threading.Lock()

        # Pre-computed embeddings cache
        self.embedding_cache: dict[str, np.ndarray] = {}
        self.cache_lock = threading.Lock()

    def get_embedding(self, name: str) -> np.ndarray:
        """Get embedding for a name (cached)."""
        with self.cache_lock:
            if name in self.embedding_cache:
                return self.embedding_cache[name]

        # Compute embedding
        embedding = self.model.encode([name], normalize_embeddings=True)[0]

        with self.cache_lock:
            self.embedding_cache[name] = embedding

        return embedding

    def score_pair(self, pair: dict) -> dict:
        """Score a single pair."""
        name_a = pair['name_a']
        name_b = pair['name_b']
        entity_a = pair['entity_a']
        entity_b = pair['entity_b']
        same_entity = pair['same_entity']

        result = {
            'name_a': name_a,
            'name_b': name_b,
            'entity_a': entity_a,
            'entity_b': entity_b,
            'same_entity': same_entity,
            'pair_type': pair.get('pair_type', 'unknown'),
            'gnr_score': None,
            'gnr_bucket': None,
            'cosine_sim': None
        }

        # Compute cosine similarity
        try:
            emb_a = self.get_embedding(name_a)
            emb_b = self.get_embedding(name_b)
            result['cosine_sim'] = compute_cosine_similarity(emb_a, emb_b)
        except Exception as e:
            logger.warning(f"Error computing cosine for '{name_a}' vs '{name_b}': {e}")

        # Get GNR score (if not skipping)
        if not self.skip_gnr:
            # Get entity ID for entity_b
            senzing_entity_id = self.record_id_to_entity.get(entity_b)

            if senzing_entity_id is not None:
                gnr_score, gnr_bucket = get_gnr_score_for_pair(
                    self.engine,
                    name_a,
                    senzing_entity_id,
                    self.record_type,
                    self.data_source
                )
                result['gnr_score'] = gnr_score
                result['gnr_bucket'] = gnr_bucket
            else:
                logger.debug(f"No entity mapping for record {entity_b}")

        return result


def main():
    parser = argparse.ArgumentParser(
        prog='sz_score_pairs_gnr',
        description='Score name pairs with GNR and cosine similarity'
    )
    parser.add_argument(
        '--pairs', '-p',
        type=str,
        required=True,
        help='Path to pairs JSONL file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output scored pairs JSONL file path'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--data_source',
        type=str,
        default='WIKIDATA',
        help='Data source name in Senzing (default: WIKIDATA)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help='Number of worker threads (default: 8)'
    )
    parser.add_argument(
        '--skip_gnr',
        action='store_true',
        help='Skip GNR scoring (cosine only, faster)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size for progress reporting (default: 1000)'
    )

    args = parser.parse_args()

    input_path = Path(args.pairs)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.pairs}")
        sys.exit(1)

    # Load pairs
    logger.info(f"Loading pairs from {args.pairs}")
    pairs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    logger.info(f"Loaded {len(pairs):,} pairs")

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)

    # Initialize Senzing (if not skipping GNR)
    engine = None
    record_id_to_entity = {}

    if not args.skip_gnr:
        logger.info("Initializing Senzing")
        sys.path.insert(0, str(Path(__file__).parent))
        from sz_utils import get_senzing_config
        from senzing_core import SzAbstractFactoryCore

        settings = get_senzing_config()
        sz_factory = SzAbstractFactoryCore("PairScorer", settings, verbose_logging=0)
        engine = sz_factory.create_engine()

        # Build record ID to entity ID mapping
        record_id_to_entity = get_entity_id_mapping(engine, args.data_source)

        if not record_id_to_entity:
            logger.error("No records found in Senzing. Did you load the data first?")
            sys.exit(1)

    # Create scorer
    scorer = PairScorer(
        engine=engine,
        model=model,
        record_id_to_entity=record_id_to_entity,
        data_source=args.data_source,
        record_type="ORGANIZATION",
        skip_gnr=args.skip_gnr
    )

    # Score pairs
    logger.info(f"Scoring {len(pairs):,} pairs with {args.threads} threads")
    start_time = time.time()

    results = []
    completed = 0
    lock = threading.Lock()

    def process_pair(pair):
        nonlocal completed
        result = scorer.score_pair(pair)
        with lock:
            completed += 1
            if completed % args.batch_size == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = len(pairs) - completed
                eta = remaining / rate if rate > 0 else 0
                logger.info(
                    f"Processed {completed:,}/{len(pairs):,} pairs "
                    f"({100*completed/len(pairs):.1f}%) - "
                    f"{rate:.1f} pairs/sec - ETA: {eta/60:.1f}min"
                )
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(executor.map(process_pair, pairs))

    elapsed = time.time() - start_time
    logger.info(f"Scoring completed in {elapsed:.1f}s ({len(pairs)/elapsed:.1f} pairs/sec)")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing to {args.output}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Summary statistics
    gnr_scores = [r['gnr_score'] for r in results if r['gnr_score'] is not None]
    cosine_scores = [r['cosine_sim'] for r in results if r['cosine_sim'] is not None]

    logger.info("Summary statistics:")
    logger.info(f"  Pairs with GNR score: {len(gnr_scores):,}/{len(results):,}")
    logger.info(f"  Pairs with cosine: {len(cosine_scores):,}/{len(results):,}")

    if gnr_scores:
        logger.info(f"  GNR score range: {min(gnr_scores)}-{max(gnr_scores)}, mean: {np.mean(gnr_scores):.1f}")

    if cosine_scores:
        logger.info(f"  Cosine range: {min(cosine_scores):.3f}-{max(cosine_scores):.3f}, mean: {np.mean(cosine_scores):.3f}")

    # Breakdown by same_entity
    same_entity_results = [r for r in results if r['same_entity']]
    diff_entity_results = [r for r in results if not r['same_entity']]

    if same_entity_results:
        same_gnr = [r['gnr_score'] for r in same_entity_results if r['gnr_score'] is not None]
        same_cos = [r['cosine_sim'] for r in same_entity_results if r['cosine_sim'] is not None]
        if same_gnr:
            logger.info(f"  Same-entity GNR mean: {np.mean(same_gnr):.1f}")
        if same_cos:
            logger.info(f"  Same-entity cosine mean: {np.mean(same_cos):.3f}")

    if diff_entity_results:
        diff_gnr = [r['gnr_score'] for r in diff_entity_results if r['gnr_score'] is not None]
        diff_cos = [r['cosine_sim'] for r in diff_entity_results if r['cosine_sim'] is not None]
        if diff_gnr:
            logger.info(f"  Diff-entity GNR mean: {np.mean(diff_gnr):.1f}")
        if diff_cos:
            logger.info(f"  Diff-entity cosine mean: {np.mean(diff_cos):.3f}")

    logger.info("Done!")


if __name__ == '__main__':
    main()
