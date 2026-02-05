#!/usr/bin/env python3
"""
sz_prepare_wikidata_for_senzing.py - Convert entity JSONL to Senzing format.

Converts sampled Wikidata entities (from sz_sample_wikidata_entities.py)
to Senzing JSONL format suitable for loading with sz_load_embeddings_onnx.py.

Usage:
    python sz_prepare_wikidata_for_senzing.py \
        --input data/gnr_alignment/wikidata_entities_20k.jsonl \
        --output data/gnr_alignment/wikidata_senzing_20k.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_entity_to_senzing(entity: dict, data_source: str = 'WIKIDATA') -> dict:
    """Convert a Wikidata entity to Senzing format."""
    entity_id = entity['entity_id']
    canonical = entity['canonical']
    aliases = entity['aliases']

    # Build NAMES array - for ORGANIZATION records, use NAME_ORG
    names = []
    for alias in aliases:
        names.append({
            'NAME_ORG': alias['name']
        })

    # Senzing record format
    senzing_record = {
        'DATA_SOURCE': data_source,
        'RECORD_ID': entity_id,
        'RECORD_TYPE': 'ORGANIZATION',
        'NAMES': names
    }

    return senzing_record


def main():
    parser = argparse.ArgumentParser(
        prog='sz_prepare_wikidata_for_senzing',
        description='Convert entity JSONL to Senzing format'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to sampled entities JSONL file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output Senzing JSONL file path'
    )
    parser.add_argument(
        '--data_source',
        type=str,
        default='WIKIDATA',
        help='Data source name for Senzing (default: WIKIDATA)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {args.input} to Senzing format")

    entity_count = 0
    total_names = 0

    with open(input_path, 'r', encoding='utf-8') as fin:
        with open(output_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                try:
                    entity = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON: {e}")
                    continue

                senzing_record = convert_entity_to_senzing(entity, args.data_source)
                fout.write(json.dumps(senzing_record, ensure_ascii=False) + '\n')

                entity_count += 1
                total_names += len(entity['aliases'])

                if entity_count % 5000 == 0:
                    logger.info(f"Converted {entity_count:,} entities")

    logger.info(f"Done! Converted {entity_count:,} entities with {total_names:,} total names")
    logger.info(f"Output: {args.output}")


if __name__ == '__main__':
    main()
