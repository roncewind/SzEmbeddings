#!/usr/bin/env python3
"""
Extract entities with aliases from Senzing JSONL data.

This script:
1. Identifies records with multiple name variants (aliases)
2. Extracts them for matching/evaluation testing
3. Optionally generates triplets (anchor, positive, negative) for model evaluation

Usage:
    # Extract all records with aliases
    python sz_extract_aliases.py -i data.jsonl -o aliases.jsonl

    # Generate triplets for evaluation
    python sz_extract_aliases.py -i data.jsonl --triplets triplets.jsonl --min_aliases 2

    # Extract from loaded Senzing database (query entities with multiple names)
    python sz_extract_aliases.py --from_senzing --data_source OPENSANCTIONS -o aliases.jsonl
"""

import argparse
import bz2
import gzip
import json
import random
import sys
from collections import defaultdict
from typing import TextIO

import orjson


def extract_aliases_from_file(
    file_handle: TextIO,
    min_aliases: int = 2
) -> tuple[list[dict], dict]:
    """
    Extract records with multiple name variants from JSONL file.

    Returns:
        - List of records with aliases
        - Statistics dict
    """
    records_with_aliases = []
    stats = {
        "total_records": 0,
        "with_aliases": 0,
        "persons": 0,
        "orgs": 0,
        "alias_counts": defaultdict(int)
    }

    for line_num, line in enumerate(file_handle, 1):
        line = line.strip()
        if not line or len(line) < 10:
            continue

        # Strip trailing comma (for JSON array format)
        if line.endswith(","):
            line = line[:-1]

        try:
            record = orjson.loads(line)
            stats["total_records"] += 1

            names = record.get("NAMES", [])
            alias_count = len(names)

            if alias_count >= min_aliases:
                records_with_aliases.append(record)
                stats["with_aliases"] += 1
                stats["alias_counts"][alias_count] += 1

                record_type = record.get("RECORD_TYPE", "UNKNOWN")
                if record_type == "PERSON":
                    stats["persons"] += 1
                elif record_type == "ORGANIZATION":
                    stats["orgs"] += 1

            if line_num % 10000 == 0:
                print(f"  {line_num:,} records processed, {stats['with_aliases']:,} with aliases...", end='\r')

        except Exception as e:
            print(f"⚠️  Error parsing line {line_num}: {e}", file=sys.stderr)
            continue

    print()  # newline after progress
    return records_with_aliases, stats


def generate_triplets(
    records: list[dict],
    max_triplets_per_record: int = 5
) -> list[dict]:
    """
    Generate triplets (anchor, positive, negative) from records with aliases.

    For each record with multiple names:
    - anchor: first name variant
    - positive: another name variant from same entity
    - negative: name from different entity

    Returns list of triplet dicts: {"anchor": str, "positive": str, "negative": str, "record_type": str}
    """
    triplets = []

    # Separate by record type for negative sampling
    persons = [r for r in records if r.get("RECORD_TYPE") == "PERSON"]
    orgs = [r for r in records if r.get("RECORD_TYPE") == "ORGANIZATION"]

    def create_triplets_for_record(record: dict, negative_pool: list[dict]) -> list[dict]:
        """Create triplets for a single record."""
        record_type = record.get("RECORD_TYPE", "UNKNOWN")
        names = record.get("NAMES", [])

        if len(names) < 2:
            return []

        # Determine name field based on record type
        if record_type == "PERSON":
            name_field = "NAME_FULL"
        elif record_type == "ORGANIZATION":
            name_field = "NAME_ORG"
        else:
            name_field = "NAME_FULL"

        # Extract name values
        name_values = []
        for name_entry in names:
            name_value = name_entry.get(name_field)
            if name_value:
                name_values.append(name_value)

        if len(name_values) < 2:
            return []

        result = []

        # Generate multiple triplets per record (cycling through name pairs)
        num_triplets = min(max_triplets_per_record, len(name_values) - 1)

        for i in range(num_triplets):
            anchor = name_values[0]
            positive = name_values[(i + 1) % len(name_values)]

            # Sample negative from different entity
            if negative_pool:
                neg_record = random.choice(negative_pool)
                neg_names = neg_record.get("NAMES", [])
                if neg_names:
                    neg_name_value = neg_names[0].get(name_field)
                    if neg_name_value:
                        result.append({
                            "anchor": anchor,
                            "positive": positive,
                            "negative": neg_name_value,
                            "record_type": record_type,
                            "record_id": record.get("RECORD_ID", "unknown")
                        })

        return result

    # Generate triplets for persons
    for record in persons:
        if len(persons) > 1:  # Need at least 2 records for negative sampling
            negative_pool = [r for r in persons if r != record]
            triplets.extend(create_triplets_for_record(record, negative_pool))

    # Generate triplets for organizations
    for record in orgs:
        if len(orgs) > 1:
            negative_pool = [r for r in orgs if r != record]
            triplets.extend(create_triplets_for_record(record, negative_pool))

    return triplets


def extract_from_senzing(
    data_source: str,
    sz_engine
) -> tuple[list[dict], dict]:
    """
    Extract entities with aliases directly from Senzing database.

    This queries Senzing for entities and identifies those with multiple name features.
    """
    # This is a placeholder - actual implementation would query Senzing
    # for entities and extract those with multiple name features
    raise NotImplementedError("Senzing extraction not yet implemented")


def main():
    parser = argparse.ArgumentParser(
        description="Extract entities with aliases for testing and evaluation"
    )

    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input", help="Input JSONL file (supports .json, .jsonl, .gz, .bz2)")
    input_group.add_argument("--from_senzing", action="store_true", help="Extract from Senzing database")

    # Output options
    parser.add_argument("-o", "--output", help="Output file for records with aliases (JSONL)")
    parser.add_argument("--triplets", help="Output file for evaluation triplets (JSONL)")

    # Filtering
    parser.add_argument("--min_aliases", type=int, default=2, help="Minimum number of name variants required (default: 2)")
    parser.add_argument("--max_triplets_per_record", type=int, default=5, help="Max triplets per record (default: 5)")

    # Senzing options
    parser.add_argument("--data_source", help="Senzing data source (required with --from_senzing)")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed for triplet generation (default: 42)")

    args = parser.parse_args()

    # Validate arguments
    if args.from_senzing and not args.data_source:
        parser.error("--from_senzing requires --data_source")

    if not args.output and not args.triplets:
        parser.error("Must specify at least one of --output or --triplets")

    # Set random seed
    random.seed(args.seed)

    # Extract aliases
    if args.from_senzing:
        print("⚠️  Senzing extraction not yet implemented")
        print("    Use file-based extraction with -i instead")
        sys.exit(1)
    else:
        # File-based extraction
        input_path = args.input

        if input_path.endswith(".bz2"):
            file_handle = bz2.open(input_path, 'rt')
        elif input_path.endswith(".gz"):
            file_handle = gzip.open(input_path, 'rt')
        else:
            file_handle = open(input_path, 'r')

        print(f"📖 Reading from {input_path}...")
        print(f"   Looking for records with at least {args.min_aliases} name variants...")

        try:
            records_with_aliases, stats = extract_aliases_from_file(file_handle, args.min_aliases)
        finally:
            file_handle.close()

        print(f"\n✅ Processed {stats['total_records']:,} total records")
        print(f"   Found {stats['with_aliases']:,} records with {args.min_aliases}+ aliases ({100*stats['with_aliases']/max(stats['total_records'],1):.1f}%)")
        print(f"   - Persons: {stats['persons']:,}")
        print(f"   - Organizations: {stats['orgs']:,}")

        print(f"\n📊 Alias count distribution:")
        for count, freq in sorted(stats['alias_counts'].items())[:10]:
            print(f"   {count} aliases: {freq:,} records")

        # Write records with aliases
        if args.output:
            print(f"\n✍️  Writing records to {args.output}...")
            with open(args.output, 'w') as out:
                for record in records_with_aliases:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"   Wrote {len(records_with_aliases):,} records")

        # Generate and write triplets
        if args.triplets:
            print(f"\n🎲 Generating triplets...")
            triplets = generate_triplets(records_with_aliases, args.max_triplets_per_record)
            print(f"   Generated {len(triplets):,} triplets")

            print(f"✍️  Writing triplets to {args.triplets}...")
            with open(args.triplets, 'w') as out:
                for triplet in triplets:
                    out.write(json.dumps(triplet, ensure_ascii=False) + "\n")
            print(f"   Wrote {len(triplets):,} triplets")

            # Sample triplets
            if triplets:
                print(f"\n📝 Sample triplets:")
                for i, triplet in enumerate(random.sample(triplets, min(3, len(triplets)))):
                    print(f"\n   Triplet {i+1} ({triplet['record_type']}):")
                    print(f"     Anchor:   {triplet['anchor']}")
                    print(f"     Positive: {triplet['positive']}")
                    print(f"     Negative: {triplet['negative']}")


if __name__ == "__main__":
    main()
