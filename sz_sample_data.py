#!/usr/bin/env python3
"""
Sample a subset of Senzing JSONL data for testing.

This script creates a stratified sample based on:
- Entity type (PERSON vs ORGANIZATION)
- Name complexity (word count, special characters)
- Alias count (entities with multiple name variants)
- Character script diversity (Latin, Cyrillic, Arabic, CJK, etc.)

Usage:
    python sz_sample_data.py -i input.jsonl -o subset.jsonl --size 10000
    python sz_sample_data.py -i input.json.gz -o subset.jsonl --size 50000 --person_ratio 0.5
"""

import argparse
import bz2
import gzip
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import TextIO

import orjson


def detect_script(text: str) -> str:
    """Detect the primary script/alphabet used in text."""
    if not text:
        return "empty"

    # Count characters by Unicode ranges
    scripts = {
        "latin": 0,
        "cyrillic": 0,
        "arabic": 0,
        "cjk": 0,
        "other": 0
    }

    for char in text:
        code = ord(char)
        if (0x0041 <= code <= 0x007A) or (0x00C0 <= code <= 0x024F):  # Latin
            scripts["latin"] += 1
        elif 0x0400 <= code <= 0x04FF:  # Cyrillic
            scripts["cyrillic"] += 1
        elif (0x0600 <= code <= 0x06FF) or (0x0750 <= code <= 0x077F):  # Arabic
            scripts["arabic"] += 1
        elif (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF):  # CJK
            scripts["cjk"] += 1
        elif not char.isspace():
            scripts["other"] += 1

    # Return the dominant script
    return max(scripts.items(), key=lambda x: x[1])[0]


def get_name_complexity(name: str) -> str:
    """Categorize name complexity."""
    if not name:
        return "empty"

    word_count = len(name.split())
    has_special = bool(re.search(r'[^\w\s-]', name))

    if word_count <= 2 and not has_special:
        return "simple"
    elif word_count <= 4 and not has_special:
        return "medium"
    else:
        return "complex"


def analyze_record(record: dict) -> dict:
    """Extract features for stratified sampling."""
    record_type = record.get("RECORD_TYPE", "UNKNOWN")
    names = record.get("NAMES", [])

    # Get primary name based on record type
    if record_type == "PERSON":
        name_field = "NAME_FULL"
    elif record_type == "ORGANIZATION":
        name_field = "NAME_ORG"
    else:
        name_field = "NAME_FULL"  # fallback

    # Extract primary name
    primary_name = ""
    if names:
        primary_name = names[0].get(name_field, "")

    alias_count = len(names)
    script = detect_script(primary_name)
    complexity = get_name_complexity(primary_name)

    return {
        "record": record,
        "record_type": record_type,
        "alias_count": alias_count,
        "script": script,
        "complexity": complexity,
        "has_aliases": alias_count > 1
    }


def stratified_sample(
    records: list[dict],
    target_size: int,
    person_ratio: float = 0.5
) -> list[dict]:
    """
    Create a stratified sample ensuring diversity across multiple dimensions.

    Args:
        records: List of analyzed records (output from analyze_record)
        target_size: Desired number of records in sample
        person_ratio: Target ratio of person records (0.0-1.0)
    """
    # Separate by record type
    persons = [r for r in records if r["record_type"] == "PERSON"]
    orgs = [r for r in records if r["record_type"] == "ORGANIZATION"]
    others = [r for r in records if r["record_type"] not in ["PERSON", "ORGANIZATION"]]

    # Calculate target counts
    person_target = int(target_size * person_ratio)
    org_target = target_size - person_target

    # Adjust if we don't have enough of one type
    if len(persons) < person_target:
        person_target = len(persons)
        org_target = min(target_size - person_target, len(orgs))
    elif len(orgs) < org_target:
        org_target = len(orgs)
        person_target = min(target_size - org_target, len(persons))

    print(f"📊 Sampling {person_target:,} persons, {org_target:,} organizations")

    def sample_group(group: list[dict], count: int) -> list[dict]:
        """Sample from a group with preference for diversity."""
        if len(group) <= count:
            return group

        # Priority sampling: prefer records with aliases and diverse scripts
        high_value = [r for r in group if r["has_aliases"]]
        diverse_scripts = [r for r in group if r["script"] != "latin"]
        complex_names = [r for r in group if r["complexity"] == "complex"]

        # Start with high-value records
        selected = []

        # Add all multi-alias records first (up to 30% of target)
        alias_quota = min(len(high_value), int(count * 0.3))
        if high_value:
            selected.extend(random.sample(high_value, alias_quota))

        # Add diverse scripts (up to 20% of target)
        script_quota = min(len(diverse_scripts), int(count * 0.2))
        diverse_remaining = [r for r in diverse_scripts if r not in selected]
        if diverse_remaining:
            selected.extend(random.sample(diverse_remaining, min(script_quota, len(diverse_remaining))))

        # Fill remaining with random sampling from all records
        remaining = [r for r in group if r not in selected]
        remaining_needed = count - len(selected)
        if remaining and remaining_needed > 0:
            selected.extend(random.sample(remaining, min(remaining_needed, len(remaining))))

        return selected

    # Sample each group
    sampled_persons = sample_group(persons, person_target)
    sampled_orgs = sample_group(orgs, org_target)

    # Combine and shuffle (only include persons and orgs, skip "others" like VESSEL, AIRCRAFT, and None)
    result = sampled_persons + sampled_orgs
    random.shuffle(result)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Sample a subset of Senzing JSONL data for testing"
    )
    parser.add_argument("-i", "--input", required=True, help="Input file (supports .json, .jsonl, .gz, .bz2, or - for stdin)")
    parser.add_argument("-o", "--output", help="Output JSONL file (required unless --analyze_only)")
    parser.add_argument("--size", type=int, default=10000, help="Target number of records in sample (default: 10000)")
    parser.add_argument("--person_ratio", type=float, default=0.5, help="Ratio of person records (0.0-1.0, default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze data distribution without sampling")

    args = parser.parse_args()

    # Validate arguments
    if not args.analyze_only and not args.output:
        parser.error("the following arguments are required: -o/--output (unless --analyze_only is specified)")

    # Set random seed
    random.seed(args.seed)

    # Open input file
    input_path = args.input
    if input_path == "-":
        file_handle = sys.stdin
    elif input_path.endswith(".bz2"):
        file_handle = bz2.open(input_path, 'rt')
    elif input_path.endswith(".gz"):
        file_handle = gzip.open(input_path, 'rt')
    else:
        file_handle = open(input_path, 'r')

    print(f"📖 Reading from {input_path}...")

    # Read and analyze all records
    records = []
    stats = defaultdict(lambda: defaultdict(int))

    try:
        for line_num, line in enumerate(file_handle, 1):
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # Strip trailing comma (for JSON array format)
            if line.endswith(","):
                line = line[:-1]

            try:
                record = orjson.loads(line)
                analyzed = analyze_record(record)
                records.append(analyzed)

                # Update statistics
                stats["record_type"][analyzed["record_type"]] += 1
                stats["script"][analyzed["script"]] += 1
                stats["complexity"][analyzed["complexity"]] += 1
                if analyzed["has_aliases"]:
                    stats["has_aliases"]["true"] += 1
                else:
                    stats["has_aliases"]["false"] += 1

                if line_num % 10000 == 0:
                    print(f"  {line_num:,} records read...", end='\r')

            except Exception as e:
                print(f"⚠️  Error parsing line {line_num}: {e}", file=sys.stderr)
                continue

    finally:
        if file_handle != sys.stdin:
            file_handle.close()

    print(f"\n✅ Read {len(records):,} total records")

    # Print statistics
    print("\n📊 Data Distribution:")
    print(f"  Record Types:")
    for rtype, count in sorted(stats["record_type"].items(), key=lambda x: (x[0] is None, x[0])):
        pct = 100 * count / len(records)
        print(f"    {rtype if rtype else 'None/Missing'}: {count:,} ({pct:.1f}%)")

    print(f"  Scripts:")
    for script, count in sorted(stats["script"].items(), key=lambda x: -x[1]):
        pct = 100 * count / len(records)
        print(f"    {script}: {count:,} ({pct:.1f}%)")

    print(f"  Complexity:")
    for complexity, count in sorted(stats["complexity"].items(), key=lambda x: (x[0] is None, x[0])):
        pct = 100 * count / len(records)
        print(f"    {complexity}: {count:,} ({pct:.1f}%)")

    print(f"  Aliases:")
    for has_alias, count in sorted(stats["has_aliases"].items(), key=lambda x: str(x[0])):
        pct = 100 * count / len(records)
        print(f"    Has aliases: {has_alias}: {count:,} ({pct:.1f}%)")

    if args.analyze_only:
        print("\n✅ Analysis complete (--analyze_only mode)")
        return

    # Perform stratified sampling
    print(f"\n🎲 Creating stratified sample of {args.size:,} records...")
    sampled = stratified_sample(records, args.size, args.person_ratio)

    # Write output
    print(f"✍️  Writing to {args.output}...")
    with open(args.output, 'w') as out:
        for item in sampled:
            # Write just the record (not the analysis metadata)
            out.write(json.dumps(item["record"], ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(sampled):,} records to {args.output}")

    # Print sample statistics
    sample_stats = defaultdict(lambda: defaultdict(int))
    for item in sampled:
        sample_stats["record_type"][item["record_type"]] += 1
        sample_stats["script"][item["script"]] += 1
        sample_stats["complexity"][item["complexity"]] += 1
        if item["has_aliases"]:
            sample_stats["has_aliases"]["true"] += 1
        else:
            sample_stats["has_aliases"]["false"] += 1

    print("\n📊 Sample Distribution:")
    print(f"  Record Types:")
    for rtype, count in sorted(sample_stats["record_type"].items(), key=lambda x: (x[0] is None, x[0])):
        pct = 100 * count / len(sampled)
        print(f"    {rtype if rtype else 'None/Missing'}: {count:,} ({pct:.1f}%)")

    print(f"  Scripts:")
    for script, count in sorted(sample_stats["script"].items(), key=lambda x: -x[1]):
        pct = 100 * count / len(sampled)
        print(f"    {script}: {count:,} ({pct:.1f}%)")

    print(f"  Has aliases: {sample_stats['has_aliases']['true']:,} ({100*sample_stats['has_aliases']['true']/len(sampled):.1f}%)")


if __name__ == "__main__":
    main()
