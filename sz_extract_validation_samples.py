#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Extract validation test samples from loaded data file.
#
# Reads JSONL records, samples N records, extracts all names/aliases,
# and creates one test case per name/alias for production validation.
#
# Usage:
#   python sz_extract_validation_samples.py \
#     --input opensanctions_test_500.jsonl \
#     --output validation_samples_100.jsonl \
#     --sample_size 100 \
#     --filter both \
#     --seed 42
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson


def parse_line(line: str) -> str | None:
    """Parse a JSONL line, handling common formatting issues."""
    stripped_line = line.strip()
    if len(stripped_line) < 10:
        return None
    # Handle trailing commas (from JSON arrays)
    if stripped_line.endswith(","):
        stripped_line = stripped_line[:-1]
    return stripped_line


def detect_script(text: str) -> str:
    """
    Detect the primary script/language of text.
    Returns: "Latin", "Cyrillic", "Georgian", "Arabic", "Chinese", "Mixed", or "Unknown"
    """
    if not text:
        return "Unknown"

    # Count characters by Unicode ranges
    latin = sum(1 for c in text if '\u0041' <= c <= '\u007A' or '\u00C0' <= c <= '\u00FF')
    cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    georgian = sum(1 for c in text if '\u10A0' <= c <= '\u10FF')
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    chinese = sum(1 for c in text if '\u4E00' <= c <= '\u9FFF')

    total = latin + cyrillic + georgian + arabic + chinese
    if total == 0:
        return "Unknown"

    # Determine dominant script (>60% threshold)
    threshold = 0.6 * total
    if latin > threshold:
        return "Latin"
    elif cyrillic > threshold:
        return "Cyrillic"
    elif georgian > threshold:
        return "Georgian"
    elif arabic > threshold:
        return "Arabic"
    elif chinese > threshold:
        return "Chinese"
    else:
        return "Mixed"


def extract_names_from_record(record: dict[str, Any], record_type: str) -> list[tuple[str, str]]:
    """
    Extract all names/aliases from a record's NAMES array.

    Args:
        record: Record dictionary
        record_type: "PERSON" or "ORGANIZATION"

    Returns:
        List of (name, name_type) tuples where name_type is "primary_<script>" or "alias_<script>"
    """
    name_field = "NAME_FULL" if record_type == "PERSON" else "NAME_ORG"
    names = []

    for i, name_entry in enumerate(record.get("NAMES", [])):
        name_value = name_entry.get(name_field)
        if not name_value:
            continue

        # Determine if primary or alias
        name_type_raw = name_entry.get("NAME_TYPE", "").upper()
        if name_type_raw == "PRIMARY" or i == 0:  # First is primary if not specified
            base_type = "primary"
        else:
            base_type = "alias"

        # Detect script
        script = detect_script(name_value)

        # Combine: "primary_Latin", "alias_Cyrillic", etc.
        name_type = f"{base_type}_{script}"

        names.append((name_value, name_type))

    return names


def create_test_case(
    record: dict[str, Any],
    name: str,
    name_type: str,
    all_names: list[str],
    index: int,
    original_index: int,
) -> dict[str, Any]:
    """
    Create a test case dictionary for a single name/alias.

    Args:
        record: Original record
        name: The query name for this test case
        name_type: Name type (e.g., "primary_Latin")
        all_names: All names/aliases from this record
        index: Index of this name within the record (0, 1, 2, ...)
        original_index: Original record index in the data file

    Returns:
        Test case dictionary
    """
    record_id = record.get("RECORD_ID", "unknown")
    data_source = record.get("DATA_SOURCE", "OPEN_SANCTIONS")
    record_type = record.get("RECORD_TYPE", "PERSON")

    return {
        "test_case_id": f"{record_id}_name_{index}",
        "record_id": record_id,
        "data_source": data_source,
        "record_type": record_type,
        "query_name": name,
        "query_name_type": name_type,
        "all_aliases": all_names,
        "expected_entity_id": None,  # Will be filled in during validation
        "metadata": {
            "original_index": original_index,
            "sample_date": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        }
    }


def load_and_sample_records(
    input_file: str,
    sample_size: int,
    filter_type: str,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Load records from JSONL file and randomly sample N records.

    Args:
        input_file: Path to input JSONL file
        sample_size: Number of records to sample (0 = all)
        filter_type: "person", "organization", or "both"
        seed: Random seed for reproducibility

    Returns:
        List of sampled record dictionaries
    """
    print(f"⏳ Loading records from {input_file}...")

    # Load all records
    records = []
    line_count = 0

    with open(input_file, 'r') as f:
        for line in f:
            line_count += 1
            parsed = parse_line(line)
            if not parsed:
                continue

            try:
                record = orjson.loads(parsed)
                record_type = record.get("RECORD_TYPE", "").upper()

                # Apply filter
                if filter_type == "person" and record_type != "PERSON":
                    continue
                elif filter_type == "organization" and record_type != "ORGANIZATION":
                    continue
                elif filter_type == "both" and record_type not in ("PERSON", "ORGANIZATION"):
                    continue

                # Store with original index
                record['_original_index'] = line_count
                records.append(record)

            except Exception as e:
                print(f"⚠️  Error parsing line {line_count}: {e}", file=sys.stderr)
                continue

    print(f"✅ Loaded {len(records):,} records (from {line_count:,} lines)")

    # Sample if requested
    if sample_size > 0 and sample_size < len(records):
        random.seed(seed)
        sampled_records = random.sample(records, sample_size)
        print(f"📊 Sampled {sample_size:,} records (seed={seed})")
        return sampled_records
    else:
        print(f"📊 Using all {len(records):,} records")
        return records


def extract_test_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract all test cases from sampled records.
    Creates one test case per name/alias.

    Args:
        records: List of record dictionaries

    Returns:
        List of test case dictionaries
    """
    print(f"⏳ Extracting test cases from {len(records):,} records...")

    test_cases = []
    skipped_records = 0

    for record in records:
        record_type = record.get("RECORD_TYPE", "PERSON")
        original_index = record.get('_original_index', 0)

        # Extract all names/aliases
        names_with_types = extract_names_from_record(record, record_type)

        if not names_with_types:
            skipped_records += 1
            continue

        # Create list of just names (for all_aliases field)
        all_names = [name for name, _ in names_with_types]

        # Create one test case per name/alias
        for idx, (name, name_type) in enumerate(names_with_types):
            test_case = create_test_case(
                record,
                name,
                name_type,
                all_names,
                idx,
                original_index,
            )
            test_cases.append(test_case)

    print(f"✅ Extracted {len(test_cases):,} test cases from {len(records):,} records")
    if skipped_records > 0:
        print(f"⚠️  Skipped {skipped_records:,} records (no names found)")

    # Compute statistics
    avg_names_per_record = len(test_cases) / len(records) if records else 0
    print(f"📊 Average {avg_names_per_record:.1f} names per record")

    return test_cases


def write_test_cases(test_cases: list[dict[str, Any]], output_file: str) -> None:
    """
    Write test cases to JSONL output file.

    Args:
        test_cases: List of test case dictionaries
        output_file: Path to output JSONL file
    """
    print(f"⏳ Writing {len(test_cases):,} test cases to {output_file}...")

    with open(output_file, 'w') as f:
        for test_case in test_cases:
            # Use orjson for fast serialization
            json_bytes = orjson.dumps(test_case)
            f.write(json_bytes.decode('utf-8') + '\n')

    print(f"✅ Test cases saved to {output_file}")


def print_statistics(test_cases: list[dict[str, Any]]) -> None:
    """Print statistics about the extracted test cases."""
    if not test_cases:
        return

    # Count by record type
    type_counts = {}
    for tc in test_cases:
        rt = tc.get("record_type", "Unknown")
        type_counts[rt] = type_counts.get(rt, 0) + 1

    # Count by script
    script_counts = {}
    for tc in test_cases:
        name_type = tc.get("query_name_type", "unknown")
        script = name_type.split("_")[-1] if "_" in name_type else "Unknown"
        script_counts[script] = script_counts.get(script, 0) + 1

    # Count by name type (primary vs alias)
    primary_count = sum(1 for tc in test_cases if tc.get("query_name_type", "").startswith("primary"))
    alias_count = sum(1 for tc in test_cases if tc.get("query_name_type", "").startswith("alias"))

    print("\n" + "=" * 80)
    print("TEST CASE STATISTICS")
    print("=" * 80)

    print(f"\nTotal test cases: {len(test_cases):,}")

    print("\nBy record type:")
    for rt, count in sorted(type_counts.items()):
        pct = count / len(test_cases) * 100
        print(f"  {rt:<20} {count:>6,} ({pct:>5.1f}%)")

    print("\nBy name type:")
    print(f"  {'Primary':<20} {primary_count:>6,} ({primary_count/len(test_cases)*100:>5.1f}%)")
    print(f"  {'Alias':<20} {alias_count:>6,} ({alias_count/len(test_cases)*100:>5.1f}%)")

    print("\nBy script:")
    for script, count in sorted(script_counts.items(), key=lambda x: -x[1]):
        pct = count / len(test_cases) * 100
        print(f"  {script:<20} {count:>6,} ({pct:>5.1f}%)")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        prog="sz_extract_validation_samples",
        description="Extract validation test samples from loaded data file"
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Path to input JSONL data file")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Path to output validation samples JSONL file")
    parser.add_argument("--sample_size", type=int, default=100,
                       help="Number of RECORDS to sample (0 = all, default: 100)")
    parser.add_argument("--filter", type=str, default="both",
                       choices=["person", "organization", "both"],
                       help="Filter by record type (default: both)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling (default: 42)")

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load and sample records
    records = load_and_sample_records(
        args.input,
        args.sample_size,
        args.filter,
        args.seed,
    )

    if not records:
        print("❌ No records found matching filter criteria", file=sys.stderr)
        sys.exit(1)

    # Extract test cases
    test_cases = extract_test_cases(records)

    if not test_cases:
        print("❌ No test cases extracted", file=sys.stderr)
        sys.exit(1)

    # Write output
    write_test_cases(test_cases, args.output)

    # Print statistics
    print_statistics(test_cases)

    print(f"\n✅ Extraction complete! Output: {args.output}")


if __name__ == "__main__":
    main()
