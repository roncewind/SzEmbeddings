#!/usr/bin/env python3
"""
Generate fuzzy name variants for validation testing.

Combines automated variant generation with manual curation to create
test cases that demonstrate embedding value beyond exact name matching.
"""

import json
import argparse
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class VariantTestCase:
    """A variant test case with metadata."""
    original_name: str
    variant_query: str
    expected_record_id: str  # Stable across reloads
    data_source: str
    record_type: str
    variant_type: str
    description: str
    source: str  # "automated" or "manual"


# Legal entity suffixes to remove
LEGAL_SUFFIXES = [
    "LIMITED", "LTD", "LTD.", "LLC", "L.L.C.", "INC", "INC.", "INCORPORATED",
    "CORP", "CORP.", "CORPORATION", "PLC", "P.L.C.", "SA", "S.A.",
    "COMPANY", "CO", "CO.", "AG", "GMBH", "N.V.", "B.V.",
    "PUBLIC COMPANY", "JOINT-STOCK COMPANY", "JSC", "OJSC", "CJSC"
]

# Common business word abbreviations
BUSINESS_ABBREVIATIONS = {
    "MANAGEMENT": "MGMT",
    "INTERNATIONAL": "INTL",
    "CORPORATION": "CORP",
    "COMPANY": "CO",
    "LIMITED": "LTD",
    "INCORPORATED": "INC",
    "ENTERPRISE": "ENTERPR",
    "INVESTMENT": "INVEST",
    "INVESTMENTS": "INVESTS",
    "DEVELOPMENT": "DEV",
    "TECHNOLOGIES": "TECH",
    "MANUFACTURING": "MFG",
    "SERVICES": "SVCS",
}

# Common prepositions to drop
PREPOSITIONS = ["OF", "THE", "AND", "&", "FOR", "TO", "IN", "AT", "BY", "WITH"]


def remove_legal_suffix(name: str) -> str:
    """Remove legal entity suffix from business name."""
    name_upper = name.upper()
    for suffix in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        # Match suffix at end, possibly with punctuation
        pattern = rf'\b{re.escape(suffix)}[.,]?\s*$'
        if re.search(pattern, name_upper):
            result = re.sub(pattern, '', name, flags=re.IGNORECASE).strip()
            if result and result != name:  # Only return if we actually changed something
                return result
    return name


def abbreviate_business_words(name: str) -> str:
    """Replace common business words with abbreviations."""
    words = name.split()
    changed = False
    for i, word in enumerate(words):
        word_upper = word.upper().rstrip('.,')
        if word_upper in BUSINESS_ABBREVIATIONS:
            words[i] = BUSINESS_ABBREVIATIONS[word_upper]
            changed = True
    return ' '.join(words) if changed else name


def expand_abbreviations(name: str) -> str:
    """Expand abbreviations to full words."""
    reverse_map = {v: k for k, v in BUSINESS_ABBREVIATIONS.items()}
    words = name.split()
    changed = False
    for i, word in enumerate(words):
        word_upper = word.upper().rstrip('.,')
        if word_upper in reverse_map:
            words[i] = reverse_map[word_upper].title()
            changed = True
    return ' '.join(words) if changed else name


def remove_prepositions(name: str) -> str:
    """Remove common prepositions from name."""
    words = name.split()
    if len(words) <= 2:  # Don't remove from very short names
        return name

    result = [w for w in words if w.upper() not in PREPOSITIONS]
    if len(result) < len(words) and result:
        return ' '.join(result)
    return name


def drop_words(name: str, n: int = 1) -> str:
    """Drop n words from the name (preferably descriptive words)."""
    words = name.split()
    if len(words) <= n + 1:  # Keep at least one meaningful word
        return name

    # Prefer dropping words that are not the first or last
    if len(words) > n + 2:
        # Drop from middle
        start = len(words) // 2
        result = words[:start] + words[start+n:]
        return ' '.join(result)
    else:
        # Drop from end
        return ' '.join(words[:-n])


def drop_patronymic(name: str) -> str:
    """Drop middle name/patronymic from person name."""
    words = name.split()
    if len(words) <= 2:
        return name

    # If 3+ words, drop middle word(s), keep first and last
    return f"{words[0]} {words[-1]}"


def reverse_name_order(name: str) -> str:
    """Reverse name order (First Last -> Last First)."""
    words = name.split()
    if len(words) < 2:
        return name

    # Simple reversal
    return f"{words[-1]} {words[0]}"


def generate_business_variants(name: str, record_id: str, data_source: str) -> List[VariantTestCase]:
    """Generate variants for business names."""
    variants = []

    # Remove legal suffix
    variant = remove_legal_suffix(name)
    if variant != name:
        variants.append(VariantTestCase(
            original_name=name,
            variant_query=variant,
            expected_record_id=record_id,
            data_source=data_source,
            record_type="ORGANIZATION",
            variant_type="legal_suffix_removed",
            description=f"Legal suffix removed from '{name}'",
            source="automated"
        ))

    # Abbreviate words
    variant = abbreviate_business_words(name)
    if variant != name:
        variants.append(VariantTestCase(
            original_name=name,
            variant_query=variant,
            expected_record_id=record_id,
            data_source=data_source,
            record_type="ORGANIZATION",
            variant_type="words_abbreviated",
            description=f"Common words abbreviated in '{name}'",
            source="automated"
        ))

    # Expand abbreviations
    variant = expand_abbreviations(name)
    if variant != name:
        variants.append(VariantTestCase(
            original_name=name,
            variant_query=variant,
            expected_record_id=record_id,
            data_source=data_source,
            record_type="ORGANIZATION",
            variant_type="abbreviations_expanded",
            description=f"Abbreviations expanded in '{name}'",
            source="automated"
        ))

    # Remove prepositions
    variant = remove_prepositions(name)
    if variant != name:
        variants.append(VariantTestCase(
            original_name=name,
            variant_query=variant,
            expected_record_id=record_id,
            data_source=data_source,
            record_type="ORGANIZATION",
            variant_type="prepositions_removed",
            description=f"Prepositions removed from '{name}'",
            source="automated"
        ))

    # Drop one word (for longer names)
    if len(name.split()) > 3:
        variant = drop_words(name, 1)
        if variant != name:
            variants.append(VariantTestCase(
                original_name=name,
                variant_query=variant,
                expected_record_id=record_id,
                data_source=data_source,
                record_type="ORGANIZATION",
                variant_type="word_dropped",
                description=f"One word dropped from '{name}'",
                source="automated"
            ))

    return variants


def generate_person_variants(name: str, record_id: str, data_source: str) -> List[VariantTestCase]:
    """Generate variants for person names."""
    variants = []

    # Drop patronymic/middle name
    if len(name.split()) > 2:
        variant = drop_patronymic(name)
        if variant != name:
            variants.append(VariantTestCase(
                original_name=name,
                variant_query=variant,
                expected_record_id=record_id,
                data_source=data_source,
                record_type="PERSON",
                variant_type="patronymic_dropped",
                description=f"Middle name/patronymic dropped from '{name}'",
                source="automated"
            ))

    # Reverse name order
    if len(name.split()) >= 2:
        variant = reverse_name_order(name)
        if variant != name:
            variants.append(VariantTestCase(
                original_name=name,
                variant_query=variant,
                expected_record_id=record_id,
                data_source=data_source,
                record_type="PERSON",
                variant_type="name_order_reversed",
                description=f"Name order reversed from '{name}'",
                source="automated"
            ))

    return variants


def get_manual_variants(loaded_record_ids: set = None) -> List[VariantTestCase]:
    """Return manually curated high-value test cases.

    NOTE: These use actual record IDs from the opensanctions dataset.
    They will only be included if the record exists in the loaded sample.

    Args:
        loaded_record_ids: Set of record_ids from loaded samples. If provided,
                          only variants for records in this set will be returned.
    """
    manual_variants = [
        # Test 1: Fuzzy company name (RNPK = Ryazan Oil Refining)
        VariantTestCase(
            original_name="RNPK JSC",
            variant_query="Ryazan Refinery",
            expected_record_id="NK-6JHQE4LmZLd9PATFPLNcMW",
            data_source="OPEN_SANCTIONS",
            record_type="ORGANIZATION",
            variant_type="fuzzy_company_name",
            description="Query missing multiple significant words",
            source="manual"
        ),

        # Test 7: Company name variation (confirmed working)
        VariantTestCase(
            original_name="ABERDEEN ASSET MANAGERS LIMITED",
            variant_query="Aberdeen Asset Management Ltd",
            expected_record_id="NK-52eqKSj5yz2z6yoxpJGnq6",
            data_source="OPEN_SANCTIONS",
            record_type="ORGANIZATION",
            variant_type="word_form_variation",
            description="'Managers' vs 'Management', 'Limited' vs 'Ltd'",
            source="manual"
        ),

        # Additional business name variants
        VariantTestCase(
            original_name="Custody Bank of Japan, Ltd.",
            variant_query="Custody Bank Japan",
            expected_record_id="NK-39MNPnyuTJTsWuf4Prniyc",
            data_source="OPEN_SANCTIONS",
            record_type="ORGANIZATION",
            variant_type="legal_suffix_removed",
            description="Removed 'of' and 'Ltd' from company name",
            source="manual"
        ),

        # Person name variants (Bychkov)
        VariantTestCase(
            original_name="Бычков Владимир Петрович",
            variant_query="Vladimir Bychkov",
            expected_record_id="NK-3NULJaQxASzh2kVjbFYdcx",
            data_source="OPEN_SANCTIONS",
            record_type="PERSON",
            variant_type="patronymic_dropped",
            description="Latin transliteration without patronymic",
            source="manual"
        ),

        VariantTestCase(
            original_name="Бычков Владимир Петрович",
            variant_query="Bychkov Vladimir",
            expected_record_id="NK-3NULJaQxASzh2kVjbFYdcx",
            data_source="OPEN_SANCTIONS",
            record_type="PERSON",
            variant_type="name_order_variation",
            description="Last name first, no patronymic",
            source="manual"
        ),

        # Typo/spelling variation (shown to work)
        VariantTestCase(
            original_name="Бычков Владимир Петрович",
            variant_query="Vladamir Bychkov",
            expected_record_id="NK-3NULJaQxASzh2kVjbFYdcx",
            data_source="OPEN_SANCTIONS",
            record_type="PERSON",
            variant_type="spelling_variation",
            description="Typo: 'Vladamir' vs 'Vladimir' (a instead of i)",
            source="manual"
        ),
    ]

    # Filter manual variants to only include records in the loaded sample
    if loaded_record_ids is not None:
        all_manual = manual_variants
        manual_variants = [v for v in all_manual if v.expected_record_id in loaded_record_ids]
        filtered_count = len(all_manual) - len(manual_variants)
        if filtered_count > 0:
            print(f"⚠️  Filtered {filtered_count} manual variants (records not in loaded sample)")
            print(f"   Kept {len(manual_variants)} manual variants from loaded sample")

    return manual_variants


def load_validation_samples(input_file: str) -> List[Dict[str, Any]]:
    """Load existing validation samples."""
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def generate_variants_from_samples(samples: List[Dict[str, Any]],
                                   max_per_record: int = 3) -> List[VariantTestCase]:
    """Generate automated variants from validation samples."""
    all_variants = []

    for sample in samples:
        record_type = sample.get("record_type", "")
        record_id = sample.get("record_id")
        data_source = sample.get("data_source", "")
        query_name = sample.get("query_name", "")

        if not record_id or not query_name or not data_source:
            continue

        # Generate variants for this name
        record_variants = []
        if record_type == "ORGANIZATION":
            record_variants.extend(generate_business_variants(query_name, record_id, data_source))
        elif record_type == "PERSON":
            record_variants.extend(generate_person_variants(query_name, record_id, data_source))

        # Limit variants per record
        all_variants.extend(record_variants[:max_per_record])

    return all_variants


def save_variants(variants: List[VariantTestCase], output_file: str):
    """Save variants to JSONL file."""
    with open(output_file, 'w') as f:
        for variant in variants:
            record = {
                "original_name": variant.original_name,
                "variant_query": variant.variant_query,
                "expected_record_id": variant.expected_record_id,
                "data_source": variant.data_source,
                "record_type": variant.record_type,
                "variant_type": variant.variant_type,
                "description": variant.description,
                "source": variant.source
            }
            f.write(json.dumps(record) + '\n')


def print_summary(variants: List[VariantTestCase]):
    """Print summary of generated variants."""
    print(f"\n{'='*80}")
    print("VARIANT GENERATION SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal variants: {len(variants)}")

    # By source
    automated = sum(1 for v in variants if v.source == "automated")
    manual = sum(1 for v in variants if v.source == "manual")
    print(f"\nBy source:")
    print(f"  Automated: {automated}")
    print(f"  Manual:    {manual}")

    # By record type
    orgs = sum(1 for v in variants if v.record_type == "ORGANIZATION")
    persons = sum(1 for v in variants if v.record_type == "PERSON")
    print(f"\nBy record type:")
    print(f"  Organizations: {orgs}")
    print(f"  Persons:       {persons}")

    # By variant type
    print(f"\nBy variant type:")
    variant_types = {}
    for v in variants:
        variant_types[v.variant_type] = variant_types.get(v.variant_type, 0) + 1
    for vtype, count in sorted(variant_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {vtype}: {count}")

    # Show some examples
    print(f"\n{'='*80}")
    print("SAMPLE VARIANTS (first 10)")
    print(f"{'='*80}\n")
    for i, variant in enumerate(variants[:10], 1):
        print(f"{i}. [{variant.source.upper()}] {variant.variant_type}")
        print(f"   Original:  {variant.original_name}")
        print(f"   Variant:   {variant.variant_query}")
        print(f"   Record ID: {variant.expected_record_id}")
        print(f"   Source:    {variant.data_source}")
        print(f"   Desc:      {variant.description}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate fuzzy name variants for validation testing")
    parser.add_argument("-i", "--input", type=str, required=True,
                       help="Input validation samples JSONL file")
    parser.add_argument("-o", "--output", type=str, required=True,
                       help="Output variants JSONL file")
    parser.add_argument("--max-per-record", type=int, default=3,
                       help="Maximum automated variants per record (default: 3)")
    parser.add_argument("--manual-only", action="store_true",
                       help="Only include manually curated variants")
    parser.add_argument("--automated-only", action="store_true",
                       help="Only include automated variants")

    args = parser.parse_args()

    print("⏳ Loading validation samples...")
    samples = load_validation_samples(args.input)
    print(f"✅ Loaded {len(samples)} validation samples")

    all_variants = []

    # Add automated variants
    if not args.manual_only:
        print(f"\n⏳ Generating automated variants (max {args.max_per_record} per record)...")
        automated_variants = generate_variants_from_samples(samples, args.max_per_record)
        all_variants.extend(automated_variants)
        print(f"✅ Generated {len(automated_variants)} automated variants")

    # Add manual variants (filtered to only include records from loaded sample)
    if not args.automated_only:
        print("\n⏳ Adding manually curated variants...")
        # Get set of record IDs from loaded samples
        loaded_record_ids = {s["record_id"] for s in samples if "record_id" in s}
        manual_variants = get_manual_variants(loaded_record_ids)
        all_variants.extend(manual_variants)
        print(f"✅ Added {len(manual_variants)} manual variants from loaded sample")

    # Remove duplicates (same variant_query + record_id)
    seen = set()
    unique_variants = []
    for v in all_variants:
        key = (v.variant_query.lower(), v.expected_record_id)
        if key not in seen:
            seen.add(key)
            unique_variants.append(v)

    if len(unique_variants) < len(all_variants):
        print(f"\n⚠️  Removed {len(all_variants) - len(unique_variants)} duplicate variants")

    print(f"\n⏳ Saving {len(unique_variants)} variants to {args.output}...")
    save_variants(unique_variants, args.output)
    print("✅ Variants saved")

    # Print summary
    print_summary(unique_variants)


if __name__ == "__main__":
    main()
