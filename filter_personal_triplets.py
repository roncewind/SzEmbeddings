#!/usr/bin/env python3
"""
Filter personal names test triplets to match loaded records.
"""
import sys
import orjson

def get_loaded_record_ids(input_file):
    """Extract personal record IDs from the loaded dataset file."""
    record_ids = set()

    print(f"📖 Reading PERSON record IDs from {input_file}...")
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                print(f"  Processed {line_num} lines...", end='\r')

            record = orjson.loads(line)
            if record.get('RECORD_TYPE') == 'PERSON':
                record_id = record.get('RECORD_ID')
                if record_id:
                    record_ids.add(record_id)

    print(f"\n✓ Found {len(record_ids)} PERSON record IDs")
    return record_ids

def filter_triplets(triplets_file, loaded_ids, output_file):
    """Filter triplets to only include those with anchor_group in loaded_ids."""
    print(f"\n📖 Filtering personal triplets from {triplets_file}...")

    matched_triplets = []
    total = 0
    matched = 0

    with open(triplets_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"  Processed {line_num} triplets, matched {matched}...", end='\r')

            triplet = orjson.loads(line)
            total += 1
            anchor_group = triplet.get('anchor_group')

            if anchor_group in loaded_ids:
                matched += 1
                matched_triplets.append(triplet)

    print(f"\n✓ Matched {matched} triplets out of {total} total ({matched/total*100:.1f}%)")

    # Write triplets
    with open(output_file, 'wb') as f:
        for triplet in matched_triplets:
            f.write(orjson.dumps(triplet) + b'\n')
    print(f"✓ Wrote {len(matched_triplets)} triplets to {output_file}")

    return len(matched_triplets)

if __name__ == '__main__':
    input_file = 'opensanctions_test_5k_final.jsonl'
    triplets_file = '/home/roncewind/roncewind.git/PersonalNames/output/20250821/open_sanctions_test_triplets.jsonl'
    output_file = 'opensanctions_test_5k_triplets_personal.jsonl'

    # Get loaded record IDs
    loaded_ids = get_loaded_record_ids(input_file)

    # Filter triplets
    count = filter_triplets(triplets_file, loaded_ids, output_file)

    print(f"\n{'='*60}")
    print(f"✓ Filtered {count} personal triplets successfully!")
    print(f"{'='*60}")
