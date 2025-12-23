#!/usr/bin/env python3
"""
Create filtered test triplets that only include records present in the loaded dataset.
"""
import sys
import orjson

def get_loaded_record_ids(input_file):
    """Extract all record IDs from the loaded dataset file."""
    record_ids = set()
    record_types = {}  # Track record type for each ID

    print(f"📖 Reading record IDs from {input_file}...")
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                print(f"  Processed {line_num} lines...", end='\r')

            record = orjson.loads(line)
            record_id = record.get('RECORD_ID')
            record_type = record.get('RECORD_TYPE')

            if record_id:
                record_ids.add(record_id)
                record_types[record_id] = record_type

    print(f"\n✓ Found {len(record_ids)} unique record IDs")
    print(f"  Business: {sum(1 for rt in record_types.values() if rt == 'ORGANIZATION')}")
    print(f"  Personal: {sum(1 for rt in record_types.values() if rt == 'PERSON')}")
    return record_ids, record_types

def filter_triplets(triplets_file, loaded_ids, output_file, record_types):
    """Filter triplets to only include those with anchor_group in loaded_ids."""
    print(f"\n📖 Filtering triplets from {triplets_file}...")

    business_triplets = []
    personal_triplets = []
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
                record_type = record_types.get(anchor_group)

                if record_type == 'ORGANIZATION':
                    business_triplets.append(triplet)
                elif record_type == 'PERSON':
                    personal_triplets.append(triplet)

    print(f"\n✓ Matched {matched} triplets out of {total} total ({matched/total*100:.1f}%)")
    print(f"  Business triplets: {len(business_triplets)}")
    print(f"  Personal triplets: {len(personal_triplets)}")

    # Write business triplets
    biz_output = output_file.replace('.jsonl', '_business.jsonl')
    with open(biz_output, 'wb') as f:
        for triplet in business_triplets:
            f.write(orjson.dumps(triplet) + b'\n')
    print(f"\n✓ Wrote {len(business_triplets)} business triplets to {biz_output}")

    # Write personal triplets
    personal_output = output_file.replace('.jsonl', '_personal.jsonl')
    with open(personal_output, 'wb') as f:
        for triplet in personal_triplets:
            f.write(orjson.dumps(triplet) + b'\n')
    print(f"✓ Wrote {len(personal_triplets)} personal triplets to {personal_output}")

    return len(business_triplets), len(personal_triplets)

if __name__ == '__main__':
    input_file = 'opensanctions_test_5k_final.jsonl'
    triplets_file = '/home/roncewind/roncewind.git/BizNames/output/opensanctions_test_triplets.jsonl'
    output_file = 'opensanctions_test_5k_triplets.jsonl'

    # Get loaded record IDs
    loaded_ids, record_types = get_loaded_record_ids(input_file)

    # Filter triplets
    biz_count, personal_count = filter_triplets(triplets_file, loaded_ids, output_file, record_types)

    print(f"\n{'='*60}")
    print("✓ Filtered triplets created successfully!")
    print(f"  Business: opensanctions_test_5k_triplets_business.jsonl ({biz_count} triplets)")
    print(f"  Personal: opensanctions_test_5k_triplets_personal.jsonl ({personal_count} triplets)")
    print(f"{'='*60}")
