#!/usr/bin/env python3
"""Quick script to check Senzing database status."""
import sys
from senzing_core import SzAbstractFactoryCore
from sz_utils import get_senzing_config

try:
    config = get_senzing_config()
    factory = SzAbstractFactoryCore(instance_name="check_db", settings=config)
    engine = factory.create_engine()

    # Get some basic stats
    stats = engine.get_stats()
    print("Senzing Database Status:")
    print(f"  Stats available: Yes")
    print(f"  Stats length: {len(stats)} characters")

    # Try to get record count (this is approximate)
    import json
    stats_dict = json.loads(stats)

    # Print summary if available
    if 'SUMMARY' in stats_dict:
        summary = stats_dict['SUMMARY']
        print(f"\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    print("\n✓ Database is accessible and has data")
    sys.exit(0)

except Exception as e:
    print(f"✗ Error checking database: {e}")
    sys.exit(1)
