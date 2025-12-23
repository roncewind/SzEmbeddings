#!/usr/bin/env python3
"""
Cross-validate Senzing entity resolution results against PostgreSQL test results.

This script:
1. Queries which OpenSanctions entities were loaded into Senzing
2. Retrieves PostgreSQL test results for those same entities
3. Compares entity resolution behavior between Senzing and PostgreSQL
4. Generates a comparison report

Usage:
    python sz_cross_validate.py \\
        --pg_biz_db embeddings_db \\
        --pg_names_db personalnames_db \\
        --output cross_validation_report.txt
"""

import argparse
import json
import logging
import sys
from collections import defaultdict

import psycopg2
from senzing_core import SzAbstractFactoryCore

from sz_utils import get_senzing_config

# -----------------------------------------------------------------------------
# Logging setup (will be reconfigured based on --debug flag in main)
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def get_loaded_entities_from_senzing(sz_engine) -> dict:
    """
    Query Senzing to get all loaded OpenSanctions entities.

    Returns dict: {record_id: {entity_id, record_type, names}}
    """
    print("📖 Querying Senzing for loaded entities...")

    # For now, we'll need to query through the database directly
    # as the Senzing API doesn't have a "get all records" method

    # Connect to Senzing database
    config = get_senzing_config()
    config_dict = json.loads(config)

    # Extract DB connection from config
    db_config = config_dict.get('SQL', {}).get('CONNECTION')

    # This is a placeholder - actual implementation would parse the connection string
    # and query the dsrc_record table

    print("⚠️  Direct database query needed - not yet implemented")
    print("    Manual query: SELECT dsrc_id, etype_id, dsrc_record_id FROM dsrc_record WHERE dsrc_id = (SELECT dsrc_id FROM dsrc WHERE dsrc_code = 'OPEN_SANCTIONS');")

    return {}


def get_postgresql_results(db_params: dict, test_set: str, record_ids: set) -> dict:
    """
    Query PostgreSQL test database for results on specific record IDs.

    Returns dict: {record_id: {test_results}}
    """
    print(f"📖 Querying PostgreSQL {db_params['database']} for test results...")

    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    # Get embeddings and test results for the loaded records
    # Note: anchor_group in ground_truth_triplets maps to RECORD_ID in senzing.json

    results = {}

    # Query for matching records
    if record_ids:
        placeholders = ','.join(['%s'] * len(record_ids))

        # Get test triplets where these records are anchors
        cur.execute(f"""
            SELECT anchor_group, COUNT(*) as triplet_count
            FROM ground_truth_triplets
            WHERE test_set = %s
              AND anchor_group IN ({placeholders})
            GROUP BY anchor_group;
        """, (test_set,) + tuple(record_ids))

        for row in cur.fetchall():
            record_id, triplet_count = row
            results[record_id] = {
                'triplet_count': triplet_count
            }

    cur.close()
    conn.close()

    print(f"✅ Found {len(results):,} records with test data")
    return results


def compare_results(senzing_results: dict, pg_biz_results: dict, pg_names_results: dict) -> dict:
    """
    Compare entity resolution results between Senzing and PostgreSQL.

    Returns comparison statistics and details.
    """
    comparison = {
        'total_in_senzing': len(senzing_results),
        'total_in_pg_biz': len(pg_biz_results),
        'total_in_pg_names': len(pg_names_results),
        'entities_in_both': 0,
        'details': []
    }

    # Compare entity resolution behavior
    # This is where we'd analyze if Senzing is resolving entities similarly to PostgreSQL

    return comparison


def generate_report(comparison: dict, output_file: str = None):
    """Generate comparison report."""

    lines = []
    lines.append("=" * 80)
    lines.append("Cross-Validation Report: Senzing vs PostgreSQL")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Summary:")
    lines.append(f"  Entities in Senzing: {comparison['total_in_senzing']:,}")
    lines.append(f"  Entities in PG BizNames: {comparison['total_in_pg_biz']:,}")
    lines.append(f"  Entities in PG PersonalNames: {comparison['total_in_pg_names']:,}")
    lines.append(f"  Entities in both: {comparison['entities_in_both']:,}")
    lines.append("")

    report = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"✅ Report written to {output_file}")
    else:
        print(report)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate Senzing results against PostgreSQL test results"
    )

    # Database connections
    parser.add_argument("--pg_biz_db", default="embeddings_db", help="PostgreSQL BizNames database")
    parser.add_argument("--pg_names_db", default="personalnames_db", help="PostgreSQL PersonalNames database")
    parser.add_argument("--pg_user", default="senzing", help="PostgreSQL user")
    parser.add_argument("--pg_password", default="senzing", help="PostgreSQL password")
    parser.add_argument("--pg_host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--pg_port", type=int, default=5432, help="PostgreSQL port")

    # Test set
    parser.add_argument("--test_set", default="opensanctions", help="Test set name")

    # Output
    parser.add_argument("-o", "--output", help="Output report file")

    # Logging and verbosity
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (default: INFO level)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Senzing logging")

    args = parser.parse_args()

    # Configure logging based on --debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        force=True  # Override any existing configuration
    )

    print("🔍 Starting cross-validation...")
    print()

    # Initialize Senzing
    print("⏳ Initializing Senzing engine...")
    settings = get_senzing_config()
    verbose_logging = 1 if args.verbose else 0
    sz_factory = SzAbstractFactoryCore("CrossValidation", settings, verbose_logging=verbose_logging)
    sz_engine = sz_factory.create_engine()

    # Get loaded entities from Senzing
    senzing_results = get_loaded_entities_from_senzing(sz_engine)

    # Get record IDs that were loaded
    loaded_record_ids = set(senzing_results.keys())

    # Get PostgreSQL results for business names
    pg_biz_params = {
        "database": args.pg_biz_db,
        "user": args.pg_user,
        "password": args.pg_password,
        "host": args.pg_host,
        "port": args.pg_port
    }
    pg_biz_results = get_postgresql_results(pg_biz_params, args.test_set, loaded_record_ids)

    # Get PostgreSQL results for personal names
    pg_names_params = {
        "database": args.pg_names_db,
        "user": args.pg_user,
        "password": args.pg_password,
        "host": args.pg_host,
        "port": args.pg_port
    }
    pg_names_results = get_postgresql_results(pg_names_params, args.test_set, loaded_record_ids)

    # Compare results
    print()
    print("📊 Comparing results...")
    comparison = compare_results(senzing_results, pg_biz_results, pg_names_results)

    # Generate report
    print()
    generate_report(comparison, args.output)

    print()
    print("✅ Cross-validation complete!")


if __name__ == "__main__":
    main()
