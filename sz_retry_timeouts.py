#!/usr/bin/env python3
"""
Retry records that failed with timeout errors during initial load.

Usage:
    python sz_retry_timeouts.py -i timeout_errors.jsonl --threads 1
"""

import argparse
import json
import logging
import time
from datetime import datetime

from senzing import SzEngine
from senzing.szerror import SzRetryableError, SzError
from senzing_core import SzAbstractFactoryCore

from sz_utils import get_senzing_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def retry_record(entity: dict, engine: SzEngine, attempt: int = 1) -> tuple[bool, str]:
    """
    Retry adding a single record to Senzing.
    
    Returns:
        (success, message)
    """
    record_id = entity.get('RECORD_ID', 'unknown')
    
    try:
        engine.add_record(
            entity.get("DATA_SOURCE", ""),
            record_id,
            json.dumps(entity, ensure_ascii=False)
        )
        logger.info(f"✅ Success (attempt {attempt}): {record_id}")
        return (True, "Success")

    except SzRetryableError as err:
        logger.warning(f"⏱️  Retryable error again (attempt {attempt}): {record_id} - {err}")
        return (False, "Retryable")

    except SzError as err:
        # Check for SENZ1001 "too long" errors which are also retryable
        error_msg = str(err)
        if 'too long' in error_msg.lower() or 'SENZ1001' in error_msg:
            logger.warning(f"⏱️  SENZ1001 too long again (attempt {attempt}): {record_id}")
            return (False, "Retryable")
        # Other Senzing errors
        logger.error(f"❌ Other Senzing error (attempt {attempt}): {record_id} - {err}")
        return (False, str(err))

    except Exception as err:
        logger.error(f"❌ Error (attempt {attempt}): {record_id} - {err}")
        return (False, str(err))


def main():
    parser = argparse.ArgumentParser(description='Retry timeout errors from load')
    parser.add_argument('-i', '--input', required=True, help='Input file with timeout errors (JSONL)')
    parser.add_argument('--max-attempts', type=int, default=3, help='Max retry attempts per record')
    parser.add_argument('--delay', type=int, default=5, help='Delay between retries (seconds)')
    parser.add_argument('--threads', type=int, default=1, help='Thread count (use 1 to avoid contention)')
    parser.add_argument('--output', help='Output file for records that still fail')
    args = parser.parse_args()
    
    # Initialize Senzing
    logger.info("Initializing Senzing...")
    settings = get_senzing_config()
    factory = SzAbstractFactoryCore('', settings=settings)
    engine = factory.create_engine()
    
    # Set thread count
    if args.threads > 1:
        logger.warning(f"Using {args.threads} threads - may cause lock contention!")
    
    # Load failed records
    logger.info(f"Loading failed records from {args.input}")
    failed_records = []
    with open(args.input, 'r') as f:
        for line in f:
            record = json.loads(line)
            failed_records.append(record)
    
    logger.info(f"Loaded {len(failed_records)} failed records")
    
    # Statistics
    stats = {
        'total': len(failed_records),
        'success': 0,
        'still_timeout': 0,
        'other_error': 0
    }
    
    still_failing = []
    start_time = time.time()
    
    # Retry each record
    for i, record_data in enumerate(failed_records, 1):
        entity = record_data.get('entity', record_data)
        record_id = entity.get('RECORD_ID', 'unknown')
        
        logger.info(f"\n[{i}/{len(failed_records)}] Retrying: {record_id}")
        
        # Try multiple attempts with delays
        success = False
        for attempt in range(1, args.max_attempts + 1):
            if attempt > 1:
                logger.info(f"  Waiting {args.delay} seconds before retry {attempt}...")
                time.sleep(args.delay)
            
            success, message = retry_record(entity, engine, attempt)
            
            if success:
                stats['success'] += 1
                break
            elif 'Timeout' in message:
                stats['still_timeout'] += 1
                if attempt == args.max_attempts:
                    still_failing.append(record_data)
            else:
                stats['other_error'] += 1
                if attempt == args.max_attempts:
                    still_failing.append(record_data)
        
        # Progress update every 10 records
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(failed_records) - i) / rate if rate > 0 else 0
            logger.info(f"Progress: {i}/{len(failed_records)} ({stats['success']} success, {stats['still_timeout']} timeout) - ETA: {remaining:.0f}s")
    
    # Final report
    elapsed = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("RETRY RESULTS")
    logger.info("="*80)
    logger.info(f"Total records: {stats['total']}")
    logger.info(f"✅ Successful: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    logger.info(f"⏱️  Still timeout: {stats['still_timeout']} ({stats['still_timeout']/stats['total']*100:.1f}%)")
    logger.info(f"❌ Other errors: {stats['other_error']} ({stats['other_error']/stats['total']*100:.1f}%)")
    logger.info(f"⏱️  Time elapsed: {elapsed:.1f}s ({len(failed_records)/elapsed:.2f} records/sec)")
    
    # Save still-failing records
    if args.output and still_failing:
        logger.info(f"\nSaving {len(still_failing)} still-failing records to {args.output}")
        with open(args.output, 'w') as f:
            for record in still_failing:
                f.write(json.dumps(record) + '\n')
    
    engine.destroy()
    factory.destroy()


if __name__ == '__main__':
    main()
