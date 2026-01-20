# -----------------------------------------------------------------------------
# Read Senzing JSONL file and import records with embeddings into Senzing.
# Uses ONNX models for embedding generation (faster inference, no GPU required).
#
# Example usage:
# python sz_load_embeddings_onnx.py -i data/test_samples/opensanctions_test_500.jsonl \
#   --name_model_path ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-onnx-fp16-native \
#   --biz_model_path ~/roncewind.git/BizNames/output/phase10_quantization/onnx_fp16_direct/ \
#   --truncate_dim 512
#
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import bz2
import concurrent.futures
import gzip
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import TextIO

import numpy as np
import orjson
from senzing import SzEngine, SzEngineFlags
from senzing.szerror import SzRetryableError, SzError
from senzing_core import SzAbstractFactoryCore

from onnx_sentence_transformer import ONNXSentenceTransformer, load_onnx_model
from sz_utils import format_seconds_to_hhmmss, get_senzing_config, extract_algorithm_name

# -----------------------------------------------------------------------------
# logging setup (will be reconfigured based on --debug flag in main)
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------------

number_of_lines_to_process = 0

# -----------------------------------------------------------------------------
# Global variables for failure tracking (thread-safe)
# -----------------------------------------------------------------------------

import threading

failed_records = []  # List of failed records for retry
failed_records_lock = threading.Lock()  # Thread-safe access


# =============================================================================
# Get embedding using ONNX model
def get_embedding_onnx(
    name: str, model: ONNXSentenceTransformer, truncate_dim: int | None = None
) -> np.ndarray:
    """
    Create embedding for a single name using ONNX model.

    Args:
        name: Text to embed
        model: ONNXSentenceTransformer model
        truncate_dim: Optional Matryoshka truncation dimension (e.g., 512)

    Returns:
        Embedding vector as float16 numpy array
    """
    embedding = model.encode(
        [name],
        normalize_embeddings=True,
    )

    # Apply Matryoshka truncation if specified
    if truncate_dim is not None and truncate_dim < embedding.shape[1]:
        embedding = embedding[:, :truncate_dim]
        # Re-normalize to unit length after truncation
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    return embedding[0].astype(np.float16, copy=False)


# =============================================================================
# Parse JSON and add embeddings to entity (runs in main thread for thread safety)
def prepare_entity_with_embeddings(
    line: str | bytes,
    name_model: ONNXSentenceTransformer,
    biz_model: ONNXSentenceTransformer,
    truncate_dim: int | None = None,
) -> dict | None:
    """
    Parse JSON line and compute embeddings.
    Returns the entity dict with embeddings added, or None if line should be skipped.

    Args:
        line: JSON line to parse
        name_model: Personal names ONNX model
        biz_model: Business names ONNX model
        truncate_dim: Optional Matryoshka truncation dimension (e.g., 512)
    """
    # Parse the JSON string
    entity = orjson.loads(line)
    record_type = entity.get("RECORD_TYPE")

    if not record_type:
        logger.warning(f"Missing RECORD_TYPE in record: {entity.get('RECORD_ID', 'unknown')}")
        return entity  # Still return entity to be added without embeddings

    model = None
    name_field = ""
    embed_field = ""
    label_field = ""
    if record_type == "PERSON":
        name_field = "NAME_FULL"
        embed_field = "NAME_EMBEDDING"
        label_field = "NAME_LABEL"
        model = name_model
    elif record_type == "ORGANIZATION":
        name_field = "NAME_ORG"
        embed_field = "BIZNAME_EMBEDDING"
        label_field = "BIZNAME_LABEL"
        model = biz_model
    else:
        logger.warning(f"Unknown RECORD_TYPE '{record_type}' in record: {entity.get('RECORD_ID', 'unknown')}")
        return entity  # Still return entity to be added without embeddings

    names = entity.get("NAMES", [])
    if names and model:
        # Embed names one at a time to ensure consistent embeddings
        embed_list = []
        for name_entry in names:
            name_value = name_entry.get(name_field)
            if name_value:
                embedding = get_embedding_onnx(name_value, model, truncate_dim)
                embed_list.append({
                    label_field: name_value,
                    embed_field: f"{embedding.tolist()}"
                })
            else:
                logger.debug(f"Missing {name_field} in name entry for record: {entity.get('RECORD_ID', 'unknown')}")
        if embed_list:
            entity[embed_field + 'S'] = embed_list

    return entity


# =============================================================================
# Add record to Senzing (called from thread pool)
def add_record_to_senzing(
    entity: dict,
    engine: SzEngine,
    save_failures: bool = False,
) -> tuple[bool, str | None]:
    """
    Add a prepared entity to Senzing. Safe to call from multiple threads.

    Args:
        entity: The entity record to add
        engine: Senzing engine instance
        save_failures: Whether to save failed records for retry

    Returns:
        (success, error_type) where error_type is 'retryable', 'other', or None
    """
    try:
        # Use default flags (SZ_NO_FLAGS) which skips detailed response for faster loading
        engine.add_record(
            entity.get("DATA_SOURCE", ""),
            entity.get("RECORD_ID", ""),
            json.dumps(entity, ensure_ascii=False)
        )
        return (True, None)

    except SzRetryableError as err:
        # Retryable errors (SENZ0010 timeout, SENZ1001 too long, etc.)
        error_msg = str(err)
        logger.error(f"Retryable error: {entity.get('RECORD_ID', 'unknown')} - {error_msg}")

        if save_failures:
            with failed_records_lock:
                failed_records.append({
                    'entity': entity,
                    'error': error_msg,
                    'error_type': 'retryable',
                    'timestamp': datetime.now().isoformat()
                })

        return (False, 'retryable')

    except SzError as err:
        # Other Senzing errors - check if retryable
        error_msg = str(err)
        logger.error(f"Senzing error: {entity.get('RECORD_ID', 'unknown')} - {error_msg}")

        # "too long" errors (e.g., name exceeds varchar(300) limit) are NOT retryable
        # These records will be skipped - the field would need to be truncated or the schema changed
        if 'too long' in error_msg.lower() or 'SENZ1001' in error_msg or '22001' in error_msg:
            logger.warning(f"SKIPPING record with field too long: {entity.get('RECORD_ID', 'unknown')} - this record will NOT be retried")
            if save_failures:
                with failed_records_lock:
                    failed_records.append({
                        'entity': entity,
                        'error': error_msg,
                        'error_type': 'too_long',  # Non-retryable
                        'timestamp': datetime.now().isoformat()
                    })
            return (False, 'too_long')

        # Other non-retryable Senzing errors
        if save_failures:
            with failed_records_lock:
                failed_records.append({
                    'entity': entity,
                    'error': error_msg,
                    'error_type': 'other',
                    'timestamp': datetime.now().isoformat()
                })
        return (False, 'other')

    except Exception as err:
        logger.error(f"{err}")
        return (False, 'other')


# =============================================================================
# Parse a line from the file
def parse_line(line: str) -> str | None:
    """Strip and clean a line. Returns None if line should be skipped."""
    stripped_line = line.strip()
    if len(stripped_line) < 10:
        return None
    # strip off trailing comma (for JSON array format)
    if stripped_line.endswith(","):
        stripped_line = stripped_line[:-1]
    return stripped_line


# =============================================================================
# Read a jsonl file and process
# - Embeddings computed in main thread (thread-safe, consistent results)
# - add_record() calls threaded (benefits from I/O parallelism)
STATUS_PRINT_LINES = 100
MAX_PENDING_RECORDS = 100  # Max records waiting to be added to Senzing


def read_file_futures(
    file_handle: TextIO,
    name_model: ONNXSentenceTransformer,
    biz_model: ONNXSentenceTransformer,
    engine: SzEngine,
    max_workers: int | None = None,
    truncate_dim: int | None = None,
    save_failures: bool = False,
) -> int:
    """
    Process file with embeddings computed in main thread and DB writes threaded.

    Args:
        file_handle: Input file handle
        name_model: Personal names ONNX model
        biz_model: Business names ONNX model
        engine: Senzing engine
        max_workers: Number of worker threads
        truncate_dim: Optional Matryoshka truncation dimension (e.g., 512)
        save_failures: Whether to save failed records for retry
    """
    line_count = 0
    records_submitted = 0
    start_time = time.time()

    # Use explicit max_workers or let ThreadPoolExecutor choose default
    effective_max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
        print(f"   Threads for DB writes: {effective_max_workers}")
        print(" ", flush=True, end='\r')

        futures: dict[concurrent.futures.Future, int] = {}

        for line in file_handle:
            # Parse and embed in main thread (thread-safe)
            parsed = parse_line(line)
            if parsed is None:
                continue

            try:
                entity = prepare_entity_with_embeddings(parsed, name_model, biz_model, truncate_dim)
                if entity is None:
                    continue
            except Exception as e:
                logger.exception(f"Error preparing entity: {e}")
                continue

            # Submit add_record to thread pool (I/O-bound, benefits from threading)
            future = executor.submit(add_record_to_senzing, entity, engine, save_failures)
            futures[future] = records_submitted
            records_submitted += 1

            # Limit pending futures to avoid memory buildup
            while len(futures) >= MAX_PENDING_RECORDS:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    try:
                        success, error_type = f.result()
                        if success:
                            line_count += 1
                        # Failures are tracked in failed_records list if save_failures=True
                    except Exception as e:
                        logger.exception(f"Error adding record: {e}")
                    del futures[f]

            # Progress reporting
            if records_submitted % 1000 == 0:
                logger.debug(engine.get_stats())
            if records_submitted % STATUS_PRINT_LINES == 0:
                print(f"   {records_submitted:,} records processed, {line_count:,} added in {format_seconds_to_hhmmss(time.time() - start_time)}", flush=True, end='\r')

            # Check line limit
            if number_of_lines_to_process > 0 and records_submitted >= number_of_lines_to_process:
                break

        # Wait for remaining futures to complete
        for f in concurrent.futures.as_completed(futures):
            try:
                success, error_type = f.result()
                if success:
                    line_count += 1
                # Failures are tracked in failed_records list if save_failures=True
            except Exception as e:
                logger.exception(f"Error adding record: {e}")

    print("\n")
    print(f"{records_submitted:,} records processed, {line_count:,} successfully added", flush=True)
    print(f"{format_seconds_to_hhmmss(time.time() - start_time)} total time", flush=True)
    return line_count


# =============================================================================
# Retry failed records (single-threaded to avoid lock contention)
def retry_failed_records(
    failed_records_list: list,
    engine: SzEngine,
    retry_delay: int = 0,
) -> list:
    """
    Retry records that failed during initial load.

    Args:
        failed_records_list: List of failed record dicts
        engine: Senzing engine instance
        retry_delay: Seconds to wait before starting retry (default: 0)

    Returns:
        List of records that still failed after retry
    """
    if not failed_records_list:
        return []

    print(f"\n{'='*80}")
    print(f"RETRY PHASE: {len(failed_records_list)} failed records")
    print(f"{'='*80}")

    if retry_delay > 0:
        print(f"Waiting {retry_delay} seconds before retry...")
        time.sleep(retry_delay)

    still_failing = []
    stats = {
        'total': len(failed_records_list),
        'success': 0,
        'timeout': 0,
        'too_long': 0,
        'other': 0
    }

    start_time = time.time()

    # Retry each record single-threaded
    for i, record_data in enumerate(failed_records_list, 1):
        entity = record_data.get('entity', record_data)
        record_id = entity.get('RECORD_ID', 'unknown')
        original_error = record_data.get('error_type', 'unknown')

        if i % 10 == 0 or i == 1:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(failed_records_list) - i) / rate if rate > 0 else 0
            print(f"[{i}/{len(failed_records_list)}] Retrying {record_id[:50]}... (ETA: {remaining:.0f}s)", flush=True, end='\r')

        # Retry with save_failures=False (don't recursively track retry failures)
        success, error_type = add_record_to_senzing(entity, engine, save_failures=False)

        if success:
            stats['success'] += 1
            logger.info(f"Retry success: {record_id} (was: {original_error})")
        else:
            # Still failing
            stats[error_type] += 1
            still_failing.append(record_data)
            logger.warning(f"Still failing: {record_id} ({error_type})")

    # Final report
    elapsed = time.time() - start_time
    print("\n")
    print(f"{'='*80}")
    print(f"RETRY RESULTS")
    print(f"{'='*80}")
    print(f"Total retried:     {stats['total']}")
    print(f"Successful:        {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Still timeout:     {stats['timeout']} ({stats['timeout']/stats['total']*100:.1f}%)")
    print(f"Still too long:    {stats['too_long']} ({stats['too_long']/stats['total']*100:.1f}%)")
    print(f"Other errors:      {stats['other']} ({stats['other']/stats['total']*100:.1f}%)")
    print(f"Time elapsed:      {format_seconds_to_hhmmss(elapsed)} ({len(failed_records_list)/elapsed:.2f} records/sec)")
    print(f"{'='*80}\n")

    return still_failing


# =============================================================================
# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sz_load_embeddings_onnx",
        description="Loads business name and personal name embeddings using ONNX models."
    )

    parser.add_argument("-i", "--infile", action="store", required=True, help='Path to Senzing JSON file.')
    parser.add_argument('--name_model_path', type=str, required=True, help='Path to personal names ONNX model.')
    parser.add_argument('--biz_model_path', type=str, required=True, help='Path to business names ONNX model.')
    parser.add_argument('--truncate_dim', type=int, default=None, help='Matryoshka truncation dimension (e.g., 512 for 768d models)')
    parser.add_argument('--threads', type=int, default=12, help='Number of worker threads for database writes (default: 12)')
    parser.add_argument('--retry', choices=['auto', 'manual', 'none'], default='auto',
                        help='Retry strategy: auto=retry automatically after load (default), manual=save to file for manual retry, none=skip retry')
    parser.add_argument('--retry-delay', type=int, default=0,
                        help='Seconds to wait before starting retry (default: 0)')
    parser.add_argument('--retry-output', type=str, default='still_failing.jsonl',
                        help='Output file for records that still fail after retry (default: still_failing.jsonl)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging (default: INFO level)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose Senzing logging')
    args = parser.parse_args()

    # Configure logging based on --debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        force=True  # Override any existing configuration
    )

    infile_path = args.infile

    # Extract algorithm names for tracking
    name_algo = extract_algorithm_name(args.name_model_path)
    biz_algo = extract_algorithm_name(args.biz_model_path)

    print("=" * 80)
    print("ONNX EMBEDDING LOADER")
    print("=" * 80)
    print(f"   Personal names model: {args.name_model_path}")
    print(f"   Personal names algo:  {name_algo}")
    print(f"   Business names model: {args.biz_model_path}")
    print(f"   Business names algo:  {biz_algo}")
    print("=" * 80)

    print("\nLoading ONNX models...")
    name_model = load_onnx_model(args.name_model_path)
    print(f"   Personal names: {name_model.embedding_dimension}d, max_seq={name_model.max_seq_length}")

    biz_model = load_onnx_model(args.biz_model_path)
    print(f"   Business names: {biz_model.embedding_dimension}d, max_seq={biz_model.max_seq_length}")

    print("\nConnecting to Senzing...")
    settings = get_senzing_config()
    verbose_logging = 1 if args.verbose else 0
    sz_factory = SzAbstractFactoryCore("SzEmbeddingsLoaderONNX", settings, verbose_logging=verbose_logging)
    sz_engine = sz_factory.create_engine()

    start_time = datetime.now()

    truncate_dim = args.truncate_dim
    if truncate_dim:
        print(f"   Matryoshka truncation: {truncate_dim} dimensions")

    # Determine if we should save failures for retry
    save_failures = args.retry != 'none'
    if save_failures:
        print(f"   Retry mode: {args.retry}")

    print("\n" + "=" * 80)
    print("LOADING RECORDS")
    print("=" * 80)

    try:
        if infile_path.endswith(".bz2"):
            logger.info(f"Opening {infile_path}...")
            with bz2.open(infile_path, 'rt') as f:
                read_file_futures(f, name_model, biz_model, sz_engine, max_workers=args.threads, truncate_dim=truncate_dim, save_failures=save_failures)
        elif infile_path.endswith(".gz"):
            logger.info(f"Opening {infile_path}...")
            with gzip.open(infile_path, 'rt') as f:
                read_file_futures(f, name_model, biz_model, sz_engine, max_workers=args.threads, truncate_dim=truncate_dim, save_failures=save_failures)
        elif infile_path.endswith(".json") or infile_path.endswith(".jsonl"):
            logger.info(f"Opening {infile_path}...")
            with open(infile_path, 'rt') as f:
                read_file_futures(f, name_model, biz_model, sz_engine, max_workers=args.threads, truncate_dim=truncate_dim, save_failures=save_failures)
        elif infile_path == "-":
            logger.info("Opening stdin...")
            with open(sys.stdin.fileno(), 'rt') as f:
                read_file_futures(f, name_model, biz_model, sz_engine, max_workers=args.threads, truncate_dim=truncate_dim, save_failures=save_failures)
        else:
            logger.warning("Unrecognized file type.")
    except Exception:
        logger.exception("Error processing file")

    end_time = datetime.now()
    print(f"Input read from {infile_path}.", flush=True)
    print(f"    Started at: {start_time}, ended at: {end_time}.", flush=True)

    # Handle retry based on retry mode
    if args.retry == 'auto' and failed_records:
        # Separate "too_long" errors (non-retryable) from other errors
        too_long_records = [r for r in failed_records if r.get('error_type') == 'too_long']
        retryable_records = [r for r in failed_records if r.get('error_type') != 'too_long']

        if too_long_records:
            print(f"\n{'='*80}")
            print(f"SKIPPED RECORDS (field too long - NOT retryable): {len(too_long_records)}")
            print(f"{'='*80}")
            for r in too_long_records:
                record_id = r.get('entity', {}).get('RECORD_ID', 'unknown')
                logger.warning(f"   SKIPPED: {record_id} - field exceeds database column limit")
            print(f"{'='*80}\n")

        # Only retry records that are actually retryable
        still_failing = retry_failed_records(retryable_records, sz_engine, retry_delay=args.retry_delay)

        # Save still-failing records and too_long records to file if any
        all_unresolved = still_failing + too_long_records
        if all_unresolved:
            print(f"WARNING: {len(all_unresolved)} records unresolved ({len(still_failing)} retry failures + {len(too_long_records)} too_long)")
            print(f"Saving to {args.retry_output}")
            with open(args.retry_output, 'w') as f:
                for record in all_unresolved:
                    f.write(json.dumps(record) + '\n')
        elif retryable_records:
            print(f"All retryable records resolved successfully!")
        else:
            print(f"No retryable records (only {len(too_long_records)} too_long records skipped)")

    elif args.retry == 'manual' and failed_records:
        # Save to file for manual retry
        retry_file = 'timeout_errors.jsonl'
        print(f"\nWARNING: {len(failed_records)} records failed during load")
        print(f"Saving to {retry_file} for manual retry")
        with open(retry_file, 'w') as f:
            for record in failed_records:
                f.write(json.dumps(record) + '\n')
        print(f"To retry: python sz_retry_timeouts.py -i {retry_file}")

    elif failed_records:
        print(f"\nWARNING: {len(failed_records)} records failed during load (retry disabled)")

    print("\n" + "-" * 80 + "\n")
