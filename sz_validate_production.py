#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Production validation for SzEmbeddings.
#
# Validates that loaded records can be found by searching for their
# names/aliases. Tests both name-only and embedding attribute searches.
#
# Usage:
#   python sz_validate_production.py \
#     --input validation_samples_100.jsonl \
#     --name_model_path ~/PersonalNames/output/model \
#     --biz_model_path ~/BizNames/output/model \
#     --output validation_results.json
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import psycopg2
import torch
from sentence_transformers import SentenceTransformer
from senzing import SzEngine, SzEngineFlags, SzError
from senzing_core import SzAbstractFactoryCore

from sz_utils import format_seconds_to_hhmmss, get_embedding, get_senzing_config

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class SearchResult:
    """Results from a single search query."""
    entity_ids: list[int]           # List of ENTITY_IDs in rank order
    gnr_scores: dict[int, float]    # entity_id -> GNR score
    emb_scores: dict[int, float]    # entity_id -> embedding score
    query_time_ms: float            # Query latency in milliseconds


@dataclass
class ValidationResult:
    """Validation result for a single test case."""
    test_case_id: str
    query_name: str
    query_name_type: str
    record_type: str
    expected_entity_id: int | None

    # Name-only search results
    name_only_found: bool
    name_only_rank: int | None
    name_only_score: float | None
    name_only_total_results: int
    name_only_query_time_ms: float

    # Embedding search results
    embedding_found: bool
    embedding_rank: int | None
    embedding_score: float | None
    embedding_total_results: int
    embedding_query_time_ms: float

    # All entity scores (for threshold analysis)
    name_only_all_scores: dict[int, float]  # entity_id -> score for all entities
    embedding_all_scores: dict[int, float]  # entity_id -> score for all entities


@dataclass
class PostgreSQLValidationResult:
    """PostgreSQL validation result for a single test case."""
    test_case_id: str
    query_name: str
    record_type: str
    expected_aliases: list[str]  # All aliases from the record

    # Top-K results from PostgreSQL
    top_k_labels: list[str]      # Top K labels returned
    top_k_distances: list[float]  # Cosine distances

    # Sibling recall metrics
    aliases_found_in_top_k: int   # How many expected aliases were in top K
    sibling_recall: float         # Percentage of expected aliases found

    # Query time
    query_time_ms: float


# -----------------------------------------------------------------------------
# Search functions (reused from sz_evaluate_model.py)
# -----------------------------------------------------------------------------
def search_by_name_only(
    engine: SzEngine,
    record_type: str,
    name: str,
) -> SearchResult:
    """
    Search using name attribute only (no embeddings).
    Tests traditional Senzing name matching performance (baseline).
    """
    search_attr = "NAME_FULL" if record_type == "PERSON" else "NAME_ORG"

    attributes = json.dumps({
        search_attr: name
    })

    flags = (
        SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS
        | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
        | SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES  # Threshold-independent
    )

    start_time = time.time()
    try:
        result = engine.search_by_attributes(attributes, flags)
        query_time_ms = (time.time() - start_time) * 1000
        search_result = orjson.loads(result)
        return parse_search_result(search_result, query_time_ms)
    except SzError as err:
        logger.error(f"Search error: {err}")
        return SearchResult([], {}, {}, 0.0)


def search_with_embedding(
    engine: SzEngine,
    model: SentenceTransformer,
    record_type: str,
    name: str,
    truncate_dim: int | None = None,
) -> SearchResult:
    """
    Search using both name attribute and embedding.
    Tests combined performance of name matching + semantic similarity.
    """
    search_emb = "NAME_EMBEDDING" if record_type == "PERSON" else "BIZNAME_EMBEDDING"
    search_label = "NAME_LABEL" if record_type == "PERSON" else "BIZNAME_LABEL"
    search_attr = "NAME_FULL" if record_type == "PERSON" else "NAME_ORG"

    embedding = get_embedding(name, model, truncate_dim)
    attributes = json.dumps({
        search_attr: name,
        search_label: name,
        search_emb: f"{embedding.tolist()}"
    })

    flags = (
        SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS
        | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
        | SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES  # Threshold-independent
    )

    start_time = time.time()
    try:
        result = engine.search_by_attributes(attributes, flags)
        query_time_ms = (time.time() - start_time) * 1000
        search_result = orjson.loads(result)
        return parse_search_result(search_result, query_time_ms)
    except SzError as err:
        logger.error(f"Search error: {err}")
        return SearchResult([], {}, {}, 0.0)


def parse_search_result(result_dict: dict[str, Any], query_time_ms: float) -> SearchResult:
    """Parse Senzing search result into SearchResult structure.

    NOTE: Senzing returns results in unsorted order. We must sort by score
    to get proper rankings.
    """
    entity_scores = []  # List of (entity_id, combined_score) tuples
    gnr_scores = {}
    emb_scores = {}

    entities = result_dict.get("RESOLVED_ENTITIES", [])
    for entity in entities:
        entity_id = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID")
        if entity_id is None:
            continue

        # Extract feature scores
        feature_scores = entity.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})

        # GNR score (NAME feature)
        gnr_feature = feature_scores.get("NAME", [])
        if gnr_feature:
            gnr_scores[entity_id] = gnr_feature[0].get("SCORE", 0)

        # Embedding score
        emb_feature = (feature_scores.get("NAME_EMBEDDING", []) or
                      feature_scores.get("BIZNAME_EMBEDDING", []))
        if emb_feature:
            emb_scores[entity_id] = emb_feature[0].get("SCORE", 0)

        # Calculate combined score for sorting
        # Prefer embedding score if available (though it usually isn't due to Senzing bug),
        # otherwise use GNR score
        combined_score = emb_scores.get(entity_id, gnr_scores.get(entity_id, 0))
        entity_scores.append((entity_id, combined_score))

    # Sort by score (descending) - highest scores first
    entity_scores.sort(key=lambda x: x[1], reverse=True)
    entity_ids = [entity_id for entity_id, score in entity_scores]

    return SearchResult(entity_ids, gnr_scores, emb_scores, query_time_ms)


def get_entity_by_record_id(
    engine: SzEngine,
    data_source: str,
    record_id: str,
) -> int | None:
    """
    Look up the entity ID for a given record.
    Returns None if record not found or error occurs.
    """
    try:
        flags = SzEngineFlags.SZ_ENTITY_DEFAULT_FLAGS
        result = engine.get_entity_by_record_id(data_source, record_id, flags)
        entity_dict = orjson.loads(result)
        entity_id = entity_dict.get("RESOLVED_ENTITY", {}).get("ENTITY_ID")
        return entity_id
    except SzError:
        return None


# -----------------------------------------------------------------------------
# Validation functions
# -----------------------------------------------------------------------------
def load_validation_samples(input_file: str) -> list[dict[str, Any]]:
    """Load validation test cases from JSONL file."""
    print(f"⏳ Loading validation samples from {input_file}...")

    test_cases = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                test_case = orjson.loads(stripped)
                test_cases.append(test_case)
            except Exception as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue

    print(f"✅ Loaded {len(test_cases):,} test cases")
    return test_cases


def lookup_entity_ids(
    engine: SzEngine,
    test_cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """
    Look up expected entity IDs for all test cases.
    Returns (test_cases_with_entity_ids, skipped_count)
    """
    print(f"⏳ Looking up entity IDs for {len(test_cases):,} test cases...")

    updated_cases = []
    skipped = 0

    for i, tc in enumerate(test_cases):
        if (i + 1) % 100 == 0:
            print(f"⏳ Progress: {i+1:,}/{len(test_cases):,} ({(i+1)/len(test_cases)*100:.1f}%)",
                  end='\r', flush=True)

        data_source = tc.get("data_source", "OPEN_SANCTIONS")
        record_id = tc.get("record_id")

        if not record_id:
            skipped += 1
            continue

        entity_id = get_entity_by_record_id(engine, data_source, record_id)

        if entity_id is None:
            logger.debug(f"Entity not found for record {record_id}")
            skipped += 1
            continue

        # Update test case with entity ID
        tc['expected_entity_id'] = entity_id
        updated_cases.append(tc)

    print()  # New line after progress
    print(f"✅ Found entity IDs for {len(updated_cases):,} test cases")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped:,} test cases (records not found in Senzing)")

    return updated_cases, skipped


def validate_test_case(
    engine: SzEngine,
    name_model: SentenceTransformer,
    biz_model: SentenceTransformer,
    test_case: dict[str, Any],
    truncate_dim: int | None,
) -> ValidationResult:
    """
    Validate a single test case by performing both name-only and embedding searches.
    """
    test_case_id = test_case["test_case_id"]
    query_name = test_case["query_name"]
    query_name_type = test_case["query_name_type"]
    record_type = test_case["record_type"]
    expected_entity_id = test_case["expected_entity_id"]

    # Select appropriate model
    model = name_model if record_type == "PERSON" else biz_model

    # Perform name-only search
    name_result = search_by_name_only(engine, record_type, query_name)
    name_found = expected_entity_id in name_result.entity_ids
    name_rank = name_result.entity_ids.index(expected_entity_id) + 1 if name_found else None
    name_score = name_result.gnr_scores.get(expected_entity_id) if name_found else None

    # Perform embedding search
    emb_result = search_with_embedding(engine, model, record_type, query_name, truncate_dim)
    emb_found = expected_entity_id in emb_result.entity_ids
    emb_rank = emb_result.entity_ids.index(expected_entity_id) + 1 if emb_found else None
    # Prefer embedding score if available, otherwise use GNR score
    emb_score = (emb_result.emb_scores.get(expected_entity_id) or
                 emb_result.gnr_scores.get(expected_entity_id)) if emb_found else None

    # Collect all scores for threshold analysis
    # For embedding search, prefer embedding scores but fallback to GNR if not available
    embedding_all_scores = {}
    for entity_id in emb_result.entity_ids:
        score = emb_result.emb_scores.get(entity_id) or emb_result.gnr_scores.get(entity_id, 0)
        embedding_all_scores[entity_id] = score

    return ValidationResult(
        test_case_id=test_case_id,
        query_name=query_name,
        query_name_type=query_name_type,
        record_type=record_type,
        expected_entity_id=expected_entity_id,
        name_only_found=name_found,
        name_only_rank=name_rank,
        name_only_score=name_score,
        name_only_total_results=len(name_result.entity_ids),
        name_only_query_time_ms=name_result.query_time_ms,
        embedding_found=emb_found,
        embedding_rank=emb_rank,
        embedding_score=emb_score,
        embedding_total_results=len(emb_result.entity_ids),
        embedding_query_time_ms=emb_result.query_time_ms,
        name_only_all_scores=name_result.gnr_scores,
        embedding_all_scores=embedding_all_scores,
    )


def validate_postgresql_embeddings(
    pg_conn,
    model: SentenceTransformer,
    test_case: dict[str, Any],
    truncate_dim: int | None,
    top_k: int = 100,
) -> PostgreSQLValidationResult:
    """
    Validate embeddings via direct PostgreSQL query.

    Queries the embedding table and checks:
    1. Sibling recall: Are other aliases from this record in top-K?
    2. Cosine distance distribution

    Args:
        pg_conn: PostgreSQL connection
        model: Embedding model
        test_case: Test case dictionary
        truncate_dim: Matryoshka truncation dimension
        top_k: Number of neighbors to retrieve
    """
    test_case_id = test_case["test_case_id"]
    query_name = test_case["query_name"]
    record_type = test_case["record_type"]
    expected_aliases = test_case.get("all_aliases", [])

    # Select appropriate table
    table_name = "name_embedding" if record_type == "PERSON" else "bizname_embedding"

    # Generate embedding for query
    query_embedding = get_embedding(query_name, model, truncate_dim)

    # Query PostgreSQL for top-K nearest neighbors
    start_time = time.time()
    try:
        with pg_conn.cursor() as cur:
            # Use cosine distance operator <=>
            query = f"""
                SELECT LABEL, EMBEDDING <=> %s::vector AS distance
                FROM {table_name}
                ORDER BY EMBEDDING <=> %s::vector
                LIMIT %s;
            """
            # Convert numpy array to list for PostgreSQL
            embedding_list = query_embedding.tolist()
            cur.execute(query, (embedding_list, embedding_list, top_k))
            results = cur.fetchall()

        query_time_ms = (time.time() - start_time) * 1000

        # Extract labels and distances
        top_k_labels = [row[0] for row in results]
        top_k_distances = [float(row[1]) for row in results]

        # Compute sibling recall: how many expected aliases are in top-K?
        aliases_found = sum(1 for alias in expected_aliases if alias in top_k_labels)
        sibling_recall = (aliases_found / len(expected_aliases) * 100) if expected_aliases else 0.0

        return PostgreSQLValidationResult(
            test_case_id=test_case_id,
            query_name=query_name,
            record_type=record_type,
            expected_aliases=expected_aliases,
            top_k_labels=top_k_labels,
            top_k_distances=top_k_distances,
            aliases_found_in_top_k=aliases_found,
            sibling_recall=sibling_recall,
            query_time_ms=query_time_ms,
        )

    except Exception as e:
        logger.error(f"PostgreSQL query error: {e}")
        return PostgreSQLValidationResult(
            test_case_id=test_case_id,
            query_name=query_name,
            record_type=record_type,
            expected_aliases=expected_aliases,
            top_k_labels=[],
            top_k_distances=[],
            aliases_found_in_top_k=0,
            sibling_recall=0.0,
            query_time_ms=0.0,
        )


# -----------------------------------------------------------------------------
# Metrics computation
# -----------------------------------------------------------------------------
def compute_basic_metrics(results: list[ValidationResult]) -> dict[str, Any]:
    """Compute basic validation metrics."""
    if not results:
        return {}

    total = len(results)

    # Name-only metrics
    name_found_count = sum(1 for r in results if r.name_only_found)
    name_not_found = total - name_found_count
    name_ranks = [r.name_only_rank for r in results if r.name_only_rank is not None]

    name_recall_at_k = {}
    for k in [1, 5, 10, 100]:
        count = sum(1 for r in results if r.name_only_rank is not None and r.name_only_rank <= k)
        name_recall_at_k[k] = count / total * 100

    name_mrr = np.mean([1.0 / r.name_only_rank for r in results if r.name_only_rank is not None]) if name_ranks else 0.0
    name_mean_rank = np.mean(name_ranks) if name_ranks else 0.0

    # Embedding metrics
    emb_found_count = sum(1 for r in results if r.embedding_found)
    emb_not_found = total - emb_found_count
    emb_ranks = [r.embedding_rank for r in results if r.embedding_rank is not None]

    emb_recall_at_k = {}
    for k in [1, 5, 10, 100]:
        count = sum(1 for r in results if r.embedding_rank is not None and r.embedding_rank <= k)
        emb_recall_at_k[k] = count / total * 100

    emb_mrr = np.mean([1.0 / r.embedding_rank for r in results if r.embedding_rank is not None]) if emb_ranks else 0.0
    emb_mean_rank = np.mean(emb_ranks) if emb_ranks else 0.0

    # Impact metrics
    improvement = sum(1 for r in results if r.embedding_found and not r.name_only_found)
    degradation = sum(1 for r in results if r.name_only_found and not r.embedding_found)

    # Latency
    name_latencies = [r.name_only_query_time_ms for r in results]
    emb_latencies = [r.embedding_query_time_ms for r in results]

    return {
        "total_test_cases": total,

        "name_only": {
            "found_count": name_found_count,
            "not_found_count": name_not_found,
            "not_found_rate": name_not_found / total * 100,
            "recall_at_k": name_recall_at_k,
            "mrr": name_mrr,
            "mean_rank": name_mean_rank,
            "latency_avg_ms": np.mean(name_latencies),
            "latency_p50_ms": np.percentile(name_latencies, 50),
            "latency_p95_ms": np.percentile(name_latencies, 95),
        },

        "embedding": {
            "found_count": emb_found_count,
            "not_found_count": emb_not_found,
            "not_found_rate": emb_not_found / total * 100,
            "recall_at_k": emb_recall_at_k,
            "mrr": emb_mrr,
            "mean_rank": emb_mean_rank,
            "latency_avg_ms": np.mean(emb_latencies),
            "latency_p50_ms": np.percentile(emb_latencies, 50),
            "latency_p95_ms": np.percentile(emb_latencies, 95),
        },

        "impact": {
            "improvement_count": improvement,
            "improvement_rate": improvement / total * 100,
            "degradation_count": degradation,
            "degradation_rate": degradation / total * 100,
        }
    }


def compute_threshold_analysis(results: list[ValidationResult]) -> dict[str, Any]:
    """
    Compute threshold analysis for embedding searches.

    Analyzes score distributions and tests multiple closeScore thresholds
    to determine optimal threshold settings.
    """
    if not results:
        return {}

    # Collect true positive scores (correct entity) and false positive scores (wrong entities)
    true_positive_scores = []
    false_positive_scores = []

    for r in results:
        expected_id = r.expected_entity_id

        # For each entity returned in embedding search
        for entity_id, score in r.embedding_all_scores.items():
            if entity_id == expected_id:
                # This is the correct entity
                true_positive_scores.append(score)
            else:
                # This is a wrong entity (false positive candidate)
                false_positive_scores.append(score)

    if not true_positive_scores:
        return {"error": "No true positive scores found"}

    # Score distribution statistics
    tp_scores_np = np.array(true_positive_scores)
    fp_scores_np = np.array(false_positive_scores) if false_positive_scores else np.array([])

    score_distribution = {
        "true_positives": {
            "count": len(true_positive_scores),
            "min": float(np.min(tp_scores_np)),
            "max": float(np.max(tp_scores_np)),
            "mean": float(np.mean(tp_scores_np)),
            "median": float(np.median(tp_scores_np)),
            "p25": float(np.percentile(tp_scores_np, 25)),
            "p75": float(np.percentile(tp_scores_np, 75)),
        }
    }

    if len(fp_scores_np) > 0:
        score_distribution["false_positives"] = {
            "count": len(false_positive_scores),
            "min": float(np.min(fp_scores_np)),
            "max": float(np.max(fp_scores_np)),
            "mean": float(np.mean(fp_scores_np)),
            "median": float(np.median(fp_scores_np)),
            "p25": float(np.percentile(fp_scores_np, 25)),
            "p75": float(np.percentile(fp_scores_np, 75)),
        }

    # Threshold sweep - test multiple closeScore values
    threshold_sweep = []
    test_thresholds = [50, 55, 60, 65, 70, 75, 80]

    for threshold in test_thresholds:
        # Count TPs and FPs at this threshold
        tp_count = sum(1 for score in true_positive_scores if score >= threshold)
        fp_count = sum(1 for score in false_positive_scores if score >= threshold)

        # Metrics at this threshold
        total_positive = len(true_positive_scores)
        recall = (tp_count / total_positive * 100) if total_positive > 0 else 0.0
        precision = (tp_count / (tp_count + fp_count) * 100) if (tp_count + fp_count) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Average volume per query
        avg_volume = (tp_count + fp_count) / len(results)

        threshold_sweep.append({
            "threshold": threshold,
            "tp_count": tp_count,
            "fp_count": fp_count,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "avg_volume_per_query": avg_volume,
        })

    return {
        "score_distribution": score_distribution,
        "threshold_sweep": threshold_sweep,
    }


def generate_threshold_recommendations(
    threshold_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate 3 recommended threshold sets based on analysis:
    - Balanced: Optimizes F1 score
    - Conservative: Higher precision (fewer false positives)
    - Aggressive: Higher recall (catch more true positives)

    Returns complete CFG_CFRTN threshold sets.
    """
    if not threshold_analysis or "threshold_sweep" not in threshold_analysis:
        return {}

    sweep = threshold_analysis["threshold_sweep"]

    if not sweep:
        return {}

    # Find best thresholds for different objectives

    # Balanced: Highest F1 score
    best_f1 = max(sweep, key=lambda x: x["f1"])
    balanced_threshold = best_f1["threshold"]

    # Conservative: Threshold with precision >= 90% and highest recall
    conservative_candidates = [s for s in sweep if s["precision"] >= 90.0]
    if conservative_candidates:
        conservative_threshold = max(conservative_candidates, key=lambda x: x["recall"])["threshold"]
    else:
        # Fallback: highest precision
        conservative_threshold = max(sweep, key=lambda x: x["precision"])["threshold"]

    # Aggressive: Threshold with recall >= 90% and highest precision
    aggressive_candidates = [s for s in sweep if s["recall"] >= 90.0]
    if aggressive_candidates:
        aggressive_threshold = min(aggressive_candidates, key=lambda x: x["threshold"])["threshold"]
    else:
        # Fallback: highest recall
        aggressive_threshold = min(sweep, key=lambda x: x["threshold"])["threshold"]

    # Generate complete CFG_CFRTN threshold sets
    # Format: {"sameScore": X, "closeScore": Y, "likelyScore": Z, "plausibleScore": W, "unlikelyScore": V}

    def make_cfg_cfrtn(closeScore: int) -> dict[str, int]:
        """Generate complete CFG_CFRTN with standard intervals from closeScore."""
        return {
            "sameScore": 100,           # Always 100 for exact matches
            "closeScore": closeScore,   # Primary cutoff (CFUNC_CFRTN)
            "likelyScore": max(closeScore - 10, 50),
            "plausibleScore": max(closeScore - 20, 40),
            "unlikelyScore": max(closeScore - 30, 30),
        }

    # Find performance metrics for each recommendation
    def find_metrics(threshold: int) -> dict:
        matching = [s for s in sweep if s["threshold"] == threshold]
        if matching:
            return matching[0]
        # Interpolate if exact threshold not in sweep
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "avg_volume_per_query": 0.0}

    recommendations = {
        "balanced": {
            "cfg_cfrtn": make_cfg_cfrtn(balanced_threshold),
            "objective": "Maximize F1 score (balance precision and recall)",
            "metrics": find_metrics(balanced_threshold),
        },
        "conservative": {
            "cfg_cfrtn": make_cfg_cfrtn(conservative_threshold),
            "objective": "Maximize precision (minimize false positives, precision >= 90%)",
            "metrics": find_metrics(conservative_threshold),
        },
        "aggressive": {
            "cfg_cfrtn": make_cfg_cfrtn(aggressive_threshold),
            "objective": "Maximize recall (catch more true positives, recall >= 90%)",
            "metrics": find_metrics(aggressive_threshold),
        },
    }

    return recommendations


def compute_breakdown_metrics(results: list[ValidationResult]) -> dict[str, Any]:
    """
    Compute metrics with breakdowns by record type, script, and name type.

    Returns detailed metrics for each dimension:
    - by_record_type: PERSON vs ORGANIZATION
    - by_script: Latin, Cyrillic, Georgian, Arabic, Chinese, Mixed, Unknown
    - by_name_type: primary vs alias
    - rank_distribution: histogram of ranks
    """
    if not results:
        return {}

    def parse_query_name_type(query_name_type: str) -> tuple[str, str]:
        """Parse 'primary_Latin' into ('primary', 'Latin')."""
        if "_" in query_name_type:
            parts = query_name_type.split("_", 1)
            return parts[0], parts[1]
        return "unknown", "unknown"

    # Group results by dimensions
    by_record_type = defaultdict(list)
    by_script = defaultdict(list)
    by_name_type = defaultdict(list)

    for r in results:
        # By record type
        by_record_type[r.record_type].append(r)

        # By script and name type
        name_type, script = parse_query_name_type(r.query_name_type)
        by_script[script].append(r)
        by_name_type[name_type].append(r)

    # Compute metrics for each breakdown
    def compute_group_metrics(group_results: list[ValidationResult]) -> dict[str, Any]:
        """Compute metrics for a group of results."""
        if not group_results:
            return {}

        total = len(group_results)

        # Embedding metrics (focus on embedding mode)
        emb_found = sum(1 for r in group_results if r.embedding_found)
        emb_ranks = [r.embedding_rank for r in group_results if r.embedding_rank is not None]

        recall_at_k = {}
        for k in [1, 5, 10, 100]:
            count = sum(1 for r in group_results if r.embedding_rank is not None and r.embedding_rank <= k)
            recall_at_k[k] = count / total * 100

        mrr = np.mean([1.0 / r.embedding_rank for r in group_results if r.embedding_rank is not None]) if emb_ranks else 0.0
        mean_rank = np.mean(emb_ranks) if emb_ranks else 0.0
        not_found_rate = (total - emb_found) / total * 100

        return {
            "count": total,
            "found": emb_found,
            "not_found_rate": not_found_rate,
            "recall_at_k": recall_at_k,
            "mrr": mrr,
            "mean_rank": mean_rank,
        }

    # Compute breakdowns
    breakdowns = {
        "by_record_type": {
            record_type: compute_group_metrics(group_results)
            for record_type, group_results in by_record_type.items()
        },
        "by_script": {
            script: compute_group_metrics(group_results)
            for script, group_results in by_script.items()
        },
        "by_name_type": {
            name_type: compute_group_metrics(group_results)
            for name_type, group_results in by_name_type.items()
        },
    }

    # Rank distribution (embedding mode)
    rank_distribution = {
        "rank_1": sum(1 for r in results if r.embedding_rank == 1),
        "rank_2_5": sum(1 for r in results if r.embedding_rank is not None and 2 <= r.embedding_rank <= 5),
        "rank_6_10": sum(1 for r in results if r.embedding_rank is not None and 6 <= r.embedding_rank <= 10),
        "rank_11_50": sum(1 for r in results if r.embedding_rank is not None and 11 <= r.embedding_rank <= 50),
        "rank_51_100": sum(1 for r in results if r.embedding_rank is not None and 51 <= r.embedding_rank <= 100),
        "rank_gt_100": sum(1 for r in results if r.embedding_rank is not None and r.embedding_rank > 100),
        "not_found": sum(1 for r in results if r.embedding_rank is None),
    }

    breakdowns["rank_distribution"] = rank_distribution

    return breakdowns


def compute_postgresql_metrics(pg_results: list[PostgreSQLValidationResult]) -> dict[str, Any]:
    """
    Compute PostgreSQL-specific metrics.

    Focuses on:
    - Sibling recall: % of cases where all aliases from record found in top-K
    - Average sibling recall rate
    - Intra-record distance: cosine distance between aliases from same record
    """
    if not pg_results:
        return {}

    total = len(pg_results)

    # Sibling recall metrics
    perfect_sibling_recall = sum(1 for r in pg_results if r.sibling_recall == 100.0)
    avg_sibling_recall = np.mean([r.sibling_recall for r in pg_results])

    # Average query time
    avg_query_time = np.mean([r.query_time_ms for r in pg_results])

    # Compute intra-record distances (distance between aliases from same record)
    intra_record_distances = []
    for r in pg_results:
        # Find distances for expected aliases that were in top-K
        for alias in r.expected_aliases:
            if alias in r.top_k_labels:
                idx = r.top_k_labels.index(alias)
                intra_record_distances.append(r.top_k_distances[idx])

    intra_record_stats = {}
    if intra_record_distances:
        distances_np = np.array(intra_record_distances)
        intra_record_stats = {
            "count": len(intra_record_distances),
            "min": float(np.min(distances_np)),
            "max": float(np.max(distances_np)),
            "mean": float(np.mean(distances_np)),
            "median": float(np.median(distances_np)),
            "p25": float(np.percentile(distances_np, 25)),
            "p75": float(np.percentile(distances_np, 75)),
        }

    return {
        "total_queries": total,
        "perfect_sibling_recall_count": perfect_sibling_recall,
        "perfect_sibling_recall_rate": perfect_sibling_recall / total * 100,
        "avg_sibling_recall": avg_sibling_recall,
        "avg_query_time_ms": avg_query_time,
        "intra_record_distance": intra_record_stats,
    }


def print_basic_report(
    metrics: dict[str, Any],
    test_set_name: str,
    threshold_analysis: dict[str, Any] | None = None,
    recommendations: dict[str, Any] | None = None,
    breakdowns: dict[str, Any] | None = None,
    pg_metrics: dict[str, Any] | None = None,
) -> None:
    """Print validation report with threshold analysis, breakdowns, and PostgreSQL metrics."""
    print("\n" + "=" * 80)
    print("PRODUCTION VALIDATION REPORT")
    print("=" * 80)
    print(f"Test Set: {test_set_name}")
    print(f"Total Test Cases: {metrics['total_test_cases']:,}")

    name = metrics["name_only"]
    emb = metrics["embedding"]
    impact = metrics["impact"]

    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"{'':40} {'Name-Only':>15} {'Embedding':>15} {'Delta':>10}")
    print("-" * 80)

    for k in [1, 5, 10, 100]:
        name_recall = name["recall_at_k"][k]
        emb_recall = emb["recall_at_k"][k]
        delta = emb_recall - name_recall
        print(f"{'Recall@' + str(k):<40} {name_recall:>14.1f}% {emb_recall:>14.1f}% {delta:>+9.1f}%")

    print(f"{'Not Found Rate:':<40} {name['not_found_rate']:>14.1f}% {emb['not_found_rate']:>14.1f}% {emb['not_found_rate']-name['not_found_rate']:>+9.1f}%")
    print(f"{'Mean Rank (when found):':<40} {name['mean_rank']:>15.1f} {emb['mean_rank']:>15.1f} {emb['mean_rank']-name['mean_rank']:>+10.1f}")
    print(f"{'MRR:':<40} {name['mrr']:>15.4f} {emb['mrr']:>15.4f} {emb['mrr']-name['mrr']:>+10.4f}")

    print("\nEmbedding Impact:")
    print(f"  Improvement (found by EMB, not by name): {impact['improvement_rate']:>6.1f}%  ({impact['improvement_count']:,} cases)")
    print(f"  Degradation (found by name, not by EMB): {impact['degradation_rate']:>6.1f}%  ({impact['degradation_count']:,} cases)")

    print("\n" + "=" * 80)
    print("QUERY LATENCY")
    print("=" * 80)
    print(f"{'':40} {'Name-Only':>15} {'Embedding':>15}")
    print("-" * 80)
    print(f"{'Average:':<40} {name['latency_avg_ms']:>13.2f} ms {emb['latency_avg_ms']:>13.2f} ms")
    print(f"{'P50:':<40} {name['latency_p50_ms']:>13.2f} ms {emb['latency_p50_ms']:>13.2f} ms")
    print(f"{'P95:':<40} {name['latency_p95_ms']:>13.2f} ms {emb['latency_p95_ms']:>13.2f} ms")

    # Threshold Analysis Section
    if threshold_analysis and "score_distribution" in threshold_analysis:
        print("\n" + "=" * 80)
        print("THRESHOLD ANALYSIS")
        print("=" * 80)

        score_dist = threshold_analysis["score_distribution"]

        # True Positive Score Distribution
        if "true_positives" in score_dist:
            tp = score_dist["true_positives"]
            print(f"\nTrue Positive Score Distribution (Correct Entity):")
            print(f"  Count:   {tp['count']:>6,}")
            print(f"  Min:     {tp['min']:>6.1f}")
            print(f"  P25:     {tp['p25']:>6.1f}")
            print(f"  Median:  {tp['median']:>6.1f}")
            print(f"  Mean:    {tp['mean']:>6.1f}")
            print(f"  P75:     {tp['p75']:>6.1f}")
            print(f"  Max:     {tp['max']:>6.1f}")

        # False Positive Score Distribution
        if "false_positives" in score_dist:
            fp = score_dist["false_positives"]
            print(f"\nFalse Positive Score Distribution (Wrong Entities):")
            print(f"  Count:   {fp['count']:>6,}")
            print(f"  Min:     {fp['min']:>6.1f}")
            print(f"  P25:     {fp['p25']:>6.1f}")
            print(f"  Median:  {fp['median']:>6.1f}")
            print(f"  Mean:    {fp['mean']:>6.1f}")
            print(f"  P75:     {fp['p75']:>6.1f}")
            print(f"  Max:     {fp['max']:>6.1f}")

        # Threshold Sweep Table
        if "threshold_sweep" in threshold_analysis:
            sweep = threshold_analysis["threshold_sweep"]
            print(f"\nThreshold Sweep Results:")
            print(f"{'Threshold':>10} {'TP Count':>10} {'FP Count':>10} {'Recall %':>10} {'Precision %':>13} {'F1 Score':>10} {'Avg Vol':>8}")
            print("-" * 80)
            for s in sweep:
                print(f"{s['threshold']:>10} {s['tp_count']:>10,} {s['fp_count']:>10,} "
                      f"{s['recall']:>10.1f} {s['precision']:>13.1f} {s['f1']:>10.2f} {s['avg_volume_per_query']:>8.1f}")

    # Threshold Recommendations
    if recommendations:
        print("\n" + "=" * 80)
        print("THRESHOLD RECOMMENDATIONS")
        print("=" * 80)

        for rec_name, rec_data in [("BALANCED", "balanced"), ("CONSERVATIVE", "conservative"), ("AGGRESSIVE", "aggressive")]:
            if rec_data not in recommendations:
                continue

            rec = recommendations[rec_data]
            cfg = rec.get("cfg_cfrtn", {})
            metrics_data = rec.get("metrics", {})

            print(f"\n{rec_name} Configuration:")
            print(f"  Objective: {rec.get('objective', 'N/A')}")
            print(f"\n  CFG_CFRTN Thresholds:")
            print(f"    sameScore:        {cfg.get('sameScore', 0):>3}")
            print(f"    closeScore:       {cfg.get('closeScore', 0):>3}  ← Primary cutoff (CFUNC_CFRTN)")
            print(f"    likelyScore:      {cfg.get('likelyScore', 0):>3}")
            print(f"    plausibleScore:   {cfg.get('plausibleScore', 0):>3}")
            print(f"    unlikelyScore:    {cfg.get('unlikelyScore', 0):>3}")

            if metrics_data:
                print(f"\n  Expected Performance:")
                print(f"    Recall:           {metrics_data.get('recall', 0):>6.1f}%")
                print(f"    Precision:        {metrics_data.get('precision', 0):>6.1f}%")
                print(f"    F1 Score:         {metrics_data.get('f1', 0):>6.2f}")
                print(f"    Avg Volume/Query: {metrics_data.get('avg_volume_per_query', 0):>6.1f}")

    # Breakdown Metrics
    if breakdowns:
        # Rank Distribution
        if "rank_distribution" in breakdowns:
            print("\n" + "=" * 80)
            print("RANK DISTRIBUTION (Embedding Mode)")
            print("=" * 80)

            rd = breakdowns["rank_distribution"]
            total = sum(rd.values())

            print(f"\n{'Rank Range':<20} {'Count':>10} {'Percentage':>12}")
            print("-" * 80)
            print(f"{'Rank 1':<20} {rd['rank_1']:>10,} {rd['rank_1']/total*100:>11.1f}%")
            print(f"{'Rank 2-5':<20} {rd['rank_2_5']:>10,} {rd['rank_2_5']/total*100:>11.1f}%")
            print(f"{'Rank 6-10':<20} {rd['rank_6_10']:>10,} {rd['rank_6_10']/total*100:>11.1f}%")
            print(f"{'Rank 11-50':<20} {rd['rank_11_50']:>10,} {rd['rank_11_50']/total*100:>11.1f}%")
            print(f"{'Rank 51-100':<20} {rd['rank_51_100']:>10,} {rd['rank_51_100']/total*100:>11.1f}%")
            print(f"{'Rank >100':<20} {rd['rank_gt_100']:>10,} {rd['rank_gt_100']/total*100:>11.1f}%")
            print(f"{'Not Found':<20} {rd['not_found']:>10,} {rd['not_found']/total*100:>11.1f}%")

        # By Record Type
        if "by_record_type" in breakdowns:
            print("\n" + "=" * 80)
            print("BREAKDOWN BY RECORD TYPE (Embedding Mode)")
            print("=" * 80)

            print(f"\n{'Type':<15} {'Count':>8} {'Recall@1':>10} {'Recall@10':>11} {'Not Found':>11} {'MRR':>8}")
            print("-" * 80)

            for record_type in sorted(breakdowns["by_record_type"].keys()):
                m = breakdowns["by_record_type"][record_type]
                print(f"{record_type:<15} {m['count']:>8,} {m['recall_at_k'][1]:>9.1f}% "
                      f"{m['recall_at_k'][10]:>10.1f}% {m['not_found_rate']:>10.1f}% {m['mrr']:>8.4f}")

        # By Script
        if "by_script" in breakdowns:
            print("\n" + "=" * 80)
            print("BREAKDOWN BY SCRIPT (Embedding Mode)")
            print("=" * 80)

            print(f"\n{'Script':<15} {'Count':>8} {'Recall@1':>10} {'Recall@10':>11} {'Not Found':>11} {'MRR':>8}")
            print("-" * 80)

            # Sort by count descending
            sorted_scripts = sorted(
                breakdowns["by_script"].items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )

            for script, m in sorted_scripts:
                print(f"{script:<15} {m['count']:>8,} {m['recall_at_k'][1]:>9.1f}% "
                      f"{m['recall_at_k'][10]:>10.1f}% {m['not_found_rate']:>10.1f}% {m['mrr']:>8.4f}")

        # By Name Type
        if "by_name_type" in breakdowns:
            print("\n" + "=" * 80)
            print("BREAKDOWN BY NAME TYPE (Embedding Mode)")
            print("=" * 80)

            print(f"\n{'Name Type':<15} {'Count':>8} {'Recall@1':>10} {'Recall@10':>11} {'Not Found':>11} {'MRR':>8}")
            print("-" * 80)

            for name_type in sorted(breakdowns["by_name_type"].keys()):
                m = breakdowns["by_name_type"][name_type]
                print(f"{name_type:<15} {m['count']:>8,} {m['recall_at_k'][1]:>9.1f}% "
                      f"{m['recall_at_k'][10]:>10.1f}% {m['not_found_rate']:>10.1f}% {m['mrr']:>8.4f}")

    # PostgreSQL Metrics
    if pg_metrics:
        print("\n" + "=" * 80)
        print("POSTGRESQL VALIDATION")
        print("=" * 80)

        print(f"\nTotal Queries:                {pg_metrics['total_queries']:,}")
        print(f"Perfect Sibling Recall:       {pg_metrics['perfect_sibling_recall_count']:,} "
              f"({pg_metrics['perfect_sibling_recall_rate']:.1f}%)")
        print(f"Average Sibling Recall:       {pg_metrics['avg_sibling_recall']:.1f}%")
        print(f"Average Query Time:           {pg_metrics['avg_query_time_ms']:.2f} ms")

        if "intra_record_distance" in pg_metrics and pg_metrics["intra_record_distance"]:
            print("\nIntra-Record Distance (Cosine Distance Between Aliases):")
            dist = pg_metrics["intra_record_distance"]
            print(f"  Samples:  {dist['count']:,}")
            print(f"  Min:      {dist['min']:.4f}")
            print(f"  P25:      {dist['p25']:.4f}")
            print(f"  Median:   {dist['median']:.4f}")
            print(f"  Mean:     {dist['mean']:.4f}")
            print(f"  P75:      {dist['p75']:.4f}")
            print(f"  Max:      {dist['max']:.4f}")
            print("\n  Note: Lower distance = higher similarity (0.0 = identical)")

    print("\n" + "=" * 80)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="sz_validate_production",
        description="Production validation for SzEmbeddings"
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Path to validation samples JSONL file")
    parser.add_argument("--name_model_path", type=str, required=True,
                       help="Path to personal names model")
    parser.add_argument("--biz_model_path", type=str, required=True,
                       help="Path to business names model")
    parser.add_argument("--truncate_dim", type=int, default=None,
                       help="Matryoshka truncation dimension (e.g., 512)")
    parser.add_argument("--data_source", type=str, default="OPEN_SANCTIONS",
                       help="Senzing DATA_SOURCE (default: OPEN_SANCTIONS)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Optional JSON file to save results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose Senzing logging")

    # PostgreSQL validation arguments (optional)
    parser.add_argument("--validate_pg", action="store_true",
                       help="Enable PostgreSQL embedding validation")
    parser.add_argument("--pg_host", type=str, default="localhost",
                       help="PostgreSQL host (default: localhost)")
    parser.add_argument("--pg_port", type=int, default=5432,
                       help="PostgreSQL port (default: 5432)")
    parser.add_argument("--pg_database", type=str, default="senzing",
                       help="PostgreSQL database name (default: senzing)")
    parser.add_argument("--pg_user", type=str, default="senzing",
                       help="PostgreSQL username (default: senzing)")
    parser.add_argument("--pg_password", type=str, default="senzing",
                       help="PostgreSQL password (default: senzing)")

    # ONNX support
    parser.add_argument("--onnx", action="store_true",
                       help="Use ONNX models instead of PyTorch models")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        force=True
    )

    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Using device: {device}")

    if args.truncate_dim:
        print(f"📌 Matryoshka truncation: {args.truncate_dim} dimensions")

    # Load models
    if args.onnx:
        from onnx_sentence_transformer import load_onnx_model
        print(f"⏳ Loading personal names ONNX model from {args.name_model_path}...")
        name_model = load_onnx_model(args.name_model_path)
        print(f"⏳ Loading business names ONNX model from {args.biz_model_path}...")
        biz_model = load_onnx_model(args.biz_model_path)
    else:
        print(f"⏳ Loading personal names model from {args.name_model_path}...")
        name_model = SentenceTransformer(args.name_model_path)
        print(f"⏳ Loading business names model from {args.biz_model_path}...")
        biz_model = SentenceTransformer(args.biz_model_path)

    # Initialize Senzing
    print("⏳ Initializing Senzing engine...")
    settings = get_senzing_config()
    verbose_logging = 1 if args.verbose else 0
    sz_factory = SzAbstractFactoryCore("SzProductionValidation", settings, verbose_logging=verbose_logging)
    sz_engine = sz_factory.create_engine()

    # Initialize PostgreSQL connection if requested
    pg_conn = None
    if args.validate_pg:
        print(f"⏳ Connecting to PostgreSQL at {args.pg_host}:{args.pg_port}/{args.pg_database}...")
        try:
            pg_conn = psycopg2.connect(
                host=args.pg_host,
                port=args.pg_port,
                database=args.pg_database,
                user=args.pg_user,
                password=args.pg_password,
            )
            print("✅ PostgreSQL connection established")
        except Exception as e:
            print(f"⚠️  Warning: PostgreSQL connection failed: {e}", file=sys.stderr)
            print("   Continuing without PostgreSQL validation...", file=sys.stderr)
            args.validate_pg = False

    # Load test cases
    test_cases = load_validation_samples(args.input)

    if not test_cases:
        print("❌ No test cases loaded", file=sys.stderr)
        sys.exit(1)

    # Look up entity IDs
    test_cases, skipped = lookup_entity_ids(sz_engine, test_cases)

    if not test_cases:
        print("❌ No valid test cases (all records not found in Senzing)", file=sys.stderr)
        sys.exit(1)

    # Validate all test cases
    print(f"\n⏳ Validating {len(test_cases):,} test cases...")
    start_time = time.time()
    results = []

    for i, test_case in enumerate(test_cases):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(test_cases) - i - 1) / rate if rate > 0 else 0
            print(f"⏳ Progress: {i+1:,}/{len(test_cases):,} ({(i+1)/len(test_cases)*100:.1f}%) "
                  f"| Rate: {rate:.1f} queries/sec | ETA: {format_seconds_to_hhmmss(remaining)}",
                  end='\r', flush=True)

        result = validate_test_case(
            sz_engine,
            name_model,
            biz_model,
            test_case,
            args.truncate_dim,
        )
        results.append(result)

    print()  # New line after progress

    total_time = time.time() - start_time
    print(f"\n✅ Validation complete in {format_seconds_to_hhmmss(total_time)}")
    print(f"   Test cases: {len(results):,}")
    print(f"   Skipped: {skipped:,}")

    # PostgreSQL validation (optional)
    pg_results = []
    pg_metrics = None
    if args.validate_pg and pg_conn:
        print(f"\n⏳ Running PostgreSQL validation on {len(test_cases):,} test cases...")
        pg_start_time = time.time()

        for i, test_case in enumerate(test_cases):
            if (i + 1) % 10 == 0:
                elapsed = time.time() - pg_start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(test_cases) - i - 1) / rate if rate > 0 else 0
                print(f"⏳ Progress: {i+1:,}/{len(test_cases):,} ({(i+1)/len(test_cases)*100:.1f}%) "
                      f"| Rate: {rate:.1f} queries/sec | ETA: {format_seconds_to_hhmmss(remaining)}",
                      end='\r', flush=True)

            # Select appropriate model
            model = name_model if test_case["record_type"] == "PERSON" else biz_model

            pg_result = validate_postgresql_embeddings(
                pg_conn,
                model,
                test_case,
                args.truncate_dim,
                top_k=100,
            )
            pg_results.append(pg_result)

        print()  # New line after progress

        pg_total_time = time.time() - pg_start_time
        print(f"✅ PostgreSQL validation complete in {format_seconds_to_hhmmss(pg_total_time)}")

        # Compute PostgreSQL metrics
        print("⏳ Computing PostgreSQL metrics...")
        pg_metrics = compute_postgresql_metrics(pg_results)

        # Close PostgreSQL connection
        pg_conn.close()
        print("✅ PostgreSQL connection closed")

    # Compute metrics
    print("\n⏳ Computing metrics...")
    metrics = compute_basic_metrics(results)

    # Compute threshold analysis
    print("⏳ Computing threshold analysis...")
    threshold_analysis = compute_threshold_analysis(results)
    recommendations = generate_threshold_recommendations(threshold_analysis)

    # Compute breakdown metrics
    print("⏳ Computing breakdown metrics...")
    breakdowns = compute_breakdown_metrics(results)

    # Print report
    test_set_name = Path(args.input).stem
    print_basic_report(metrics, test_set_name, threshold_analysis, recommendations, breakdowns, pg_metrics)

    # Save results if requested
    if args.output:
        print(f"\n⏳ Saving results to {args.output}...")

        output_data = {
            "metadata": {
                "input_file": args.input,
                "name_model_path": args.name_model_path,
                "biz_model_path": args.biz_model_path,
                "truncate_dim": args.truncate_dim,
                "data_source": args.data_source,
                "total_test_cases": len(results),
                "skipped_test_cases": skipped,
                "validation_time_seconds": total_time,
            },
            "metrics": metrics,
            "threshold_analysis": threshold_analysis,
            "threshold_recommendations": recommendations,
            "breakdowns": breakdowns,
            "postgresql_metrics": pg_metrics,
            "results": [
                {
                    "test_case_id": r.test_case_id,
                    "query_name": r.query_name,
                    "query_name_type": r.query_name_type,
                    "record_type": r.record_type,
                    "expected_entity_id": r.expected_entity_id,
                    "name_only": {
                        "found": r.name_only_found,
                        "rank": r.name_only_rank,
                        "score": r.name_only_score,
                        "total_results": r.name_only_total_results,
                        "query_time_ms": r.name_only_query_time_ms,
                    },
                    "embedding": {
                        "found": r.embedding_found,
                        "rank": r.embedding_rank,
                        "score": r.embedding_score,
                        "total_results": r.embedding_total_results,
                        "query_time_ms": r.embedding_query_time_ms,
                    }
                }
                for r in results
            ]
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
