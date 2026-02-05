#!/usr/bin/env python3
"""
Debug utility to compare different Senzing search modes.
Shows: 0) Name-only, 1) Embedding-only, 2) PostgreSQL cosine similarity,
       3) Candidate keys, 4) Name+Embedding final results (should use BOTH), 5) Analysis

Usage:
    python sz_debug_search.py "John Smith" --type personal
    python sz_debug_search.py "Acme Corporation" --type business
    python sz_debug_search.py "Puget Sound Energy" --type business --top 50 --verbose
    python sz_debug_search.py "PSE" --type business  # Test with abbreviation
    python sz_debug_search.py "Carlyle" --type business  # Embedding-only will find it!
    python sz_debug_search.py "Korean Bank" --type business --why  # Show GNR validation
    python sz_debug_search.py "Korea" --type business --threshold 0.40  # Higher threshold

Options:
    --top N         Maximum results to display per section (default: 20)
                    Shows counts when results exceed this limit
    --threshold F   Minimum similarity for PostgreSQL search (default: 0.30)
                    0.30 = Senzing's unlikelyScore, 0.40 = plausibleScore, etc.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, List, Tuple

# Suppress false positive warning from transformers about Mistral regex
# (our XLMRoberta tokenizer is not affected - verified identical to original E5)
# The warning uses logging, not warnings module
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import psycopg2  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from senzing import SzEngine, SzEngineFlags  # noqa: E402
from senzing_core import SzAbstractFactoryCore  # noqa: E402

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from sz_utils import get_embedding, get_senzing_config  # noqa: E402


# -----------------------------------------------------------------------------
def search_postgresql_cosine(
    cursor: Any,  # psycopg2 cursor
    query_name: str,
    record_type: str,
    model: SentenceTransformer,
    truncate_dim: int,
    threshold: float = 0.30
) -> Tuple[List[Tuple[str, float]], float]:
    """Direct cosine similarity search in PostgreSQL.

    Args:
        threshold: Minimum similarity score (0-1). Default 0.30 matches Senzing's unlikelyScore.
                  Distance = 1 - similarity, so threshold 0.30 means distance <= 0.70.
    """
    table_name = "name_embedding" if record_type == "PERSON" else "bizname_embedding"

    start_time = time.time()
    query_embedding = get_embedding(query_name, model, truncate_dim)

    # Convert similarity threshold to distance threshold
    # cosine distance = 1 - cosine similarity
    max_distance = 1.0 - threshold

    query = f"""
        SELECT LABEL, EMBEDDING <=> %s::vector AS distance
        FROM {table_name}
        WHERE EMBEDDING <=> %s::vector <= %s
        ORDER BY EMBEDDING <=> %s::vector;
    """

    cursor.execute(query, (query_embedding.tolist(), query_embedding.tolist(), max_distance, query_embedding.tolist()))
    results = cursor.fetchall()
    query_time_ms = (time.time() - start_time) * 1000

    return results, query_time_ms


# -----------------------------------------------------------------------------
def search_senzing_attribute(
    engine: SzEngine,
    query_name: str,
    record_type: str,
    model: SentenceTransformer,
    truncate_dim: int,
    data_source: str,
    _top_k: int = 20  # Unused - returns all results, display limited elsewhere
) -> Tuple[List[dict[str, Any]], str, float]:
    """Senzing search with embedding attribute using ALL_CANDIDATES flag."""
    embedding = get_embedding(query_name, model, truncate_dim)
    embedding_str = json.dumps(embedding.tolist())

    if record_type == "PERSON":
        search_record = {
            "DATA_SOURCE": data_source,
            "RECORD_TYPE": record_type,
            "NAME_FULL": query_name,
            "NAME_EMBEDDINGS": [{
                "NAME_LABEL": query_name,
                "NAME_EMBEDDING": embedding_str
                # "NAME_ALGORITHM": "LaBSE"
            }]
        }
    else:  # ORGANIZATION
        search_record = {
            "DATA_SOURCE": data_source,
            "RECORD_TYPE": record_type,
            "NAME_ORG": query_name,
            "BIZNAME_EMBEDDINGS": [{
                "BIZNAME_LABEL": query_name,
                "BIZNAME_EMBEDDING": embedding_str
                # "BIZNAME_ALGORITHM": "LaBSE"
            }]
        }

    search_json = json.dumps(search_record)
    # print("\n\n===")
    # print(json.dumps(search_record, ensure_ascii=False))
    # print("===\n\n")

    start_time = time.time()
    result_json = engine.search_by_attributes(
        search_json,
        # SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS |
        SzEngineFlags.SZ_ENTITY_INCLUDE_ALL_FEATURES |
        SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_ENTITIES |
        SzEngineFlags.SZ_ENTITY_INCLUDE_ENTITY_NAME |
        SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_SUMMARY |
        SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES |
        SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES |
        SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_DATA
    )
    query_time_ms = (time.time() - start_time) * 1000

    result = json.loads(result_json)
    entities = result.get("RESOLVED_ENTITIES", [])  # Get ALL entities, limit in display

    return entities, result_json, query_time_ms


# -----------------------------------------------------------------------------
def get_why_search_gnr_score(
    engine: SzEngine,
    query_name: str,
    entity_id: int,
    record_type: str,
    model: SentenceTransformer | None = None,
    truncate_dim: int | None = None,
    data_source: str = "OPEN_SANCTIONS"
) -> Tuple[int | None, str | None, dict[str, Any] | None]:
    """
    Call why_search to get the GNR score for a query against an entity.

    Args:
        engine: Senzing engine
        query_name: Name to search for
        entity_id: Entity ID to compare against
        record_type: "PERSON" or "ORGANIZATION"
        model: SentenceTransformer model (required for embedding-based search)
        truncate_dim: Embedding dimension
        data_source: Data source name

    Returns:
        (gnr_score, gnr_bucket, additional_scores) or (None, None, None) if no match
    """
    # Build full search record with name and embedding (if model provided)
    if record_type == "PERSON":
        search_attrs: dict[str, Any] = {"NAME_FULL": query_name}
        if model is not None and truncate_dim is not None:
            embedding = get_embedding(query_name, model, truncate_dim)
            embedding_str = json.dumps(embedding.tolist())
            search_attrs["DATA_SOURCE"] = data_source
            search_attrs["RECORD_TYPE"] = record_type
            search_attrs["NAME_EMBEDDINGS"] = [{
                "NAME_LABEL": query_name,
                "NAME_EMBEDDING": embedding_str
            }]
    else:  # ORGANIZATION
        search_attrs = {"NAME_ORG": query_name}
        if model is not None and truncate_dim is not None:
            embedding = get_embedding(query_name, model, truncate_dim)
            embedding_str = json.dumps(embedding.tolist())
            search_attrs["DATA_SOURCE"] = data_source
            search_attrs["RECORD_TYPE"] = record_type
            search_attrs["BIZNAME_EMBEDDINGS"] = [{
                "BIZNAME_LABEL": query_name,
                "BIZNAME_EMBEDDING": embedding_str
            }]

    try:
        why_result = engine.why_search(
            json.dumps(search_attrs),
            entity_id,
            SzEngineFlags.SZ_WHY_SEARCH_DEFAULT_FLAGS | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
        )
        why_data = json.loads(why_result)

        if why_data.get('WHY_RESULTS'):
            match_info = why_data['WHY_RESULTS'][0].get('MATCH_INFO', {})
            feature_scores = match_info.get('FEATURE_SCORES', {})

            if 'NAME' in feature_scores and feature_scores['NAME']:
                gnr_score = feature_scores['NAME'][0].get('SCORE')
                gnr_bucket = feature_scores['NAME'][0].get('SCORE_BUCKET')
                additional = feature_scores['NAME'][0].get('ADDITIONAL_SCORES', {})
                return gnr_score, gnr_bucket, additional
    except Exception:
        pass  # Entity may not exist or other error

    return None, None, None


# -----------------------------------------------------------------------------
def search_name_only(
    engine: SzEngine,
    query_name: str,
    record_type: str,
    _top_k: int = 20  # Unused - returns all results, display limited elsewhere
) -> Tuple[List[dict[str, Any]], float]:
    """Senzing search with NAME only (no embeddings) for comparison."""
    if record_type == "PERSON":
        search_record = {
            "NAME_FULL": query_name
        }
    else:  # ORGANIZATION
        search_record = {
            "NAME_ORG": query_name
        }

    search_json = json.dumps(search_record)

    start_time = time.time()
    result_json = engine.search_by_attributes(
        search_json,
        # SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS |
        SzEngineFlags.SZ_ENTITY_INCLUDE_ALL_FEATURES |
        SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_ENTITIES |
        SzEngineFlags.SZ_ENTITY_INCLUDE_ENTITY_NAME |
        SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_SUMMARY |
        SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES |
        SzEngineFlags.SZ_SEARCH_INCLUDE_STATS |
        SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES |
        SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES |
        SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_DATA
    )
    query_time_ms = (time.time() - start_time) * 1000

    result = json.loads(result_json)
    entities = result.get("RESOLVED_ENTITIES", [])  # Get ALL entities, limit in display

    return entities, query_time_ms


# -----------------------------------------------------------------------------
def search_embedding_only(
    engine: SzEngine,
    query_name: str,
    record_type: str,
    model: SentenceTransformer,
    truncate_dim: int,
    data_source: str,
    _top_k: int = 20  # Unused - returns all results, display limited elsewhere
) -> Tuple[List[dict[str, Any]], float]:
    """Senzing search with EMBEDDING only (no name) for comparison."""
    embedding = get_embedding(query_name, model, truncate_dim)
    embedding_str = json.dumps(embedding.tolist())

    if record_type == "PERSON":
        search_record = {
            "DATA_SOURCE": data_source,
            "RECORD_TYPE": record_type,
            "NAME_EMBEDDINGS": [{
                "NAME_LABEL": query_name,
                "NAME_EMBEDDING": embedding_str
                # "NAME_ALGORITHM": "LaBSE"
            }]
        }
    else:  # ORGANIZATION
        search_record = {
            "DATA_SOURCE": data_source,
            "RECORD_TYPE": record_type,
            "BIZNAME_EMBEDDINGS": [{
                "BIZNAME_LABEL": query_name,
                "BIZNAME_EMBEDDING": embedding_str
                # "BIZNAME_ALGORITHM": "LaBSE"
            }]
        }

    search_json = json.dumps(search_record)

    start_time = time.time()
    result_json = engine.search_by_attributes(
        search_json,
        # SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS |
        SzEngineFlags.SZ_ENTITY_INCLUDE_ALL_FEATURES |
        SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_ENTITIES |
        SzEngineFlags.SZ_ENTITY_INCLUDE_ENTITY_NAME |
        SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_SUMMARY |
        SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES |
        SzEngineFlags.SZ_SEARCH_INCLUDE_STATS |
        SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES |
        SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES |
        SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_DATA
    )
    query_time_ms = (time.time() - start_time) * 1000

    result = json.loads(result_json)
    entities = result.get("RESOLVED_ENTITIES", [])  # Get ALL entities, limit in display

    return entities, query_time_ms


# -----------------------------------------------------------------------------
def print_results(
    query_name: str,
    record_type: str,
    name_only_entities: List[dict[str, Any]],
    name_only_time: float,
    emb_only_entities: List[dict[str, Any]],
    emb_only_time: float,
    pg_results: List[Tuple[str, float]],
    pg_time: float,
    sz_entities: List[dict[str, Any]],
    sz_time: float,
    sz_json: str,
    top_k: int,
    threshold: float,
    verbose: bool = False,
    show_why: bool = False,
    engine: SzEngine | None = None,
    model: SentenceTransformer | None = None,
    truncate_dim: int | None = None,
    data_source: str = "OPEN_SANCTIONS"
) -> None:
    """Print comprehensive comparison results.

    Args:
        top_k: Maximum number of results to display per section. If more results exist,
               shows a count of additional results.
        threshold: Similarity threshold used for PostgreSQL search.
    """

    embedding_feature = "NAME_EMBEDDING" if record_type == "PERSON" else "BIZNAME_EMBEDDING"

    # Helper functions for extracting scores from entities
    def get_name_score(entity: dict[str, Any]) -> int:
        feature_scores = entity.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})
        if "NAME" in feature_scores and feature_scores["NAME"]:
            return feature_scores["NAME"][0].get("SCORE", 0)
        return 0

    def get_embedding_score(entity: dict[str, Any]) -> int:
        feature_scores = entity.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})
        if embedding_feature in feature_scores and feature_scores[embedding_feature]:
            return feature_scores[embedding_feature][0].get("SCORE", 0)
        return 0

    def get_combined_score(entity: dict[str, Any]) -> float:
        """Probabilistic OR: treats scores as independent evidence for match."""
        name = get_name_score(entity) / 100.0
        emb = get_embedding_score(entity) / 100.0
        return 100.0 * (1 - (1 - name) * (1 - emb))

    print("\n" + "=" * 120)
    print(f"COMPREHENSIVE SEARCH ANALYSIS: {query_name} ({record_type})")
    print("=" * 120)

    # Section 0: Name-Only Search (baseline)
    print(f"\n{'=' * 120}")
    print("0️⃣  SENZING NAME-ONLY SEARCH (Baseline - No Embeddings)")
    print(f"{'=' * 120}")
    if len(name_only_entities) > top_k:
        print(f"Query Time: {name_only_time:.1f}ms | Total Results: {len(name_only_entities)} (showing top {top_k})")
    else:
        print(f"Query Time: {name_only_time:.1f}ms | Results: {len(name_only_entities)}")
    print()

    sorted_name_only = sorted(name_only_entities, key=get_name_score, reverse=True)

    print(f"{'Rank':<6} {'Score':<8} {'Entity ID':<12} {'Name':<60}")
    print("-" * 120)

    for i, entity in enumerate(sorted_name_only[:top_k], 1):
        entity_id = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID", "N/A")
        name_score = get_name_score(entity)

        # Extract name
        match_info = entity.get("MATCH_INFO", {})
        feature_scores = match_info.get("FEATURE_SCORES", {})
        name = "N/A"
        if "NAME" in feature_scores and feature_scores["NAME"]:
            name = feature_scores["NAME"][0].get("CANDIDATE_FEAT_DESC", "N/A")

        if len(name) > 60:
            name = name[:57] + "..."

        print(f"{i:<6} {name_score:<8} {entity_id:<12} {name:<60}")

    if len(name_only_entities) > top_k:
        print(f"\n  ... and {len(name_only_entities) - top_k} more results (use --top to show more)")

    if not name_only_entities:
        print("  (No results)")

    # Section 1: Embedding-Only Search
    print(f"\n{'=' * 120}")
    print("1️⃣  SENZING EMBEDDING-ONLY SEARCH (Semantic Similarity - No Name)")
    print(f"{'=' * 120}")
    if len(emb_only_entities) > top_k:
        print(f"Query Time: {emb_only_time:.1f}ms | Total Results: {len(emb_only_entities)} entities (showing top {top_k})")
    else:
        print(f"Query Time: {emb_only_time:.1f}ms | Results: {len(emb_only_entities)} entities")
    print()

    sorted_emb_only = sorted(emb_only_entities, key=get_embedding_score, reverse=True)

    print(f"{'Rank':<6} {'Entity ID':<10} {'Score':<7} {'Best Match':<50} {'Other Aliases':<40}")
    print("-" * 130)

    has_emb_scores = False
    for i, entity in enumerate(sorted_emb_only[:top_k], 1):
        entity_id = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID", "N/A")

        # Try to get embedding score
        match_info = entity.get("MATCH_INFO", {})
        feature_scores = match_info.get("FEATURE_SCORES", {})
        candidate_keys = match_info.get("CANDIDATE_KEYS", {})

        if embedding_feature in feature_scores and feature_scores[embedding_feature]:
            emb_score = feature_scores[embedding_feature][0].get("SCORE", 0)
            score_display = str(emb_score)
            has_emb_scores = True
        else:
            score_display = "N/A"

        # Get best match name from FEATURE_SCORES
        best_match = None
        if embedding_feature in feature_scores and feature_scores[embedding_feature]:
            best_match = feature_scores[embedding_feature][0].get("CANDIDATE_FEAT_DESC")

        # Get all candidate names (aliases that matched)
        all_aliases = []
        if embedding_feature in candidate_keys and candidate_keys[embedding_feature]:
            for ck in candidate_keys[embedding_feature]:
                alias = ck.get("FEAT_DESC", "")
                if alias and alias != best_match:
                    all_aliases.append(alias)

        # Fallback for best_match
        if not best_match:
            if all_aliases:
                best_match = all_aliases.pop(0)
            else:
                records = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("RECORDS", [])
                if records:
                    best_match = records[0].get("NAME_ORG") or records[0].get("NAME_FULL") or "N/A"
                else:
                    best_match = "N/A"

        # Truncate best match
        if len(best_match) > 50:
            best_match = best_match[:47] + "..."

        # Format other aliases
        if all_aliases:
            aliases_str = f"+{len(all_aliases)} more: " + ", ".join(a[:15] + "..." if len(a) > 15 else a for a in all_aliases[:2])
            if len(all_aliases) > 2:
                aliases_str += ", ..."
        else:
            aliases_str = ""

        if len(aliases_str) > 40:
            aliases_str = aliases_str[:37] + "..."

        print(f"{i:<6} {entity_id:<10} {score_display:<7} {best_match:<50} {aliases_str:<40}")

    if not has_emb_scores and emb_only_entities:
        print()
        print("⚠️  NOTE: Embedding scores NOT in FEATURE_SCORES")

    if len(emb_only_entities) > top_k:
        print(f"\n  ... and {len(emb_only_entities) - top_k} more entities (use --top to show more)")

    if not emb_only_entities:
        print("  (No results)")

    print()
    print("Note: Senzing groups results by entity. 'Other Aliases' shows additional names from")
    print("      the same entity that also matched via embedding similarity.")

    # Section 2: PostgreSQL Cosine Similarity
    print(f"\n{'=' * 120}")
    print("2️⃣  POSTGRESQL COSINE SIMILARITY (Direct Embedding Search)")
    print(f"{'=' * 120}")
    if len(pg_results) > top_k:
        print(f"Query Time: {pg_time:.1f}ms | Total Results: {len(pg_results)} (showing top {top_k}) | Threshold: similarity >= {threshold:.0%}")
    else:
        print(f"Query Time: {pg_time:.1f}ms | Results: {len(pg_results)} | Threshold: similarity >= {threshold:.0%}")
    print()
    print(f"{'Rank':<6} {'Similarity':<12} {'Distance':<12} {'Name':<80}")
    print("-" * 120)

    for i, (label, distance) in enumerate(pg_results[:top_k], 1):
        similarity = 1.0 - distance
        print(f"{i:<6} {similarity:<12.4f} {distance:<12.4f} {label:<80}")

    if len(pg_results) > top_k:
        print(f"\n  ... and {len(pg_results) - top_k} more results above threshold (use --top to show more)")

    if not pg_results:
        print("  (No results above threshold)")

    # Section 3: Senzing Candidate Keys
    print(f"\n{'=' * 120}")
    print("3️⃣  SENZING CANDIDATE KEYS (What Embeddings Retrieved)")
    print(f"{'=' * 120}")
    print(f"Shows which names/entities were retrieved by embedding similarity (threshold >= {threshold:.0%})")
    print()

    if sz_entities:
        # Collect all unique candidates across all entities, tracking entity ID
        all_candidates = {}  # feat_id -> (feat_desc, entity_id)
        for entity in sz_entities:
            entity_id = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID")
            match_info = entity.get("MATCH_INFO", {})
            candidate_keys = match_info.get("CANDIDATE_KEYS", {})

            if embedding_feature in candidate_keys:
                for candidate in candidate_keys[embedding_feature]:
                    feat_id = candidate.get("FEAT_ID")
                    feat_desc = candidate.get("FEAT_DESC")
                    if feat_id not in all_candidates:
                        all_candidates[feat_id] = (feat_desc, entity_id)

        if all_candidates:
            # Group by entity ID to show aliases together
            entity_groups = {}
            for feat_id, (feat_desc, entity_id) in all_candidates.items():
                if entity_id not in entity_groups:
                    entity_groups[entity_id] = []
                entity_groups[entity_id].append((feat_id, feat_desc))

            print(f"Total unique names retrieved by embeddings: {len(all_candidates)}")
            print(f"Unique entities: {len(entity_groups)}")
            print()
            print(f"{'#':<6} {'Entity ID':<12} {'Feature ID':<12} {'Name':<80}")
            print("-" * 120)

            # Sort by entity ID and show grouped (respect top_k)
            row_num = 1
            shown_entities = 0
            total_names_shown = 0
            for entity_id in sorted(entity_groups.keys()):
                if shown_entities >= top_k:
                    break
                candidates = entity_groups[entity_id]
                for feat_id, feat_desc in candidates:
                    print(f"{row_num:<6} {entity_id:<12} {feat_id:<12} {feat_desc:<80}")
                    row_num += 1
                    total_names_shown += 1
                # Add separator between entities if multiple aliases
                if len(candidates) > 1:
                    print()
                shown_entities += 1

            if len(entity_groups) > top_k:
                remaining_entities = len(entity_groups) - top_k
                remaining_names = len(all_candidates) - total_names_shown
                print(f"\n  ... and {remaining_entities} more entities with {remaining_names} names (use --top to show more)")
        else:
            print("⚠️  NO embedding candidates found (embeddings not used for retrieval)")
    else:
        print("  (No results)")

    # Section 4: Final Ranked Entities
    print(f"\n{'=' * 120}")
    print("4️⃣  SENZING FINAL RANKED ENTITIES (After Scoring)")
    print(f"{'=' * 120}")
    if len(sz_entities) > top_k:
        print(f"Query Time: {sz_time:.1f}ms | Total Results: {len(sz_entities)} (showing top {top_k})")
    else:
        print(f"Query Time: {sz_time:.1f}ms | Results: {len(sz_entities)}")
    print()

    sorted_entities = sorted(sz_entities, key=get_combined_score, reverse=True)

    print(f"{'Rank':<6} {'Combined':<9} {'Emb':<6} {'Name':<6} {'Entity ID':<12} {'Match Level':<16} {'Match Key':<25} {'Name':<40}")
    print("-" * 120)

    # Count ALL entities for summary (not just displayed ones)
    emb_retrieved_no_score = 0
    emb_retrieved_with_score = 0
    not_via_embeddings = 0

    for entity in sorted_entities:
        match_info = entity.get("MATCH_INFO", {})
        candidate_keys = match_info.get("CANDIDATE_KEYS", {})
        feature_scores = match_info.get("FEATURE_SCORES", {})

        has_emb_candidate = embedding_feature in candidate_keys
        has_emb_score = embedding_feature in feature_scores

        if has_emb_candidate and not has_emb_score:
            emb_retrieved_no_score += 1
        elif has_emb_candidate and has_emb_score:
            emb_retrieved_with_score += 1
        elif not has_emb_candidate:
            not_via_embeddings += 1

    # Display only top_k entities
    for i, entity in enumerate(sorted_entities[:top_k], 1):
        entity_id = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID", "N/A")
        match_info = entity.get("MATCH_INFO", {})
        match_level = match_info.get("MATCH_LEVEL_CODE", "N/A")
        match_key = match_info.get("MATCH_KEY", "N/A")

        # Extract name - try multiple sources
        name = None

        # Try from candidate keys first (most reliable)
        candidate_keys = match_info.get("CANDIDATE_KEYS", {})
        if embedding_feature in candidate_keys and candidate_keys[embedding_feature]:
            # Use first embedding candidate name
            name = candidate_keys[embedding_feature][0].get("FEAT_DESC")

        # Try from NAME candidate keys
        if not name and "NAME" in candidate_keys and candidate_keys["NAME"]:
            name = candidate_keys["NAME"][0].get("FEAT_DESC")

        # Try from feature scores
        if not name:
            feature_scores = match_info.get("FEATURE_SCORES", {})
            if "NAME" in feature_scores and feature_scores["NAME"]:
                name = feature_scores["NAME"][0].get("CANDIDATE_FEAT_DESC")

        # Try from records
        if not name:
            records = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("RECORDS", [])
            if records:
                name = records[0].get("NAME_ORG") or records[0].get("NAME_FULL")

        # Fallback
        if not name:
            name = "N/A"

        # Truncate name if too long
        if len(name) > 40:
            name = name[:37] + "..."

        # Get scores
        name_score = get_name_score(entity)
        emb_score = get_embedding_score(entity)
        combined_score = get_combined_score(entity)

        print(f"{i:<6} {combined_score:<9.1f} {emb_score:<6} {name_score:<6} {entity_id:<12} {match_level:<16} {match_key:<25} {name:<40}")

    if len(sz_entities) > top_k:
        print(f"\n  ... and {len(sz_entities) - top_k} more entities (use --top to show more)")

    if not sz_entities:
        print("  (No results)")
    else:
        # Print summary
        print()
        print("Embedding Contribution:")
        if emb_retrieved_no_score > 0:
            print(f"  ⚠️  {emb_retrieved_no_score} entities retrieved via {embedding_feature} but NO embedding score")
        if emb_retrieved_with_score > 0:
            print(f"  ✅ {emb_retrieved_with_score} entities retrieved via {embedding_feature} AND have embedding scores")
        if not_via_embeddings > 0:
            print(f"  ℹ️  {not_via_embeddings} entities NOT retrieved via embeddings (traditional NAME matching)")

        print()
        print("ℹ️  Combined Score Calculation:")
        print("    Formula: 100 * (1 - (1 - Name/100) * (1 - Emb/100))  [Probabilistic OR]")
        print(f"    Treats NAME and {embedding_feature} scores as independent evidence")
        print("    High score in either dimension yields high combined score")
        print("    Examples: Name=100,Emb=0→100 | Name=0,Emb=100→100 | Name=50,Emb=50→75")

    # Section 5: Analysis
    print(f"\n{'=' * 120}")
    print("5️⃣  ANALYSIS")
    print(f"{'=' * 120}")

    # Compare PostgreSQL vs Senzing
    pg_names = {label for label, _ in pg_results[:10]}
    sz_names = set()
    for entity in sz_entities[:10]:
        records = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("RECORDS", [])
        if records:
            name = records[0].get("NAME_ORG", records[0].get("NAME_FULL"))
            if name:
                sz_names.add(name)
        # Also check feature scores
        match_info = entity.get("MATCH_INFO", {})
        feature_scores = match_info.get("FEATURE_SCORES", {})
        if "NAME" in feature_scores and feature_scores["NAME"]:
            name = feature_scores["NAME"][0].get("CANDIDATE_FEAT_DESC")
            if name:
                sz_names.add(name)

    overlap = pg_names & sz_names

    print(f"\nPostgreSQL Results:     {len(pg_results)} names")
    print(f"Senzing Results:        {len(sz_entities)} entities")
    print(f"Top-10 Name Overlap:    {len(overlap)} / 10")

    if overlap:
        print("\nOverlapping names:")
        for name in sorted(list(overlap)[:5]):
            print(f"  - {name}")

    # Check embedding contribution
    print("\nEmbedding Contribution Analysis:")
    emb_retrieved = sum(
        1 for e in sz_entities
        if embedding_feature in e.get("MATCH_INFO", {}).get("CANDIDATE_KEYS", {})
    )
    emb_scored = sum(
        1 for e in sz_entities
        if embedding_feature in e.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})
    )

    print(f"  Entities retrieved via embeddings:  {emb_retrieved}/{len(sz_entities)}")
    print(f"  Entities scored with embeddings:    {emb_scored}/{len(sz_entities)}")

    if emb_retrieved > 0 and emb_scored == 0:
        print(f"  ⚠️  All {emb_retrieved} entities ranked but embedding scores not returned")
    elif emb_retrieved > 0 and emb_scored > 0:
        print("\n  ✅ Embeddings used for both retrieval and scoring")
    elif emb_retrieved == 0:
        print("\n  ⚠️  Embeddings NOT used for retrieval")

    # Verbose output
    if verbose:
        print("\n" + "=" * 120)
        print("6️⃣  VERBOSE - Full Senzing JSON Response")
        print("=" * 120)
        print(json.dumps(json.loads(sz_json), indent=2))

    # Why search analysis - validate embedding results with GNR
    if show_why and engine:
        print("\n" + "=" * 120)
        print("7️⃣  WHY_SEARCH GNR VALIDATION (Embedding Results vs GNR)")
        print("=" * 120)
        print("\nValidation: For each embedding result, call why_search to get GNR score")
        print("Use case: Confirm embedding matches with GNR >= 85 threshold")
        print()

        # Get entities from embedding-only search
        if emb_only_entities:
            print(f"{'Rank':<6} {'EntID':<8} {'EMB':<6} {'GNR':<6} {'Bucket':<10} {'Valid':<8} {'Matched Name (from embedding)':<40} {'GNR Matched Name':<30}")
            print("-" * 140)

            pass_count = 0
            fail_count = 0
            no_gnr_count = 0
            seen_entities = set()

            for i, entity in enumerate(emb_only_entities[:15], 1):
                entity_id = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID")

                # Skip duplicate entities (same entity, different alias)
                if entity_id in seen_entities:
                    continue
                seen_entities.add(entity_id)

                # Get embedding score
                match_info = entity.get("MATCH_INFO", {})
                feature_scores = match_info.get("FEATURE_SCORES", {})
                emb_score = 0
                if embedding_feature in feature_scores and feature_scores[embedding_feature]:
                    emb_score = feature_scores[embedding_feature][0].get("SCORE", 0)

                # Get entity name from embedding match
                emb_matched_name = "N/A"

                # Try from embedding candidate keys first
                candidate_keys = match_info.get("CANDIDATE_KEYS", {})
                if embedding_feature in candidate_keys and candidate_keys[embedding_feature]:
                    emb_matched_name = candidate_keys[embedding_feature][0].get("FEAT_DESC", "N/A")

                # Try from feature scores
                if emb_matched_name == "N/A" and embedding_feature in feature_scores and feature_scores[embedding_feature]:
                    emb_matched_name = feature_scores[embedding_feature][0].get("CANDIDATE_FEAT_DESC", "N/A")

                # Try from records
                if emb_matched_name == "N/A":
                    records = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("RECORDS", [])
                    if records:
                        emb_matched_name = records[0].get("NAME_ORG") or records[0].get("NAME_FULL") or "N/A"

                if len(emb_matched_name) > 40:
                    emb_matched_name = emb_matched_name[:37] + "..."

                # Call why_search for GNR score (include embeddings for proper matching)
                gnr_score, gnr_bucket, additional = get_why_search_gnr_score(
                    engine, query_name, entity_id, record_type,
                    model=model, truncate_dim=truncate_dim, data_source=data_source
                )

                # Get the GNR matched name (what GNR actually matched against)
                gnr_matched_name = ""
                if gnr_score is not None:
                    # Get the name that GNR matched - call why_search with full attributes
                    try:
                        # Build search attrs with embedding
                        if record_type == "PERSON":
                            search_attrs: dict[str, Any] = {"NAME_FULL": query_name}
                            if model is not None and truncate_dim is not None:
                                emb = get_embedding(query_name, model, truncate_dim)
                                search_attrs["DATA_SOURCE"] = data_source
                                search_attrs["RECORD_TYPE"] = record_type
                                search_attrs["NAME_EMBEDDINGS"] = [{
                                    "NAME_LABEL": query_name,
                                    "NAME_EMBEDDING": json.dumps(emb.tolist())
                                }]
                        else:
                            search_attrs = {"NAME_ORG": query_name}
                            if model is not None and truncate_dim is not None:
                                emb = get_embedding(query_name, model, truncate_dim)
                                search_attrs["DATA_SOURCE"] = data_source
                                search_attrs["RECORD_TYPE"] = record_type
                                search_attrs["BIZNAME_EMBEDDINGS"] = [{
                                    "BIZNAME_LABEL": query_name,
                                    "BIZNAME_EMBEDDING": json.dumps(emb.tolist())
                                }]
                        why_result = engine.why_search(
                            json.dumps(search_attrs),
                            entity_id,
                            SzEngineFlags.SZ_WHY_SEARCH_DEFAULT_FLAGS | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
                        )
                        why_data = json.loads(why_result)
                        if why_data.get('WHY_RESULTS'):
                            fs = why_data['WHY_RESULTS'][0].get('MATCH_INFO', {}).get('FEATURE_SCORES', {})
                            if 'NAME' in fs and fs['NAME']:
                                gnr_matched_name = fs['NAME'][0].get('CANDIDATE_FEAT_DESC', '')
                                if len(gnr_matched_name) > 30:
                                    gnr_matched_name = gnr_matched_name[:27] + "..."
                    except Exception:
                        pass

                # Determine validation status
                if gnr_score is not None:
                    if gnr_score >= 85:
                        valid = "PASS"
                        pass_count += 1
                    else:
                        valid = "FAIL"
                        fail_count += 1
                    gnr_str = str(gnr_score)
                    bucket_str = gnr_bucket or "N/A"
                else:
                    valid = "NO GNR"
                    no_gnr_count += 1
                    gnr_str = "None"
                    bucket_str = "N/A"

                print(f"{i:<6} {entity_id:<8} {emb_score:<6} {gnr_str:<6} {bucket_str:<10} {valid:<8} {emb_matched_name:<40} {gnr_matched_name:<30}")

            print()
            print("Summary:")
            print(f"  PASS (GNR >= 85): {pass_count}")
            print(f"  FAIL (GNR < 85):  {fail_count}")
            print(f"  NO GNR (no match):{no_gnr_count}")

            if no_gnr_count > 0:
                print()
                print("Note: 'NO GNR' means why_search found no lexical match between")
                print("      the query and entity. These are purely semantic matches.")

            print()
            print("Tip: Same entity may have multiple aliases. GNR matches against ALL")
            print("     entity names, so 'GNR Matched Name' may differ from 'Matched Name (from embedding)'.")
        else:
            print("No embedding results to validate")

    print("\n" + "=" * 120)


# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive search debug utility"
    )

    parser.add_argument(
        "query_name", type=str, help="Name to search for"
    )
    parser.add_argument(
        "--type", type=str, required=True, choices=["personal", "business"],
        help="Record type: personal or business"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Model path (defaults to E5 models based on type)"
    )
    parser.add_argument(
        "--truncate_dim", type=int, default=None,
        help="Embedding dimension (default: auto-detect from model, typically 384 for E5)"
    )
    parser.add_argument(
        "--data_source", type=str, default="OPEN_SANCTIONS",
        help="Data source for Senzing search (default: OPEN_SANCTIONS)"
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Maximum results to display per section (default: 20)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.30,
        help="Minimum similarity threshold for PostgreSQL (default: 0.30)"
    )
    parser.add_argument("--pg_host", type=str, default="localhost")
    parser.add_argument("--pg_port", type=int, default=5432)
    parser.add_argument("--pg_database", type=str, default="senzing")
    parser.add_argument("--pg_user", type=str, default="senzing")
    parser.add_argument("--pg_password", type=str, default="senzing")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show verbose output including full Senzing JSON"
    )
    parser.add_argument(
        "--why", action="store_true",
        help="Call why_search for each embedding result to show GNR validation scores"
    )
    parser.add_argument(
        "--dump", type=str, default=None,
        help="Dump the raw Senzing JSON response to this file path"
    )

    args = parser.parse_args()

    # Determine record type and default model
    record_type = "PERSON" if args.type == "personal" else "ORGANIZATION"

    if args.model_path:
        model_path = args.model_path
    else:
        # Default to E5 models (384 dimensions) - these should match what's loaded in the database
        if args.type == "personal":
            model_path = os.path.expanduser("~/roncewind.git/PersonalNames/output/e5_small_finetuned/FINAL-fine_tuned_model")
        else:
            model_path = os.path.expanduser("~/roncewind.git/BizNames/output/phase10_e5_small_noprefix/Epoch-001-fine_tuned_model")

    print(f"🔍 Searching for: {args.query_name}")
    print(f"📌 Type: {record_type}")
    print(f"🤖 Model: {model_path}")
    print(f"📊 Display limit: {args.top} results per section")
    print(f"📏 Similarity threshold: {args.threshold:.0%}")

    # Load model
    print("⏳ Loading model...")
    model = SentenceTransformer(model_path)

    # Auto-detect embedding dimension from model if not specified
    truncate_dim: int
    if args.truncate_dim is None:
        detected_dim = model.get_sentence_embedding_dimension()
        if detected_dim is None:
            raise ValueError("Could not auto-detect embedding dimension from model")
        truncate_dim = detected_dim
        print(f"📏 Auto-detected embedding dimension: {truncate_dim}")
    else:
        truncate_dim = args.truncate_dim
        print(f"📏 Using specified dimension: {truncate_dim}")

    # Initialize Senzing
    print("⏳ Initializing Senzing...")
    settings = get_senzing_config()
    sz_factory = SzAbstractFactoryCore("DebugSearch", settings, verbose_logging=0)
    engine = sz_factory.create_engine()

    # Connect to PostgreSQL
    print("⏳ Connecting to PostgreSQL...")
    pg_conn = psycopg2.connect(
        host=args.pg_host,
        port=args.pg_port,
        database=args.pg_database,
        user=args.pg_user,
        password=args.pg_password
    )
    pg_cursor = pg_conn.cursor()

    # Perform searches
    print("🔍 Executing searches...")

    name_only_entities, name_only_time = search_name_only(
        engine, args.query_name, record_type, args.top
    )

    emb_only_entities, emb_only_time = search_embedding_only(
        engine, args.query_name, record_type, model, truncate_dim,
        args.data_source, args.top
    )

    pg_results, pg_time = search_postgresql_cosine(
        pg_cursor, args.query_name, record_type, model, truncate_dim, args.threshold
    )

    sz_entities, sz_json, sz_time = search_senzing_attribute(
        engine, args.query_name, record_type, model, truncate_dim,
        args.data_source, args.top
    )

    # Dump raw JSON if requested
    if args.dump:
        with open(args.dump, 'w') as f:
            parsed = json.loads(sz_json)
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        print(f"💾 Raw JSON saved to: {args.dump}")

    # Print results
    print_results(
        args.query_name, record_type,
        name_only_entities, name_only_time,
        emb_only_entities, emb_only_time,
        pg_results, pg_time,
        sz_entities, sz_time,
        sz_json,
        args.top,
        args.threshold,
        args.verbose,
        args.why,
        engine,
        model=model,
        truncate_dim=truncate_dim,
        data_source=args.data_source
    )

    # Cleanup
    pg_cursor.close()
    pg_conn.close()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
