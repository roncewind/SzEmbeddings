#!/usr/bin/env python3
"""
Debug utility to compare different Senzing search modes.
Shows: 0) Name-only, 1) Embedding-only, 2) PostgreSQL cosine similarity,
       3) Candidate keys, 4) Name+Embedding final results (should use BOTH), 5) Analysis

Usage:
    python sz_debug_search.py "John Smith" --type personal
    python sz_debug_search.py "Acme Corporation" --type business
    python sz_debug_search.py "Puget Sound Energy" --type business --top 20 --verbose
    python sz_debug_search.py "PSE" --type business  # Test with abbreviation
    python sz_debug_search.py "Carlyle" --type business  # Embedding-only will find it!
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import psycopg2
from senzing import SzEngine, SzEngineFlags
from senzing_core import SzAbstractFactoryCore
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from sz_utils import get_embedding, get_senzing_config


def search_postgresql_cosine(
    cursor,
    query_name: str,
    record_type: str,
    model: SentenceTransformer,
    truncate_dim: int,
    top_k: int = 20
) -> Tuple[List[Tuple[str, float]], float]:
    """Direct cosine similarity search in PostgreSQL."""
    table_name = "name_embedding" if record_type == "PERSON" else "bizname_embedding"

    start_time = time.time()
    query_embedding = get_embedding(query_name, model, truncate_dim)

    query = f"""
        SELECT LABEL, EMBEDDING <=> %s::vector AS distance
        FROM {table_name}
        ORDER BY EMBEDDING <=> %s::vector
        LIMIT %s;
    """

    cursor.execute(query, (query_embedding.tolist(), query_embedding.tolist(), top_k))
    results = cursor.fetchall()
    query_time_ms = (time.time() - start_time) * 1000

    return results, query_time_ms


def search_senzing_attribute(
    engine: SzEngine,
    query_name: str,
    record_type: str,
    model: SentenceTransformer,
    truncate_dim: int,
    data_source: str,
    top_k: int = 20
) -> Tuple[List[dict], str, float]:
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
                "NAME_EMBEDDING": embedding_str,
                "NAME_ALGORITHM": "LaBSE"
            }]
        }
    else:  # ORGANIZATION
        search_record = {
            "DATA_SOURCE": data_source,
            "RECORD_TYPE": record_type,
            "NAME_ORG": query_name,
            "BIZNAME_EMBEDDINGS": [{
                "BIZNAME_LABEL": query_name,
                "BIZNAME_EMBEDDING": embedding_str,
                "BIZNAME_ALGORITHM": "LaBSE"
            }]
        }

    search_json = json.dumps(search_record)

    start_time = time.time()
    result_json = engine.search_by_attributes(
        search_json,
        SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS |
        SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES |
        SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES |
        SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_DATA
    )
    query_time_ms = (time.time() - start_time) * 1000

    result = json.loads(result_json)
    entities = result.get("RESOLVED_ENTITIES", [])  # Get ALL entities, limit in display

    return entities, result_json, query_time_ms


def search_name_only(
    engine: SzEngine,
    query_name: str,
    record_type: str,
    top_k: int = 20
) -> Tuple[List[dict], float]:
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
        SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS |
        SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES |
        SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES |
        SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_DATA
    )
    query_time_ms = (time.time() - start_time) * 1000

    result = json.loads(result_json)
    entities = result.get("RESOLVED_ENTITIES", [])  # Get ALL entities, limit in display

    return entities, query_time_ms


def search_embedding_only(
    engine: SzEngine,
    query_name: str,
    record_type: str,
    model: SentenceTransformer,
    truncate_dim: int,
    data_source: str,
    top_k: int = 20
) -> Tuple[List[dict], float]:
    """Senzing search with EMBEDDING only (no name) for comparison."""
    embedding = get_embedding(query_name, model, truncate_dim)
    embedding_str = json.dumps(embedding.tolist())

    if record_type == "PERSON":
        search_record = {
            "DATA_SOURCE": data_source,
            "RECORD_TYPE": record_type,
            "NAME_EMBEDDINGS": [{
                "NAME_LABEL": query_name,
                "NAME_EMBEDDING": embedding_str,
                "NAME_ALGORITHM": "LaBSE"
            }]
        }
    else:  # ORGANIZATION
        search_record = {
            "DATA_SOURCE": data_source,
            "RECORD_TYPE": record_type,
            "BIZNAME_EMBEDDINGS": [{
                "BIZNAME_LABEL": query_name,
                "BIZNAME_EMBEDDING": embedding_str,
                "BIZNAME_ALGORITHM": "LaBSE"
            }]
        }

    search_json = json.dumps(search_record)

    start_time = time.time()
    result_json = engine.search_by_attributes(
        search_json,
        SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS |
        SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES |
        SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES |
        SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_DATA
    )
    query_time_ms = (time.time() - start_time) * 1000

    result = json.loads(result_json)
    entities = result.get("RESOLVED_ENTITIES", [])  # Get ALL entities, limit in display

    return entities, query_time_ms


def print_results(
    query_name: str,
    record_type: str,
    name_only_entities: List[dict],
    name_only_time: float,
    emb_only_entities: List[dict],
    emb_only_time: float,
    pg_results: List[Tuple[str, float]],
    pg_time: float,
    sz_entities: List[dict],
    sz_time: float,
    sz_json: str,
    top_k: int,
    verbose: bool = False
):
    """Print comprehensive comparison results."""

    embedding_feature = "NAME_EMBEDDING" if record_type == "PERSON" else "BIZNAME_EMBEDDING"

    print("\n" + "=" * 120)
    print(f"COMPREHENSIVE SEARCH ANALYSIS: {query_name} ({record_type})")
    print("=" * 120)

    # Section 0: Name-Only Search (baseline)
    print(f"\n{'='*120}")
    print(f"0️⃣  SENZING NAME-ONLY SEARCH (Baseline - No Embeddings)")
    print(f"{'='*120}")
    print(f"Query Time: {name_only_time:.1f}ms | Results: {len(name_only_entities)}")
    print()

    # Sort by NAME score
    def get_name_score(entity):
        feature_scores = entity.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})
        if "NAME" in feature_scores and feature_scores["NAME"]:
            return feature_scores["NAME"][0].get("SCORE", 0)
        return 0

    sorted_name_only = sorted(name_only_entities, key=get_name_score, reverse=True)

    print(f"{'Rank':<6} {'Score':<8} {'Entity ID':<12} {'Name':<60}")
    print("-" * 120)

    for i, entity in enumerate(sorted_name_only[:10], 1):
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

    if not name_only_entities:
        print("  (No results)")

    # Section 1: Embedding-Only Search
    print(f"\n{'='*120}")
    print(f"1️⃣  SENZING EMBEDDING-ONLY SEARCH (Semantic Similarity - No Name)")
    print(f"{'='*120}")
    print(f"Query Time: {emb_only_time:.1f}ms | Results: {len(emb_only_entities)}")
    print()

    # Try to sort by embedding score if available, otherwise by entity ID
    def get_emb_score(entity):
        feature_scores = entity.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})
        if embedding_feature in feature_scores and feature_scores[embedding_feature]:
            return feature_scores[embedding_feature][0].get("SCORE", 0)
        return 0

    sorted_emb_only = sorted(emb_only_entities, key=get_emb_score, reverse=True)

    print(f"{'Rank':<6} {'Emb Score':<10} {'Entity ID':<12} {'Name':<60}")
    print("-" * 120)

    has_emb_scores = False
    for i, entity in enumerate(sorted_emb_only[:10], 1):
        entity_id = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID", "N/A")

        # Try to get embedding score
        match_info = entity.get("MATCH_INFO", {})
        feature_scores = match_info.get("FEATURE_SCORES", {})
        if embedding_feature in feature_scores and feature_scores[embedding_feature]:
            emb_score = feature_scores[embedding_feature][0].get("SCORE", 0)
            score_display = str(emb_score)
            has_emb_scores = True
        else:
            score_display = "N/A"

        # Extract name - try multiple sources
        candidate_keys = match_info.get("CANDIDATE_KEYS", {})
        name = None

        # Try from embedding candidate keys first
        if embedding_feature in candidate_keys and candidate_keys[embedding_feature]:
            name = candidate_keys[embedding_feature][0].get("FEAT_DESC")

        # Try from records
        if not name:
            records = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("RECORDS", [])
            if records:
                name = records[0].get("NAME_ORG") or records[0].get("NAME_FULL")

        # Fallback
        if not name:
            name = "N/A"

        if len(name) > 60:
            name = name[:57] + "..."

        print(f"{i:<6} {score_display:<10} {entity_id:<12} {name:<60}")

    if not has_emb_scores and emb_only_entities:
        print()
        print(f"⚠️  NOTE: Embedding scores NOT in FEATURE_SCORES (config issue - embeddings not used for scoring)")

    if not emb_only_entities:
        print("  (No results)")

    # Section 2: PostgreSQL Cosine Similarity
    print(f"\n{'='*120}")
    print(f"2️⃣  POSTGRESQL COSINE SIMILARITY (Direct Embedding Search)")
    print(f"{'='*120}")
    print(f"Query Time: {pg_time:.1f}ms | Results: {len(pg_results)}")
    print()
    print(f"{'Rank':<6} {'Similarity':<12} {'Distance':<12} {'Name':<80}")
    print("-" * 120)

    for i, (label, distance) in enumerate(pg_results, 1):
        similarity = 1.0 - distance
        print(f"{i:<6} {similarity:<12.4f} {distance:<12.4f} {label:<80}")

    if not pg_results:
        print("  (No results)")

    # Section 3: Senzing Candidate Keys
    print(f"\n{'='*120}")
    print(f"3️⃣  SENZING CANDIDATE KEYS (What Embeddings Retrieved)")
    print(f"{'='*120}")
    print(f"Shows which names/entities were retrieved by embedding similarity (threshold > 0.43)")
    print()

    if sz_entities:
        # Collect all unique candidates across all entities
        all_candidates = {}
        for entity in sz_entities:
            match_info = entity.get("MATCH_INFO", {})
            candidate_keys = match_info.get("CANDIDATE_KEYS", {})

            if embedding_feature in candidate_keys:
                for candidate in candidate_keys[embedding_feature]:
                    feat_id = candidate.get("FEAT_ID")
                    feat_desc = candidate.get("FEAT_DESC")
                    if feat_id not in all_candidates:
                        all_candidates[feat_id] = feat_desc

        if all_candidates:
            print(f"Total unique names retrieved by embeddings: {len(all_candidates)}")
            print(f"(Showing ALL candidates from all entities, not limited to top-{top_k})")
            print()
            print(f"{'#':<6} {'Feature ID':<12} {'Name':<80}")
            print("-" * 120)
            # Show ALL candidates (not limited by top_k)
            for i, (feat_id, feat_desc) in enumerate(list(all_candidates.items()), 1):
                print(f"{i:<6} {feat_id:<12} {feat_desc:<80}")
        else:
            print("⚠️  NO embedding candidates found (embeddings not used for retrieval)")
    else:
        print("  (No results)")

    # Section 4: Final Ranked Entities
    print(f"\n{'='*120}")
    print(f"4️⃣  SENZING FINAL RANKED ENTITIES (After Scoring)")
    print(f"{'='*120}")
    print(f"Query Time: {sz_time:.1f}ms | Total Results: {len(sz_entities)} | Displaying Top: {min(top_k, len(sz_entities))}")
    print()

    # Sort entities by NAME score (descending)
    def get_name_score(entity):
        feature_scores = entity.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})
        if "NAME" in feature_scores and feature_scores["NAME"]:
            return feature_scores["NAME"][0].get("SCORE", 0)
        return 0

    sorted_entities = sorted(sz_entities, key=get_name_score, reverse=True)[:top_k]

    print(f"{'Rank':<6} {'Score':<8} {'Entity ID':<12} {'Match Level':<16} {'Match Key':<25} {'Name':<40}")
    print("-" * 120)

    # Counters for summary
    emb_retrieved_no_score = 0
    emb_retrieved_with_score = 0
    not_via_embeddings = 0

    for i, entity in enumerate(sorted_entities, 1):
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

        # Get NAME score
        name_score = get_name_score(entity)

        print(f"{i:<6} {name_score:<8} {entity_id:<12} {match_level:<16} {match_key:<25} {name:<40}")

        # Count embedding contribution (but don't print per entity)
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

    if not sz_entities:
        print("  (No results)")
    else:
        # Print summary
        print()
        print("Embedding Contribution:")
        if emb_retrieved_no_score > 0:
            print(f"  ⚠️  {emb_retrieved_no_score} entities retrieved via {embedding_feature} but NO embedding score (ranked by NAME only)")
        if emb_retrieved_with_score > 0:
            print(f"  ✅ {emb_retrieved_with_score} entities retrieved via {embedding_feature} AND have embedding scores")
        if not_via_embeddings > 0:
            print(f"  ℹ️  {not_via_embeddings} entities NOT retrieved via embeddings (traditional NAME matching)")

    # Section 5: Analysis
    print(f"\n{'='*120}")
    print(f"5️⃣  ANALYSIS")
    print(f"{'='*120}")

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
        print(f"\nOverlapping names:")
        for name in sorted(list(overlap)[:5]):
            print(f"  - {name}")

    # Check embedding contribution
    print(f"\nEmbedding Contribution Analysis:")
    emb_retrieved = sum(1 for e in sz_entities
                       if embedding_feature in e.get("MATCH_INFO", {}).get("CANDIDATE_KEYS", {}))
    emb_scored = sum(1 for e in sz_entities
                    if embedding_feature in e.get("MATCH_INFO", {}).get("FEATURE_SCORES", {}))

    print(f"  Entities retrieved via embeddings:  {emb_retrieved}/{len(sz_entities)}")
    print(f"  Entities scored with embeddings:    {emb_scored}/{len(sz_entities)}")

    if emb_retrieved > 0 and emb_scored == 0:
        print(f"\n  ⚠️  ISSUE: Embeddings used for retrieval but NOT for scoring!")
        print(f"  ⚠️  All {emb_retrieved} entities ranked by NAME scores only")
    elif emb_retrieved > 0 and emb_scored > 0:
        print(f"\n  ✅ Embeddings used for both retrieval and scoring")
    elif emb_retrieved == 0:
        print(f"\n  ⚠️  Embeddings NOT used for retrieval")

    # Verbose output
    if verbose:
        print("\n" + "=" * 120)
        print("6️⃣  VERBOSE - Full Senzing JSON Response")
        print("=" * 120)
        print(json.dumps(json.loads(sz_json), indent=2))

    print("\n" + "=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive search debug utility"
    )

    parser.add_argument("query_name", type=str,
                       help="Name to search for")
    parser.add_argument("--type", type=str, required=True, choices=["personal", "business"],
                       help="Record type: personal or business")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Model path (defaults based on type)")
    parser.add_argument("--truncate_dim", type=int, default=512,
                       help="Matryoshka truncation dimension (default: 512)")
    parser.add_argument("--data_source", type=str, default="OPEN_SANCTIONS",
                       help="Data source for Senzing search (default: OPEN_SANCTIONS)")
    parser.add_argument("--top", type=int, default=20,
                       help="Number of top results to return (default: 20)")
    parser.add_argument("--pg_host", type=str, default="localhost")
    parser.add_argument("--pg_port", type=int, default=5432)
    parser.add_argument("--pg_database", type=str, default="senzing")
    parser.add_argument("--pg_user", type=str, default="senzing")
    parser.add_argument("--pg_password", type=str, default="senzing")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show verbose output including full Senzing JSON")

    args = parser.parse_args()

    # Determine record type and default model
    record_type = "PERSON" if args.type == "personal" else "ORGANIZATION"

    if args.model_path:
        model_path = args.model_path
    else:
        if args.type == "personal":
            model_path = os.path.expanduser("~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model")
        else:
            model_path = os.path.expanduser("~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model")

    print(f"🔍 Searching for: {args.query_name}")
    print(f"📌 Type: {record_type}")
    print(f"🤖 Model: {model_path}")
    print(f"📏 Truncation: {args.truncate_dim} dimensions")

    # Load model
    print("⏳ Loading model...")
    model = SentenceTransformer(model_path)

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
        engine, args.query_name, record_type, model, args.truncate_dim,
        args.data_source, args.top
    )

    pg_results, pg_time = search_postgresql_cosine(
        pg_cursor, args.query_name, record_type, model, args.truncate_dim, args.top
    )

    sz_entities, sz_json, sz_time = search_senzing_attribute(
        engine, args.query_name, record_type, model, args.truncate_dim,
        args.data_source, args.top
    )

    # Print results
    print_results(
        args.query_name, record_type,
        name_only_entities, name_only_time,
        emb_only_entities, emb_only_time,
        pg_results, pg_time,
        sz_entities, sz_time,
        sz_json,
        args.top,
        args.verbose
    )

    # Cleanup
    pg_cursor.close()
    pg_conn.close()


if __name__ == "__main__":
    main()
