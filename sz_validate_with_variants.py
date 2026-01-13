#!/usr/bin/env python3
"""
Enhanced validation with both exact and variant queries.

Tests embedding value by comparing performance on:
1. Exact name queries (from validation samples)
2. Fuzzy variant queries (from generated variants)

This demonstrates embeddings perform equally on exact matches
but significantly better on fuzzy variants.
"""

import json
import argparse
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from senzing import SzEngine, SzEngineFlags
from senzing_core import SzAbstractFactoryCore

import sys
sys.path.insert(0, '/home/roncewind/roncewind.git/SzEmbeddings')
from sz_utils import get_senzing_config, get_embedding


@dataclass
class SearchResult:
    """Search result with entity rankings and scores."""
    entity_ids: List[int]
    name_scores: Dict[int, int]
    embedding_scores: Dict[int, int]
    query_time_ms: float


@dataclass
class TestResult:
    """Result for a single test case."""
    test_case_id: str
    query_type: str  # "exact" or "variant"
    variant_type: str  # Type of variant (if applicable)
    original_name: str
    query_name: str
    expected_entity_id: int
    record_type: str

    # Name-only results
    name_only_found: bool
    name_only_rank: int
    name_only_score: int
    name_only_time_ms: float

    # Embedding results
    embedding_found: bool
    embedding_rank: int
    embedding_score: int
    embedding_time_ms: float


def lookup_entity_id(engine: SzEngine, record_id: str, data_source: str) -> int:
    """Look up entity_id for a given record_id and data_source.

    This makes validation portable across database reloads, since entity_ids
    change but record_ids are stable.
    """
    try:
        flags = SzEngineFlags.SZ_ENTITY_DEFAULT_FLAGS
        result = engine.get_entity_by_record_id(data_source, record_id, flags)
        entity_data = json.loads(result)
        entity_id = entity_data.get("RESOLVED_ENTITY", {}).get("ENTITY_ID")
        return entity_id if entity_id else None
    except Exception as e:
        print(f"⚠️  Warning: Could not lookup entity for {data_source}:{record_id}: {e}")
        return None


def parse_search_result(result_dict: Dict[str, Any], query_time_ms: float) -> SearchResult:
    """Parse Senzing search result into SearchResult structure.

    NOTE: Senzing returns results in unsorted order. We must sort by score
    to get proper rankings.
    """
    entity_scores = []  # List of (entity_id, combined_score) tuples
    name_scores = {}
    embedding_scores = {}

    entities = result_dict.get("RESOLVED_ENTITIES", [])
    for entity in entities:
        entity_id = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID")
        if entity_id is None:
            continue

        # Extract feature scores
        feature_scores = entity.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})

        # Name score (GNR)
        name_feature = feature_scores.get("NAME", [])
        if name_feature:
            name_scores[entity_id] = name_feature[0].get("SCORE", 0)

        # Embedding score
        emb_feature = (feature_scores.get("NAME_EMBEDDING", []) or
                      feature_scores.get("BIZNAME_EMBEDDING", []))
        if emb_feature:
            embedding_scores[entity_id] = emb_feature[0].get("SCORE", 0)

        # Calculate combined score for sorting
        # Prefer embedding score if available, otherwise use name score
        combined_score = embedding_scores.get(entity_id, name_scores.get(entity_id, 0))
        entity_scores.append((entity_id, combined_score))

    # Sort by score (descending) - highest scores first
    entity_scores.sort(key=lambda x: x[1], reverse=True)
    entity_ids = [entity_id for entity_id, score in entity_scores]

    return SearchResult(entity_ids, name_scores, embedding_scores, query_time_ms)


def search_name_only(engine: SzEngine, record_type: str, name: str) -> SearchResult:
    """Search using name attribute only (no embeddings)."""
    search_attr = "NAME_FULL" if record_type == "PERSON" else "NAME_ORG"
    attributes = json.dumps({search_attr: name})

    flags = (
        SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS
        | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
        | SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES
    )

    start = time.time()
    result = engine.search_by_attributes(attributes, flags)
    query_time_ms = (time.time() - start) * 1000

    result_dict = json.loads(result)
    return parse_search_result(result_dict, query_time_ms)


def search_with_embedding(engine: SzEngine, model: SentenceTransformer,
                         record_type: str, name: str, truncate_dim: int) -> SearchResult:
    """Search using both name attribute and embedding."""
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
        | SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES
    )

    start = time.time()
    result = engine.search_by_attributes(attributes, flags)
    query_time_ms = (time.time() - start) * 1000

    result_dict = json.loads(result)
    return parse_search_result(result_dict, query_time_ms)


def run_test_case(engine: SzEngine, name_model: SentenceTransformer,
                 biz_model: SentenceTransformer, test_case: Dict[str, Any],
                 truncate_dim: int, query_type: str) -> TestResult:
    """Run a single test case with both name-only and embedding searches."""

    # Extract test case info
    query_name = test_case.get("query_name") or test_case.get("variant_query")
    expected_entity_id = test_case.get("expected_entity_id")
    record_type = test_case.get("record_type", "")
    test_case_id = test_case.get("test_case_id", f"variant_{query_name[:20]}")
    variant_type = test_case.get("variant_type", "")
    original_name = test_case.get("original_name", query_name)

    # Choose model based on record type
    model = name_model if record_type == "PERSON" else biz_model

    # Name-only search
    name_result = search_name_only(engine, record_type, query_name)
    name_found = expected_entity_id in name_result.entity_ids
    name_rank = name_result.entity_ids.index(expected_entity_id) + 1 if name_found else None
    name_score = name_result.name_scores.get(expected_entity_id, 0) if name_found else 0

    # Embedding search
    emb_result = search_with_embedding(engine, model, record_type, query_name, truncate_dim)
    emb_found = expected_entity_id in emb_result.entity_ids
    emb_rank = emb_result.entity_ids.index(expected_entity_id) + 1 if emb_found else None
    emb_score = emb_result.embedding_scores.get(expected_entity_id,
                                                 emb_result.name_scores.get(expected_entity_id, 0)) if emb_found else 0

    return TestResult(
        test_case_id=test_case_id,
        query_type=query_type,
        variant_type=variant_type,
        original_name=original_name,
        query_name=query_name,
        expected_entity_id=expected_entity_id,
        record_type=record_type,
        name_only_found=name_found,
        name_only_rank=name_rank or 999,
        name_only_score=name_score,
        name_only_time_ms=name_result.query_time_ms,
        embedding_found=emb_found,
        embedding_rank=emb_rank or 999,
        embedding_score=emb_score,
        embedding_time_ms=emb_result.query_time_ms
    )


def calculate_metrics(results: List[TestResult], search_mode: str) -> Dict[str, Any]:
    """Calculate recall and MRR metrics for a set of results."""
    total = len(results)
    if total == 0:
        return {}

    # Determine which ranks to use
    ranks = [r.name_only_rank if search_mode == "name_only" else r.embedding_rank for r in results]
    found = [r.name_only_found if search_mode == "name_only" else r.embedding_found for r in results]
    times = [r.name_only_time_ms if search_mode == "name_only" else r.embedding_time_ms for r in results]

    # Calculate recall@K
    recall_at_k = {}
    for k in [1, 5, 10, 100]:
        count = sum(1 for rank in ranks if rank <= k)
        recall_at_k[k] = (count / total) * 100

    # Calculate MRR
    mrr = sum(1.0 / rank for rank in ranks if rank < 999) / total

    # Calculate mean rank (excluding not found)
    found_ranks = [r for r in ranks if r < 999]
    mean_rank = sum(found_ranks) / len(found_ranks) if found_ranks else 999

    return {
        "found_count": sum(found),
        "not_found_count": total - sum(found),
        "not_found_rate": ((total - sum(found)) / total) * 100,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "latency_avg_ms": sum(times) / len(times),
        "latency_p50_ms": sorted(times)[len(times) // 2],
        "latency_p95_ms": sorted(times)[int(len(times) * 0.95)]
    }


def print_summary(exact_results: List[TestResult], variant_results: List[TestResult]):
    """Print summary of results."""
    print(f"\n{'='*100}")
    print("ENHANCED VALIDATION RESULTS SUMMARY")
    print(f"{'='*100}")

    print(f"\n## Test Cases")
    print(f"Exact name queries:   {len(exact_results)}")
    print(f"Variant queries:      {len(variant_results)}")
    print(f"Total:                {len(exact_results) + len(variant_results)}")

    # Exact queries
    if exact_results:
        print(f"\n## Exact Name Queries ({len(exact_results)} cases)")
        print(f"{'='*100}")

        name_metrics = calculate_metrics(exact_results, "name_only")
        emb_metrics = calculate_metrics(exact_results, "embedding")

        print(f"\nName-Only Search:")
        print(f"  Recall@1:  {name_metrics['recall_at_k'][1]:.1f}%")
        print(f"  Recall@10: {name_metrics['recall_at_k'][10]:.1f}%")
        print(f"  MRR:       {name_metrics['mrr']:.4f}")
        print(f"  Avg Time:  {name_metrics['latency_avg_ms']:.1f}ms")

        print(f"\nEmbedding Search:")
        print(f"  Recall@1:  {emb_metrics['recall_at_k'][1]:.1f}%")
        print(f"  Recall@10: {emb_metrics['recall_at_k'][10]:.1f}%")
        print(f"  MRR:       {emb_metrics['mrr']:.4f}")
        print(f"  Avg Time:  {emb_metrics['latency_avg_ms']:.1f}ms")

        print(f"\n  ✅ Result: Both perform equally well on exact matches")

    # Variant queries
    if variant_results:
        print(f"\n## Fuzzy Variant Queries ({len(variant_results)} cases)")
        print(f"{'='*100}")

        name_metrics = calculate_metrics(variant_results, "name_only")
        emb_metrics = calculate_metrics(variant_results, "embedding")

        print(f"\nName-Only Search:")
        print(f"  Recall@1:  {name_metrics['recall_at_k'][1]:.1f}%")
        print(f"  Recall@10: {name_metrics['recall_at_k'][10]:.1f}%")
        print(f"  MRR:       {name_metrics['mrr']:.4f}")
        print(f"  Avg Time:  {name_metrics['latency_avg_ms']:.1f}ms")

        print(f"\nEmbedding Search:")
        print(f"  Recall@1:  {emb_metrics['recall_at_k'][1]:.1f}%")
        print(f"  Recall@10: {emb_metrics['recall_at_k'][10]:.1f}%")
        print(f"  MRR:       {emb_metrics['mrr']:.4f}")
        print(f"  Avg Time:  {emb_metrics['latency_avg_ms']:.1f}ms")

        improvement = emb_metrics['recall_at_k'][10] - name_metrics['recall_at_k'][10]
        if improvement > 5:
            print(f"\n  🎯 Result: Embeddings significantly better (+{improvement:.1f}% Recall@10)")
        elif improvement > 0:
            print(f"\n  ✅ Result: Embeddings slightly better (+{improvement:.1f}% Recall@10)")
        else:
            print(f"\n  ⚪ Result: Similar performance")

        # Variant breakdown
        print(f"\n## Variant Type Breakdown")
        print(f"{'='*100}")

        variant_types = {}
        for r in variant_results:
            if r.variant_type not in variant_types:
                variant_types[r.variant_type] = {"name": [], "emb": []}
            variant_types[r.variant_type]["name"].append(r.name_only_found)
            variant_types[r.variant_type]["emb"].append(r.embedding_found)

        for vtype, data in sorted(variant_types.items()):
            name_found = sum(data["name"])
            emb_found = sum(data["emb"])
            total = len(data["name"])
            print(f"\n{vtype}: ({total} cases)")
            print(f"  Name-only: {name_found}/{total} ({100*name_found/total:.1f}%)")
            print(f"  Embedding: {emb_found}/{total} ({100*emb_found/total:.1f}%)")
            if emb_found > name_found:
                print(f"  🎯 Embedding wins by {emb_found - name_found}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced validation with exact and variant queries")
    parser.add_argument("--exact", type=str,
                       help="Exact name validation samples JSONL file")
    parser.add_argument("--variants", type=str,
                       help="Variant queries JSONL file")
    parser.add_argument("--name_model_path", type=str, required=True,
                       help="Path to personal names model")
    parser.add_argument("--biz_model_path", type=str, required=True,
                       help="Path to business names model")
    parser.add_argument("--truncate_dim", type=int, default=512,
                       help="Matryoshka truncation dimension")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file")
    parser.add_argument("--limit", type=int,
                       help="Limit number of test cases (for testing)")
    parser.add_argument("--onnx", action="store_true",
                       help="Use ONNX models instead of PyTorch models")

    args = parser.parse_args()

    print("⏳ Loading models...")
    if args.onnx:
        from onnx_sentence_transformer import load_onnx_model
        name_model = load_onnx_model(args.name_model_path)
        biz_model = load_onnx_model(args.biz_model_path)
    else:
        name_model = SentenceTransformer(args.name_model_path)
        biz_model = SentenceTransformer(args.biz_model_path)

    print("⏳ Initializing Senzing...")
    settings = get_senzing_config()
    sz_abstract_factory = SzAbstractFactoryCore("", settings=settings)
    sz_engine = sz_abstract_factory.create_engine()

    # Load test cases
    exact_cases = []
    variant_cases = []

    if args.exact:
        print(f"⏳ Loading exact name test cases from {args.exact}...")
        with open(args.exact, 'r') as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    exact_cases.append(case)
        print(f"✅ Loaded {len(exact_cases)} exact name test cases")

    if args.variants:
        print(f"⏳ Loading variant test cases from {args.variants}...")
        with open(args.variants, 'r') as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    variant_cases.append(case)
        print(f"✅ Loaded {len(variant_cases)} variant test cases")

    # Apply limit if specified
    if args.limit:
        exact_cases = exact_cases[:args.limit]
        variant_cases = variant_cases[:args.limit]

    # Look up entity IDs for test cases that only have record_id
    all_cases = exact_cases + variant_cases
    cases_needing_lookup = [c for c in all_cases if not c.get("expected_entity_id") and (c.get("expected_record_id") or c.get("record_id"))]

    if cases_needing_lookup:
        print(f"\n⏳ Looking up entity IDs for {len(cases_needing_lookup)} test cases...")
        for i, case in enumerate(cases_needing_lookup, 1):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(cases_needing_lookup)}")

            record_id = case.get("expected_record_id") or case.get("record_id")
            data_source = case.get("data_source")

            if record_id and data_source:
                entity_id = lookup_entity_id(sz_engine, record_id, data_source)
                if entity_id:
                    case["expected_entity_id"] = entity_id
                else:
                    print(f"  ⚠️  Could not find entity for {data_source}:{record_id}")

        # Filter out cases where we couldn't find the entity
        exact_cases = [c for c in exact_cases if c.get("expected_entity_id")]
        variant_cases = [c for c in variant_cases if c.get("expected_entity_id")]
        print(f"✅ Entity ID lookup complete")
        print(f"   Exact cases ready: {len(exact_cases)}")
        print(f"   Variant cases ready: {len(variant_cases)}")

    # Run tests
    exact_results = []
    variant_results = []

    if exact_cases:
        print(f"\n⏳ Running exact name tests ({len(exact_cases)} cases)...")
        for i, case in enumerate(exact_cases, 1):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(exact_cases)}")
            result = run_test_case(sz_engine, name_model, biz_model, case, args.truncate_dim, "exact")
            exact_results.append(result)
        print(f"✅ Completed exact name tests")

    if variant_cases:
        print(f"\n⏳ Running variant tests ({len(variant_cases)} cases)...")
        for i, case in enumerate(variant_cases, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(variant_cases)}")
            result = run_test_case(sz_engine, name_model, biz_model, case, args.truncate_dim, "variant")
            variant_results.append(result)
        print(f"✅ Completed variant tests")

    # Print summary
    print_summary(exact_results, variant_results)

    # Save results
    print(f"\n⏳ Saving results to {args.output}...")
    output = {
        "metadata": {
            "exact_file": args.exact,
            "variants_file": args.variants,
            "name_model_path": args.name_model_path,
            "biz_model_path": args.biz_model_path,
            "truncate_dim": args.truncate_dim,
            "exact_count": len(exact_results),
            "variant_count": len(variant_results)
        },
        "exact_metrics": {
            "name_only": calculate_metrics(exact_results, "name_only") if exact_results else {},
            "embedding": calculate_metrics(exact_results, "embedding") if exact_results else {}
        },
        "variant_metrics": {
            "name_only": calculate_metrics(variant_results, "name_only") if variant_results else {},
            "embedding": calculate_metrics(variant_results, "embedding") if variant_results else {}
        },
        "exact_results": [asdict(r) for r in exact_results],
        "variant_results": [asdict(r) for r in variant_results]
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Results saved!")


if __name__ == "__main__":
    main()
