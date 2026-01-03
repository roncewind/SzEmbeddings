#!/usr/bin/env python3
"""
Test embedding value on fuzzy matching scenarios.

Tests cases where name-only search fails but embeddings should succeed:
- Partial names (missing parts)
- Abbreviations vs full forms
- Transliterations
- Spelling variations
- Fuzzy matches
"""

import json
import sys
import argparse
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from senzing import SzEngine, SzEngineFlags
from senzing_core import SzAbstractFactoryCore

sys.path.insert(0, '/home/roncewind/roncewind.git/SzEmbeddings')
from sz_utils import get_senzing_config, get_embedding


@dataclass
class TestCase:
    """Test case for fuzzy matching."""
    name: str
    query: str
    expected_entity_id: int
    record_type: str
    category: str
    description: str


def search_by_name_only(
    engine: SzEngine,
    record_type: str,
    name: str
) -> Tuple[List[int], Dict[int, int]]:
    """Search using name attribute only (no embeddings)."""
    search_attr = "NAME_FULL" if record_type == "PERSON" else "NAME_ORG"
    attributes = json.dumps({search_attr: name})

    flags = (
        SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS
        | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
        | SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES
    )

    try:
        result = engine.search_by_attributes(attributes, flags)
        result_dict = json.loads(result)

        entity_scores = []
        entities = result_dict.get("RESOLVED_ENTITIES", [])
        for entity in entities:
            entity_id = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID")
            if entity_id is None:
                continue

            # Get NAME score
            feature_scores = entity.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})
            score = 0
            if "NAME" in feature_scores and feature_scores["NAME"]:
                score = feature_scores["NAME"][0].get("SCORE", 0)

            entity_scores.append((entity_id, score))

        # Sort by score descending
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        entity_ids = [eid for eid, _ in entity_scores]
        scores = {eid: score for eid, score in entity_scores}

        return entity_ids, scores
    except Exception as err:
        print(f"Name-only search error: {err}")
        return [], {}


def search_with_embedding(
    engine: SzEngine,
    model: SentenceTransformer,
    record_type: str,
    name: str,
    truncate_dim: int
) -> Tuple[List[int], Dict[int, int]]:
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

    try:
        result = engine.search_by_attributes(attributes, flags)
        result_dict = json.loads(result)

        entity_scores = []
        entities = result_dict.get("RESOLVED_ENTITIES", [])
        for entity in entities:
            entity_id = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID")
            if entity_id is None:
                continue

            # Get score (prefer embedding if available, otherwise NAME)
            feature_scores = entity.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})
            score = 0

            # Try embedding score first
            emb_feature = (feature_scores.get("NAME_EMBEDDING", []) or
                          feature_scores.get("BIZNAME_EMBEDDING", []))
            if emb_feature:
                score = emb_feature[0].get("SCORE", 0)
            elif "NAME" in feature_scores and feature_scores["NAME"]:
                score = feature_scores["NAME"][0].get("SCORE", 0)

            entity_scores.append((entity_id, score))

        # Sort by score descending
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        entity_ids = [eid for eid, _ in entity_scores]
        scores = {eid: score for eid, score in entity_scores}

        return entity_ids, scores
    except Exception as err:
        print(f"Embedding search error: {err}")
        return [], {}


def get_entity_names(engine: SzEngine, entity_id: int) -> List[str]:
    """Get all names for an entity."""
    try:
        flags = SzEngineFlags.SZ_ENTITY_INCLUDE_RECORD_DATA
        result = engine.get_entity_by_entity_id(entity_id, flags)
        entity_data = json.loads(result)

        names = []
        records = entity_data.get("RESOLVED_ENTITY", {}).get("RECORDS", [])
        for record in records:
            if "NAME_FULL" in record:
                names.append(record["NAME_FULL"])
            if "NAME_ORG" in record:
                names.append(record["NAME_ORG"])

        return list(set(names))  # Deduplicate
    except Exception:
        return []


def run_test_case(
    engine: SzEngine,
    name_model: SentenceTransformer,
    biz_model: SentenceTransformer,
    test_case: TestCase,
    truncate_dim: int
) -> Dict[str, Any]:
    """Run a single test case."""
    model = name_model if test_case.record_type == "PERSON" else biz_model

    # Name-only search
    name_ids, name_scores = search_by_name_only(engine, test_case.record_type, test_case.query)
    name_found = test_case.expected_entity_id in name_ids
    name_rank = name_ids.index(test_case.expected_entity_id) + 1 if name_found else None
    name_score = name_scores.get(test_case.expected_entity_id, 0) if name_found else None

    # Embedding search
    emb_ids, emb_scores = search_with_embedding(engine, model, test_case.record_type, test_case.query, truncate_dim)
    emb_found = test_case.expected_entity_id in emb_ids
    emb_rank = emb_ids.index(test_case.expected_entity_id) + 1 if emb_found else None
    emb_score = emb_scores.get(test_case.expected_entity_id, 0) if emb_found else None

    # Get actual names from entity
    actual_names = get_entity_names(engine, test_case.expected_entity_id)

    return {
        "name": test_case.name,
        "query": test_case.query,
        "expected_entity_id": test_case.expected_entity_id,
        "actual_names": actual_names,
        "category": test_case.category,
        "description": test_case.description,
        "name_only": {
            "found": name_found,
            "rank": name_rank,
            "score": name_score,
            "total_results": len(name_ids)
        },
        "embedding": {
            "found": emb_found,
            "rank": emb_rank,
            "score": emb_score,
            "total_results": len(emb_ids)
        }
    }


def print_results(results: List[Dict[str, Any]]):
    """Print test results in a readable format."""
    print("\n" + "=" * 120)
    print("EMBEDDING VALUE TEST RESULTS")
    print("=" * 120)

    # Group by category
    by_category = {}
    for result in results:
        category = result["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(result)

    # Print summary
    total = len(results)
    name_only_success = sum(1 for r in results if r["name_only"]["found"])
    embedding_success = sum(1 for r in results if r["embedding"]["found"])
    embedding_improvement = sum(1 for r in results if r["embedding"]["found"] and not r["name_only"]["found"])

    print(f"\nOverall Summary:")
    print(f"  Total test cases: {total}")
    print(f"  Name-only found: {name_only_success}/{total} ({100*name_only_success/total:.1f}%)")
    print(f"  Embedding found: {embedding_success}/{total} ({100*embedding_success/total:.1f}%)")
    print(f"  Embedding improvement: {embedding_improvement}/{total} ({100*embedding_improvement/total:.1f}%)")

    # Print by category
    for category, cat_results in sorted(by_category.items()):
        print(f"\n{'=' * 120}")
        print(f"Category: {category.upper()}")
        print(f"{'=' * 120}")

        for result in cat_results:
            name_status = "✓" if result["name_only"]["found"] else "✗"
            emb_status = "✓" if result["embedding"]["found"] else "✗"

            # Determine if embedding helped
            if result["embedding"]["found"] and not result["name_only"]["found"]:
                improvement = "🎯 EMBEDDING WINS"
            elif result["name_only"]["found"] and not result["embedding"]["found"]:
                improvement = "⚠️  NAME-ONLY WINS"
            elif result["name_only"]["found"] and result["embedding"]["found"]:
                improvement = "✓ Both work"
            else:
                improvement = "✗ Both fail"

            print(f"\nTest: {result['name']}")
            print(f"  Description: {result['description']}")
            print(f"  Query: \"{result['query']}\"")
            print(f"  Expected Entity: {result['expected_entity_id']}")
            print(f"  Actual Names: {', '.join(result['actual_names'][:3])}{'...' if len(result['actual_names']) > 3 else ''}")
            print(f"  Name-only: {name_status} Rank={result['name_only']['rank']}, Score={result['name_only']['score']}, Results={result['name_only']['total_results']}")
            print(f"  Embedding: {emb_status} Rank={result['embedding']['rank']}, Score={result['embedding']['score']}, Results={result['embedding']['total_results']}")
            print(f"  Result: {improvement}")


def main():
    parser = argparse.ArgumentParser(description="Test embedding value on fuzzy matching scenarios")
    parser.add_argument("--name_model_path", type=str, required=True,
                       help="Path to personal names SentenceTransformer model")
    parser.add_argument("--biz_model_path", type=str, required=True,
                       help="Path to business names SentenceTransformer model")
    parser.add_argument("--truncate_dim", type=int, default=512,
                       help="Matryoshka truncation dimension")
    parser.add_argument("--output", type=str,
                       help="Optional output JSON file")

    args = parser.parse_args()

    print("⏳ Loading models...")
    name_model = SentenceTransformer(args.name_model_path)
    biz_model = SentenceTransformer(args.biz_model_path)

    print("⏳ Initializing Senzing...")
    settings = get_senzing_config()
    sz_abstract_factory = SzAbstractFactoryCore("", settings=settings)
    sz_engine = sz_abstract_factory.create_engine()

    # Define test cases
    # Note: These need to be entities that actually exist in the database
    # You'll need to update these with real entity IDs from your 10k dataset

    test_cases = [
        # Example: Partial name (missing middle name)
        TestCase(
            name="Bychkov (partial)",
            query="Bychkov Vladimir",
            expected_entity_id=2312,
            record_type="PERSON",
            category="partial_name",
            description="Query missing patronymic/middle name"
        ),

        # Example: Different word order
        TestCase(
            name="Name order variation",
            query="Vladimir Bychkov",
            expected_entity_id=2312,
            record_type="PERSON",
            category="word_order",
            description="Query with reversed name order"
        ),

        # Example: Fuzzy business name
        TestCase(
            name="Abbreviated company",
            query="Custody Bank Japan",
            expected_entity_id=1834,
            record_type="ORGANIZATION",
            category="abbreviation",
            description="Query with abbreviated form"
        ),

        # Add more test cases here based on entities in your database
    ]

    print(f"\n⏳ Running {len(test_cases)} test cases...")
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] Testing: {test_case.name}")
        result = run_test_case(sz_engine, name_model, biz_model, test_case, args.truncate_dim)
        results.append(result)

    # Print results
    print_results(results)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "metadata": {
                    "name_model_path": args.name_model_path,
                    "biz_model_path": args.biz_model_path,
                    "truncate_dim": args.truncate_dim,
                    "total_cases": len(test_cases)
                },
                "results": results
            }, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
