# -----------------------------------------------------------------------------
# Search Senzing using both a name attribute search and an embedding search.
#
# Example usage:
# python sz_search_embeddings.py --type personal --model_path output/20250814/FINAL-fine_tuned_model "john doe"
# python sz_search_embeddings.py --type business --model_path /home/roncewind/senzing-garage.git/bizname-research/spike/ron/embedding/output/20250722/FINAL-fine_tuned_model "Koryo Group"


# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import orjson
import torch
from sentence_transformers import SentenceTransformer
from senzing import SzEngine, SzEngineFlags, SzError
from senzing_core import SzAbstractFactoryCore

from sz_utils import get_embedding, get_senzing_config

# -----------------------------------------------------------------------------
# logging setup (will be reconfigured based on --debug flag in main)
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# =============================================================================
# output some results
def print_results(search_type: str, results_dict: dict[str, Any]) -> None:
    entities = results_dict.get("RESOLVED_ENTITIES", {})
    print("\n----------------------------------------------")
    print(f'{search_type} resolved {len(entities)} entities:\n')
    if len(entities) < 1:
        return
    print("              ID NAME EMB NAME")
    for index, e in enumerate(entities):
        logger.debug(f"Match: {json.dumps(e, ensure_ascii=False)}")
        e_id = e.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID", "***Something went wrong.***")
        feature_scores = e.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})
        e_gnr_score = feature_scores.get("NAME", [])
        if not e_gnr_score:
            e_gnr_score = -1
        else:
            e_gnr_score = e_gnr_score[0].get("SCORE", -2)
        # Check both embedding types (personal and business names)
        e_emb_score = feature_scores.get("NAME_EMBEDDING", []) or feature_scores.get("BIZNAME_EMBEDDING", [])
        if not e_emb_score:
            e_emb_score = -1
        else:
            e_emb_score = e_emb_score[0].get("SCORE", -2)
        e_name = e.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_NAME", "***Something went wrong.***")
        print(f"[{index:02}]: {e_id:>10} {e_gnr_score:>3} {e_emb_score:>3} {e_name:<60}")


# =============================================================================
# search by full name attribute
def search_by_name(engine: SzEngine, model_type: str, name: str) -> None:
    logger.debug("search_by_name:")
    search_attr = "NAME_FULL"
    if model_type == "business":
        search_attr = "NAME_ORG"
    attributes = json.dumps({search_attr: name})

    flags = (
        SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS
        | SzEngineFlags.SZ_SEARCH_INCLUDE_REQUEST
        | SzEngineFlags.SZ_SEARCH_INCLUDE_REQUEST_DETAILS
        | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
        | SzEngineFlags.SZ_SEARCH_INCLUDE_STATS
        | SzEngineFlags.SZ_INCLUDE_MATCH_KEY_DETAILS
    )

    try:
        result = engine.search_by_attributes(attributes, flags)
        search_result = orjson.loads(result)
        logger.debug(f"\nName result:\n{json.dumps(result, ensure_ascii=False)}\n")
        print_results("Name-only search", search_result)
    except SzError as err:
        logger.error(f"Search error: {err}")


# =============================================================================
# run a dual search for the given name using name only attribute search and embedding search
def search_by_embedding(
    engine: SzEngine,
    model: SentenceTransformer,
    model_type: str,
    name: str,
    truncate_dim: int | None = None,
) -> None:
    logger.debug("search_by_embedding:")
    search_emb = "NAME_EMBEDDING"
    search_label = "NAME_LABEL"
    search_attr = "NAME_FULL"
    if model_type == "business":
        search_emb = "BIZNAME_EMBEDDING"
        search_label = "BIZNAME_LABEL"
        search_attr = "NAME_ORG"

    embedding = get_embedding(name, model, truncate_dim)
    logger.debug(f"{name}: {embedding.tolist()}")

    attributes = json.dumps(
        {search_label: name, search_attr: name, search_emb: f"{embedding.tolist()}"},
        ensure_ascii=False,
    )
    logger.debug("attributes:")
    logger.debug(attributes)
    flags = (
        SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS
        | SzEngineFlags.SZ_SEARCH_INCLUDE_REQUEST
        | SzEngineFlags.SZ_SEARCH_INCLUDE_REQUEST_DETAILS
        | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
        | SzEngineFlags.SZ_SEARCH_INCLUDE_STATS
        | SzEngineFlags.SZ_INCLUDE_MATCH_KEY_DETAILS
        # | SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES
    )
    try:
        result = engine.search_by_attributes(attributes, flags)
        search_result = orjson.loads(result)
        logger.debug(f"\nEmbedding result:\n{json.dumps(search_result, ensure_ascii=False)}\n")
        print_results("Embedding search", search_result)
    except SzError as err:
        logger.error(f"Search error: {err}")


# =============================================================================
# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sz_search_embeddings", description="Search Senzing using name attributes and embeddings."
    )

    parser.add_argument('--model_path', type=str, required=True, help='Path to names model.')
    parser.add_argument('--type', type=str, required=False, default="personal", help='Either "business" or "personal". Default: personal')
    parser.add_argument('--truncate_dim', type=int, default=None, help='Matryoshka truncation dimension (e.g., 512 for 768d models)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging (default: WARNING level)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose Senzing logging')
    parser.add_argument('search_name', type=str, help='Name to search Senzing for.')
    args = parser.parse_args()

    # Configure logging based on --debug flag
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        force=True  # Override any existing configuration
    )

    # check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Using device: {device}")

    print("⏳ Loading model...")
    model = SentenceTransformer(args.model_path)

    print("⏳ Get Senzing engine...")
    settings = get_senzing_config()
    verbose_logging = 1 if args.verbose else 0
    sz_factory = SzAbstractFactoryCore("SzEmbeddingsSearch", settings, verbose_logging=verbose_logging)
    sz_engine = sz_factory.create_engine()

    truncate_dim = args.truncate_dim
    if truncate_dim:
        print(f"📌 Matryoshka truncation: {truncate_dim} dimensions")

    print(f"\nSearch for: {args.search_name}")
    search_by_name(sz_engine, args.type, args.search_name)
    search_by_embedding(sz_engine, model, args.type, args.search_name, truncate_dim)
    sys.exit(0)

# python sz_search_embeddings.py --type personal --model_path output/20250814/FINAL-fine_tuned_model "john doe"
# python sz_search_embeddings.py --type business --model_path /path/to/biz_model "Koryo Group"
# python sz_search_embeddings.py --type business --truncate_dim 512 --model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model "Koryo Group"
# python sz_search_embeddings.py --type personal --truncate_dim 512 --model_path ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model "სალომე ახვლედიანი"


#
# "PERSON", "Salome Akhvlediani" "სალომე ახვლედიანი" "Саломе Ахвледиани"
#   "سالومي أخفلدياني"
# "PERSON" "Levan Zhorzholiani" "Леван Жоржолиани" "ლევან ჟორჟოლიანი" "L. Zhorzholiani"
#    "ليڤان جورجولياني"
