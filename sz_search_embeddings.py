# -----------------------------------------------------------------------------
# Search Senzing using both a name attribute search and an embedding search.
#
# Example usage:
# python sz_search_embeddings.py --type personal --model_path output/20250814/FINAL-fine_tuned_model "john doe"
# python sz_search_embeddings.py --type business --model_path /home/roncewind/senzing-garage.git/bizname-research/spike/ron/embedding/output/20250722/FINAL-fine_tuned_model "Koryo Group"


# -----------------------------------------------------------------------------
import argparse
import json
import os
import sys

import numpy as np
import numpy.typing as npt
import orjson
import torch
from sentence_transformers import SentenceTransformer
from senzing import SzEngineFlags, SzError
from senzing_core import SzAbstractFactoryCore

# -----------------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------------

file_path = ""
debug_on = True
number_of_lines_to_process = 0
number_of_names_to_process = 0
status_print_lines = 10000


# =============================================================================
def debug(text):
    if debug_on:
        print(text, file=sys.stderr, flush=True)


# =============================================================================
def format_seconds_to_hhmmss(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


# =============================================================================
# create embedding for all the names in the list
def get_embedding(name, model, batch_size=256) -> npt.NDArray[np.float16]:
    embedding = model.encode(
        list(name),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float16, copy=False)
    return embedding


# =============================================================================
# create embedding for all the names in the list
def get_embeddings(names, model, batch_size) -> npt.NDArray[np.float16]:
    embeddings = model.encode(
        names,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float16, copy=False)
    return embeddings


# =============================================================================
# output some results
def print_results(search_type, results_dict):
    entities = results_dict.get("RESOLVED_ENTITIES", {})
    print("\n----------------------------------------------")
    print(f'{search_type} resolved {len(entities)} entities:\n')
    if len(entities) < 1:
        return
    print("              ID GNR EMB NAME")
    for index, e in enumerate(entities):
        debug(f"Match: {json.dumps(e, ensure_ascii=False)}")
        e_id = e.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID", "***Something went wrong.***")
        e_gnr_score = e.get("MATCH_INFO", {}).get("FEATURE_SCORES", {}).get("NAME", [])
        if not e_gnr_score:
            e_gnr_score = 0
        else:
            e_gnr_score = e_gnr_score[0].get("SCORE", 0)
        e_emb_score = e.get("MATCH_INFO", {}).get("FEATURE_SCORES", {}).get("BIZNAME_EMBEDDING", [])
        if not e_emb_score:
            e_emb_score = 0
        else:
            e_emb_score = e_emb_score[0].get("SCORE", 0)
        e_name = e.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_NAME", "***Something went wrong.***")
        print(f"[{index:02}]: {e_id:>10} {e_gnr_score:>3} {e_emb_score:>3} {e_name:<60}")


# =============================================================================
# search by full name attribue
def search_by_name(engine, model_type, name):

    search_attr = "NAME_FULL"
    if model_type == "business":
        search_attr = "NAME_ORG"
    attributes = json.dumps({search_attr: name})

    flags = SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS | SzEngineFlags.SZ_SEARCH_INCLUDE_REQUEST | SzEngineFlags.SZ_SEARCH_INCLUDE_REQUEST_DETAILS | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES | SzEngineFlags.SZ_SEARCH_INCLUDE_STATS | SzEngineFlags.SZ_INCLUDE_MATCH_KEY_DETAILS

    search_result = {}
    try:
        result = engine.search_by_attributes(attributes, flags)
        search_result = orjson.loads(result)
        debug(f"\nName result:\n{json.dumps(result, ensure_ascii=False)}\n")
        print_results("Name search", search_result)
    except SzError as err:
        print(f"\nERROR: {err}\n")

    # print(f'Resolved entities: {len(search_result.get("RESOLVED_ENTITIES", {}))}')
    return


# =============================================================================
# run a dual search for the given name using name only attribute search and embedding search
def search_by_embedding(engine, model, model_type, name, batch_size=256):

    search_emb = "NAME_EMBEDDING"
    search_label = "NAME_LABEL"
    search_attr = "NAME_FULL"
    if model_type == "business":
        search_emb = "BIZNAME_EMBEDDING"
        search_label = "BIZNAME_LABEL"
        search_attr = "NAME_ORG"

    n_list = [name]
    e_list = get_embeddings(n_list, model, batch_size)
    debug(f"n_list: {len(n_list)}  e_list: {len(e_list)}")
    for n, e in zip(n_list, e_list):
        debug(f"{n}: {e.tolist()}")

    debug(f"{name}: {e_list.tolist()[0]}")
    # FIXME: do we want NAME_FULL and NAME_ORG in the embedding search?
    attributes = json.dumps({search_label: f"{n_list[0]}", search_attr: f"{n_list[0]}", search_emb: f"{e_list.tolist()[0]}"}, ensure_ascii=False)
    # attributes = json.dumps({search_label: f"{n_list[0]}", search_emb: f"{e_list.tolist()[0]}"}, ensure_ascii=False)
    debug(attributes)
    flags = SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS | SzEngineFlags.SZ_SEARCH_INCLUDE_REQUEST | SzEngineFlags.SZ_SEARCH_INCLUDE_REQUEST_DETAILS | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES | SzEngineFlags.SZ_SEARCH_INCLUDE_STATS | SzEngineFlags.SZ_INCLUDE_MATCH_KEY_DETAILS
    # flags = SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS | SzEngineFlags.SZ_SEARCH_INCLUDE_REQUEST | SzEngineFlags.SZ_SEARCH_INCLUDE_REQUEST_DETAILS | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES | SzEngineFlags.SZ_SEARCH_INCLUDE_STATS | SzEngineFlags.SZ_INCLUDE_MATCH_KEY_DETAILS | SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES
    search_result = {}
    try:
        result = engine.search_by_attributes(attributes, flags)
        search_result = orjson.loads(result)
        debug(f"\nEmbedding result:\n{json.dumps(search_result, ensure_ascii=False)}\n")
        print_results("Embedding search", search_result)
    except SzError as err:
        print(f"\nERROR: {err}\n")

    return


# =============================================================================
# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sz_search_open_sactions_names", description="Search Senzing for Open Sanctions names."
    )

    parser.add_argument('--model_path', type=str, required=True, help='Path to names model.')
    parser.add_argument('--type', type=str, required=False, default="personal", help='Either "business" or "personal". Default: personal')
    parser.add_argument('search_name', type=str, help='Name to search Senzing for.')
    parser.add_argument('--batch_size', type=int, required=False, help='Training batch size. Default: auto tune')
    args = parser.parse_args()

    # check for CUDA
    device = "cpu"
    batch_size = 64
    if torch.cuda.is_available():
        batch_size = 256 if not args.batch_size else args.batch_size
        device = "cuda"
    print(f"📌 Using device: {device}")
    print(f"📌 Batch size: {batch_size}")

    print("⏳ Loading model...")
    model = SentenceTransformer(args.model_path)

    print("⏳ Get Senzing engine...")
    settings = os.getenv("SENZING_ENGINE_CONFIGURATION_JSON")
    sz_factory = SzAbstractFactoryCore("OpenSanctionsLoader", settings, verbose_logging=0)
    sz_engine = sz_factory.create_engine()
    print(f"\nSearch for: {args.search_name}")
    search_by_name(sz_engine, args.type, args.search_name)
    search_by_embedding(sz_engine, model, args.type, args.search_name, batch_size)
    exit(0)

# python sz_search_embeddings.py --type personal --model_path output/20250814/FINAL-fine_tuned_model "john doe"
# python sz_search_embeddings.py --type business --model_path /home/roncewind/senzing-garage.git/bizname-research/spike/ron/embedding/output/20250722/FINAL-fine_tuned_model "Koryo Group"
