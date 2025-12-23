#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Evaluate embedding models through Senzing entity resolution.
#
# Tests models on ground truth triplets by:
# 1. Searching for anchor names in two modes (embedding-only, GNR+embedding)
# 2. Checking if positive's entity ranks higher than negative's entity
# 3. Computing comprehensive metrics (Accuracy, Recall@K, Precision@K, MRR, etc.)
#
# Example usage:
# python sz_evaluate_model.py \
#   --type business \
#   --model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
#   --triplets ~/roncewind.git/BizNames/output/opensanctions_test_triplets.jsonl \
#   --test_set opensanctions \
#   --sample 1000
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import orjson
import torch
from sentence_transformers import SentenceTransformer
from senzing import SzEngine, SzEngineFlags, SzError
from senzing_core import SzAbstractFactoryCore

from sz_utils import format_seconds_to_hhmmss, get_embedding, get_senzing_config

# -----------------------------------------------------------------------------
# Logging setup (will be reconfigured based on --debug flag in main)
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
K_VALUES = [1, 5, 10, 100]  # K values for Recall@K, Precision@K, etc.


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class SearchResult:
    """Results from a single search query."""
    entity_ids: list[int]  # List of ENTITY_IDs in rank order
    gnr_scores: dict[int, float]  # entity_id -> GNR score
    emb_scores: dict[int, float]  # entity_id -> embedding score
    query_time_ms: float  # Query latency in milliseconds


@dataclass
class TripletResult:
    """Evaluation result for a single triplet (including variant information)."""
    anchor: str
    positive: str
    negative: str
    anchor_group: str

    # Variant tracking (NEW)
    variant_type: str = "exact"  # Type of variant: "exact", "alias_Latin", "synthetic_abbreviated", etc.
    original_anchor: str = ""    # Original anchor before variant generation

    # GNR-only search results (baseline)
    gnr_positive_rank: int | None = None  # Rank of positive entity (None if not found)
    gnr_negative_rank: int | None = None  # Rank of negative entity (None if not found)
    gnr_positive_score: float | None = None  # Score for positive entity (NEW)
    gnr_negative_score: float | None = None  # Score for negative entity (NEW)
    gnr_correct: bool = False  # True if positive ranks first (legacy)
    gnr_relative_correct: bool = False  # True if positive ranks > negative (NEW)
    gnr_result_count: int = 0  # Total results returned (NEW)
    gnr_scenario: str = ""  # Scenario classification (NEW)
    gnr_query_time_ms: float = 0.0

    # GNR+Embedding combined search results
    emb_positive_rank: int | None = None  # Rank of positive entity with embeddings
    emb_negative_rank: int | None = None  # Rank of negative entity with embeddings
    emb_positive_score: float | None = None  # Score for positive entity (NEW)
    emb_negative_score: float | None = None  # Score for negative entity (NEW)
    emb_correct: bool = False  # True if positive ranks first (legacy)
    emb_relative_correct: bool = False  # True if positive ranks > negative (NEW)
    emb_result_count: int = 0  # Total results returned (NEW)
    emb_scenario: str = ""  # Scenario classification (NEW)
    emb_query_time_ms: float = 0.0


@dataclass
class RelativeRankingMetrics:
    """Metrics focused on relative ranking quality (positive vs negative)."""
    positive_gt_negative_rate: float  # % where positive ranks > negative
    both_found_correct_order: float   # % where both found and correctly ordered
    both_found_wrong_order: float     # % where both found but wrong order (BAD)
    only_positive_found: float        # % where only positive found (GOOD)
    only_negative_found: float        # % where only negative found (BAD)
    neither_found: float              # % where neither found

    avg_rank_separation: float        # Average (negative_rank - positive_rank) when both found
    median_rank_separation: float

    avg_positive_rank: float          # Average rank of positive entity
    avg_negative_rank: float          # Average rank of negative entity
    avg_result_count: float           # Average total results returned


@dataclass
class VariantTypeMetrics:
    """Metrics for a specific variant type."""
    variant_type: str
    count: int

    # Core relative ranking metrics
    relative_ranking: RelativeRankingMetrics

    # Legacy top-1 metrics (for comparison)
    accuracy_top1: float
    recall_at_k: dict[int, float]
    mrr: float

    # Latency
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float


# -----------------------------------------------------------------------------
# Variant Generation Functions
# -----------------------------------------------------------------------------
def detect_script(text: str) -> str:
    """
    Detect the primary script/language of text.
    Returns: "Latin", "Cyrillic", "Georgian", "Arabic", "Chinese", "Mixed", or "Unknown"
    """
    if not text:
        return "Unknown"

    # Count characters by Unicode ranges
    latin = sum(1 for c in text if '\u0041' <= c <= '\u007A' or '\u00C0' <= c <= '\u00FF')
    cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    georgian = sum(1 for c in text if '\u10A0' <= c <= '\u10FF')
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    chinese = sum(1 for c in text if '\u4E00' <= c <= '\u9FFF')

    total = latin + cyrillic + georgian + arabic + chinese
    if total == 0:
        return "Unknown"

    # Determine dominant script (>60% threshold)
    threshold = 0.6 * total
    if latin > threshold:
        return "Latin"
    elif cyrillic > threshold:
        return "Cyrillic"
    elif georgian > threshold:
        return "Georgian"
    elif arabic > threshold:
        return "Arabic"
    elif chinese > threshold:
        return "Chinese"
    else:
        return "Mixed"


def find_record_by_id(data_file: str, record_id: str) -> dict | None:
    """
    Find a record in the JSONL data file by RECORD_ID.
    Returns the record dict or None if not found.
    """
    try:
        with open(data_file, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or len(stripped) < 10:
                    continue
                record = orjson.loads(stripped)
                if record.get("RECORD_ID") == record_id:
                    return record
    except Exception as e:
        logger.warning(f"Error reading data file {data_file}: {e}")
    return None


def extract_aliases_for_triplet(
    triplet: dict,
    data_file: str,
    model_type: str,
) -> list[tuple[str, str]]:
    """
    Extract all NAME_FULL/NAME_ORG aliases from the anchor_group record.
    Returns list of (alias_name, variant_type) tuples.
    variant_type format: "alias_Latin", "alias_Cyrillic", etc.
    """
    if not data_file:
        return [(triplet["anchor"], "exact")]

    record = find_record_by_id(data_file, triplet["anchor_group"])
    if not record:
        logger.debug(f"Record {triplet['anchor_group']} not found in {data_file}")
        return [(triplet["anchor"], "exact")]

    name_field = "NAME_FULL" if model_type == "personal" else "NAME_ORG"
    aliases = []

    for name_entry in record.get("NAMES", []):
        name_val = name_entry.get(name_field)
        if name_val:
            script = detect_script(name_val)
            aliases.append((name_val, f"alias_{script}"))

    # If no aliases found, use the original anchor
    if not aliases:
        aliases.append((triplet["anchor"], "exact"))

    return aliases


def generate_abbreviated_variants(name: str, model_type: str) -> list[tuple[str, str]]:
    """
    Generate abbreviated name variants.
    Returns list of (variant_name, variant_type) tuples.
    """
    variants = []

    if model_type == "personal":
        # Personal name abbreviations
        parts = name.split()
        if len(parts) >= 2:
            # First Initial + Last Name: "John Smith" → "J Smith"
            abbreviated = f"{parts[0][0]} {parts[-1]}"
            variants.append((abbreviated, "synthetic_abbreviated_initial_last"))

            # First Initial with period + Last: "John Smith" → "J. Smith"
            abbreviated_period = f"{parts[0][0]}. {parts[-1]}"
            variants.append((abbreviated_period, "synthetic_abbreviated_initial_period"))

        if len(parts) >= 3:
            # First + Middle Initial + Last: "John William Smith" → "John W Smith"
            abbreviated_middle = f"{parts[0]} {parts[1][0]} {parts[-1]}"
            variants.append((abbreviated_middle, "synthetic_abbreviated_middle_initial"))

    elif model_type == "business":
        # Business name abbreviations - remove legal entity suffixes
        import re
        legal_entities = [
            r'\bLimited Liability Company\b', r'\bLLC\b', r'\bLtd\.?\b', r'\bInc\.?\b',
            r'\bCorporation\b', r'\bCorp\.?\b', r'\bCompany\b', r'\bCo\.?\b',
            r'\bLimited\b', r'\bLLP\b', r'\bL\.L\.C\.\b', r'\bIncorporated\b'
        ]

        cleaned = name
        for entity in legal_entities:
            cleaned = re.sub(entity, '', cleaned, flags=re.IGNORECASE)

        cleaned = cleaned.strip(' ,"')
        if cleaned and cleaned != name:
            variants.append((cleaned, "synthetic_abbreviated_no_legal_entity"))

    return variants


def generate_fuzzy_variants(name: str) -> list[tuple[str, str]]:
    """
    Generate fuzzy variants (punctuation/spacing changes).
    Returns list of (variant_name, variant_type) tuples.
    """
    import re
    variants = []

    # Remove all punctuation
    no_punct = re.sub(r'[^\w\s]', '', name)
    if no_punct != name:
        variants.append((no_punct, "synthetic_fuzzy_no_punctuation"))

    # Normalize spaces (multiple spaces → single space)
    normalized = ' '.join(name.split())
    if normalized != name and normalized != no_punct:
        variants.append((normalized, "synthetic_fuzzy_normalized_spaces"))

    # Replace hyphens with spaces
    if '-' in name:
        no_hyphen = name.replace('-', ' ')
        variants.append((no_hyphen, "synthetic_fuzzy_hyphen_to_space"))

    return variants


def generate_partial_variants(name: str, model_type: str) -> list[tuple[str, str]]:
    """
    Generate partial name variants (first+last only, last only, etc.).
    Returns list of (variant_name, variant_type) tuples.
    """
    variants = []

    if model_type == "personal":
        parts = name.split()
        if len(parts) >= 3:
            # First + Last only (skip middle): "John William Smith" → "John Smith"
            first_last = f"{parts[0]} {parts[-1]}"
            variants.append((first_last, "synthetic_partial_first_last"))

        if len(parts) >= 2:
            # Last name only: "John Smith" → "Smith"
            last_only = parts[-1]
            variants.append((last_only, "synthetic_partial_last_only"))

    return variants


def generate_synthetic_variants(name: str, model_type: str) -> list[tuple[str, str]]:
    """
    Generate all synthetic variants (abbreviated, fuzzy, partial).
    Returns list of (variant_name, variant_type) tuples.
    """
    variants = []
    variants.extend(generate_abbreviated_variants(name, model_type))
    variants.extend(generate_fuzzy_variants(name))
    variants.extend(generate_partial_variants(name, model_type))
    return variants


def generate_all_variants_for_triplet(
    triplet: dict[str, str],
    data_file: str | None,
    model_type: str,
    variant_mode: str,
    synthetic_types: list[str],
) -> list[tuple[str, str]]:
    """
    Generate all variants for a triplet based on configuration.

    Args:
        triplet: Triplet dict with anchor, positive, negative, anchor_group
        data_file: Path to JSONL data file (required for aliases)
        model_type: "personal" or "business"
        variant_mode: "aliases", "synthetics", "both", or "none"
        synthetic_types: List of synthetic types to generate (e.g., ["abbreviated", "fuzzy"])

    Returns:
        List of (variant_query, variant_type) tuples. Always includes ("exact", anchor) as first variant.
    """
    variants = []
    anchor = triplet["anchor"]

    # Always include the original anchor as "exact" variant
    variants.append((anchor, "exact"))

    if variant_mode == "none":
        return variants

    # Generate aliases from data file
    if variant_mode in ("aliases", "both"):
        if data_file:
            try:
                aliases = extract_aliases_for_triplet(triplet, data_file, model_type)
                # Filter out the original anchor (avoid duplicates)
                aliases = [(name, vtype) for name, vtype in aliases if name != anchor]
                variants.extend(aliases)
            except Exception as e:
                logger.warning(f"Failed to extract aliases for {anchor}: {e}")
        else:
            logger.warning("--variants includes 'aliases' but --data_file not provided")

    # Generate synthetic variants
    if variant_mode in ("synthetics", "both"):
        synthetic_variants = []

        if "abbreviated" in synthetic_types:
            synthetic_variants.extend(generate_abbreviated_variants(anchor, model_type))

        if "fuzzy" in synthetic_types:
            synthetic_variants.extend(generate_fuzzy_variants(anchor))

        if "partial" in synthetic_types:
            synthetic_variants.extend(generate_partial_variants(anchor, model_type))

        # Filter out duplicates that match the original anchor
        synthetic_variants = [(name, vtype) for name, vtype in synthetic_variants if name != anchor]
        variants.extend(synthetic_variants)

    return variants


# -----------------------------------------------------------------------------
# Search functions
# -----------------------------------------------------------------------------
def search_gnr_only(
    engine: SzEngine,
    model_type: str,
    name: str,
) -> SearchResult:
    """
    Search using GNR (name attribute) only - no embeddings.
    Tests traditional Senzing name matching performance (baseline).
    """
    search_attr = "NAME_FULL" if model_type == "personal" else "NAME_ORG"

    attributes = json.dumps({
        search_attr: name
    })

    flags = (
        SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS
        | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
        | SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES  # NEW: Return all candidates (threshold-independent)
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


def search_gnr_and_embedding(
    engine: SzEngine,
    model: SentenceTransformer,
    model_type: str,
    name: str,
    truncate_dim: int | None = None,
) -> SearchResult:
    """
    Search using both GNR attribute and embedding.
    Tests combined performance of name matching + semantic similarity.

    Args:
        engine: Senzing engine
        model: SentenceTransformer model
        model_type: "personal" or "business"
        name: Name to search for
        truncate_dim: Optional Matryoshka truncation dimension (e.g., 512)
    """
    search_emb = "NAME_EMBEDDING" if model_type == "personal" else "BIZNAME_EMBEDDING"
    search_label = "NAME_LABEL" if model_type == "personal" else "BIZNAME_LABEL"
    search_attr = "NAME_FULL" if model_type == "personal" else "NAME_ORG"

    embedding = get_embedding(name, model, truncate_dim)
    attributes = json.dumps({
        search_attr: name,
        search_label: name,
        search_emb: f"{embedding.tolist()}"
    })

    flags = (
        SzEngineFlags.SZ_SEARCH_BY_ATTRIBUTES_DEFAULT_FLAGS
        | SzEngineFlags.SZ_INCLUDE_FEATURE_SCORES
        | SzEngineFlags.SZ_SEARCH_INCLUDE_ALL_CANDIDATES  # NEW: Return all candidates (threshold-independent)
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
    """Parse Senzing search result into SearchResult structure."""
    entity_ids = []
    gnr_scores = {}
    emb_scores = {}

    entities = result_dict.get("RESOLVED_ENTITIES", [])
    for entity in entities:
        entity_id = entity.get("ENTITY", {}).get("RESOLVED_ENTITY", {}).get("ENTITY_ID")
        if entity_id is None:
            continue

        entity_ids.append(entity_id)

        # Extract feature scores
        feature_scores = entity.get("MATCH_INFO", {}).get("FEATURE_SCORES", {})

        # GNR score
        gnr_feature = feature_scores.get("NAME", [])
        if gnr_feature:
            gnr_scores[entity_id] = gnr_feature[0].get("SCORE", 0)

        # Embedding score (check both types)
        emb_feature = feature_scores.get("NAME_EMBEDDING", []) or feature_scores.get("BIZNAME_EMBEDDING", [])
        if emb_feature:
            emb_scores[entity_id] = emb_feature[0].get("SCORE", 0)

    return SearchResult(entity_ids, gnr_scores, emb_scores, query_time_ms)


# -----------------------------------------------------------------------------
# Entity lookup
# -----------------------------------------------------------------------------
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
# Helper functions for triplet evaluation
# -----------------------------------------------------------------------------
def find_first_non_positive_rank(entity_ids: list[int], positive_entity_id: int) -> int | None:
    """
    Find the rank of the first entity that is NOT the positive entity.
    This serves as a proxy for the negative entity rank.

    Args:
        entity_ids: List of entity IDs in rank order (1-indexed conceptually)
        positive_entity_id: The entity ID of the positive (correct) entity

    Returns:
        Rank (1-indexed) of first non-positive entity, or None if no other entities
    """
    for i, entity_id in enumerate(entity_ids):
        if entity_id != positive_entity_id:
            return i + 1  # 1-indexed
    return None  # All entities are the positive entity (shouldn't happen)


def classify_ranking_scenario(positive_rank: int | None, negative_rank: int | None) -> str:
    """
    Classify the ranking scenario based on positive and negative ranks.

    Returns:
        Scenario string: "both_found_correct_order", "both_found_wrong_order",
                        "only_positive_found", "only_negative_found", "neither_found"
    """
    if positive_rank is not None and negative_rank is not None:
        if positive_rank < negative_rank:
            return "both_found_correct_order"
        else:
            return "both_found_wrong_order"
    elif positive_rank is not None:
        return "only_positive_found"
    elif negative_rank is not None:
        return "only_negative_found"
    else:
        return "neither_found"


# -----------------------------------------------------------------------------
# Triplet evaluation
# -----------------------------------------------------------------------------
def evaluate_triplet(
    engine: SzEngine,
    model: SentenceTransformer,
    model_type: str,
    data_source: str,
    triplet: dict[str, str],
    truncate_dim: int | None = None,
) -> TripletResult | None:
    """
    Evaluate a single triplet by searching for the anchor and checking ranks.
    Returns None if the ground truth entities cannot be found in Senzing.

    Args:
        engine: Senzing engine
        model: SentenceTransformer model
        model_type: "personal" or "business"
        data_source: Senzing data source name
        triplet: Triplet dict with anchor, positive, negative, anchor_group
        truncate_dim: Optional Matryoshka truncation dimension (e.g., 512)
    """
    anchor = triplet["anchor"]
    positive = triplet["positive"]
    negative = triplet["negative"]
    anchor_group = triplet["anchor_group"]

    # Extract variant metadata (if present from variant generation)
    variant_type = triplet.get("variant_type", "exact")
    original_anchor = triplet.get("original_anchor", anchor)

    # Look up the entity ID for the anchor_group (positive should match this)
    positive_entity_id = get_entity_by_record_id(engine, data_source, anchor_group)
    if positive_entity_id is None:
        logger.debug(f"Skipping triplet: anchor_group {anchor_group} not found in Senzing")
        return None

    # Search with GNR only (baseline)
    gnr_result = search_gnr_only(engine, model_type, anchor)
    gnr_positive_rank = None
    gnr_negative_rank = None
    gnr_positive_score = None
    gnr_negative_score = None

    # Find positive rank and score
    if positive_entity_id in gnr_result.entity_ids:
        gnr_positive_rank = gnr_result.entity_ids.index(positive_entity_id) + 1  # 1-indexed
        gnr_positive_score = gnr_result.gnr_scores.get(positive_entity_id)

    # Find negative rank (first non-positive entity as proxy)
    gnr_negative_rank = find_first_non_positive_rank(gnr_result.entity_ids, positive_entity_id)
    if gnr_negative_rank is not None and len(gnr_result.entity_ids) >= gnr_negative_rank:
        negative_entity_id = gnr_result.entity_ids[gnr_negative_rank - 1]  # 0-indexed access
        gnr_negative_score = gnr_result.gnr_scores.get(negative_entity_id)

    # Check if positive ranks first (correct) vs not first (incorrect)
    gnr_correct = gnr_positive_rank == 1 if gnr_positive_rank is not None else False

    # Compute relative ranking correctness
    gnr_relative_correct = (
        gnr_positive_rank < gnr_negative_rank
        if gnr_positive_rank is not None and gnr_negative_rank is not None
        else False
    )

    # Classify scenario and count results
    gnr_scenario = classify_ranking_scenario(gnr_positive_rank, gnr_negative_rank)
    gnr_result_count = len(gnr_result.entity_ids)

    # Search with GNR + embedding (combined)
    emb_result = search_gnr_and_embedding(engine, model, model_type, anchor, truncate_dim)
    emb_positive_rank = None
    emb_negative_rank = None
    emb_positive_score = None
    emb_negative_score = None

    # Find positive rank and score
    if positive_entity_id in emb_result.entity_ids:
        emb_positive_rank = emb_result.entity_ids.index(positive_entity_id) + 1
        # Prefer embedding score if available, otherwise use GNR score
        emb_positive_score = emb_result.emb_scores.get(positive_entity_id) or emb_result.gnr_scores.get(positive_entity_id)

    # Find negative rank (first non-positive entity as proxy)
    emb_negative_rank = find_first_non_positive_rank(emb_result.entity_ids, positive_entity_id)
    if emb_negative_rank is not None and len(emb_result.entity_ids) >= emb_negative_rank:
        negative_entity_id = emb_result.entity_ids[emb_negative_rank - 1]  # 0-indexed access
        emb_negative_score = emb_result.emb_scores.get(negative_entity_id) or emb_result.gnr_scores.get(negative_entity_id)

    emb_correct = emb_positive_rank == 1 if emb_positive_rank is not None else False

    # Compute relative ranking correctness
    emb_relative_correct = (
        emb_positive_rank < emb_negative_rank
        if emb_positive_rank is not None and emb_negative_rank is not None
        else False
    )

    # Classify scenario and count results
    emb_scenario = classify_ranking_scenario(emb_positive_rank, emb_negative_rank)
    emb_result_count = len(emb_result.entity_ids)

    return TripletResult(
        anchor=anchor,
        positive=positive,
        negative=negative,
        anchor_group=anchor_group,
        variant_type=variant_type,
        original_anchor=original_anchor,
        gnr_positive_rank=gnr_positive_rank,
        gnr_negative_rank=gnr_negative_rank,
        gnr_positive_score=gnr_positive_score,
        gnr_negative_score=gnr_negative_score,
        gnr_correct=gnr_correct,
        gnr_relative_correct=gnr_relative_correct,
        gnr_result_count=gnr_result_count,
        gnr_scenario=gnr_scenario,
        gnr_query_time_ms=gnr_result.query_time_ms,
        emb_positive_rank=emb_positive_rank,
        emb_negative_rank=emb_negative_rank,
        emb_positive_score=emb_positive_score,
        emb_negative_score=emb_negative_score,
        emb_correct=emb_correct,
        emb_relative_correct=emb_relative_correct,
        emb_result_count=emb_result_count,
        emb_scenario=emb_scenario,
        emb_query_time_ms=emb_result.query_time_ms,
    )


# -----------------------------------------------------------------------------
# Metrics computation - New functions for Phase 4
# -----------------------------------------------------------------------------
def compute_relative_ranking_metrics(results: list[TripletResult], mode: str) -> RelativeRankingMetrics:
    """
    Compute relative ranking metrics (positive vs negative ranking).

    Args:
        results: List of TripletResult objects
        mode: "gnr" or "emb" to select which results to analyze

    Returns:
        RelativeRankingMetrics object with all relative ranking statistics
    """
    if not results:
        return RelativeRankingMetrics(
            positive_gt_negative_rate=0.0,
            both_found_correct_order=0.0,
            both_found_wrong_order=0.0,
            only_positive_found=0.0,
            only_negative_found=0.0,
            neither_found=0.0,
            avg_rank_separation=0.0,
            median_rank_separation=0.0,
            avg_positive_rank=0.0,
            avg_negative_rank=0.0,
            avg_result_count=0.0,
        )

    total = len(results)

    # Count scenarios
    scenario_counts = {
        "both_found_correct_order": 0,
        "both_found_wrong_order": 0,
        "only_positive_found": 0,
        "only_negative_found": 0,
        "neither_found": 0,
    }

    positive_ranks = []
    negative_ranks = []
    rank_separations = []
    result_counts = []

    for r in results:
        # Select mode-specific fields
        if mode == "gnr":
            pos_rank = r.gnr_positive_rank
            neg_rank = r.gnr_negative_rank
            scenario = r.gnr_scenario
            result_count = r.gnr_result_count
        else:  # emb
            pos_rank = r.emb_positive_rank
            neg_rank = r.emb_negative_rank
            scenario = r.emb_scenario
            result_count = r.emb_result_count

        # Count scenario
        if scenario in scenario_counts:
            scenario_counts[scenario] += 1

        # Track ranks and separations
        if pos_rank is not None:
            positive_ranks.append(pos_rank)
        if neg_rank is not None:
            negative_ranks.append(neg_rank)
        if pos_rank is not None and neg_rank is not None:
            rank_separations.append(neg_rank - pos_rank)

        result_counts.append(result_count)

    # Compute percentages
    positive_gt_negative_count = scenario_counts["both_found_correct_order"] + scenario_counts["only_positive_found"]
    positive_gt_negative_rate = (positive_gt_negative_count / total * 100) if total > 0 else 0.0

    return RelativeRankingMetrics(
        positive_gt_negative_rate=positive_gt_negative_rate,
        both_found_correct_order=scenario_counts["both_found_correct_order"] / total * 100,
        both_found_wrong_order=scenario_counts["both_found_wrong_order"] / total * 100,
        only_positive_found=scenario_counts["only_positive_found"] / total * 100,
        only_negative_found=scenario_counts["only_negative_found"] / total * 100,
        neither_found=scenario_counts["neither_found"] / total * 100,
        avg_rank_separation=np.mean(rank_separations) if rank_separations else 0.0,
        median_rank_separation=np.median(rank_separations) if rank_separations else 0.0,
        avg_positive_rank=np.mean(positive_ranks) if positive_ranks else 0.0,
        avg_negative_rank=np.mean(negative_ranks) if negative_ranks else 0.0,
        avg_result_count=np.mean(result_counts) if result_counts else 0.0,
    )


def compute_ndcg_at_k(results: list[TripletResult], mode: str, k: int) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain) for ranking quality.

    NDCG measures how well the positive entity is ranked, with higher scores
    for better rankings. Score of 1.0 means perfect ranking (positive at rank 1).

    Args:
        results: List of TripletResult objects
        mode: "gnr" or "emb" to select which results to analyze
        k: Cutoff rank (e.g., 5, 10, 20, 100)

    Returns:
        Mean NDCG@K score (0.0 to 1.0)
    """
    if not results:
        return 0.0

    ndcg_scores = []

    for r in results:
        # Select mode-specific positive rank
        positive_rank = r.gnr_positive_rank if mode == "gnr" else r.emb_positive_rank

        if positive_rank is None:
            # Not found = worst case
            ndcg_scores.append(0.0)
        elif positive_rank <= k:
            # DCG = relevance / log2(rank+1)
            # For binary relevance (1 for positive, 0 for all others):
            dcg = 1.0 / np.log2(positive_rank + 1)
            # IDCG (ideal) = 1.0 / log2(2) = 1.0 (positive at rank 1)
            idcg = 1.0
            ndcg_scores.append(dcg / idcg)
        else:
            # Beyond K
            ndcg_scores.append(0.0)

    return np.mean(ndcg_scores)


def compute_rescue_rates(results: list[TripletResult]) -> dict[str, float]:
    """
    Compute rescue rate metrics - cases where embeddings succeed when GNR fails.

    Returns:
        Dict with rescue rate statistics
    """
    if not results:
        return {
            "rescue_rate": 0.0,
            "both_correct": 0.0,
            "both_wrong": 0.0,
            "gnr_only_correct": 0.0,
            "emb_only_correct": 0.0,
        }

    total = len(results)
    rescue_count = 0  # EMB correct when GNR wrong
    both_correct = 0
    both_wrong = 0
    gnr_only_correct = 0
    emb_only_correct = 0

    for r in results:
        gnr_found = r.gnr_positive_rank is not None
        emb_found = r.emb_positive_rank is not None

        if not gnr_found and emb_found:
            rescue_count += 1

        if r.gnr_correct and r.emb_correct:
            both_correct += 1
        elif not r.gnr_correct and not r.emb_correct:
            both_wrong += 1
        elif r.gnr_correct and not r.emb_correct:
            gnr_only_correct += 1
        elif not r.gnr_correct and r.emb_correct:
            emb_only_correct += 1

    return {
        "rescue_rate": rescue_count / total * 100,
        "both_correct": both_correct / total * 100,
        "both_wrong": both_wrong / total * 100,
        "gnr_only_correct": gnr_only_correct / total * 100,
        "emb_only_correct": emb_only_correct / total * 100,
    }


def compute_volume_metrics(results: list[TripletResult], mode: str) -> dict[str, Any]:
    """
    Compute result volume metrics and correlation with ranking quality.

    Args:
        results: List of TripletResult objects
        mode: "gnr" or "emb" to select which results to analyze

    Returns:
        Dict with volume statistics
    """
    if not results:
        return {
            "avg_result_count": 0.0,
            "median_result_count": 0.0,
            "min_result_count": 0,
            "max_result_count": 0,
            "volume_rank_correlation": 0.0,
        }

    result_counts = []
    positive_ranks = []

    for r in results:
        if mode == "gnr":
            result_counts.append(r.gnr_result_count)
            if r.gnr_positive_rank is not None:
                positive_ranks.append((r.gnr_result_count, r.gnr_positive_rank))
        else:  # emb
            result_counts.append(r.emb_result_count)
            if r.emb_positive_rank is not None:
                positive_ranks.append((r.emb_result_count, r.emb_positive_rank))

    # Compute correlation between volume and rank
    correlation = 0.0
    if len(positive_ranks) > 1:
        counts, ranks = zip(*positive_ranks)
        correlation = np.corrcoef(counts, ranks)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

    return {
        "avg_result_count": np.mean(result_counts),
        "median_result_count": np.median(result_counts),
        "min_result_count": int(np.min(result_counts)),
        "max_result_count": int(np.max(result_counts)),
        "volume_rank_correlation": correlation,
    }


def compute_metrics_by_variant(results: list[TripletResult]) -> dict[str, VariantTypeMetrics]:
    """
    Compute metrics broken down by variant type.

    Returns:
        Dict mapping variant_type -> VariantTypeMetrics
    """
    if not results:
        return {}

    # Group results by variant type
    variants: dict[str, list[TripletResult]] = {}
    for r in results:
        vtype = r.variant_type
        if vtype not in variants:
            variants[vtype] = []
        variants[vtype].append(r)

    # Compute metrics for each variant type
    variant_metrics = {}
    for vtype, vresults in variants.items():
        total = len(vresults)

        # Accuracy
        emb_correct_count = sum(1 for r in vresults if r.emb_correct)
        accuracy_top1 = emb_correct_count / total * 100

        # Recall@K
        recall_at_k = {}
        for k in K_VALUES:
            recall_count = sum(1 for r in vresults if r.emb_positive_rank is not None and r.emb_positive_rank <= k)
            recall_at_k[k] = recall_count / total * 100

        # MRR
        mrr_values = [1.0 / r.emb_positive_rank for r in vresults if r.emb_positive_rank is not None]
        mrr = np.mean(mrr_values) if mrr_values else 0.0

        # Latency
        query_times = [r.emb_query_time_ms for r in vresults]
        latency_avg = np.mean(query_times)
        latency_p50 = np.percentile(query_times, 50)
        latency_p95 = np.percentile(query_times, 95)

        # Relative ranking metrics
        relative_ranking = compute_relative_ranking_metrics(vresults, "emb")

        variant_metrics[vtype] = VariantTypeMetrics(
            variant_type=vtype,
            count=total,
            relative_ranking=relative_ranking,
            accuracy_top1=accuracy_top1,
            recall_at_k=recall_at_k,
            mrr=mrr,
            latency_avg_ms=latency_avg,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
        )

    return variant_metrics


def compute_threshold_analysis(results: list[TripletResult], mode: str) -> dict[str, Any]:
    """
    Analyze score distributions and recommend optimal thresholds.

    Args:
        results: List of TripletResult objects
        mode: "gnr" or "emb" to select which scores to analyze

    Returns:
        Dict with score distributions and threshold recommendations
    """
    if not results:
        return {}

    positive_scores = []
    negative_scores = []

    for r in results:
        if mode == "gnr":
            if r.gnr_positive_score is not None:
                positive_scores.append(r.gnr_positive_score)
            if r.gnr_negative_score is not None:
                negative_scores.append(r.gnr_negative_score)
        else:  # emb
            if r.emb_positive_score is not None:
                positive_scores.append(r.emb_positive_score)
            if r.emb_negative_score is not None:
                negative_scores.append(r.emb_negative_score)

    if not positive_scores:
        return {}

    # Score distribution statistics
    pos_mean = np.mean(positive_scores)
    pos_median = np.median(positive_scores)
    pos_std = np.std(positive_scores)
    pos_p05 = np.percentile(positive_scores, 5)
    pos_p10 = np.percentile(positive_scores, 10)
    pos_p25 = np.percentile(positive_scores, 25)
    pos_p75 = np.percentile(positive_scores, 75)
    pos_p90 = np.percentile(positive_scores, 90)
    pos_p95 = np.percentile(positive_scores, 95)

    neg_mean = np.mean(negative_scores) if negative_scores else 0.0
    neg_median = np.median(negative_scores) if negative_scores else 0.0
    neg_std = np.std(negative_scores) if negative_scores else 0.0

    separation = pos_mean - neg_mean

    # Generate threshold recommendations
    # Test different closeScore values and evaluate their impact
    threshold_options = []

    for close_score in [50, 60, 65, 70, 75, 80]:
        # Estimate positive recall at this threshold
        pos_recall = sum(1 for s in positive_scores if s >= close_score) / len(positive_scores) * 100

        # Estimate pos>neg rate (simplified: assume scores above threshold)
        pos_above = sum(1 for s in positive_scores if s >= close_score)
        neg_above = sum(1 for s in negative_scores if s >= close_score)
        total_above = pos_above + neg_above
        pos_gt_neg_rate = (pos_above / total_above * 100) if total_above > 0 else 0.0

        # Create full 5-level threshold set
        same_score = min(100, close_score + 15)
        likely_score = max(20, close_score - 10)
        plausible_score = max(15, close_score - 20)
        unlikely_score = max(10, close_score - 30)

        threshold_options.append({
            "name": f"Option (close={close_score})",
            "sameScore": same_score,
            "closeScore": close_score,
            "likelyScore": likely_score,
            "plausibleScore": plausible_score,
            "unlikelyScore": unlikely_score,
            "estimated_pos_recall": pos_recall,
            "estimated_pos_gt_neg_rate": pos_gt_neg_rate,
        })

    # Recommend best options
    # Option A: Balanced (95% positive recall)
    balanced = next((opt for opt in threshold_options if opt["estimated_pos_recall"] >= 95), threshold_options[-1])

    # Option B: Conservative (90% positive recall, higher precision)
    conservative = next((opt for opt in threshold_options if opt["estimated_pos_recall"] >= 90 and opt["closeScore"] > balanced["closeScore"]), balanced)

    return {
        "positive_scores": {
            "mean": pos_mean,
            "median": pos_median,
            "std": pos_std,
            "p05": pos_p05,
            "p10": pos_p10,
            "p25": pos_p25,
            "p75": pos_p75,
            "p90": pos_p90,
            "p95": pos_p95,
        },
        "negative_scores": {
            "mean": neg_mean,
            "median": neg_median,
            "std": neg_std,
        },
        "separation": separation,
        "threshold_options": threshold_options,
        "recommended_balanced": balanced,
        "recommended_conservative": conservative,
    }


# -----------------------------------------------------------------------------
# Metrics computation - Main function
# -----------------------------------------------------------------------------
def compute_metrics(results: list[TripletResult]) -> dict[str, Any]:
    """
    Compute comprehensive evaluation metrics from triplet results.
    Compares GNR-only (baseline) vs GNR+Embedding (combined) performance.
    """
    if not results:
        return {}

    # Basic counts
    total = len(results)

    # GNR-only metrics (baseline)
    gnr_correct_count = sum(1 for r in results if r.gnr_correct)
    gnr_accuracy = gnr_correct_count / total * 100

    gnr_recall_at_k = {}
    for k in K_VALUES:
        recall_count = sum(1 for r in results if r.gnr_positive_rank is not None and r.gnr_positive_rank <= k)
        gnr_recall_at_k[k] = recall_count / total * 100

    gnr_query_times = [r.gnr_query_time_ms for r in results]
    gnr_latency_avg = np.mean(gnr_query_times)
    gnr_latency_p50 = np.percentile(gnr_query_times, 50)
    gnr_latency_p95 = np.percentile(gnr_query_times, 95)
    gnr_latency_p99 = np.percentile(gnr_query_times, 99)

    # GNR+Embedding metrics (combined)
    emb_correct_count = sum(1 for r in results if r.emb_correct)
    emb_accuracy = emb_correct_count / total * 100

    emb_recall_at_k = {}
    for k in K_VALUES:
        recall_count = sum(1 for r in results if r.emb_positive_rank is not None and r.emb_positive_rank <= k)
        emb_recall_at_k[k] = recall_count / total * 100

    emb_query_times = [r.emb_query_time_ms for r in results]
    emb_latency_avg = np.mean(emb_query_times)
    emb_latency_p50 = np.percentile(emb_query_times, 50)
    emb_latency_p95 = np.percentile(emb_query_times, 95)
    emb_latency_p99 = np.percentile(emb_query_times, 99)

    # MRR (Mean Reciprocal Rank)
    gnr_mrr_values = [1.0 / r.gnr_positive_rank for r in results if r.gnr_positive_rank is not None]
    gnr_mrr = np.mean(gnr_mrr_values) if gnr_mrr_values else 0.0

    emb_mrr_values = [1.0 / r.emb_positive_rank for r in results if r.emb_positive_rank is not None]
    emb_mrr = np.mean(emb_mrr_values) if emb_mrr_values else 0.0

    # Rescue rates (where adding embeddings changes the result)
    emb_improvement = sum(1 for r in results if r.emb_correct and not r.gnr_correct)
    emb_degradation = sum(1 for r in results if r.gnr_correct and not r.emb_correct)
    emb_improvement_rate = emb_improvement / total * 100
    emb_degradation_rate = emb_degradation / total * 100

    # NEW METRICS (Phase 4)
    # Relative ranking metrics
    gnr_relative_ranking = compute_relative_ranking_metrics(results, "gnr")
    emb_relative_ranking = compute_relative_ranking_metrics(results, "emb")

    # NDCG@K for ranking quality
    gnr_ndcg = {}
    emb_ndcg = {}
    for k in [5, 10, 20, 100]:
        gnr_ndcg[k] = compute_ndcg_at_k(results, "gnr", k)
        emb_ndcg[k] = compute_ndcg_at_k(results, "emb", k)

    # Rescue rates (detailed)
    rescue_rates = compute_rescue_rates(results)

    # Volume metrics
    gnr_volume = compute_volume_metrics(results, "gnr")
    emb_volume = compute_volume_metrics(results, "emb")

    # Metrics by variant type
    variant_metrics = compute_metrics_by_variant(results)

    # Threshold analysis
    gnr_threshold_analysis = compute_threshold_analysis(results, "gnr")
    emb_threshold_analysis = compute_threshold_analysis(results, "emb")

    return {
        "total_triplets": total,

        # GNR-only (baseline) - Legacy metrics
        "gnr_accuracy": gnr_accuracy,
        "gnr_recall_at_k": gnr_recall_at_k,
        "gnr_mrr": gnr_mrr,
        "gnr_latency_avg_ms": gnr_latency_avg,
        "gnr_latency_p50_ms": gnr_latency_p50,
        "gnr_latency_p95_ms": gnr_latency_p95,
        "gnr_latency_p99_ms": gnr_latency_p99,

        # GNR-only (baseline) - NEW Phase 4 metrics
        "gnr_relative_ranking": {
            "positive_gt_negative_rate": gnr_relative_ranking.positive_gt_negative_rate,
            "both_found_correct_order": gnr_relative_ranking.both_found_correct_order,
            "both_found_wrong_order": gnr_relative_ranking.both_found_wrong_order,
            "only_positive_found": gnr_relative_ranking.only_positive_found,
            "only_negative_found": gnr_relative_ranking.only_negative_found,
            "neither_found": gnr_relative_ranking.neither_found,
            "avg_rank_separation": gnr_relative_ranking.avg_rank_separation,
            "median_rank_separation": gnr_relative_ranking.median_rank_separation,
            "avg_positive_rank": gnr_relative_ranking.avg_positive_rank,
            "avg_negative_rank": gnr_relative_ranking.avg_negative_rank,
            "avg_result_count": gnr_relative_ranking.avg_result_count,
        },
        "gnr_ndcg": gnr_ndcg,
        "gnr_volume": gnr_volume,
        "gnr_threshold_analysis": gnr_threshold_analysis,

        # GNR+Embedding (combined) - Legacy metrics
        "emb_accuracy": emb_accuracy,
        "emb_recall_at_k": emb_recall_at_k,
        "emb_mrr": emb_mrr,
        "emb_latency_avg_ms": emb_latency_avg,
        "emb_latency_p50_ms": emb_latency_p50,
        "emb_latency_p95_ms": emb_latency_p95,
        "emb_latency_p99_ms": emb_latency_p99,

        # GNR+Embedding (combined) - NEW Phase 4 metrics
        "emb_relative_ranking": {
            "positive_gt_negative_rate": emb_relative_ranking.positive_gt_negative_rate,
            "both_found_correct_order": emb_relative_ranking.both_found_correct_order,
            "both_found_wrong_order": emb_relative_ranking.both_found_wrong_order,
            "only_positive_found": emb_relative_ranking.only_positive_found,
            "only_negative_found": emb_relative_ranking.only_negative_found,
            "neither_found": emb_relative_ranking.neither_found,
            "avg_rank_separation": emb_relative_ranking.avg_rank_separation,
            "median_rank_separation": emb_relative_ranking.median_rank_separation,
            "avg_positive_rank": emb_relative_ranking.avg_positive_rank,
            "avg_negative_rank": emb_relative_ranking.avg_negative_rank,
            "avg_result_count": emb_relative_ranking.avg_result_count,
        },
        "emb_ndcg": emb_ndcg,
        "emb_volume": emb_volume,
        "emb_threshold_analysis": emb_threshold_analysis,

        # Impact of embeddings (legacy)
        "emb_improvement_rate": emb_improvement_rate,  # Cases where embeddings helped
        "emb_degradation_rate": emb_degradation_rate,  # Cases where embeddings hurt

        # NEW Phase 4: Rescue rates (detailed)
        "rescue_rates": rescue_rates,

        # NEW Phase 4: Variant type breakdown
        "variant_metrics": {
            vtype: {
                "count": vm.count,
                "relative_ranking": {
                    "positive_gt_negative_rate": vm.relative_ranking.positive_gt_negative_rate,
                    "both_found_correct_order": vm.relative_ranking.both_found_correct_order,
                    "both_found_wrong_order": vm.relative_ranking.both_found_wrong_order,
                    "only_positive_found": vm.relative_ranking.only_positive_found,
                    "only_negative_found": vm.relative_ranking.only_negative_found,
                    "neither_found": vm.relative_ranking.neither_found,
                    "avg_rank_separation": vm.relative_ranking.avg_rank_separation,
                },
                "accuracy_top1": vm.accuracy_top1,
                "recall_at_k": vm.recall_at_k,
                "mrr": vm.mrr,
                "latency_avg_ms": vm.latency_avg_ms,
                "latency_p50_ms": vm.latency_p50_ms,
                "latency_p95_ms": vm.latency_p95_ms,
            }
            for vtype, vm in variant_metrics.items()
        },
    }


def print_metrics(metrics: dict[str, Any], model_name: str, test_set: str) -> None:
    """Print evaluation metrics in a readable format with new Phase 4 metrics."""
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {model_name} on {test_set}")
    print("=" * 80)
    print("NOTE: Evaluation uses SZ_SEARCH_INCLUDE_ALL_CANDIDATES (threshold-independent)")

    print(f"\nTotal queries evaluated: {metrics['total_triplets']:,}")

    # RELATIVE RANKING (PRIMARY METRIC)
    print("\n" + "=" * 80)
    print("RELATIVE RANKING (PRIMARY)")
    print("=" * 80)
    gnr_rr = metrics['gnr_relative_ranking']
    emb_rr = metrics['emb_relative_ranking']

    print(f"{'':40} {'Name-Only':>15} {'Embeddings':>15} {'Delta':>10}")
    print("-" * 80)
    print(f"{'Positive > Negative Rate:':<40} {gnr_rr['positive_gt_negative_rate']:>14.1f}% {emb_rr['positive_gt_negative_rate']:>14.1f}% {emb_rr['positive_gt_negative_rate'] - gnr_rr['positive_gt_negative_rate']:>+9.1f}%")

    print("\nScenarios (Embeddings):")
    print(f"  ✓ Both found, correct order       {emb_rr['both_found_correct_order']:>6.1f}%")
    print(f"  ✗ Both found, wrong order          {emb_rr['both_found_wrong_order']:>6.1f}%")
    print(f"  ✓ Only positive found              {emb_rr['only_positive_found']:>6.1f}%")
    print(f"  ✗ Only negative found              {emb_rr['only_negative_found']:>6.1f}%")
    print(f"  ○ Neither found                    {emb_rr['neither_found']:>6.1f}%")

    print(f"\nAverage rank separation:          {gnr_rr['avg_rank_separation']:>+6.1f} ranks   {emb_rr['avg_rank_separation']:>+6.1f} ranks")

    # RANKING QUALITY (NDCG)
    print("\n" + "=" * 80)
    print("RANKING QUALITY (NDCG@K)")
    print("=" * 80)
    print(f"{'':20} {'Name-Only':>15} {'Embeddings':>15} {'Delta':>10}")
    print("-" * 80)

    for k in [5, 10, 20, 100]:
        gnr_ndcg = metrics['gnr_ndcg'].get(k, 0.0)
        emb_ndcg = metrics['emb_ndcg'].get(k, 0.0)
        delta = emb_ndcg - gnr_ndcg
        print(f"NDCG@{k:<3}:            {gnr_ndcg:>15.4f} {emb_ndcg:>15.4f} {delta:>+10.4f}")

    # Compute mean NDCG
    gnr_ndcg_mean = sum(metrics['gnr_ndcg'].values()) / len(metrics['gnr_ndcg'])
    emb_ndcg_mean = sum(metrics['emb_ndcg'].values()) / len(metrics['emb_ndcg'])
    print("-" * 80)
    print(f"Mean NDCG (all K):  {gnr_ndcg_mean:>15.4f} {emb_ndcg_mean:>15.4f} {emb_ndcg_mean - gnr_ndcg_mean:>+10.4f}")

    # THRESHOLD ANALYSIS
    if metrics.get('emb_threshold_analysis'):
        print("\n" + "=" * 80)
        print("THRESHOLD ANALYSIS")
        print("=" * 80)

        ta = metrics['emb_threshold_analysis']
        pos_scores = ta.get('positive_scores', {})
        neg_scores = ta.get('negative_scores', {})

        print("Score Distribution (Embeddings):")
        print(f"  Positive entities:    Mean={pos_scores.get('mean', 0):.1f}, Median={pos_scores.get('median', 0):.1f}, Std={pos_scores.get('std', 0):.1f}")
        print(f"  Negative entities:    Mean={neg_scores.get('mean', 0):.1f}, Median={neg_scores.get('median', 0):.1f}, Std={neg_scores.get('std', 0):.1f}")
        print(f"  Separation:           {ta.get('separation', 0):.1f} points")

        print("\nRecommended Thresholds (CFG_CFRTN format - all 5 levels):")

        balanced = ta.get('recommended_balanced', {})
        if balanced:
            print(f"\n  Option A (Balanced - {balanced.get('estimated_pos_recall', 0):.0f}% pos recall):")
            print(f"    sameScore:        {balanced.get('sameScore', 0)}")
            print(f"    closeScore:       {balanced.get('closeScore', 0)}  (cutoff)")
            print(f"    likelyScore:      {balanced.get('likelyScore', 0)}")
            print(f"    plausibleScore:   {balanced.get('plausibleScore', 0)}")
            print(f"    unlikelyScore:    {balanced.get('unlikelyScore', 0)}")
            print(f"    Pos>Neg rate:     {balanced.get('estimated_pos_gt_neg_rate', 0):.1f}%")

        conservative = ta.get('recommended_conservative', {})
        if conservative and conservative.get('closeScore') != balanced.get('closeScore'):
            print(f"\n  Option B (Conservative - {conservative.get('estimated_pos_recall', 0):.0f}% pos recall):")
            print(f"    sameScore:        {conservative.get('sameScore', 0)}")
            print(f"    closeScore:       {conservative.get('closeScore', 0)}  (cutoff)")
            print(f"    likelyScore:      {conservative.get('likelyScore', 0)}")
            print(f"    plausibleScore:   {conservative.get('plausibleScore', 0)}")
            print(f"    unlikelyScore:    {conservative.get('unlikelyScore', 0)}")
            print(f"    Pos>Neg rate:     {conservative.get('estimated_pos_gt_neg_rate', 0):.1f}%")

    # VARIANT TYPE BREAKDOWN
    if metrics.get('variant_metrics'):
        print("\n" + "=" * 80)
        print("VARIANT TYPE BREAKDOWN")
        print("=" * 80)
        print(f"{'Variant Type':<30} {'Count':>8} {'Pos>Neg%':>10} {'Top-1':>8} {'R@10':>8} {'MRR':>8}")
        print("-" * 80)

        # Sort variants by type (exact first, then alphabetically)
        sorted_variants = sorted(
            metrics['variant_metrics'].items(),
            key=lambda x: (x[0] != "exact", x[0])
        )

        for vtype, vm in sorted_variants:
            pos_gt_neg = vm['relative_ranking']['positive_gt_negative_rate']
            top1 = vm['accuracy_top1']
            recall_10 = vm['recall_at_k'].get(10, 0.0)
            mrr = vm['mrr']
            count = vm['count']

            print(f"{vtype:<30} {count:>8,} {pos_gt_neg:>9.1f}% {top1:>7.1f}% {recall_10:>7.1f}% {mrr:>8.4f}")

    # RESCUE ANALYSIS
    print("\n" + "=" * 80)
    print("RESCUE ANALYSIS")
    print("=" * 80)
    rescue = metrics.get('rescue_rates', {})
    print(f"Embedding rescue rate:      {rescue.get('rescue_rate', 0):.1f}%   (EMB found when name-only failed)")
    print(f"Both correct:               {rescue.get('both_correct', 0):.1f}%")
    print(f"Both wrong:                 {rescue.get('both_wrong', 0):.1f}%")
    print(f"Name-only correct:          {rescue.get('gnr_only_correct', 0):.1f}%")
    print(f"Embedding-only correct:     {rescue.get('emb_only_correct', 0):.1f}%")

    # VOLUME ANALYSIS
    print("\n" + "=" * 80)
    print("VOLUME ANALYSIS")
    print("=" * 80)
    gnr_vol = metrics.get('gnr_volume', {})
    emb_vol = metrics.get('emb_volume', {})

    gnr_avg = gnr_vol.get('avg_result_count', 0)
    emb_avg = emb_vol.get('avg_result_count', 0)
    delta = emb_avg - gnr_avg

    print(f"Avg results returned:       {gnr_avg:.1f} → {emb_avg:.1f} ({delta:+.1f})")
    print(f"Volume×rank correlation:    {emb_vol.get('volume_rank_correlation', 0):.3f} (negative = more results → better ranks)")

    # LEGACY METRICS (for backward compatibility)
    print("\n" + "=" * 80)
    print("LEGACY METRICS (Top-1, MRR, Recall@K)")
    print("=" * 80)
    print(f"{'':40} {'Name-Only':>15} {'Embeddings':>15} {'Delta':>10}")
    print("-" * 80)
    print(f"{'Accuracy (top-1):':<40} {metrics['gnr_accuracy']:>14.2f}% {metrics['emb_accuracy']:>14.2f}% {metrics['emb_accuracy'] - metrics['gnr_accuracy']:>+9.2f}%")
    print(f"{'MRR:':<40} {metrics['gnr_mrr']:>15.4f} {metrics['emb_mrr']:>15.4f} {metrics['emb_mrr'] - metrics['gnr_mrr']:>+10.4f}")

    print("\nRecall@K:")
    for k in K_VALUES:
        gnr_recall = metrics['gnr_recall_at_k'][k]
        emb_recall = metrics['emb_recall_at_k'][k]
        print(f"  Recall@{k:<3}:                        {gnr_recall:>14.2f}% {emb_recall:>14.2f}% {emb_recall - gnr_recall:>+9.2f}%")

    # LATENCY
    print("\n" + "=" * 80)
    print("QUERY LATENCY")
    print("=" * 80)
    print(f"{'':40} {'Name-Only':>15} {'Embeddings':>15}")
    print("-" * 80)
    print(f"{'Average:':<40} {metrics['gnr_latency_avg_ms']:>13.2f} ms {metrics['emb_latency_avg_ms']:>13.2f} ms")
    print(f"{'P50:':<40} {metrics['gnr_latency_p50_ms']:>13.2f} ms {metrics['emb_latency_p50_ms']:>13.2f} ms")
    print(f"{'P95:':<40} {metrics['gnr_latency_p95_ms']:>13.2f} ms {metrics['emb_latency_p95_ms']:>13.2f} ms")
    print(f"{'P99:':<40} {metrics['gnr_latency_p99_ms']:>13.2f} ms {metrics['emb_latency_p99_ms']:>13.2f} ms")

    print("\n" + "=" * 80)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sz_evaluate_model",
        description="Evaluate embedding model performance through Senzing entity resolution.",
    )

    parser.add_argument("--type", type=str, required=True, choices=["personal", "business"],
                       help="Model type: 'personal' or 'business'")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the SentenceTransformer model")
    parser.add_argument("--triplets", type=str, required=True,
                       help="Path to test triplets JSONL file")
    parser.add_argument("--test_set", type=str, required=True,
                       help="Test set name (e.g., 'opensanctions', 'wikidata')")
    parser.add_argument("--data_source", type=str, default="OPEN_SANCTIONS",
                       help="Senzing DATA_SOURCE to look up records (default: OPEN_SANCTIONS)")
    parser.add_argument("--sample", type=int, default=0,
                       help="Evaluate only first N triplets (0 = all)")
    parser.add_argument("--output", type=str, default=None,
                       help="Optional JSON file to save detailed results")
    parser.add_argument("--truncate_dim", type=int, default=None,
                       help="Matryoshka truncation dimension (e.g., 512 for 768d models)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging (default: INFO level)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose Senzing logging")

    # Variant generation control (NEW)
    parser.add_argument("--data_file", type=str, default=None,
                       help="Path to JSONL data file for alias extraction (e.g., opensanctions_test_5k_final.jsonl)")
    parser.add_argument("--variants", type=str, default="both", choices=["aliases", "synthetics", "both", "none"],
                       help="Variant types to test: 'aliases' (from data), 'synthetics' (generated), 'both', or 'none' (default: both)")
    parser.add_argument("--synthetic-types", type=str, default="abbreviated,fuzzy,partial",
                       help="Comma-separated list of synthetic variant types: abbreviated, fuzzy, partial (default: all)")

    args = parser.parse_args()

    # Parse synthetic variant types
    synthetic_types = [t.strip() for t in args.synthetic_types.split(',') if t.strip()]

    # Configure logging based on --debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        force=True  # Override any existing configuration
    )

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Using device: {device}")

    if args.truncate_dim:
        print(f"📌 Matryoshka truncation: {args.truncate_dim} dimensions")

    # Load model
    print(f"⏳ Loading {args.type} model from {args.model_path}...")
    model = SentenceTransformer(args.model_path)

    # Initialize Senzing
    print("⏳ Initializing Senzing engine...")
    settings = get_senzing_config()
    verbose_logging = 1 if args.verbose else 0
    sz_factory = SzAbstractFactoryCore("SzEmbeddingsEval", settings, verbose_logging=verbose_logging)
    sz_engine = sz_factory.create_engine()

    # Load triplets
    print(f"⏳ Loading triplets from {args.triplets}...")
    triplets = []
    with open(args.triplets, 'r') as f:
        for line in f:
            triplet = orjson.loads(line.strip())
            triplets.append(triplet)
            if args.sample > 0 and len(triplets) >= args.sample:
                break

    print(f"📊 Evaluating {len(triplets):,} triplets...")

    # Print variant generation info
    if args.variants != "none":
        print(f"📝 Variant mode: {args.variants}")
        if args.variants in ("synthetics", "both"):
            print(f"   Synthetic types: {', '.join(synthetic_types)}")
        if args.variants in ("aliases", "both"):
            if args.data_file:
                print(f"   Alias data file: {args.data_file}")
            else:
                print(f"   ⚠️  WARNING: --data_file not provided, aliases will be skipped")

    # Evaluate triplets
    start_time = time.time()
    results = []
    skipped = 0
    total_queries = 0  # Track total number of queries (including variants)

    for i, triplet in enumerate(triplets):
        # Generate variants for this triplet
        variants = generate_all_variants_for_triplet(
            triplet,
            args.data_file,
            args.type,
            args.variants,
            synthetic_types,
        )

        # Evaluate each variant
        for variant_query, variant_type in variants:
            # Create variant triplet
            variant_triplet = triplet.copy()
            variant_triplet["anchor"] = variant_query
            variant_triplet["variant_type"] = variant_type
            variant_triplet["original_anchor"] = triplet["anchor"]

            result = evaluate_triplet(
                sz_engine,
                model,
                args.type,
                args.data_source,
                variant_triplet,
                args.truncate_dim,
            )

            if result is not None:
                results.append(result)
            else:
                skipped += 1

            total_queries += 1

        # Progress reporting (per base triplet)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(triplets) - i - 1) / rate if rate > 0 else 0
            avg_variants = total_queries / (i + 1)
            print(f"⏳ Progress: {i+1:,}/{len(triplets):,} triplets ({(i+1)/len(triplets)*100:.1f}%) "
                  f"| {total_queries:,} queries ({avg_variants:.1f} variants/triplet avg) "
                  f"| Rate: {rate:.1f} triplets/sec | ETA: {format_seconds_to_hhmmss(remaining)}",
                  flush=True, end='\r')

    print()  # New line after progress

    total_time = time.time() - start_time
    print(f"\n✅ Evaluation complete in {format_seconds_to_hhmmss(total_time)}")
    print(f"   Base triplets: {len(triplets):,}")
    print(f"   Total queries: {total_queries:,} (avg {total_queries/len(triplets):.1f} variants per triplet)")
    print(f"   Evaluated: {len(results):,} queries")
    print(f"   Skipped: {skipped:,} queries (records not found in Senzing)")

    # Compute and print metrics
    if results:
        metrics = compute_metrics(results)
        print_metrics(metrics, args.model_path.split('/')[-1], args.test_set)

        # Save detailed results if requested
        if args.output:
            output_data = {
                "model_path": args.model_path,
                "model_type": args.type,
                "test_set": args.test_set,
                "data_source": args.data_source,
                "total_triplets": len(triplets),
                "evaluated_triplets": len(results),
                "skipped_triplets": skipped,
                "evaluation_time_seconds": total_time,
                "metrics": metrics,
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Detailed results saved to: {args.output}")
    else:
        print("\n⚠️  No results to evaluate (all triplets skipped)")
        sys.exit(1)
