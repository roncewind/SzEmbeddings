# Testing Embedding Models Through Senzing Entity Resolution

**Created:** 2025-12-15
**Purpose:** Evaluate how embedding models perform when integrated with Senzing's entity resolution engine

---

## Table of Contents

1. [Overview](#overview)
2. [Why Test Through Senzing?](#why-test-through-senzing)
3. [Test Data Sources](#test-data-sources)
4. [Two Search Modes](#two-search-modes)
5. [Testing Workflow](#testing-workflow)
6. [Metrics Explained](#metrics-explained)
7. [Interpreting Results](#interpreting-results)
8. [Production Validation](#production-validation)
9. [Example Commands](#example-commands)

---

## Overview

This testing framework evaluates embedding models by:
1. Loading real-world entity data into Senzing with embeddings
2. Searching for test queries using ground truth triplets
3. Measuring how well the models find correct matches
4. Comparing embedding-only search vs combined GNR+embedding search

**Key Goal:** Determine if embedding models improve Senzing's entity resolution accuracy and how much they contribute beyond traditional name matching (GNR).

---

## Why Test Through Senzing?

### Traditional Testing vs Senzing Testing

**Traditional PostgreSQL+pgvector testing** (what we do in BizNames/PersonalNames projects):
- Direct vector similarity search in PostgreSQL
- Fast, isolated testing of embedding quality
- Measures pure semantic similarity performance
- Simpler infrastructure

**Senzing integration testing** (this project):
- Tests models in the actual production environment
- Combines embeddings with Senzing's GNR (Generic Name Resolution)
- Tests how embeddings interact with entity resolution logic
- Measures real-world performance with all Senzing features

### What Senzing Adds

Senzing provides:
- **Entity resolution**: Merging records that refer to the same entity
- **GNR (Generic Name Resolution)**: Sophisticated name matching algorithms
- **Feature scoring**: Weighted combination of multiple evidence types
- **Production-ready infrastructure**: Optimized for large-scale entity resolution

By testing through Senzing, we answer: *"Do embeddings actually improve entity resolution in production?"*

---

## Test Data Sources

We test on two datasets to measure different aspects of model performance:

### 1. OpenSanctions (Primary - Generalization Test)

**What:** Real-world sanctions and risk entities
**Size:** ~2M records, ~424K test triplets
**Location:** `/data/OpenSanctions/senzing.json`
**Test triplets:** `~/roncewind.git/BizNames/output/opensanctions_test_triplets.jsonl`

**Why this is critical:**
- Models are trained on Wikidata
- OpenSanctions is **out-of-distribution** (never seen during training)
- Tests **generalization** to real-world data
- More representative of production use cases

**Example entity:**
```json
{
  "DATA_SOURCE": "OPEN_SANCTIONS",
  "RECORD_ID": "NK-223yQP6hRaMuiALDCJ6xbY",
  "RECORD_TYPE": "ORGANIZATION",
  "NAMES": [
    {"NAME_TYPE": "PRIMARY", "NAME_ORG": "Limited Liability Company \"Zelinsky Group\""},
    {"NAME_TYPE": "ALIAS", "NAME_ORG": "Общество с ограниченной ответственностью \"Зелинский Групп\""},
    {"NAME_TYPE": "ALIAS", "NAME_ORG": "Tovarystvo z obmezhenoiu vidpovidalnistiu \"Zelinskyi Hrupp\""}
  ]
}
```

**Test triplet format:**
```json
{
  "anchor": "Limited Liability Company \"Zelinsky Group\"",
  "positive": "Общество с ограниченной ответственностью \"Зелинский Групп\"",
  "negative": "Баухаус Лтд",
  "anchor_group": "NK-223yQP6hRaMuiALDCJ6xbY"
}
```

The test checks: When we search for "anchor", does the entity containing "positive" rank higher than unrelated entities?

### 2. Wikidata (Secondary - In-Distribution Test)

**What:** Multilingual entity data from Wikidata
**Size:** ~625K test triplets
**Location:** Need to create from CSV files
**Source CSV:** `~/roncewind.git/BizNames/output/20251009_full/20250901_biznames_wikidata.csv`

**Why this matters:**
- Tests **in-distribution** performance
- Confirms model learned training data correctly
- Useful baseline to compare with OpenSanctions
- If Wikidata performance is good but OpenSanctions is poor → generalization problem

---

## Two Search Modes

We test embedding models by comparing **two search strategies** to measure their contribution:

### Mode 1: GNR-Only Search (Baseline)

**What's sent to Senzing:**
```json
{
  "NAME_ORG": "Limited Liability Company \"Zelinsky Group\""
}
```

**Tests:** Traditional Senzing name matching performance
**Uses:** GNR (Generic Name Resolution) only
**Answers:** "How well does Senzing's traditional name matching work?"

### Mode 2: GNR + Embedding Combined Search

**What's sent to Senzing:**
```json
{
  "NAME_ORG": "Limited Liability Company \"Zelinsky Group\"",
  "NAME_LABEL": "Limited Liability Company \"Zelinsky Group\"",
  "NAME_EMBEDDING": "[0.123, -0.456, 0.789, ...]"
}
```

**Tests:** Combined performance with embeddings added
**Uses:** Both GNR name matching AND embedding semantic similarity
**Answers:** "Do embeddings improve results over GNR alone?"

**Important:** Senzing automatically combines all available features. We cannot isolate embeddings completely - we can only compare GNR alone vs GNR+embeddings together.

### Key Questions We Answer

1. **GNR-only accuracy** - How accurate is traditional name matching?
2. **GNR+Embedding accuracy** - Does adding embeddings improve accuracy?
3. **Delta (difference)** - How much do embeddings help? (positive = they help, negative = they hurt)
4. **Rescue rates** - How complementary are the methods?
   - GNR rescue rate: % where GNR+embedding found correct match but GNR-only missed
   - Embedding rescue rate: % where GNR-only found correct match but GNR+embedding missed (rare - would indicate embeddings hurting performance)

---

## Testing Workflow

### Phase 1: Environment Setup

```bash
# 1. Source Senzing environment
source ~/senzingv4/setupEnv

# 2. Activate Python virtual environment
source venv/bin/activate

# 3. Set Senzing configuration (if not already set)
source DO_NOT_PUSH_sz_env_vars.sh
```

### Phase 2: Purge and Load Data

**IMPORTANT:** Senzing database must be purged before loading new test data.

```bash
# Purge Senzing database (user does this manually)
# ... database purge commands ...

# Load OpenSanctions data with embeddings
python sz_load_embeddings.py \
  -i /data/OpenSanctions/senzing.json \
  --name_model_path ~/roncewind.git/PersonalNames/output/<MODEL_NAME> \
  --biz_model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  2> load.log
```

**What happens during loading:**
1. Reads each record from senzing.json
2. For each name in the NAMES array:
   - Generates 512-dim embedding using appropriate model (personal or business)
   - Normalizes embedding (unit length for cosine similarity)
   - Converts to float16 for efficiency
3. Adds NAME_EMBEDDINGS or BIZNAME_EMBEDDINGS array to record
4. Inserts record into Senzing
5. Senzing stores embeddings in PostgreSQL NAME_EMBEDDING/BIZNAME_EMBEDDING tables

**Expected duration:** ~2-4 hours for 2M records (depends on GPU)

### Phase 3: Quick Validation Test

Before running full evaluation, test on a small sample to catch errors early:

```bash
python sz_evaluate_model.py \
  --type business \
  --model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --triplets ~/roncewind.git/BizNames/output/opensanctions_test_triplets.jsonl \
  --test_set opensanctions \
  --sample 100 \
  --output results/quick_test.json
```

**Checks:**
- Script runs without errors
- Searches complete successfully
- Results look reasonable (accuracy > 50%)
- Latencies are acceptable (< 1 second per query)

### Phase 4: Full Evaluation

Run complete evaluation on all test triplets:

```bash
# Create results directory
mkdir -p results

# Evaluate business names model
python sz_evaluate_model.py \
  --type business \
  --model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --triplets ~/roncewind.git/BizNames/output/opensanctions_test_triplets.jsonl \
  --test_set opensanctions \
  --output results/opensanctions_business_phase9b.json
```

**What happens during evaluation:**

For each test triplet:
1. **Look up ground truth:** Find entity ID for anchor_group in Senzing
2. **Search (embedding-only):**
   - Generate embedding for anchor name
   - Search Senzing with embedding only
   - Record which entities returned and their ranks
   - Check if correct entity (positive) was found
3. **Search (GNR+embedding):**
   - Same embedding, but also include NAME_ORG/NAME_FULL attribute
   - Search Senzing with both GNR and embedding
   - Record results
4. **Record metrics:**
   - Did positive entity rank higher than negatives?
   - At what rank was positive found? (for Recall@K)
   - Query latency
   - Feature scores (GNR score, embedding score)

**Expected duration:** ~1-3 hours for 424K triplets (depends on query performance)

### Phase 5: Compare Results

```bash
# Compare all evaluation results
python sz_compare_models.py results/*.json

# Or save to file
python sz_compare_models.py -o comparison_report.txt results/*.json
```

**Output:** Side-by-side comparison of all models tested, grouped by test set and model type.

---

## Metrics Explained

### ⭐ PRIMARY METRIC: Relative Ranking (Positive > Negative Rate)

**NEW in December 2025:** The primary metric for evaluation is now **Positive > Negative Rate**.

| Metric | Description | Good Value | Why It's Better |
|--------|-------------|------------|-----------------|
| **Positive > Negative Rate** | % of queries where correct entity ranks above incorrect entity | >95% | Measures true ranking quality, not just top-1 |

**Why this changed:**
- **Old approach (top-1 accuracy):** Binary pass/fail - did correct entity rank #1?
- **New approach (relative ranking):** Did correct entity rank *higher* than incorrect entities?
- **Key insight:** Ranking 5 correct entities and 3 incorrect ones is valuable even if correct isn't #1!

**Example:**
```
Query results: [Entity A, Entity B, Entity C (correct), Entity D, Entity E (incorrect)]

Old metric (top-1): ❌ FAIL (correct not #1)
New metric (Pos>Neg): ✅ PASS (correct ranks 3rd, incorrect ranks 5th)
```

### Scenario Breakdown

The system classifies each query result into one of these scenarios:

| Scenario | Description | Quality |
|----------|-------------|---------|
| **Both found, correct order** | Positive ranks above negative | ✅ BEST |
| **Both found, wrong order** | Negative ranks above positive | ❌ WORST |
| **Only positive found** | Only correct entity returned | ✅ GOOD |
| **Only negative found** | Only incorrect entity returned | ❌ BAD |
| **Neither found** | Neither entity returned | ○ NEUTRAL |

**Good model:** >85% "both found, correct order" + high "only positive found"

### NDCG@K (Ranking Quality)

**NEW:** Normalized Discounted Cumulative Gain measures overall ranking quality.

| Metric | Description | Range | Good Value |
|--------|-------------|-------|------------|
| **NDCG@5** | Ranking quality in top 5 results | 0.0-1.0 | >0.85 |
| **NDCG@10** | Ranking quality in top 10 results | 0.0-1.0 | >0.90 |
| **NDCG@20** | Ranking quality in top 20 results | 0.0-1.0 | >0.92 |
| **NDCG@100** | Ranking quality in top 100 results | 0.0-1.0 | >0.95 |

**Interpretation:**
- **1.0** = Perfect ranking (correct entity at position #1)
- **0.8-1.0** = Excellent (correct entity in top few positions)
- **0.5-0.8** = Good (correct entity found but not optimally ranked)
- **<0.5** = Poor (correct entity ranked very low or not found)

**Why NDCG matters:** Top-1 accuracy can be harsh. NDCG rewards having correct entities ranked high even if not #1.

### Threshold Analysis & Recommendations

**NEW:** Evaluation now runs **threshold-independent** using `SZ_SEARCH_INCLUDE_ALL_CANDIDATES`, then recommends optimal threshold settings.

The system analyzes score distributions and recommends all 5 Senzing threshold levels:

| Threshold Level | Purpose | Typical Range |
|-----------------|---------|---------------|
| **sameScore** | Very high confidence matches | 85-95 |
| **closeScore** | **CUTOFF** - minimum to return | 60-80 |
| **likelyScore** | Moderate confidence | 40-60 |
| **plausibleScore** | Lower confidence | 25-45 |
| **unlikelyScore** | Floor threshold | 15-25 |

**Example recommendation:**
```
Option A (Balanced - 95% pos recall):
  sameScore:        85
  closeScore:       65  (cutoff)
  likelyScore:      50
  plausibleScore:   35
  unlikelyScore:    20
  Pos>Neg rate:     98.2%
```

**How to use:** Update your `szConfig.json` CFG_CFRTN section with recommended thresholds for your feature type.

### Variant Type Breakdown

**NEW:** Tests query variants automatically to measure robustness:

**Alias variants** (from loaded data):
- `alias_Latin` - Latin script variants
- `alias_Cyrillic` - Cyrillic script variants
- `alias_Georgian` - Georgian script (ქართული)
- `alias_Arabic` - Arabic script (العربية)
- `alias_Chinese` - Chinese script (中文)

**Synthetic variants** (generated for testing):
- `synthetic_abbreviated` - "J Smith", "J. Smith", "John W. Smith"
- `synthetic_fuzzy` - Punctuation/spacing changes
- `synthetic_partial` - "John Smith" (first+last only)

**Why test variants:**
- Real users search with abbreviations, typos, different scripts
- Tests model robustness beyond exact name matching
- Identifies which query types work well vs poorly

**Good model:** Performs well across all variant types, especially cross-lingual aliases.

### Legacy Metrics (Backward Compatibility)

These metrics are still computed for comparison with previous evaluations:

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Accuracy (top-1)** | % where correct entity ranks #1 | >90% |
| **Recall@K** | % where correct in top K (K=1,5,10,100) | R@10 >95% |
| **MRR** | Mean Reciprocal Rank | >0.85 |

### Rescue Rates

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Rescue Rate** | % where embeddings found correct when name-only failed | 5-20% |
| **Both Correct** | % where both methods found correct | >80% |
| **Both Wrong** | % where both methods failed | <10% |

**Interpretation:**
- High rescue rate → Embeddings provide valuable additional coverage
- Both methods correct → Complementary approaches working well

### Latency Metrics

| Metric | Description | Acceptable |
|--------|-------------|------------|
| **Average** | Mean query time | < 500ms |
| **P50** | Median (50th percentile) | < 300ms |
| **P95** | 95th percentile | < 1000ms |
| **P99** | 99th percentile | < 2000ms |

**Note:** Threshold-independent evaluation (`SZ_SEARCH_INCLUDE_ALL_CANDIDATES`) returns more results, so latencies may be higher than production with configured thresholds.

---

## Interpreting Results

### Scenario 1: Embeddings Help Significantly

```
GNR-only Accuracy: 85.3%
GNR+EMB Accuracy: 91.7%
Delta: +6.4%
Improvement Rate: 8.2% (embeddings found correct match that GNR missed)
Degradation Rate: 1.8% (embeddings caused GNR to miss a match)
```

**Interpretation:**
- Embeddings improve accuracy by 6.4%
- Embeddings help in 8.2% of cases (finding matches GNR alone missed)
- Embeddings hurt in only 1.8% of cases
- **Conclusion:** Embeddings provide meaningful improvement, strong positive contribution

### Scenario 2: Embeddings Don't Help Much

```
GNR-only Accuracy: 88.5%
GNR+EMB Accuracy: 89.1%
Delta: +0.6%
Improvement Rate: 3.4%
Degradation Rate: 2.8%
```

**Interpretation:**
- Only 0.6% improvement from embeddings
- Small improvement and degradation rates that nearly cancel out
- **Conclusion:** GNR already very good, embeddings add marginal value

### Scenario 3: Embeddings Hurt Performance

```
GNR-only Accuracy: 82.3%
GNR+EMB Accuracy: 79.8%
Delta: -2.5%
Improvement Rate: 1.2%
Degradation Rate: 3.7%
```

**Interpretation:**
- Adding embeddings reduces accuracy by 2.5%
- Degradation rate exceeds improvement rate
- **Conclusion:** Model may be undertrained, misconfigured, or poorly calibrated for this data

### What to Look For

**Good signs:**
- Positive delta (GNR+EMB > EMB alone)
- High Recall@10 (>90%)
- Reasonable latencies (P95 < 1s)
- Balanced rescue rates (both methods contribute)

**Warning signs:**
- Negative delta (embeddings hurt)
- Low Recall@10 (<85%)
- High latencies (P95 > 2s)
- Very high GNR rescue rate, very low EMB rescue rate → embeddings not working

---

## Production Validation

Production validation differs from model evaluation (triplet-based testing). While model evaluation tests **relative ranking** (does positive rank above negative?), production validation tests **absolute recall** (can we find loaded records?).

### Why Production Validation Matters

**Model Evaluation** (`sz_evaluate_model.py`):
- Tests model training quality
- Uses ground truth triplets (anchor, positive, negative)
- Measures relative ranking performance
- Answers: "Does the model rank correct matches higher than incorrect matches?"

**Production Validation** (`sz_validate_production.py`):
- Tests production readiness
- Uses loaded records with their actual names/aliases
- Measures absolute recall (found or not found)
- Answers: "Can we find loaded records by searching for their names?"

### Production Validation Workflow

```bash
# 1. Extract validation samples from loaded data
python sz_extract_validation_samples.py \
  --input data/test_samples/opensanctions_test_500.jsonl \
  --output data/test_samples/validation_samples_100.jsonl \
  --sample_size 100 \
  --filter both \
  --seed 42

# 2. Run validation (Senzing + optional PostgreSQL)
python sz_validate_production.py \
  --input data/test_samples/validation_samples_100.jsonl \
  --name_model_path ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model \
  --biz_model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --truncate_dim 512 \
  --validate_pg \
  --pg_database senzing \
  --output results/validation_results.json

# 3. Compare multiple validation runs
python sz_compare_validations.py results/validation_*.json
```

### Production Validation Metrics

**Primary Metrics:**
- **Recall@10**: % where expected entity found in top-10 (TARGET: >85%)
- **Recall@100**: % where expected entity found in top-100 (TARGET: >95%)
- **Not Found Rate**: % where expected entity not in results (TARGET: <5%)
- **MRR**: Mean Reciprocal Rank (higher is better)

**Threshold Analysis:**
- Tests multiple `closeScore` thresholds (50, 55, 60, 65, 70, 75, 80)
- Computes precision, recall, F1 at each threshold
- Recommends 3 complete CFG_CFRTN threshold sets:
  - **Balanced**: Maximizes F1 score (best precision/recall balance)
  - **Conservative**: Maximizes precision (≥90% target, minimizes false positives)
  - **Aggressive**: Maximizes recall (≥90% target, catches more true positives)

**Breakdowns:**
- By Record Type: PERSON vs ORGANIZATION
- By Script: Latin, Cyrillic, Georgian, Arabic, Chinese, Mixed
- By Name Type: primary vs alias

**PostgreSQL Validation (optional `--validate_pg`):**
- **Sibling Recall**: % of aliases from same record found in top-K
- **Intra-Record Distance**: Cosine distance between aliases from same record
- Tests embedding quality directly via pgvector

### When to Use Each Testing Method

**Use Model Evaluation** (`sz_evaluate_model.py`) when:
- Testing a new model before deployment
- Comparing models during training
- Measuring relative ranking quality
- Testing on synthetic variants (fuzzy matching, abbreviations)

**Use Production Validation** (`sz_validate_production.py`) when:
- Validating after loading data
- Tuning Senzing thresholds
- Smoke testing production readiness
- Measuring absolute recall performance
- Debugging "why can't we find this record?"

---

## Example Commands

### Full Testing Sequence

```bash
# ============================================================================
# SETUP
# ============================================================================
source ~/senzingv4/setupEnv
source venv/bin/activate
mkdir -p results

# ============================================================================
# LOAD DATA (do once per model test)
# ============================================================================
# Purge Senzing database first!
python sz_load_embeddings.py \
  -i /data/OpenSanctions/senzing.json \
  --name_model_path ~/roncewind.git/PersonalNames/output/20250812/FINAL-fine_tuned_model \
  --biz_model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  2> load_opensanctions.log

# Monitor progress in another terminal:
tail -f load_opensanctions.log

# ============================================================================
# QUICK TEST (100 triplets, exact queries only)
# ============================================================================
python sz_evaluate_model.py \
  --type business \
  --model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --triplets ~/roncewind.git/BizNames/output/opensanctions_test_triplets.jsonl \
  --test_set opensanctions \
  --sample 100 \
  --variants none \
  --output results/quick_test_biz.json

# ============================================================================
# VARIANT TESTING (NEW - tests query variants for robustness)
# ============================================================================
# Test with BOTH aliases (from data) and synthetics (generated)
python sz_evaluate_model.py \
  --type personal \
  --model_path ~/roncewind.git/PersonalNames/output/20250812/FINAL-fine_tuned_model \
  --triplets data/test_samples/opensanctions_test_5k_triplets_personal.jsonl \
  --test_set opensanctions \
  --sample 100 \
  --data_file data/test_samples/opensanctions_test_5k_final.jsonl \
  --variants both \
  --synthetic-types abbreviated,fuzzy,partial \
  --output results/variant_test.json

# Test ONLY real aliases (cross-lingual matching)
python sz_evaluate_model.py \
  --type personal \
  --model_path ~/roncewind.git/PersonalNames/output/20250812/FINAL-fine_tuned_model \
  --triplets data/test_samples/opensanctions_test_5k_triplets_personal.jsonl \
  --test_set opensanctions \
  --sample 100 \
  --data_file data/test_samples/opensanctions_test_5k_final.jsonl \
  --variants aliases \
  --output results/alias_test.json

# Test ONLY synthetic variants (fuzzy matching of unknown queries)
python sz_evaluate_model.py \
  --type business \
  --model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --triplets data/test_samples/opensanctions_test_5k_triplets_business.jsonl \
  --test_set opensanctions \
  --sample 100 \
  --variants synthetics \
  --synthetic-types abbreviated \
  --output results/synthetic_test.json

# ============================================================================
# FULL EVALUATION
# ============================================================================
# Business names
python sz_evaluate_model.py \
  --type business \
  --model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --triplets ~/roncewind.git/BizNames/output/opensanctions_test_triplets.jsonl \
  --test_set opensanctions \
  --data_source OPEN_SANCTIONS \
  --output results/opensanctions_business_phase9b.json

# If you have personal names in OpenSanctions too:
python sz_evaluate_model.py \
  --type personal \
  --model_path ~/roncewind.git/PersonalNames/output/20250812/FINAL-fine_tuned_model \
  --triplets ~/roncewind.git/PersonalNames/output/opensanctions_test_triplets.jsonl \
  --test_set opensanctions \
  --data_source OPEN_SANCTIONS \
  --output results/opensanctions_personal_distiluse.json

# ============================================================================
# COMPARE RESULTS
# ============================================================================
# Quick summary to stdout
python sz_compare_models.py results/*.json

# Full report to file
python sz_compare_models.py -o comparison_report.txt results/*.json

# View report
cat comparison_report.txt
```

### Testing Multiple Models

To test different models, you must reload data with each model's embeddings:

```bash
# Model A
purge_senzing_db
python sz_load_embeddings.py -i ... --biz_model_path MODEL_A ...
python sz_evaluate_model.py ... --output results/model_a.json

# Model B
purge_senzing_db
python sz_load_embeddings.py -i ... --biz_model_path MODEL_B ...
python sz_evaluate_model.py ... --output results/model_b.json

# Compare
python sz_compare_models.py results/model_a.json results/model_b.json
```

---

## Summary

This testing framework answers the critical question: **"Do embedding models improve Senzing's entity resolution in production?"**

By testing on OpenSanctions (out-of-distribution data), we measure real-world generalization. By comparing embedding-only vs GNR+embedding searches, we isolate the contribution of embeddings.

The metrics tell us:
- How accurate the models are (Accuracy, Recall@K)
- How well they rank results (MRR)
- How fast they perform (latency percentiles)
- Whether they complement GNR (rescue rates)

**Next steps:** Once the new PersonalNames model finishes training, run the full evaluation workflow and compare results with the existing baseline.

---

## AAR Experiment (Alias-Augmented Retrieval)

**Date:** 2026-01-30
**Status:** Complete - Not Recommended

### Overview

AAR (Alias-Augmented Retrieval) was evaluated as a potential improvement to business name retrieval. The approach generates stripped aliases (legal forms removed like LLC, Inc, GmbH, etc.) at index time to improve recall.

### Results Summary

| Metric | Baseline | AAR | Delta | Verdict |
|--------|----------|-----|-------|---------|
| **Recall@0.85** | 30.5% | 30.5% | +0.0% | Neutral |
| **Avg Candidates** | 0.9 | 1.1 | +0.2 | Worse |
| **Index Size** | 19K rows | 30K rows | +58% | Worse |

**Conclusion:** AAR provides **no recall benefit** while increasing index size by 58% and candidate volume by 22%. **Not recommended for production.**

### Key Findings

1. **Legal form stripping doesn't help**: The E5 model already handles legal form variations adequately
2. **Hard cases remain hard**: CJK names (7.9% recall) and location-prefixed names (3.6% recall) need different solutions
3. **30% ceiling**: Overall recall at 0.85 threshold suggests the threshold may be too aggressive or the model needs improvement

### Files

- **Results**: `results/AAR_EXPERIMENT_RESULTS.md` (full analysis)
- **Script**: `sz_retrieval_experiment.py`
- **Runner**: `run_retrieval_experiment.sh`
- **JSON Output**: `results/retrieval_experiment.json`

### Methodology

The experiment used a "black-box scorer framework" with:
- 10,000 business entities from OpenSanctions
- 1,000 stratified queries (weighted toward hard cases)
- 3-seed confidence check (seeds: 42, 123, 456)
- Leave-one-out evaluation (query excluded from index)
- Threshold sweep: 0.80, 0.83, 0.85, 0.88, 0.90
- Per-slice analysis: Script type, name length, location prefix, multi-alias

See `results/AAR_EXPERIMENT_RESULTS.md` for complete methodology and detailed breakdown.
