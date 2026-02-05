# GNR Score Alignment Training Pipeline

This directory contains data and results for training embedding models to correlate with Senzing's GNR (Generic Name Resolver) scores.

## Goal

Train the embedding model so that cosine similarity scores correlate with GNR pairwise scores:
- **Current state**: Embedding similarity ≠ GNR score (poor correlation)
- **Desired state**: High cosine ≈ high GNR, low cosine ≈ low GNR
- **Benefit**: Could potentially skip expensive GNR scoring if embeddings are good enough

## Pipeline Overview

```
Wikidata CSV → Sample Entities → Senzing Format → Load to Senzing
                    ↓
              Generate Pairs → Score with GNR → Analyze Correlation
                    ↓
              Mine Triplets → Fine-tune Model → Validate Improvement
```

## Quick Start

```bash
# Activate environment
source ~/senzingv4/setupEnv
source venv/bin/activate
source secrets/DO_NOT_PUSH_sz_env_vars.sh

# Step 1: Sample entities from Wikidata
python sz_sample_wikidata_entities.py \
  --input ~/roncewind.git/BizNames/data/20250901_biznames_wikidata.csv \
  --output data/gnr_alignment/wikidata_entities_20k.jsonl \
  --sample 20000 \
  --min_aliases 3 \
  --prefer_cross_script \
  --seed 42

# Step 2: Convert to Senzing format
python sz_prepare_wikidata_for_senzing.py \
  --input data/gnr_alignment/wikidata_entities_20k.jsonl \
  --output data/gnr_alignment/wikidata_senzing_20k.jsonl

# Step 3: Load into Senzing (purge first)
python sz_load_embeddings_onnx.py \
  -i data/gnr_alignment/wikidata_senzing_20k.jsonl \
  --name_model_path ~/999gz.git/name_model/personalnames_model/onnx_int8 \
  --biz_model_path ~/999gz.git/name_model/biznames_model/onnx_int8 \
  --threads 24

# Step 4: Generate name pairs
python sz_generate_gnr_pairs.py \
  --entities data/gnr_alignment/wikidata_entities_20k.jsonl \
  --output data/gnr_alignment/name_pairs_for_gnr.jsonl \
  --within_entity_max_pairs 6 \
  --cross_entity_candidates 5 \
  --model_path ~/999gz.git/name_model/biznames_model/onnx_int8 \
  --threshold 0.43 \
  --prioritize_cross_script

# Step 5: Score pairs with GNR
python sz_score_pairs_gnr.py \
  --pairs data/gnr_alignment/name_pairs_for_gnr.jsonl \
  --model_path ~/999gz.git/name_model/biznames_model/onnx_int8 \
  --output data/gnr_alignment/pairs_with_gnr_scores.jsonl \
  --threads 8

# Step 6: Analyze correlation (baseline)
python sz_analyze_gnr_correlation.py \
  --input data/gnr_alignment/pairs_with_gnr_scores.jsonl \
  --output results/gnr_correlation_baseline.json

# Step 7: Mine triplets for training
python sz_mine_gnr_triplets.py \
  --input data/gnr_alignment/pairs_with_gnr_scores.jsonl \
  --output data/gnr_alignment/triplets_gnr_aligned.jsonl \
  --positive_gnr_min 85 \
  --negative_gnr_max 50 \
  --negative_cosine_min 0.50 \
  --max_triplets_per_anchor 5

# Step 8: Fine-tune model (in BizNames repo)
cd ~/roncewind.git/BizNames
python train_model.py \
  --base_model intfloat/e5-small-v2 \
  --triplets ~/roncewind.git/SzEmbeddings/data/gnr_alignment/triplets_gnr_aligned.jsonl \
  --output output/gnr_aligned_e5_small \
  --epochs 3 \
  --learning_rate 2e-5

# Step 9: Validate improvement
cd ~/roncewind.git/SzEmbeddings
python sz_score_pairs_gnr.py \
  --pairs data/gnr_alignment/name_pairs_for_gnr.jsonl \
  --model_path ~/roncewind.git/BizNames/output/gnr_aligned_e5_small/FINAL-onnx-fp32 \
  --output data/gnr_alignment/pairs_with_gnr_scores_v2.jsonl \
  --skip_gnr

python sz_analyze_gnr_correlation.py \
  --input data/gnr_alignment/pairs_with_gnr_scores_v2.jsonl \
  --baseline results/gnr_correlation_baseline.json \
  --output results/gnr_correlation_after_training.json
```

## Scripts

| Script | Purpose |
|--------|---------|
| `sz_sample_wikidata_entities.py` | Sample entities with good alias diversity from CSV |
| `sz_prepare_wikidata_for_senzing.py` | Convert entity JSONL to Senzing format |
| `sz_generate_gnr_pairs.py` | Generate within-entity and cross-entity pairs |
| `sz_score_pairs_gnr.py` | Score pairs with why_search + compute cosine |
| `sz_analyze_gnr_correlation.py` | Analyze cosine vs GNR correlation |
| `sz_mine_gnr_triplets.py` | Mine triplets from scored pairs |

## Data Files

| File | Description |
|------|-------------|
| `wikidata_entities_20k.jsonl` | Sampled entities with aliases |
| `wikidata_senzing_20k.jsonl` | Senzing format for loading |
| `name_pairs_for_gnr.jsonl` | Pairs to score |
| `pairs_with_gnr_scores.jsonl` | Pairs with GNR + cosine |
| `triplets_gnr_aligned.jsonl` | Final training triplets |

## File Formats

### Entity Format (wikidata_entities_20k.jsonl)
```json
{
  "entity_id": "Q4995",
  "canonical": "sabmiller",
  "aliases": [
    {"name": "SABMiller", "language": "mk"},
    {"name": "SABミラー", "language": "ja"},
    {"name": "南非米勒", "language": "zh"}
  ]
}
```

### Scored Pair Format (pairs_with_gnr_scores.jsonl)
```json
{
  "name_a": "SABMiller",
  "name_b": "SABミラー",
  "entity_a": "Q4995",
  "entity_b": "Q4995",
  "same_entity": true,
  "gnr_score": 92,
  "gnr_bucket": "CLOSE",
  "cosine_sim": 0.65,
  "pair_type": "within_entity_cross_script"
}
```

### Triplet Format (triplets_gnr_aligned.jsonl)
```json
{
  "anchor": "SABMiller",
  "positive": "SABミラー",
  "negative": "SAB Industries"
}
```

## Key Metrics

### Baseline Targets
- **Spearman correlation** (cosine vs GNR): Current model baseline
- **ROC-AUC**: Can cosine separate same-entity from different-entity?
- **False positive rate**: % of cross-entity pairs with cosine > 0.70

### Training Targets
- Spearman: 0.5 → 0.75+
- AUC: 0.85 → 0.95+
- Cross-script recall: Significant improvement for CJK/Cyrillic

### Ultimate Success Criteria
If Spearman > 0.85:
1. High-cosine matches (>0.90) could skip GNR entirely
2. Medium-cosine matches (0.70-0.90) use GNR for confirmation
3. Low-cosine matches (<0.70) rejected without GNR

## Expected Data Volumes

| Stage | Records |
|-------|---------|
| Source Wikidata rows | 1,060,369 |
| Source unique entities | 364,554 |
| Sampled entities (with 3+ aliases) | 20,000 |
| Total aliases in sample | ~80,000 |
| Within-entity pairs | ~60,000 |
| Cross-entity candidate pairs | ~100,000 |
| Total pairs to score | ~160,000 |
| Final triplets | ~40,000-60,000 |

## Performance Notes

- GNR scoring via `why_search`: ~50ms per call
- 160,000 pairs ≈ 2-3 hours with 8 threads
- Embedding computation: Fast with ONNX INT8 on CPU

## Cross-Script Focus

The Wikidata data is rich in cross-script examples. These are prioritized because they're where current models struggle most:

| Script Pair | Examples |
|------------|----------|
| Latin ↔ CJK | SABMiller ↔ 南非米勒 |
| Latin ↔ Cyrillic | Nintendo ↔ Нинтендо |
| Latin ↔ Arabic | Nintendo ↔ نينتندو |
| Latin ↔ Korean | Nintendo ↔ 닌텐도 |

## Troubleshooting

### No records found in Senzing
The Wikidata entities must be loaded before scoring. Run:
```bash
python sz_load_embeddings_onnx.py -i data/gnr_alignment/wikidata_senzing_20k.jsonl ...
```

### Low triplet count
Adjust thresholds:
- Lower `--negative_cosine_min` (include easier negatives)
- Raise `--negative_gnr_max` (allow higher GNR negatives)
- Lower `--positive_gnr_min` (include weaker positives)

### GNR scoring is slow
- Increase `--threads` (default 8)
- Use `--skip_gnr` to only compute cosine (for validation after training)
