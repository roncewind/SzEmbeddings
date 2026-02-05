# File Inventory

Complete inventory of scripts and files in the SzEmbeddings repository.

---

## Core Scripts

### Data Loading

| Script | Purpose |
|--------|---------|
| `sz_load_embeddings.py` | Load records with PyTorch models into Senzing with embeddings |
| `sz_load_embeddings_onnx.py` | Load records with ONNX models (FP16/INT8) |

### Model Evaluation & Validation

| Script | Purpose |
|--------|---------|
| `sz_evaluate_model.py` | Evaluate model accuracy on test triplets (relative ranking) |
| `sz_validate_production.py` | Validate recall for loaded records (Senzing + PostgreSQL) |
| `sz_validate_with_variants.py` | Validate with exact queries and synthetic variants |
| `sz_validate_with_gnr_comparison.py` | Validate with GNR correlation analysis |
| `sz_compare_models.py` | Compare evaluation results across models |
| `sz_compare_validations.py` | Compare multiple validation runs |

### Interactive Search & Debugging

| Script | Purpose |
|--------|---------|
| `sz_debug_search.py` | Comprehensive interactive search with detailed analysis |
| `sz_cross_validate.py` | Compare Senzing vs PostgreSQL results |

### ONNX Support

| Script | Purpose |
|--------|---------|
| `onnx_sentence_transformer.py` | ONNX model wrapper with SentenceTransformer-compatible API |
| `sz_convert_to_onnx_generic.py` | Convert PyTorch models to ONNX FP32 |
| `sz_convert_to_int8.py` | Quantize ONNX models to INT8 |
| `sz_convert_to_ort_fp16.py` | Convert to ORT-optimized FP16 |

---

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `sz_sample_data.py` | Sample stratified subsets from JSONL files |
| `sz_extract_validation_samples.py` | Extract validation samples from loaded data |
| `sz_extract_aliases.py` | Extract aliases from Senzing JSONL data |
| `sz_generate_variants.py` | Generate synthetic fuzzy variants for testing |
| `shuffle_jsonl.py` | Shuffle JSONL files randomly |
| `sz_utils.py` | Shared utilities (embedding generation, config, formatting) |
| `sz_simple_redoer.py` | Handle Senzing redo queue |
| `sz_retry_timeouts.py` | Retry failed load operations |
| `script_classifier.py` | Detect Unicode script of names (Latin, Cyrillic, CJK, etc.) |

---

## GNR Score Alignment Training (`gnr_training/`)

Scripts for training embedding models to correlate with Senzing's GNR scores.

| Script | Purpose |
|--------|---------|
| `sz_sample_wikidata_entities.py` | Sample entities with cross-script alias diversity from Wikidata |
| `sz_prepare_wikidata_for_senzing.py` | Convert sampled entities to Senzing JSONL format |
| `sz_generate_gnr_pairs.py` | Generate within-entity and cross-entity name pairs |
| `sz_score_pairs_gnr.py` | Score pairs with GNR (via why_search) and compute cosine |
| `sz_analyze_gnr_correlation.py` | Analyze correlation between cosine and GNR scores |
| `sz_mine_gnr_triplets.py` | Mine hard triplets from scored pairs for training |
| `sz_mine_hard_positives.py` | Mine cross-script hard positives for training |

See `gnr_training/README.md` for the full pipeline documentation.

---

## Setup & Configuration

| File | Purpose |
|------|---------|
| `test_setup.py` | Verify environment setup |
| `check_db_status.py` | Check database status and counts |
| `szConfig.json` | Senzing configuration with embedding features |
| `CLAUDE.md` | Claude Code guidance for this repository |

---

## Shell Scripts

| Script | Purpose |
|--------|---------|
| `run_validation_pipeline.sh` | End-to-end validation pipeline |
| `package_models.sh` | Package models for deployment to name_model repo |
| `check_redoer_status.sh` | Monitor Senzing redo queue status |

---

## Directory Structure

```
SzEmbeddings/
├── docs/                          # All documentation
│   ├── README.md                  # Setup guide, PostgreSQL/Senzing config
│   ├── TESTING.md                 # Testing workflow and metrics
│   ├── LOADING.md                 # Loading instructions
│   ├── ONNX_MODELS.md             # ONNX model evaluation results
│   ├── POSTGRESQL_TUNING.md       # PostgreSQL tuning
│   ├── FILE_INVENTORY.md          # This file
│   ├── archive/                   # Historical documentation
│   │   ├── VALIDATION_UPDATES_SUMMARY.md
│   │   └── VARIANT_TESTING.md
│   └── results/                   # Results documentation
│       ├── GNR_ALIGNMENT_RESULTS.md
│       ├── E5_VALIDATION_RESULTS.md
│       └── AAR_EXPERIMENT_RESULTS.md
├── gnr_training/                  # GNR score alignment training pipeline
│   ├── README.md                  # Pipeline documentation
│   ├── sz_sample_wikidata_entities.py
│   ├── sz_prepare_wikidata_for_senzing.py
│   ├── sz_generate_gnr_pairs.py
│   ├── sz_score_pairs_gnr.py
│   ├── sz_analyze_gnr_correlation.py
│   ├── sz_mine_gnr_triplets.py
│   └── sz_mine_hard_positives.py
├── archive/                       # Archived scripts
│   ├── experiments/               # AAR and other experiments
│   │   ├── sz_aar_prototype.py
│   │   ├── sz_load_embeddings_aar.py
│   │   ├── sz_retrieval_experiment.py
│   │   ├── sz_threshold_analysis.py
│   │   ├── sz_compare_models_retrieval.py
│   │   ├── benchmark_embedding_time.py
│   │   ├── sz_test_embedding_value.py
│   │   └── *.sh (experiment runners)
│   └── docs/                      # Archived documentation
│       ├── E5_MIGRATION_PLAN.md
│       ├── GNR_VS_EMBEDDING_ANALYSIS.md
│       ├── COMPARISON_DOCUMENT.md
│       └── GNR_ALIGNMENT_PIPELINE_README.md
├── data/                          # Test data and samples
│   ├── test_samples/              # Test JSONL files
│   └── gnr_alignment/             # GNR training data
├── results/                       # Validation results (JSON)
│   ├── e5_validation/
│   └── baseline_onnx_comparison/
├── secrets/                       # Environment variables (not tracked)
└── venv/                          # Python virtual environment
```

---

## Data Directories

### data/test_samples/

Test data for validation:
- `opensanctions_test_500.jsonl` - Small test set (500 records)
- `opensanctions_test_5k_final.jsonl` - Medium test set (5K records)
- `sample_20k_shuffled.jsonl` - Large test set (20K records)
- `validation_*.jsonl` - Extracted validation samples
- `variants_*.jsonl` - Generated variant test cases

### data/gnr_alignment/

GNR alignment training data (pipeline complete):
- `wikidata_entities_20k.jsonl` - Sampled Wikidata entities
- `wikidata_senzing_20k.jsonl` - Senzing format
- `name_pairs_for_gnr.jsonl` - Name pairs for scoring
- `pairs_with_gnr_scores.jsonl` - Scored pairs
- `triplets_gnr_aligned.jsonl` - Training triplets

### results/

Validation and evaluation results:
- `*.json` - Validation results
- `e5_validation/` - E5 model validation results
- `baseline_onnx_comparison/` - ONNX format comparison results

---

## Script Dependencies

### Loading Data

```
sz_load_embeddings.py
├── sz_utils.py (embedding generation)
├── sentence_transformers (PyTorch models)
└── senzing (Senzing SDK)

sz_load_embeddings_onnx.py
├── onnx_sentence_transformer.py (ONNX wrapper)
└── senzing (Senzing SDK)
```

### Validation

```
sz_validate_with_variants.py
├── sz_utils.py
├── sz_generate_variants.py (variant generation)
└── onnx_sentence_transformer.py (if --onnx)

sz_validate_production.py
├── sz_utils.py
└── onnx_sentence_transformer.py (if --onnx)
```

### Evaluation

```
sz_evaluate_model.py
├── sz_utils.py
└── sentence_transformers

sz_compare_models.py (standalone)
sz_compare_validations.py (standalone)
```

---

## Common Workflows

### 1. Load and Validate (Quick Test)

```bash
# Load 500 records
python sz_load_embeddings_onnx.py \
  -i data/test_samples/opensanctions_test_500.jsonl \
  --name_model_path ~/999gz.git/name_model/personalnames_model/onnx_int8 \
  --biz_model_path ~/999gz.git/name_model/biznames_model/onnx_int8

# Extract validation samples
python sz_extract_validation_samples.py \
  --input data/test_samples/opensanctions_test_500.jsonl \
  --output data/test_samples/validation_500.jsonl

# Run validation
python sz_validate_production.py \
  --input data/test_samples/validation_500.jsonl \
  --name_model_path ~/999gz.git/name_model/personalnames_model/onnx_int8 \
  --biz_model_path ~/999gz.git/name_model/biznames_model/onnx_int8 \
  --output results/validation_500.json \
  --onnx
```

### 2. Full Validation with Variants

```bash
# Generate variants
python sz_generate_variants.py \
  --input data/test_samples/validation_20k.jsonl \
  --output data/test_samples/variants_20k.jsonl

# Run validation with variants
python sz_validate_with_variants.py \
  --exact data/test_samples/validation_20k.jsonl \
  --variants data/test_samples/variants_20k.jsonl \
  --name_model_path ~/999gz.git/name_model/personalnames_model/onnx_int8 \
  --biz_model_path ~/999gz.git/name_model/biznames_model/onnx_int8 \
  --output results/validation_with_variants.json \
  --onnx
```

### 3. Interactive Debugging

```bash
# Debug a specific search
python sz_debug_search.py "Carlyle Group" --type business \
  --model_path ~/999gz.git/name_model/biznames_model/onnx_int8 \
  --onnx
```

---

## See Also

- [CLAUDE.md](../CLAUDE.md) - Quick reference for common commands
- [docs/README.md](README.md) - Setup and configuration guide
- [docs/TESTING.md](TESTING.md) - Testing workflow details
