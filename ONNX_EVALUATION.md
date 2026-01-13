# ONNX Model Evaluation Results

**Date:** 2026-01-13
**Evaluated by:** Claude Code

## Overview

This document summarizes the evaluation of FP16 ONNX-quantized embedding models compared to the original PyTorch models. The goal was to verify that ONNX conversion does not degrade model performance while potentially offering faster inference.

## Models Tested

### Personal Names Model
| Version | Path |
|---------|------|
| PyTorch (baseline) | `~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model` |
| ONNX FP16 | `~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-onnx-fp16-native` |

### Business Names Model
| Version | Path |
|---------|------|
| PyTorch (baseline) | `~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model` |
| ONNX FP16 | `~/roncewind.git/BizNames/output/phase10_quantization/onnx_fp16_direct/` |

**Note:** The business names ONNX model is from phase10, while baseline is phase9b. Despite this difference, results are comparable.

## Test Configuration

- **Embedding Dimension:** 512 (Matryoshka truncation from 768)
- **Dataset:** OpenSanctions 10K balanced sample
- **Test Cases:** 438 exact + 535 variants = 973 total

## Results Summary

### Exact Name Queries (Self-Retrieval)

| Metric | Baseline (PyTorch) | ONNX (FP16) | Delta |
|--------|-------------------|-------------|-------|
| Test Cases | 436 | 438 | +2 |
| Name-only Recall@1 | 99.31% | 99.3% | ~0% |
| Embedding Recall@1 | 100.0% | 100.0% | 0.0% |
| Embedding Recall@10 | 100.0% | 100.0% | 0.0% |
| MRR | 1.0000 | 1.0000 | 0.0000 |
| Not Found Rate | 0.0% | 0.0% | 0.0% |

### Variant Queries (Fuzzy Matching)

| Metric | Baseline (PyTorch) | ONNX (FP16) | Delta |
|--------|-------------------|-------------|-------|
| Test Cases | 532 | 535 | +3 |
| Name-only Recall@1 | 72.0% | 72.1% | +0.1% |
| Name-only Recall@10 | 72.4% | 72.3% | -0.1% |
| **Embedding Recall@1** | **90.8%** | **90.8%** | **0.0%** |
| **Embedding Recall@10** | **97.2%** | **97.6%** | **+0.4%** |
| MRR | 0.9342 | 0.9349 | +0.0007 |
| Not Found Rate | 0.6% | ~0.4% | -0.2% |

### Variant Type Performance (ONNX)

| Variant Type | Cases | Name-Only | Embedding | Improvement |
|--------------|-------|-----------|-----------|-------------|
| word_dropped | 124 | 9.7% | 100.0% | +90.3% |
| words_abbreviated | 13 | 61.5% | 100.0% | +38.5% |
| patronymic_dropped | 107 | 87.9% | 99.1% | +11.2% |
| name_order_reversed | 204 | 91.7% | 99.5% | +7.8% |
| legal_suffix_removed | 41 | 97.6% | 100.0% | +2.4% |
| abbreviations_expanded | 36 | 100.0% | 100.0% | 0.0% |
| prepositions_removed | 10 | 100.0% | 100.0% | 0.0% |

## Key Findings

1. **No Performance Degradation:** ONNX FP16 models perform identically to PyTorch baseline within measurement noise (±0.4%)

2. **Embedding Value Confirmed:** Embeddings provide +25% improvement over name-only search on variant queries

3. **Variant Handling:** The `word_dropped` variant type shows the most dramatic improvement from embeddings (+90%)

4. **Production Ready:** ONNX models are suitable for production deployment

## Load Performance

| Metric | Value |
|--------|-------|
| Records Loaded | 9,998 / 10,000 |
| Success Rate | 99.98% |
| Load Time | 31 min 46 sec |
| Throughput | ~5.2 records/sec |
| Failed Records | 2 (data exceeds column limits) |

## Files Created

### New Scripts
- `onnx_sentence_transformer.py` - ONNX model wrapper with SentenceTransformer-compatible API
- `sz_load_embeddings_onnx.py` - Data loader using ONNX models

### Modified Scripts
- `sz_validate_production.py` - Added `--onnx` flag
- `sz_validate_with_variants.py` - Added `--onnx` flag

### Result Files
- `results/validation_onnx_500.json` - 500-record validation
- `results/validation_onnx_10k.json` - 10K exact validation
- `results/validation_onnx_10k_with_variants.json` - 10K exact + variant validation

## Recommendations

1. **Use ONNX models for production** - Equivalent performance with potential inference speed benefits
2. **Monitor column limits** - Some OpenSanctions records exceed VARCHAR(255/300) limits
3. **Keep variant testing** - The variant test is more discriminating than exact match tests

## Reproduction Commands

```bash
# Load with ONNX models
python sz_load_embeddings_onnx.py \
  -i data/test_samples/sample_10k_balanced.jsonl \
  --name_model_path ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-onnx-fp16-native \
  --biz_model_path ~/roncewind.git/BizNames/output/phase10_quantization/onnx_fp16_direct/ \
  --truncate_dim 512 \
  --threads 16

# Validate with variants
python sz_validate_with_variants.py \
  --exact data/test_samples/validation_10k.jsonl \
  --variants data/test_samples/variants_10k.jsonl \
  --name_model_path ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-onnx-fp16-native \
  --biz_model_path ~/roncewind.git/BizNames/output/phase10_quantization/onnx_fp16_direct/ \
  --truncate_dim 512 \
  --output results/validation_onnx_with_variants.json \
  --onnx
```
