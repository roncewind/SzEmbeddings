# Loading OpenSanctions Data into Senzing with Embeddings

## Overview
This document describes how to load OpenSanctions entity data into Senzing with semantic embeddings for both personal names and business names.

## Prerequisites

1. **Senzing v4** installed and configured
2. **Python virtual environment** with dependencies
3. **Environment variables** configured (SENZING_ENGINE_CONFIGURATION_JSON)
4. **Model paths** accessible:
   - Personal names: `~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model`
   - Business names: `~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model`

## Quick Start - Small Test (500 records)

For testing Senzing configuration and investigating issues:

```bash
# Setup environment
source ~/senzingv4/setupEnv
source venv/bin/activate
source secrets/DO_NOT_PUSH_sz_env_vars.sh

# Run load with full output capture
python sz_load_embeddings.py \
  -i opensanctions_test_500.jsonl \
  --name_model_path ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model \
  --biz_model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --truncate_dim 512 \
  --threads 24 \
  > load_500_stdout.log 2> load_500_stderr.log

# Monitor progress in another terminal
tail -f load_500_stderr.log
```

**Expected time:** ~5-10 minutes (depending on Senzing issues)

## Full Test Load (5K records)

For validating models with cross-validation against PostgreSQL test results:

```bash
# Setup environment
source ~/senzingv4/setupEnv
source venv/bin/activate
source secrets/DO_NOT_PUSH_sz_env_vars.sh

# Run load with full output capture
python sz_load_embeddings.py \
  -i opensanctions_test_5k_final.jsonl \
  --name_model_path ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model \
  --biz_model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --truncate_dim 512 \
  --threads 24 \
  > load_5k_stdout.log 2> load_5k_stderr.log

# Monitor progress in another terminal
tail -f load_5k_stderr.log
```

**Expected time:** ~30-60 minutes (at 90-156 rec/min with current issues)

## Understanding the Output

### Standard Output (stdout)
Contains progress messages:
```
📌 Using device: cuda
⏳ Loading models...
📌 Matryoshka truncation: 512 dimensions
📌 Threads for DB writes: 24
☺ 100 records processed, 11 added in 00:00:01
☺ 200 records processed, 111 added in 00:00:05
...
```

### Standard Error (stderr)
Contains detailed logs and errors:
```
2025-12-18 13:49:27,908 - INFO - Load pretrained SentenceTransformer: ...
2025-12-18 13:54:44,541 - ERROR - SENZ0010|Retry timeout exceeded ...
```

### Key Metrics to Track

**Success Rate:**
```bash
# Count successful vs processed
grep "records processed" load_5k_stdout.log | tail -1
```

**Error Rate:**
```bash
# Count Senzing errors
grep "SENZ0010" load_5k_stderr.log | wc -l
```

**Load Speed:**
```bash
# Calculate records per minute
grep "records processed" load_5k_stdout.log
```

## Common Issues

### Issue 1: SENZ0010 Retry Timeout Exceeded

**Error:**
```
SzRetryTimeoutExceededError: SENZ0010|Retry timeout exceeded resolved entity locklist []
```

**Impact:** Records fail to load (~33% failure rate observed)

**Investigation Steps:**
1. Check Senzing database connection pool settings
2. Review entity resolution configuration
3. Monitor database locks: `SELECT * FROM pg_locks WHERE NOT granted;`
4. Consider reducing thread count (try `--threads 12` or `--threads 6`)

**Workarounds Attempted:**
- Increased thread count to 24 (helped speed but not errors)
- Results: 69 rec/min → 90-156 rec/min improvement
- But 33.5% failure rate remains

### Issue 2: Out of Memory

**Symptoms:** Process killed, no error message

**Solutions:**
- Use `--truncate_dim 512` (reduces memory for embeddings)
- Reduce batch size in code (currently hardcoded to 100)
- Reduce thread count

### Issue 3: Model Loading Errors

**Symptoms:** "Model not found" or "Invalid model path"

**Solutions:**
- Verify model paths exist: `ls ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model`
- Check permissions on model directories
- Ensure models were trained with compatible sentence-transformers version

## Checking Load Results

After loading, verify what was successfully loaded:

```bash
source ~/senzingv4/setupEnv
source venv/bin/activate
source secrets/DO_NOT_PUSH_sz_env_vars.sh

# Connect to Senzing database
psql -d G2 -c "
SELECT
    COUNT(*) as records,
    (SELECT COUNT(*) FROM bizname_embedding) as biz_embeddings,
    (SELECT COUNT(*) FROM name_embedding) as name_embeddings
FROM dsrc_record;
"
```

**Example output:**
```
 records | biz_embeddings | name_embeddings
---------+----------------+-----------------
    3300 |           4129 |            7975
```

**Note:** Embedding counts > record counts is **normal** - each record can have multiple name variants (aliases), and each name gets its own embedding.

## Parameters Explained

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|----------------|
| `-i` | Input JSONL file | Required | Start with 500-record test file |
| `--name_model_path` | Personal names model | Required | Use fine-tuned LaBSE model |
| `--biz_model_path` | Business names model | Required | Use fine-tuned LaBSE model |
| `--truncate_dim` | Matryoshka truncation dimension | None | Use 512 for memory efficiency |
| `--threads` | Thread count for DB writes | 12 | Try 24 for speed, reduce to 6-12 if errors |

## File Descriptions

### Input Files
- `opensanctions_test_500.jsonl` - 500 records for quick testing
- `opensanctions_test_5k_final.jsonl` - 4,965 records (2,750 persons + 2,215 orgs) for validation
- Both files contain entities from test databases with existing PostgreSQL results for cross-validation

### Output Files
- `load_*_stdout.log` - Progress messages and statistics
- `load_*_stderr.log` - Detailed logs and error messages

## Next Steps After Loading

1. **Verify load results** (see "Checking Load Results" above)
2. **Cross-validate with PostgreSQL results** - Compare Senzing entity resolution with pgvector test results
3. **Investigate errors** - Analyze stderr logs for patterns in failures
4. **Optimize if needed** - Adjust thread count, batch size, or Senzing configuration

## Troubleshooting Command Reference

```bash
# Check if models are accessible
ls ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model
ls ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model

# Check GPU availability
nvidia-smi

# Check Senzing environment
echo $SENZING_ENGINE_CONFIGURATION_JSON | python -m json.tool

# Monitor database activity
psql -d G2 -c "SELECT * FROM pg_stat_activity WHERE application_name LIKE 'SzEmbeddings%';"

# Check database locks
psql -d G2 -c "SELECT * FROM pg_locks WHERE NOT granted;"

# Clear Senzing database (DESTRUCTIVE - use with caution)
# ... database purge commands would go here ...
```

## Performance Benchmarks

Based on testing with opensanctions_test_5k_final.jsonl (4,965 records):

| Metric | Value |
|--------|-------|
| **Load speed** | 90-156 records/min |
| **Success rate** | 66.5% (3,300/4,965) |
| **Failure rate** | 33.5% (1,665/4,965) |
| **Primary error** | SENZ0010 Retry timeout |
| **Embeddings created** | 12,104 (avg 2.4 per record) |
| **GPU utilization** | Yes (CUDA enabled) |

## Contact & Support

For Senzing issues:
- Report to Senzing support with error logs
- Include: stderr logs, database lock queries, configuration

For model/embedding issues:
- Check BizNames and PersonalNames project documentation
- Verify model training completed successfully
