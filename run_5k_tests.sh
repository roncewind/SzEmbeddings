#!/bin/bash
# Comprehensive validation and evaluation for 5k OpenSanctions load
# Run after redoer completes

set -e  # Exit on error

echo "=================================================="
echo "5k OpenSanctions Test Suite"
echo "=================================================="
echo ""

# Setup environment
source ~/senzingv4/setupEnv
source venv/bin/activate
source secrets/DO_NOT_PUSH_sz_env_vars.sh

# Configuration
INPUT_FILE="opensanctions_test_5k_final.jsonl"
VALIDATION_SAMPLES="validation_samples_200.jsonl"
NAME_MODEL="$HOME/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model"
BIZ_MODEL="$HOME/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model"
RESULTS_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=================================================="
echo "Step 1: Verify Database State"
echo "=================================================="
echo ""

PGPASSWORD=senzing psql -h localhost -U senzing -d senzing -c "
SELECT
    COUNT(*) as total_records,
    (SELECT COUNT(*) FROM name_embedding) as name_embeddings,
    (SELECT COUNT(*) FROM bizname_embedding) as biz_embeddings
FROM dsrc_record;
"

echo ""
echo "=================================================="
echo "Step 2: Extract Validation Samples (200 records)"
echo "=================================================="
echo ""

python sz_extract_validation_samples.py \
  --input "$INPUT_FILE" \
  --output "$VALIDATION_SAMPLES" \
  --sample_size 200 \
  --filter both \
  --seed 42

echo ""
echo "✅ Validation samples extracted to: $VALIDATION_SAMPLES"
echo ""

echo "=================================================="
echo "Step 3: Production Validation (Senzing + PostgreSQL)"
echo "=================================================="
echo "Testing recall on loaded data..."
echo ""

python sz_validate_production.py \
  --input "$VALIDATION_SAMPLES" \
  --name_model_path "$NAME_MODEL" \
  --biz_model_path "$BIZ_MODEL" \
  --truncate_dim 512 \
  --validate_pg \
  --pg_database senzing \
  --pg_user senzing \
  --pg_password senzing \
  --output "$RESULTS_DIR/validation_5k_${TIMESTAMP}.json" \
  | tee "$RESULTS_DIR/validation_5k_${TIMESTAMP}.log"

echo ""
echo "✅ Production validation complete"
echo ""

echo "=================================================="
echo "Step 4: Model Evaluation - Business Names"
echo "=================================================="
echo "Evaluating business model on triplets (sample 500 for speed)..."
echo ""

python sz_evaluate_model.py \
  --type business \
  --model_path "$BIZ_MODEL" \
  --triplets opensanctions_test_5k_triplets_business.jsonl \
  --test_set opensanctions_5k \
  --data_source OPEN_SANCTIONS \
  --sample 500 \
  --output "$RESULTS_DIR/eval_business_5k_${TIMESTAMP}.json" \
  | tee "$RESULTS_DIR/eval_business_5k_${TIMESTAMP}.log"

echo ""
echo "✅ Business model evaluation complete"
echo ""

echo "=================================================="
echo "Step 5: Model Evaluation - Personal Names"
echo "=================================================="
echo "Evaluating personal model on triplets (sample 500 for speed)..."
echo ""

python sz_evaluate_model.py \
  --type personal \
  --model_path "$NAME_MODEL" \
  --triplets opensanctions_test_5k_triplets_personal.jsonl \
  --test_set opensanctions_5k \
  --data_source OPEN_SANCTIONS \
  --sample 500 \
  --output "$RESULTS_DIR/eval_personal_5k_${TIMESTAMP}.json" \
  | tee "$RESULTS_DIR/eval_personal_5k_${TIMESTAMP}.log"

echo ""
echo "✅ Personal model evaluation complete"
echo ""

echo "=================================================="
echo "All Tests Complete!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - Production Validation: $RESULTS_DIR/validation_5k_${TIMESTAMP}.json"
echo "  - Business Evaluation:   $RESULTS_DIR/eval_business_5k_${TIMESTAMP}.json"
echo "  - Personal Evaluation:   $RESULTS_DIR/eval_personal_5k_${TIMESTAMP}.json"
echo ""
echo "Logs saved to:"
echo "  - $RESULTS_DIR/validation_5k_${TIMESTAMP}.log"
echo "  - $RESULTS_DIR/eval_business_5k_${TIMESTAMP}.log"
echo "  - $RESULTS_DIR/eval_personal_5k_${TIMESTAMP}.log"
echo ""
echo "To view validation results:"
echo "  cat $RESULTS_DIR/validation_5k_${TIMESTAMP}.log | less"
echo ""
echo "To compare with other runs (after multiple tests):"
echo "  python sz_compare_validations.py $RESULTS_DIR/validation_5k_*.json"
echo "  python sz_compare_models.py $RESULTS_DIR/eval_*_5k_*.json"
echo ""
