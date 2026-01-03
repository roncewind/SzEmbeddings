#!/bin/bash
#
# Reproducible Validation Pipeline
#
# This script runs the complete validation workflow:
# 1. Sample records from source data
# 2. Extract validation test cases
# 3. Generate fuzzy variants
# 4. Purge and reload database
# 5. Run comprehensive validation
#
# This makes validation fully reproducible across database reloads.
#

set -e  # Exit on error

# Configuration
SAMPLE_SIZE=${SAMPLE_SIZE:-10000}
VALIDATION_SIZE=${VALIDATION_SIZE:-200}
VARIANTS_PER_RECORD=${VARIANTS_PER_RECORD:-2}
SEED=${SEED:-42}
VERSION=${VERSION:-$(date +%Y%m%d_%H%M%S)}

# Paths
SOURCE_DATA=${SOURCE_DATA:-"/data/OpenSanctions/senzing.json"}
NAME_MODEL=${NAME_MODEL:-"$HOME/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model"}
BIZ_MODEL=${BIZ_MODEL:-"$HOME/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model"}
TRUNCATE_DIM=${TRUNCATE_DIM:-512}

# Output directories
DATA_DIR="data/test_samples"
RESULTS_DIR="results"

# Output files (versioned)
SAMPLE_FILE="${DATA_DIR}/sample_${SAMPLE_SIZE}_v${VERSION}.jsonl"
VALIDATION_FILE="${DATA_DIR}/validation_samples_${VALIDATION_SIZE}_v${VERSION}.jsonl"
VARIANTS_FILE="${DATA_DIR}/validation_variants_v${VERSION}.jsonl"
RESULTS_FILE="${RESULTS_DIR}/validation_results_v${VERSION}.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Reproducible Validation Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Sample Size:       $SAMPLE_SIZE"
echo "  Validation Size:   $VALIDATION_SIZE"
echo "  Variants/Record:   $VARIANTS_PER_RECORD"
echo "  Seed:              $SEED"
echo "  Version:           $VERSION"
echo ""
echo "Output Files:"
echo "  Sample:       $SAMPLE_FILE"
echo "  Validation:   $VALIDATION_FILE"
echo "  Variants:     $VARIANTS_FILE"
echo "  Results:      $RESULTS_FILE"
echo ""

# Setup environment
echo -e "${YELLOW}⏳ Setting up environment...${NC}"
source ~/senzingv4/setupEnv
source venv/bin/activate
source secrets/DO_NOT_PUSH_sz_env_vars.sh

# Verify environment
if [ -z "$SENZING_ENGINE_CONFIGURATION_JSON" ]; then
    echo -e "${RED}✗ SENZING_ENGINE_CONFIGURATION_JSON is NOT set${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Environment ready${NC}"
echo ""

# Step 1: Sample records from source
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 1: Sample Records${NC}"
echo -e "${BLUE}========================================${NC}"
if [ -f "$SAMPLE_FILE" ]; then
    echo -e "${YELLOW}Sample file already exists: $SAMPLE_FILE${NC}"
    read -p "Overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing sample file"
    else
        echo -e "${YELLOW}⏳ Sampling $SAMPLE_SIZE records from $SOURCE_DATA...${NC}"
        python sz_sample_data.py \
            -i "$SOURCE_DATA" \
            -o "$SAMPLE_FILE" \
            --sample_size $SAMPLE_SIZE \
            --seed $SEED
        echo -e "${GREEN}✓ Sampling complete${NC}"
    fi
else
    echo -e "${YELLOW}⏳ Sampling $SAMPLE_SIZE records from $SOURCE_DATA...${NC}"
    python sz_sample_data.py \
        -i "$SOURCE_DATA" \
        -o "$SAMPLE_FILE" \
        --sample_size $SAMPLE_SIZE \
        --seed $SEED
    echo -e "${GREEN}✓ Sampling complete${NC}"
fi
echo ""

# Step 2: Extract validation test cases
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 2: Extract Validation Test Cases${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}⏳ Extracting $VALIDATION_SIZE validation cases...${NC}"
python sz_extract_validation_samples.py \
    --input "$SAMPLE_FILE" \
    --output "$VALIDATION_FILE" \
    --sample_size $VALIDATION_SIZE \
    --filter both \
    --seed $SEED
echo -e "${GREEN}✓ Validation cases extracted${NC}"
echo ""

# Step 3: Generate fuzzy variants
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 3: Generate Fuzzy Variants${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}⏳ Generating variants (max $VARIANTS_PER_RECORD per record)...${NC}"
python sz_generate_variants.py \
    -i "$VALIDATION_FILE" \
    -o "$VARIANTS_FILE" \
    --max-per-record $VARIANTS_PER_RECORD
echo -e "${GREEN}✓ Variants generated${NC}"
echo ""

# Step 4: Purge and reload database
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 4: Database Purge and Reload${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${RED}⚠️  WARNING: This will PURGE the Senzing database!${NC}"
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping database reload"
    echo "To run validation with existing data, use:"
    echo "  python sz_validate_with_variants.py --exact $VALIDATION_FILE --variants $VARIANTS_FILE ..."
    exit 0
fi

echo -e "${YELLOW}⏳ Purging database...${NC}"
python -c "
from senzing_core import SzAbstractFactoryCore
import os
settings = os.environ.get('SENZING_ENGINE_CONFIGURATION_JSON')
factory = SzAbstractFactoryCore('', settings=settings)
engine = factory.create_engine()
engine.purge_repository()
print('✓ Database purged')
"
echo -e "${GREEN}✓ Database purged${NC}"
echo ""

echo -e "${YELLOW}⏳ Loading $SAMPLE_SIZE records with embeddings...${NC}"
python sz_load_embeddings.py \
    -i "$SAMPLE_FILE" \
    --name_model_path "$NAME_MODEL" \
    --biz_model_path "$BIZ_MODEL" \
    --truncate_dim $TRUNCATE_DIM \
    --threads 24 \
    2> "${RESULTS_DIR}/load_v${VERSION}_stderr.log"
echo -e "${GREEN}✓ Loading complete${NC}"
echo ""

# Step 5: Run comprehensive validation
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 5: Run Comprehensive Validation${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}⏳ Running validation (exact + variants)...${NC}"
python sz_validate_with_variants.py \
    --exact "$VALIDATION_FILE" \
    --variants "$VARIANTS_FILE" \
    --name_model_path "$NAME_MODEL" \
    --biz_model_path "$BIZ_MODEL" \
    --truncate_dim $TRUNCATE_DIM \
    --output "$RESULTS_FILE"
echo -e "${GREEN}✓ Validation complete${NC}"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Pipeline Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Generated Files:"
echo "  Sample Data:      $SAMPLE_FILE"
echo "  Validation Cases: $VALIDATION_FILE"
echo "  Variants:         $VARIANTS_FILE"
echo "  Results:          $RESULTS_FILE"
echo "  Load Log:         ${RESULTS_DIR}/load_v${VERSION}_stderr.log"
echo ""
echo "To view results:"
echo "  python sz_compare_validations.py $RESULTS_FILE"
echo ""
echo -e "${GREEN}✓ All done!${NC}"
