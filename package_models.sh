#!/bin/bash
# Package ONNX models for distribution in the name_model repository
#
# This script copies the recommended ONNX models (FP16 for GPU, INT8 for CPU)
# to the name_model repository structure.
#
# Usage: ./package_models.sh [--dry-run]

set -e

# Configuration
SOURCE_PERSONAL_FP16=~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-onnx-fp16-native
SOURCE_PERSONAL_INT8=~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-onnx-int8-v3
SOURCE_BIZ_FP16=~/roncewind.git/BizNames/output/phase10_quantization/onnx_fp16_direct
SOURCE_BIZ_INT8=~/roncewind.git/BizNames/output/phase10_quantization/onnx_int8

TARGET_REPO=~/999gz.git/name_model

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be copied ==="
    echo ""
fi

# Function to copy model directory
copy_model() {
    local src="$1"
    local dst="$2"
    local name="$3"

    echo "Copying $name..."
    echo "  From: $src"
    echo "  To:   $dst"

    if [[ ! -d "$src" ]]; then
        echo "  ERROR: Source directory does not exist!"
        return 1
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would copy $(du -sh "$src" | cut -f1)"
        return 0
    fi

    # Create target directory
    mkdir -p "$dst"

    # Copy model.onnx (handle different source filenames)
    if [[ -f "$src/model.onnx" ]]; then
        cp "$src/model.onnx" "$dst/model.onnx"
    elif [[ -f "$src/transformer_fp16.onnx" ]]; then
        cp "$src/transformer_fp16.onnx" "$dst/model.onnx"
    else
        echo "  ERROR: No ONNX model file found!"
        return 1
    fi

    # Copy dense layer (handle different filenames)
    if [[ -f "$src/dense_512d.pt" ]]; then
        cp "$src/dense_512d.pt" "$dst/dense_layer.pt"
    elif [[ -f "$src/dense_layer.pt" ]]; then
        cp "$src/dense_layer.pt" "$dst/dense_layer.pt"
    fi

    # Copy config files
    [[ -f "$src/model_config.json" ]] && cp "$src/model_config.json" "$dst/"
    [[ -f "$src/pooling_config.json" ]] && cp "$src/pooling_config.json" "$dst/"
    [[ -f "$src/sentence_transformers_config.json" ]] && cp "$src/sentence_transformers_config.json" "$dst/"

    # Copy tokenizer directory
    if [[ -d "$src/tokenizer" ]]; then
        cp -r "$src/tokenizer" "$dst/"
    fi

    echo "  Done: $(du -sh "$dst" | cut -f1)"
}

# Function to create model_config.json if missing
create_config() {
    local dst="$1"
    local model_type="$2"
    local config_file="$dst/model_config.json"

    if [[ ! -f "$config_file" ]]; then
        echo "  Creating model_config.json for $model_type..."
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [DRY RUN] Would create $config_file"
            return 0
        fi
        cat > "$config_file" << EOF
{
  "max_seq_length": 32,
  "embedding_dimension": 512,
  "model_type": "$model_type",
  "base_model": "sentence-transformers/LaBSE",
  "matryoshka_dim": 512
}
EOF
    fi
}

echo "=========================================="
echo "  ONNX Model Packaging Script"
echo "=========================================="
echo ""
echo "Target repository: $TARGET_REPO"
echo ""

# Verify source models exist
echo "Verifying source models..."
for src in "$SOURCE_PERSONAL_FP16" "$SOURCE_PERSONAL_INT8" "$SOURCE_BIZ_FP16" "$SOURCE_BIZ_INT8"; do
    if [[ ! -d "$src" ]]; then
        echo "ERROR: Source not found: $src"
        exit 1
    fi
done
echo "All source models found."
echo ""

# Copy Personal Names models
echo "--- Personal Names Models ---"
copy_model "$SOURCE_PERSONAL_FP16" "$TARGET_REPO/personalnames_model/onnx_fp16" "Personal Names ONNX FP16"
create_config "$TARGET_REPO/personalnames_model/onnx_fp16" "onnx_fp16"
echo ""

copy_model "$SOURCE_PERSONAL_INT8" "$TARGET_REPO/personalnames_model/onnx_int8" "Personal Names ONNX INT8"
create_config "$TARGET_REPO/personalnames_model/onnx_int8" "onnx_int8"
echo ""

# Copy Business Names models
echo "--- Business Names Models ---"
copy_model "$SOURCE_BIZ_FP16" "$TARGET_REPO/biznames_model/onnx_fp16" "Business Names ONNX FP16"
create_config "$TARGET_REPO/biznames_model/onnx_fp16" "onnx_fp16"
echo ""

copy_model "$SOURCE_BIZ_INT8" "$TARGET_REPO/biznames_model/onnx_int8" "Business Names ONNX INT8"
create_config "$TARGET_REPO/biznames_model/onnx_int8" "onnx_int8"
echo ""

# Summary
echo "=========================================="
echo "  Summary"
echo "=========================================="
if [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    echo "Models copied to $TARGET_REPO:"
    echo ""
    du -sh "$TARGET_REPO/personalnames_model/"*/ 2>/dev/null || true
    du -sh "$TARGET_REPO/biznames_model/"*/ 2>/dev/null || true
    echo ""
    echo "Total repository size:"
    du -sh "$TARGET_REPO"
    echo ""
    echo "Next steps:"
    echo "  1. cd $TARGET_REPO"
    echo "  2. git add -A"
    echo "  3. git status  # Verify LFS tracking"
    echo "  4. git commit -m 'Add ONNX FP16 and INT8 models for optimized deployment'"
    echo "  5. git push"
else
    echo ""
    echo "[DRY RUN] No files were copied."
    echo "Run without --dry-run to perform the actual copy."
fi
