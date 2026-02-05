#!/usr/bin/env python
"""
Convert PyTorch SentenceTransformer models to generic ONNX FP32 format.

This script creates ONNX models with standard ops only (no ONNX Runtime-specific
optimizations), making them compatible with any ONNX runtime implementation.

Designed for E5 models (native 384 dimensions, no dense layer) but works with
any SentenceTransformer model.

Pipeline:
    PyTorch SentenceTransformer
             |
        ONNX Export (FP32, standard ops)
             |
        Package with Tokenizer + Configs

Usage:
    # E5 model (no dense layer, native 384d)
    python sz_convert_to_onnx_generic.py \
        --input ~/roncewind.git/PersonalNames/output/e5_small_finetuned/FINAL-fine_tuned_model \
        --output ~/roncewind.git/PersonalNames/output/e5_small_finetuned/FINAL-onnx-fp32 \
        --validate

    # LaBSE model (with dense layer truncation)
    python sz_convert_to_onnx_generic.py \
        --input ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model \
        --output ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-onnx-fp32-generic \
        --truncate_dim 512 \
        --validate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_pytorch_model(model_path: str):
    """Load PyTorch SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading PyTorch model from: {model_path}")
    model = SentenceTransformer(model_path)
    model.eval()

    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"   Max sequence length: {model.max_seq_length}")

    # Check model architecture
    print(f"   Modules:")
    for idx, (name, module) in enumerate(model._modules.items()):
        module_type = type(module).__name__
        print(f"      [{idx}] {name}: {module_type}")

    return model


def get_dense_layer_info(model):
    """
    Check if model has a Dense layer and extract its info.
    E5 models don't have a dense layer; LaBSE models do.
    """
    for idx, module in model._modules.items():
        if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
            in_features = module.linear.in_features
            out_features = module.linear.out_features
            has_bias = module.linear.bias is not None
            activation = None
            if hasattr(module, 'activation_function'):
                activation = str(type(module.activation_function).__name__)
            return {
                'has_dense': True,
                'in_features': in_features,
                'out_features': out_features,
                'has_bias': has_bias,
                'activation': activation,
                'module': module
            }
    return {'has_dense': False}


def truncate_dense_layer(model, target_dim: int):
    """
    Truncate the Dense layer to output fewer dimensions (Matryoshka truncation).
    Only applicable for models with a dense layer (LaBSE, not E5).
    """
    print(f"\nTruncating Dense layer to {target_dim}d...")

    dense_info = get_dense_layer_info(model)
    if not dense_info['has_dense']:
        print("   No Dense layer found - model outputs native dimensions")
        return model, None

    dense_module = dense_info['module']
    original_out = dense_info['out_features']
    print(f"   Original: {dense_info['in_features']} -> {original_out}")

    if target_dim >= original_out:
        print(f"   Target dim {target_dim} >= current {original_out}, skipping")
        return model, None

    weight = dense_module.linear.weight.data
    bias = dense_module.linear.bias.data if dense_module.linear.bias is not None else None

    weight_truncated = weight[:target_dim, :].clone()
    bias_truncated = bias[:target_dim].clone() if bias is not None else None

    new_linear = nn.Linear(dense_info['in_features'], target_dim, bias=(bias is not None))
    new_linear.weight.data = weight_truncated
    if bias_truncated is not None:
        new_linear.bias.data = bias_truncated

    dense_module.linear = new_linear
    dense_module.out_features = target_dim

    print(f"   Truncated: {dense_info['in_features']} -> {target_dim}")

    return model, {
        'weight': weight_truncated.cpu(),
        'bias': bias_truncated.cpu() if bias_truncated is not None else None,
        'in_features': dense_info['in_features'],
        'out_features': target_dim,
        'original_out_features': original_out,
        'activation': dense_info['activation']
    }


def export_to_onnx(model, output_path: str, opset_version: int = 17, include_token_type_ids: bool = False):
    """
    Export transformer component to ONNX format with standard ops only.
    No ONNX Runtime-specific optimizations are applied.

    Args:
        model: SentenceTransformer model
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version (default: 17 for broad compatibility)
        include_token_type_ids: Whether to include token_type_ids in model inputs.
                               Set False for models like E5 that don't need it.
    """
    print(f"\nExporting to ONNX (generic FP32, opset {opset_version})...")

    transformer = model[0].auto_model
    config = transformer.config
    max_seq_length = model.max_seq_length

    print(f"   Model type: {config.model_type}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Max seq length: {max_seq_length}")

    device = torch.device('cpu')
    transformer = transformer.to(device)
    transformer.eval()

    # Create dummy inputs
    batch_size = 1
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_seq_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(batch_size, max_seq_length, dtype=torch.long, device=device)

    # Only include token_type_ids if explicitly requested
    # Most modern models (E5, etc.) don't require it even if config has type_vocab_size > 1
    if include_token_type_ids:
        dummy_token_type_ids = torch.zeros(batch_size, max_seq_length, dtype=torch.long, device=device)
        dummy_inputs = (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
        input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        dynamic_axes = {
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'token_type_ids': {0: 'batch', 1: 'sequence'},
            'last_hidden_state': {0: 'batch', 1: 'sequence'},
            'pooler_output': {0: 'batch'},
        }
    else:
        dummy_inputs = (dummy_input_ids, dummy_attention_mask)
        input_names = ['input_ids', 'attention_mask']
        dynamic_axes = {
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'last_hidden_state': {0: 'batch', 1: 'sequence'},
            'pooler_output': {0: 'batch'},
        }

    output_names = ['last_hidden_state', 'pooler_output']
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"   Exporting with opset version {opset_version}...")
    print(f"   Input names: {input_names}")
    print(f"   Output names: {output_names}")

    with torch.no_grad():
        torch.onnx.export(
            transformer,
            dummy_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Exported: {output_path} ({size_mb:.1f} MB)")

    # Verify the model is valid
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"   Verified: ONNX model is valid")

    # List op types used (for debugging)
    op_types = set()
    for node in onnx_model.graph.node:
        op_types.add(node.op_type)
    print(f"   Op types used ({len(op_types)}): {', '.join(sorted(op_types)[:10])}...")

    return output_path


def save_supporting_files(model, output_dir: str, dense_info: dict = None, base_model_name: str = None):
    """Save tokenizer, configs, and optional dense layer."""
    print(f"\nSaving supporting files...")

    output_dir = Path(output_dir)

    # 1. Save tokenizer
    print("   Saving tokenizer...")
    tokenizer_dir = output_dir / 'tokenizer'
    model.tokenizer.save_pretrained(str(tokenizer_dir))

    # 2. Save pooling config
    print("   Saving pooling config...")
    pooling_module = model[1]
    pooling_config = {
        'word_embedding_dimension': pooling_module.word_embedding_dimension,
        'pooling_mode_cls_token': pooling_module.pooling_mode_cls_token,
        'pooling_mode_mean_tokens': pooling_module.pooling_mode_mean_tokens,
        'pooling_mode_max_tokens': pooling_module.pooling_mode_max_tokens,
        'pooling_mode_mean_sqrt_len_tokens': pooling_module.pooling_mode_mean_sqrt_len_tokens,
    }
    with open(output_dir / 'pooling_config.json', 'w') as f:
        json.dump(pooling_config, f, indent=2)

    # 3. Save dense layer if present
    saved_dense = False
    if dense_info is not None and dense_info.get('weight') is not None:
        print(f"   Saving dense layer: {dense_info['in_features']} -> {dense_info['out_features']}...")
        torch.save({
            'weight': dense_info['weight'],
            'bias': dense_info['bias'],
            'in_features': dense_info['in_features'],
            'out_features': dense_info['out_features'],
            'activation_function': dense_info.get('activation'),
        }, output_dir / 'dense_layer.pt')
        saved_dense = True

    # 4. Save model config
    print("   Saving model config...")
    embedding_dim = model.get_sentence_embedding_dimension()

    # Determine base model from config or argument
    if base_model_name is None:
        config = model[0].auto_model.config
        if hasattr(config, '_name_or_path'):
            base_model_name = config._name_or_path
        else:
            base_model_name = 'unknown'

    model_config = {
        'max_seq_length': model.max_seq_length,
        'embedding_dimension': embedding_dim,
        'model_type': 'onnx_fp32_generic',
        'base_model': base_model_name,
        'has_dense_layer': saved_dense,
    }

    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    files_saved = ["tokenizer/", "pooling_config.json", "model_config.json"]
    if saved_dense:
        files_saved.append("dense_layer.pt")
    print(f"   Done: {', '.join(files_saved)}")


def validate_conversion(onnx_model_path: str, reference_model_path: str):
    """Validate ONNX model produces similar embeddings to PyTorch reference."""
    print(f"\nValidating ONNX model against PyTorch reference...")

    from sentence_transformers import SentenceTransformer

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from onnx_sentence_transformer import ONNXSentenceTransformer

    test_texts = [
        "Vladimir Putin",
        "Apple Inc.",
        "John Smith",
        "Gazprom",
        "Maria Garcia",
        "Sberbank",
        "Deutsche Bank AG",
        "Toyota Motor Corporation",
        "Jose Garcia",
        "Владимир Путин",
    ]

    print(f"   Loading PyTorch reference model...")
    ref_model = SentenceTransformer(reference_model_path)

    print(f"   Loading ONNX model...")
    onnx_model = ONNXSentenceTransformer(onnx_model_path, providers=['CPUExecutionProvider'])

    print(f"   Generating embeddings for {len(test_texts)} test texts...")

    # Both models should output same dimensions
    ref_embeddings = ref_model.encode(test_texts, normalize_embeddings=True)
    onnx_embeddings = onnx_model.encode(test_texts, normalize_embeddings=True)

    print(f"   PyTorch shape: {ref_embeddings.shape}")
    print(f"   ONNX shape:    {onnx_embeddings.shape}")

    if ref_embeddings.shape != onnx_embeddings.shape:
        print(f"   WARNING: Shape mismatch!")
        min_dim = min(ref_embeddings.shape[1], onnx_embeddings.shape[1])
        ref_embeddings = ref_embeddings[:, :min_dim]
        onnx_embeddings = onnx_embeddings[:, :min_dim]
        # Re-normalize after truncation
        ref_embeddings = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
        onnx_embeddings = onnx_embeddings / np.linalg.norm(onnx_embeddings, axis=1, keepdims=True)

    # Calculate similarity metrics
    cosine_sims = np.sum(ref_embeddings * onnx_embeddings, axis=1)
    max_diff = np.abs(ref_embeddings - onnx_embeddings).max()

    print(f"\n   Validation Results:")
    print(f"   {'='*45}")
    print(f"   Embedding dimensions: {onnx_embeddings.shape[1]}")
    print(f"   Mean cosine similarity: {cosine_sims.mean():.6f}")
    print(f"   Min cosine similarity:  {cosine_sims.min():.6f}")
    print(f"   Max cosine similarity:  {cosine_sims.max():.6f}")
    print(f"   Max absolute diff:      {max_diff:.6f}")
    print(f"   {'='*45}")

    # FP32 should have very high similarity (>0.999)
    passed = cosine_sims.min() > 0.999

    if passed:
        print(f"   PASSED: All cosine similarities > 0.999")
        print(f"   ONNX model is functionally equivalent to PyTorch!")
    else:
        print(f"   WARNING: Some cosine similarities < 0.999")
        print(f"   The ONNX model may have conversion differences.")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch SentenceTransformer to generic ONNX FP32 format"
    )

    parser.add_argument("--input", "-i", required=True,
                        help="Path to PyTorch SentenceTransformer model")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for ONNX model")
    parser.add_argument("--truncate_dim", type=int, default=None,
                        help="Matryoshka truncation dimension (only for models with dense layer)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17 for broad compatibility)")
    parser.add_argument("--include_token_type_ids", action="store_true",
                        help="Include token_type_ids in model inputs (needed for some BERT models)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate converted model against PyTorch reference")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name for config (auto-detected if not specified)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input model not found: {input_path}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generic ONNX FP32 Model Conversion")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Opset:  {args.opset}")
    if args.truncate_dim:
        print(f"Truncation: {args.truncate_dim}d")
    print("=" * 60)

    # Step 1: Load PyTorch model
    model = load_pytorch_model(str(input_path))

    # Step 2: Check for and optionally truncate dense layer
    dense_info = None
    if args.truncate_dim:
        model, dense_info = truncate_dense_layer(model, args.truncate_dim)
    else:
        # Just check if there's a dense layer
        info = get_dense_layer_info(model)
        if info['has_dense']:
            print(f"\n   Note: Model has Dense layer ({info['in_features']} -> {info['out_features']})")
            print(f"   Use --truncate_dim to truncate output dimensions")

    # Step 3: Export to ONNX (no optimizations)
    onnx_path = output_dir / "model.onnx"
    export_to_onnx(model, str(onnx_path), opset_version=args.opset,
                   include_token_type_ids=args.include_token_type_ids)

    # Step 4: Save supporting files
    save_supporting_files(model, str(output_dir), dense_info, args.base_model)

    # Step 5: Validate if requested
    if args.validate:
        validate_conversion(str(output_dir), str(input_path))

    # Summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)

    onnx_size = (output_dir / "model.onnx").stat().st_size / (1024 * 1024)
    print(f"Output directory: {output_dir}")
    print(f"Model size: {onnx_size:.1f} MB")
    print(f"Model type: onnx_fp32_generic (standard ops only)")

    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size / 1024
            print(f"   {f.name}: {size:.1f} KB")
        elif f.is_dir():
            print(f"   {f.name}/")

    print("\nTo use this model:")
    print(f"   from onnx_sentence_transformer import ONNXSentenceTransformer")
    print(f"   model = ONNXSentenceTransformer('{output_dir}')")
    print(f"   embeddings = model.encode(['test text'], normalize_embeddings=True)")

    return 0


if __name__ == "__main__":
    exit(main())
