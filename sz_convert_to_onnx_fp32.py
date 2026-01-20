#!/usr/bin/env python
"""
Convert PyTorch SentenceTransformer models to ONNX FP32 format.

FP32 ONNX format is useful for:
- CPU deployment where FP16 is not supported
- Baseline comparison against INT8 quantized models
- Maximum accuracy (no quantization loss)

Pipeline:
    PyTorch SentenceTransformer
             |
        ONNX Export (FP32)
             |
        Transformer Optimization
             |
        Package with Tokenizer + Configs

Usage:
    python sz_convert_to_onnx_fp32.py \
        --input ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model \
        --output ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-onnx-fp32 \
        --truncate_dim 512
"""

import argparse
import json
import tempfile
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

    return model


def truncate_dense_layer(model, target_dim: int):
    """
    Truncate the Dense layer to output fewer dimensions (Matryoshka truncation).
    """
    print(f"\nTruncating Dense layer to {target_dim}d (Matryoshka)...")

    dense_layer = None
    for idx, module in model._modules.items():
        if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
            dense_layer = module
            break

    if dense_layer is None:
        print("   No Dense layer found, skipping truncation")
        return model, None

    original_out = dense_layer.linear.out_features
    print(f"   Original: {dense_layer.linear.in_features} -> {original_out}")

    if target_dim >= original_out:
        print(f"   Target dim {target_dim} >= current {original_out}, skipping")
        return model, None

    weight = dense_layer.linear.weight.data
    bias = dense_layer.linear.bias.data if dense_layer.linear.bias is not None else None

    weight_truncated = weight[:target_dim, :].clone()
    bias_truncated = bias[:target_dim].clone() if bias is not None else None

    new_linear = nn.Linear(dense_layer.linear.in_features, target_dim, bias=(bias is not None))
    new_linear.weight.data = weight_truncated
    if bias_truncated is not None:
        new_linear.bias.data = bias_truncated

    dense_layer.linear = new_linear
    dense_layer.out_features = target_dim

    print(f"   Truncated: {dense_layer.linear.in_features} -> {target_dim}")

    dense_info = {
        'weight': weight_truncated.cpu(),
        'bias': bias_truncated.cpu() if bias_truncated is not None else None,
        'in_features': dense_layer.linear.in_features,
        'out_features': target_dim,
        'original_out_features': original_out
    }

    return model, dense_info


def export_to_onnx_fp32(model, output_path: str, opset_version: int = 14):
    """Export transformer component to ONNX FP32 format."""
    print(f"\nExporting to ONNX FP32...")

    transformer = model[0].auto_model
    config = transformer.config
    max_seq_length = model.max_seq_length

    print(f"   Model type: {config.model_type}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Max seq length: {max_seq_length}")

    device = torch.device('cpu')
    transformer = transformer.to(device)
    transformer.eval()

    batch_size = 1
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_seq_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(batch_size, max_seq_length, dtype=torch.long, device=device)

    if hasattr(config, 'type_vocab_size') and config.type_vocab_size > 1:
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
            dynamo=False,
            external_data=False,
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Exported: {output_path} ({size_mb:.1f} MB)")

    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"   Verified: ONNX model is valid")

    return output_path


def apply_transformer_optimizations(input_path: str, output_path: str, num_heads: int = 12, hidden_size: int = 768):
    """
    Apply BERT/transformer-specific optimizations (FP32).
    """
    print(f"\nApplying transformer optimizations (FP32)...")
    print(f"   Input: {input_path}")
    print(f"   Num heads: {num_heads}, Hidden size: {hidden_size}")

    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.fusion_options import FusionOptions

    fusion_options = FusionOptions('bert')
    fusion_options.enable_attention = True
    fusion_options.enable_embed_layer_norm = True
    fusion_options.enable_skip_layer_norm = True
    fusion_options.enable_bias_skip_layer_norm = True
    fusion_options.enable_gelu_approximation = True
    fusion_options.enable_bias_gelu = True

    print(f"   Optimizing for CPU (use_gpu=False)...")

    optimized_model = optimizer.optimize_model(
        str(input_path),
        model_type='bert',
        num_heads=num_heads,
        hidden_size=hidden_size,
        opt_level=99,
        optimization_options=fusion_options,
        use_gpu=False,
        only_onnxruntime=True,
    )

    stats = optimized_model.get_fused_operator_statistics()
    if stats:
        print(f"   Fused operators:")
        for op_type, count in stats.items():
            if count > 0:
                print(f"      {op_type}: {count}")

    output_path = Path(output_path)
    optimized_model.save_model_to_file(str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Optimized (FP32): {output_path} ({size_mb:.1f} MB)")

    return output_path


def save_supporting_files(model, output_dir: str, dense_info: dict = None, truncate_dim: int = None):
    """Save tokenizer, configs, and dense layer."""
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

    # 3. Save dense layer
    saved_dense = False
    if dense_info is not None:
        print("   Saving dense layer (truncated)...")
        torch.save({
            'weight': dense_info['weight'],
            'bias': dense_info['bias'],
            'in_features': dense_info['in_features'],
            'out_features': dense_info['out_features'],
        }, output_dir / 'dense_layer.pt')
        saved_dense = True

        if truncate_dim == 512:
            torch.save({
                'weight': dense_info['weight'],
                'bias': dense_info['bias'],
            }, output_dir / 'dense_512d.pt')
    else:
        try:
            dense_module = model[2]
            if hasattr(dense_module, 'linear'):
                print("   Saving dense layer (from model)...")
                activation_func = None
                if hasattr(dense_module, 'activation_function'):
                    activation_func = str(type(dense_module.activation_function).__name__)

                torch.save({
                    'weight': dense_module.linear.weight.data.cpu(),
                    'bias': dense_module.linear.bias.data.cpu() if dense_module.linear.bias is not None else None,
                    'in_features': dense_module.linear.in_features,
                    'out_features': dense_module.linear.out_features,
                    'activation_function': activation_func,
                }, output_dir / 'dense_layer.pt')
                saved_dense = True
        except (IndexError, AttributeError):
            pass

    # 4. Save model config - mark as FP32
    print("   Saving model config...")
    embedding_dim = truncate_dim if truncate_dim else model.get_sentence_embedding_dimension()
    model_config = {
        'max_seq_length': model.max_seq_length,
        'embedding_dimension': embedding_dim,
        'model_type': 'onnx_fp32',  # FP32 type - CPU compatible
        'base_model': 'sentence-transformers/LaBSE',
    }
    if truncate_dim:
        model_config['matryoshka_dim'] = truncate_dim

    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    print(f"   Done: tokenizer, pooling_config.json, model_config.json" +
          (", dense_layer.pt" if saved_dense else ""))


def validate_conversion(fp32_model_path: str, reference_model_path: str, truncate_dim: int = None):
    """Validate FP32 model produces similar embeddings to PyTorch reference."""
    print(f"\nValidating FP32 model against PyTorch reference...")

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
    ]

    print(f"   Loading PyTorch reference model...")
    ref_model = SentenceTransformer(reference_model_path)

    print(f"   Loading FP32 ONNX model...")
    fp32_model = ONNXSentenceTransformer(fp32_model_path, providers=['CPUExecutionProvider'])

    print(f"   Generating embeddings for {len(test_texts)} test texts...")

    ref_embeddings = ref_model.encode(test_texts, normalize_embeddings=True)
    fp32_embeddings = fp32_model.encode(test_texts, normalize_embeddings=True)

    if truncate_dim and ref_embeddings.shape[1] > truncate_dim:
        ref_embeddings = ref_embeddings[:, :truncate_dim]
        ref_embeddings = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True)

    if ref_embeddings.shape != fp32_embeddings.shape:
        print(f"   Warning: Shape mismatch - ref: {ref_embeddings.shape}, fp32: {fp32_embeddings.shape}")
        min_dim = min(ref_embeddings.shape[1], fp32_embeddings.shape[1])
        ref_embeddings = ref_embeddings[:, :min_dim]
        fp32_embeddings = fp32_embeddings[:, :min_dim]

    cosine_sims = np.sum(ref_embeddings * fp32_embeddings, axis=1)
    max_diff = np.abs(ref_embeddings - fp32_embeddings).max()

    print(f"\n   Validation Results:")
    print(f"   {'='*40}")
    print(f"   Mean cosine similarity: {cosine_sims.mean():.6f}")
    print(f"   Min cosine similarity:  {cosine_sims.min():.6f}")
    print(f"   Max cosine similarity:  {cosine_sims.max():.6f}")
    print(f"   Max absolute diff:      {max_diff:.6f}")
    print(f"   {'='*40}")

    # FP32 should have very high similarity (near 1.0)
    passed = cosine_sims.min() > 0.99

    if passed:
        print(f"   PASSED: All cosine similarities > 0.99")
    else:
        print(f"   WARNING: Some cosine similarities < 0.99")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch SentenceTransformer to ONNX FP32 format"
    )

    parser.add_argument("--input", "-i", required=True,
                       help="Path to PyTorch SentenceTransformer model")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for FP32 model")
    parser.add_argument("--truncate_dim", type=int, default=None,
                       help="Matryoshka truncation dimension (e.g., 512)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate converted model against PyTorch reference")
    parser.add_argument("--num_heads", type=int, default=12,
                       help="Number of attention heads (default: 12 for BERT-base/LaBSE)")
    parser.add_argument("--hidden_size", type=int, default=768,
                       help="Hidden dimension (default: 768 for BERT-base/LaBSE)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input model not found: {input_path}")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ONNX FP32 Model Conversion")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    if args.truncate_dim:
        print(f"Truncation: {args.truncate_dim}d")
    print("=" * 60)

    # Step 1: Load PyTorch model
    model = load_pytorch_model(str(input_path))

    # Step 2: Truncate dense layer if requested
    dense_info = None
    if args.truncate_dim:
        model, dense_info = truncate_dense_layer(model, args.truncate_dim)

    # Step 3: Export to ONNX FP32
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        onnx_fp32_path = tmpdir / "model_fp32.onnx"
        export_to_onnx_fp32(model, str(onnx_fp32_path))

        # Step 4: Apply transformer optimizations (FP32)
        onnx_optimized_path = output_dir / "model.onnx"
        apply_transformer_optimizations(
            str(onnx_fp32_path),
            str(onnx_optimized_path),
            num_heads=args.num_heads,
            hidden_size=args.hidden_size
        )

    # Step 5: Save supporting files
    save_supporting_files(model, str(output_dir), dense_info, args.truncate_dim)

    # Step 6: Validate if requested
    if args.validate:
        validate_conversion(str(output_dir), str(input_path), args.truncate_dim)

    # Summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)

    onnx_size = (output_dir / "model.onnx").stat().st_size / (1024 * 1024)
    print(f"Output directory: {output_dir}")
    print(f"Model size: {onnx_size:.1f} MB")
    print(f"Model type: onnx_fp32")

    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size / 1024
            print(f"   {f.name}: {size:.1f} KB")
        elif f.is_dir():
            print(f"   {f.name}/")

    print("\nTo use this model:")
    print(f"   from onnx_sentence_transformer import load_onnx_model")
    print(f"   model = load_onnx_model('{output_dir}', providers=['CPUExecutionProvider'])")
    print(f"   embeddings = model.encode(['test text'])")

    return 0


if __name__ == "__main__":
    exit(main())
