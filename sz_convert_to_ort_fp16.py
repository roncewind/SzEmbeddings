#!/usr/bin/env python
"""
Convert PyTorch SentenceTransformer models to optimized ONNX FP16 format.

This script creates optimized FP16 ONNX models with:
- Transformer-specific optimizations (attention fusion, GELU approximation, etc.)
- FP16 quantization for 50% storage reduction
- Compatible with CUDA execution on modern GPUs including Blackwell (RTX 50-series)

NOTE: ORT format (.ort) is NOT recommended for Blackwell GPUs (compute capability 12.0)
as it causes inference crashes. Use FP16 ONNX instead, which works correctly.

Pipeline:
    PyTorch SentenceTransformer
             |
        ONNX Export (FP32)
             |
        Transformer Optimization (attention fusion, etc.)
             |
        FP16 Quantization
             |
        [Optional] ORT Format Conversion
             |
        Package with Tokenizer + Configs

Usage:
    python sz_convert_to_ort_fp16.py \
        --input ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model \
        --output ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-onnx-fp16 \
        --truncate_dim 512 \
        --validate

    # Optionally also generate ORT format (not recommended for Blackwell GPUs)
    python sz_convert_to_ort_fp16.py \
        --input /path/to/model \
        --output /path/to/output \
        --ort
"""

import argparse
import json
import shutil
import tempfile
import time
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

    Args:
        model: SentenceTransformer model
        target_dim: Target output dimension (e.g., 512)

    Returns:
        Modified model with truncated dense layer
    """
    print(f"\nTruncating Dense layer to {target_dim}d (Matryoshka)...")

    # Find the Dense layer (usually module index 2)
    dense_layer = None
    dense_idx = None
    for idx, module in model._modules.items():
        if hasattr(module, 'linear') and hasattr(module.linear, 'weight'):
            dense_layer = module
            dense_idx = idx
            break

    if dense_layer is None:
        print("   No Dense layer found, skipping truncation")
        return model, None

    original_out = dense_layer.linear.out_features
    print(f"   Original: {dense_layer.linear.in_features} -> {original_out}")

    if target_dim >= original_out:
        print(f"   Target dim {target_dim} >= current {original_out}, skipping")
        return model, None

    # Get weights and truncate
    weight = dense_layer.linear.weight.data  # Shape: [out_features, in_features]
    bias = dense_layer.linear.bias.data if dense_layer.linear.bias is not None else None

    weight_truncated = weight[:target_dim, :].clone()
    bias_truncated = bias[:target_dim].clone() if bias is not None else None

    # Create new layer
    new_linear = nn.Linear(dense_layer.linear.in_features, target_dim, bias=(bias is not None))
    new_linear.weight.data = weight_truncated
    if bias_truncated is not None:
        new_linear.bias.data = bias_truncated

    # Replace in model
    dense_layer.linear = new_linear
    dense_layer.out_features = target_dim

    print(f"   Truncated: {dense_layer.linear.in_features} -> {target_dim}")

    # Return dense layer info for saving
    dense_info = {
        'weight': weight_truncated.cpu(),
        'bias': bias_truncated.cpu() if bias_truncated is not None else None,
        'in_features': dense_layer.linear.in_features,
        'out_features': target_dim,
        'original_out_features': original_out
    }

    return model, dense_info


def export_to_onnx_fp32(model, output_path: str, opset_version: int = 14):
    """
    Export transformer component to ONNX FP32 format.

    Exports in FP32 so that transformer optimization patterns can be
    recognized correctly. FP16 conversion happens during optimization.

    Args:
        model: SentenceTransformer model
        output_path: Path for ONNX file
        opset_version: ONNX opset version

    Returns:
        Path to exported ONNX file
    """
    print(f"\nExporting to ONNX FP32...")

    # Get transformer module
    transformer = model[0].auto_model
    config = transformer.config
    max_seq_length = model.max_seq_length

    print(f"   Model type: {config.model_type}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Max seq length: {max_seq_length}")

    # Use CPU for FP32 export (consistent behavior)
    device = torch.device('cpu')
    print(f"   Device: {device}")

    transformer = transformer.to(device)
    transformer.eval()

    # Create dummy inputs as tensors on same device
    batch_size = 1
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_seq_length), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(batch_size, max_seq_length, dtype=torch.long, device=device)

    # Build input tuple and names based on model config
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

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"   Exporting with opset version {opset_version}...")

    # Use torch.no_grad and legacy JIT-based exporter with embedded weights
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
            export_params=True,  # Embed weights in the model file
            dynamo=False,        # Use legacy JIT-based exporter (not dynamo)
            external_data=False, # Embed weights directly, not as external files
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Exported: {output_path} ({size_mb:.1f} MB)")

    # Verify
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"   Verified: ONNX model is valid")

    return output_path


def apply_transformer_optimizations(input_path: str, output_path: str, num_heads: int = 12, hidden_size: int = 768, convert_to_fp16: bool = True):
    """
    Apply BERT/transformer-specific optimizations to reduce Memcpy nodes.

    This uses ONNX Runtime's transformer optimizer which:
    - Fuses multi-head attention patterns
    - Optimizes for GPU execution (reduces CPU-GPU data transfers)
    - Applies GELU approximation
    - Fuses layer normalization
    - Eliminates redundant memory copies

    IMPORTANT: Optimizations are applied BEFORE FP16 conversion to ensure
    fusion patterns are recognized correctly (FP16 casts can break pattern matching).

    Args:
        input_path: Path to input ONNX model (FP32 or FP16)
        output_path: Path for optimized ONNX model
        num_heads: Number of attention heads (default 12 for BERT-base/LaBSE)
        hidden_size: Hidden dimension (default 768 for BERT-base/LaBSE)
        convert_to_fp16: Whether to convert to FP16 after optimization (default True)

    Returns:
        Path to optimized model
    """
    print(f"\nApplying transformer optimizations...")
    print(f"   Input: {input_path}")
    print(f"   Num heads: {num_heads}, Hidden size: {hidden_size}")
    print(f"   Convert to FP16: {convert_to_fp16}")

    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.fusion_options import FusionOptions

    # Configure fusion options for maximum optimization
    fusion_options = FusionOptions('bert')
    fusion_options.enable_attention = True
    fusion_options.enable_embed_layer_norm = True
    fusion_options.enable_skip_layer_norm = True
    fusion_options.enable_bias_skip_layer_norm = True
    fusion_options.enable_gelu_approximation = True
    fusion_options.enable_bias_gelu = True
    fusion_options.use_multi_head_attention = True  # Use MHA op for better GPU perf

    print(f"   Optimizing for GPU (use_gpu=True)...")

    # Optimize the model - use_gpu=True is critical for reducing Memcpy nodes
    optimized_model = optimizer.optimize_model(
        str(input_path),
        model_type='bert',
        num_heads=num_heads,
        hidden_size=hidden_size,
        opt_level=99,  # Maximum optimization
        optimization_options=fusion_options,
        use_gpu=True,  # Critical: optimize for GPU to reduce Memcpy
        only_onnxruntime=True,  # Use only ONNX Runtime ops
    )

    # Get optimization stats
    stats = optimized_model.get_fused_operator_statistics()
    if stats:
        print(f"   Fused operators:")
        for op_type, count in stats.items():
            if count > 0:  # Only show non-zero
                print(f"      {op_type}: {count}")

    # Convert to FP16 AFTER optimization (preserves fusion patterns)
    if convert_to_fp16:
        print(f"   Converting to FP16 (after optimization)...")
        optimized_model.convert_float_to_float16(
            keep_io_types=True,  # Keep I/O as FP32 for compatibility
            force_fp16_initializers=True,  # Convert weights to FP16
        )

    # Save optimized model
    output_path = Path(output_path)
    optimized_model.save_model_to_file(str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Optimized: {output_path} ({size_mb:.1f} MB)")

    return output_path


def convert_to_ort(input_path: str, output_dir: str, optimization_style: str = "Runtime", cuda_optimize_first: bool = True):
    """
    Convert ONNX model to ORT format.

    For Blackwell GPUs (sm_120), the model must be CUDA-optimized first to avoid
    NaN values during inference. This is done by loading with CUDA EP and saving
    the optimized model before ORT conversion.

    Args:
        input_path: Path to ONNX model
        output_dir: Output directory for ORT model
        optimization_style: "Fixed" or "Runtime" (Runtime recommended for Blackwell GPUs)
        cuda_optimize_first: If True, optimize for CUDA before ORT conversion (required for Blackwell)

    Returns:
        Path to ORT model file
    """
    import onnxruntime as ort

    print(f"\nConverting to ORT format (optimization: {optimization_style})...")

    from onnxruntime.tools.convert_onnx_models_to_ort import (
        convert_onnx_models_to_ort,
        OptimizationStyle
    )

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # For Blackwell GPUs, we need to CUDA-optimize the model first
    # This creates CUDA-specific graph optimizations that prevent NaN values
    onnx_to_convert = input_path
    if cuda_optimize_first:
        print("   CUDA-optimizing ONNX for Blackwell compatibility...")
        cuda_opt_path = output_dir / 'model_cuda_optimized.onnx'

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.optimized_model_filepath = str(cuda_opt_path)

        # Load with CUDA EP to trigger CUDA-specific optimizations
        session = ort.InferenceSession(
            str(input_path),
            sess_options=so,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        del session  # Close session, optimized model is saved

        print(f"   CUDA-optimized: {cuda_opt_path.name}")
        onnx_to_convert = cuda_opt_path

    # Map string to enum
    style = OptimizationStyle.Fixed if optimization_style == "Fixed" else OptimizationStyle.Runtime

    # Convert
    convert_onnx_models_to_ort(
        model_path_or_dir=onnx_to_convert,
        output_dir=output_dir,
        optimization_styles=[style],
        save_optimized_onnx_model=False,
        allow_conversion_failures=False
    )

    # Find the generated .ort file
    # Runtime style creates .with_runtime_opt.ort, Fixed creates .ort
    # Use onnx_to_convert.stem since that's what was actually converted
    if optimization_style == "Runtime":
        expected_name = onnx_to_convert.stem + '.with_runtime_opt.ort'
    else:
        expected_name = onnx_to_convert.stem + '.ort'

    ort_file = output_dir / expected_name
    if not ort_file.exists():
        # Try to find any .ort file
        ort_files = list(output_dir.glob('*.ort'))
        if ort_files:
            ort_file = ort_files[0]

    if ort_file.exists():
        size_mb = ort_file.stat().st_size / (1024 * 1024)
        print(f"   Converted: {ort_file.name} ({size_mb:.1f} MB)")

        # Rename to standard name
        final_path = output_dir / 'model.ort'
        if ort_file != final_path:
            shutil.move(str(ort_file), str(final_path))
            print(f"   Renamed to: {final_path.name}")
            ort_file = final_path
    else:
        raise RuntimeError(f"ORT conversion failed - no .ort file found in {output_dir}")

    # Clean up intermediate CUDA-optimized file (after ORT is found/renamed)
    if cuda_optimize_first and onnx_to_convert != input_path:
        cuda_opt_path.unlink(missing_ok=True)
        # Also clean up any .onnx.data external data file
        cuda_opt_data = output_dir / (cuda_opt_path.name + '.data')
        cuda_opt_data.unlink(missing_ok=True)

    return ort_file


def save_supporting_files(model, output_dir: str, dense_info: dict = None, truncate_dim: int = None, generate_ort: bool = False):
    """
    Save tokenizer, configs, and dense layer.

    Args:
        model: SentenceTransformer model
        output_dir: Output directory
        dense_info: Dense layer weights (if truncated) - overrides model's dense layer
        truncate_dim: Target dimension (if truncated)
        generate_ort: Whether ORT format was also generated
    """
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

    # 3. Save dense layer - ALWAYS save if model has one, or if dense_info provided
    saved_dense = False
    if dense_info is not None:
        # Use provided dense_info (from truncation)
        print("   Saving dense layer (truncated)...")
        torch.save({
            'weight': dense_info['weight'],
            'bias': dense_info['bias'],
            'in_features': dense_info['in_features'],
            'out_features': dense_info['out_features'],
        }, output_dir / 'dense_layer.pt')
        saved_dense = True

        # Also save as dense_512d.pt for compatibility
        if truncate_dim == 512:
            torch.save({
                'weight': dense_info['weight'],
                'bias': dense_info['bias'],
            }, output_dir / 'dense_512d.pt')
    else:
        # Check if model has a Dense layer (usually at index 2)
        try:
            dense_module = model[2]
            if hasattr(dense_module, 'linear'):
                print("   Saving dense layer (from model)...")
                # Get activation function if present
                activation_func = None
                if hasattr(dense_module, 'activation_function'):
                    activation_func = str(type(dense_module.activation_function).__name__)
                    print(f"      Activation function: {activation_func}")

                torch.save({
                    'weight': dense_module.linear.weight.data.cpu(),
                    'bias': dense_module.linear.bias.data.cpu() if dense_module.linear.bias is not None else None,
                    'in_features': dense_module.linear.in_features,
                    'out_features': dense_module.linear.out_features,
                    'activation_function': activation_func,
                }, output_dir / 'dense_layer.pt')
                saved_dense = True
        except (IndexError, AttributeError):
            pass  # No dense layer in model

    # 4. Save model config
    print("   Saving model config...")
    embedding_dim = truncate_dim if truncate_dim else model.get_sentence_embedding_dimension()
    model_config = {
        'max_seq_length': model.max_seq_length,
        'embedding_dimension': embedding_dim,
        'model_type': 'ort_fp16' if generate_ort else 'onnx_fp16',
        'base_model': 'sentence-transformers/LaBSE',
    }
    if truncate_dim:
        model_config['matryoshka_dim'] = truncate_dim

    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    print(f"   Done: tokenizer, pooling_config.json, model_config.json" +
          (", dense_layer.pt" if saved_dense else ""))


def validate_conversion(ort_model_path: str, reference_model_path: str, truncate_dim: int = None):
    """
    Validate ORT model produces similar embeddings to PyTorch reference.

    Args:
        ort_model_path: Path to ORT model directory
        reference_model_path: Path to PyTorch reference model
        truncate_dim: Dimension truncation (for Matryoshka)

    Returns:
        True if validation passes
    """
    print(f"\nValidating ORT model against PyTorch reference...")

    from sentence_transformers import SentenceTransformer

    # Import our ONNX wrapper (should support .ort files)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from onnx_sentence_transformer import ONNXSentenceTransformer

    # Test texts
    test_texts = [
        "Vladimir Putin",
        "Apple Inc.",
        "John Smith",
        "Gazprom",
        "Maria Garcia",
        "Sberbank",
        "北京大学",
        "Владимир Путин",
        "Deutsche Bank AG",
        "محمد علی"
    ]

    print(f"   Loading PyTorch reference model...")
    ref_model = SentenceTransformer(reference_model_path)

    print(f"   Loading ORT model...")
    ort_model = ONNXSentenceTransformer(ort_model_path)

    print(f"   Generating embeddings for {len(test_texts)} test texts...")

    # Generate embeddings
    ref_embeddings = ref_model.encode(test_texts, normalize_embeddings=True)
    ort_embeddings = ort_model.encode(test_texts, normalize_embeddings=True)

    # Apply truncation to reference if needed
    if truncate_dim and ref_embeddings.shape[1] > truncate_dim:
        ref_embeddings = ref_embeddings[:, :truncate_dim]
        # Re-normalize after truncation
        ref_embeddings = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True)

    # Ensure same dimension
    if ref_embeddings.shape != ort_embeddings.shape:
        print(f"   Warning: Shape mismatch - ref: {ref_embeddings.shape}, ort: {ort_embeddings.shape}")
        min_dim = min(ref_embeddings.shape[1], ort_embeddings.shape[1])
        ref_embeddings = ref_embeddings[:, :min_dim]
        ort_embeddings = ort_embeddings[:, :min_dim]

    # Calculate cosine similarity (embeddings are normalized)
    cosine_sims = np.sum(ref_embeddings * ort_embeddings, axis=1)

    # Calculate max absolute difference
    max_diff = np.abs(ref_embeddings - ort_embeddings).max()

    print(f"\n   Validation Results:")
    print(f"   {'='*40}")
    print(f"   Mean cosine similarity: {cosine_sims.mean():.6f}")
    print(f"   Min cosine similarity:  {cosine_sims.min():.6f}")
    print(f"   Max cosine similarity:  {cosine_sims.max():.6f}")
    print(f"   Max absolute diff:      {max_diff:.6f}")
    print(f"   {'='*40}")

    # Check pass criteria
    passed = cosine_sims.min() > 0.99

    if passed:
        print(f"   PASSED: All cosine similarities > 0.99")
    else:
        print(f"   FAILED: Some cosine similarities < 0.99")
        print(f"   This may be expected with FP16 quantization")

    return passed


def benchmark_load_time(model_path: str, n_trials: int = 3):
    """
    Benchmark model load time.

    Args:
        model_path: Path to model directory
        n_trials: Number of trials

    Returns:
        Average load time in seconds
    """
    print(f"\nBenchmarking load time ({n_trials} trials)...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from onnx_sentence_transformer import ONNXSentenceTransformer

    times = []
    for i in range(n_trials):
        start = time.time()
        model = ONNXSentenceTransformer(model_path)
        elapsed = time.time() - start
        times.append(elapsed)
        del model
        print(f"   Trial {i+1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    print(f"   Average: {avg_time:.3f}s")

    return avg_time


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch SentenceTransformer to optimized ONNX FP16 format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert personal names model to FP16 ONNX
  python sz_convert_to_ort_fp16.py \\
      --input ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model \\
      --output ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-onnx-fp16

  # Convert with Matryoshka truncation (512d)
  python sz_convert_to_ort_fp16.py \\
      --input ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \\
      --output ~/roncewind.git/BizNames/output/phase10_quantization/onnx_fp16_512d \\
      --truncate_dim 512 \\
      --validate

  # Also generate ORT format (not recommended for Blackwell/RTX 50-series GPUs)
  python sz_convert_to_ort_fp16.py \\
      --input /path/to/model \\
      --output /path/to/output \\
      --ort

  # Skip validation for faster conversion
  python sz_convert_to_ort_fp16.py \\
      --input /path/to/model \\
      --output /path/to/output \\
      --no-validate
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Path to PyTorch SentenceTransformer model')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for ONNX FP16 model')
    parser.add_argument('--truncate_dim', type=int, default=None,
                        help='Optional Matryoshka truncation dimension (e.g., 512)')
    parser.add_argument('--opset_version', type=int, default=14,
                        help='ONNX opset version (default: 14)')
    parser.add_argument('--ort', action='store_true',
                        help='Also generate ORT format')
    parser.add_argument('--optimization_style', choices=['Fixed', 'Runtime'], default='Runtime',
                        help='ORT optimization style if --ort is used (default: Runtime for Blackwell compatibility)')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads for transformer optimization (default: 12 for BERT/LaBSE)')
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='Hidden dimension for transformer optimization (default: 768 for BERT/LaBSE)')
    parser.add_argument('--validate', dest='validate', action='store_true', default=True,
                        help='Validate conversion against reference (default: True)')
    parser.add_argument('--no-validate', dest='validate', action='store_false',
                        help='Skip validation')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark load time after conversion')

    args = parser.parse_args()

    print("=" * 80)
    print("PYTORCH TO ONNX FP16 CONVERSION")
    print("=" * 80)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    if args.truncate_dim:
        print(f"Truncate to: {args.truncate_dim}d (Matryoshka)")
    print(f"Opset version: {args.opset_version}")
    print(f"Generate ORT: {args.ort}" + (" (not recommended for Blackwell GPUs)" if args.ort else ""))
    print(f"Validate: {args.validate}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Step 1: Load PyTorch model
        print("\n" + "=" * 80)
        print("STEP 1: LOAD PYTORCH MODEL")
        print("=" * 80)
        model = load_pytorch_model(args.input)

        # Step 2: Truncate if requested
        dense_info = None
        if args.truncate_dim:
            print("\n" + "=" * 80)
            print(f"STEP 2: TRUNCATE TO {args.truncate_dim}D (MATRYOSHKA)")
            print("=" * 80)
            model, dense_info = truncate_dense_layer(model, args.truncate_dim)

        # Step 3: Export to ONNX FP32 (patterns need FP32 for proper fusion)
        print("\n" + "=" * 80)
        print("STEP 3: EXPORT TO ONNX FP32")
        print("=" * 80)
        onnx_fp32_path = temp_dir / 'model_fp32.onnx'
        export_to_onnx_fp32(model, onnx_fp32_path, args.opset_version)

        # Step 4: Apply transformer optimizations + FP16 conversion
        # (Optimize first so fusion patterns are recognized, then convert to FP16)
        # Save directly to output directory as model.onnx
        print("\n" + "=" * 80)
        print("STEP 4: OPTIMIZE + CONVERT TO FP16")
        print("=" * 80)
        onnx_fp16_path = output_dir / 'model.onnx'
        apply_transformer_optimizations(
            onnx_fp32_path,
            onnx_fp16_path,
            num_heads=args.num_heads,
            hidden_size=args.hidden_size,
            convert_to_fp16=True
        )

        # Step 5: Optionally convert to ORT format
        if args.ort:
            print("\n" + "=" * 80)
            print("STEP 5: CONVERT TO ORT FORMAT (OPTIONAL)")
            print("=" * 80)
            if args.optimization_style == 'Fixed':
                print("   WARNING: 'Fixed' optimization may crash on Blackwell GPUs - use 'Runtime' instead")
            else:
                print(f"   Using '{args.optimization_style}' optimization (Blackwell compatible)")
            ort_path = convert_to_ort(onnx_fp16_path, output_dir, args.optimization_style)

        # Step 6: Save supporting files
        print("\n" + "=" * 80)
        print("STEP 5: SAVE SUPPORTING FILES" if not args.ort else "STEP 6: SAVE SUPPORTING FILES")
        print("=" * 80)
        save_supporting_files(model, output_dir, dense_info, args.truncate_dim, generate_ort=args.ort)

    # Step 7: Validate
    if args.validate:
        print("\n" + "=" * 80)
        print("STEP 7: VALIDATE CONVERSION")
        print("=" * 80)
        validate_conversion(args.output, args.input, args.truncate_dim)

    # Step 8: Benchmark
    if args.benchmark:
        print("\n" + "=" * 80)
        print("STEP 8: BENCHMARK LOAD TIME")
        print("=" * 80)
        benchmark_load_time(args.output)

    # Summary
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)

    # List output files
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.rglob('*')):
        if f.is_file():
            rel_path = f.relative_to(output_dir)
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} bytes"
            print(f"   {rel_path}: {size_str}")

    # Show model format info
    model_format = "ONNX FP16" + (f" + ORT ({args.optimization_style})" if args.ort else "")
    print(f"\nModel format: {model_format}")
    if args.ort:
        if args.optimization_style == "Runtime":
            print("   ORT format with Runtime optimization is compatible with Blackwell GPUs")
        else:
            print("   WARNING: ORT format with Fixed optimization may crash on Blackwell GPUs")

    print(f"""
Next steps:

1. Test with sz_debug_search.py:
   python sz_debug_search.py --type personal \\
       --model_path {args.output} \\
       --truncate_dim {args.truncate_dim or 768} \\
       "Vladimir Putin"

2. Load embeddings into Senzing:
   python sz_load_embeddings.py \\
       -i <data_file> \\
       --name_model_path {args.output} \\
       --truncate_dim {args.truncate_dim or 512}
""")

    return 0


if __name__ == '__main__':
    exit(main())
