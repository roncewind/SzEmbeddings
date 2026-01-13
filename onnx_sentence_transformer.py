#!/usr/bin/env python
"""
ONNX wrapper for sentence-transformers models.

Provides a compatible API with SentenceTransformer for ONNX models.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Union

import torch
import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXSentenceTransformer:
    """
    Wrapper class for ONNX sentence-transformer models.

    Provides the same .encode() API as SentenceTransformer.
    """

    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Load ONNX model and tokenizer.

        Args:
            model_path: Path to directory containing model.onnx and configs
            providers: ONNX Runtime providers (default: ['CPUExecutionProvider'])
        """
        self.model_path = Path(model_path)

        # Default to CPU
        if providers is None:
            providers = ['CPUExecutionProvider']

        # Load ONNX model - try both naming conventions
        onnx_file = self.model_path / 'model.onnx'
        if not onnx_file.exists():
            # Try alternate name used by some models
            onnx_file = self.model_path / 'transformer_fp16.onnx'
            if not onnx_file.exists():
                raise FileNotFoundError(f"ONNX model not found: expected 'model.onnx' or 'transformer_fp16.onnx' in {self.model_path}")

        # For FP16 models, prefer CUDA if available
        try:
            import onnx
            model_proto = onnx.load(str(onnx_file))
            has_fp16 = any(init.data_type == 10 for init in model_proto.graph.initializer)  # 10 = FLOAT16
            if has_fp16 and 'CUDAExecutionProvider' not in providers:
                print(f"   ⚠️  FP16 model detected, adding CUDAExecutionProvider for better support")
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        except:
            pass

        self.session = ort.InferenceSession(str(onnx_file), providers=providers)

        # Load tokenizer - may be in subdirectory
        tokenizer_path = self.model_path / 'tokenizer'
        if tokenizer_path.exists() and tokenizer_path.is_dir():
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        # Load configs - try both naming conventions
        model_config_file = self.model_path / 'model_config.json'
        sentence_config_file = self.model_path / 'sentence_transformers_config.json'

        if model_config_file.exists():
            # Business model format
            with open(model_config_file, 'r') as f:
                self.config = json.load(f)
            self.max_seq_length = self.config['max_seq_length']
            self.embedding_dimension = self.config['embedding_dimension']
        elif sentence_config_file.exists():
            # Personal names model format
            with open(sentence_config_file, 'r') as f:
                sent_config = json.load(f)
            self.max_seq_length = sent_config['max_seq_length']
            self.embedding_dimension = sent_config['embedding_dimension']
            # Create compatible config dict
            self.config = {
                'max_seq_length': self.max_seq_length,
                'embedding_dimension': self.embedding_dimension
            }
        else:
            raise FileNotFoundError(f"Model config not found: expected 'model_config.json' or 'sentence_transformers_config.json' in {self.model_path}")

        # Load pooling config - may not exist for all models
        pooling_config_file = self.model_path / 'pooling_config.json'
        if pooling_config_file.exists():
            with open(pooling_config_file, 'r') as f:
                self.pooling_config = json.load(f)
        elif sentence_config_file.exists():
            # Extract pooling config from sentence_transformers_config
            with open(sentence_config_file, 'r') as f:
                sent_config = json.load(f)
            # Find pooling module
            pooling_module = None
            for module in sent_config.get('modules', []):
                if module['type'] == 'Pooling':
                    pooling_module = module['config']
                    break
            if pooling_module:
                self.pooling_config = pooling_module
            else:
                # Default to CLS pooling
                self.pooling_config = {'pooling_mode_cls_token': True}
        else:
            # Default to CLS pooling
            self.pooling_config = {'pooling_mode_cls_token': True}

        # Get input names from ONNX model
        self.input_names = [inp.name for inp in self.session.get_inputs()]

        # Check for Dense layer (may be for Matryoshka truncation or post-processing)
        # Try multiple naming conventions
        dense_file = None
        for dense_name in ['dense_512d.pt', 'dense_layer.pt']:
            candidate = self.model_path / dense_name
            if candidate.exists():
                dense_file = candidate
                break

        if dense_file is not None:
            print(f"   📦 Loading Dense layer from {dense_file.name}...")
            dense_state = torch.load(dense_file, map_location='cpu', weights_only=False)
            # Handle tensors that may require grad
            weight = dense_state['weight']
            self.dense_weight = weight.detach().numpy() if weight.requires_grad else weight.numpy()
            if dense_state.get('bias') is not None:
                bias = dense_state['bias']
                self.dense_bias = bias.detach().numpy() if bias.requires_grad else bias.numpy()
            else:
                self.dense_bias = None
            self.dense_in_features = dense_state.get('in_features', self.dense_weight.shape[1])
            self.dense_out_features = dense_state.get('out_features', self.dense_weight.shape[0])
            self.dense_activation = dense_state.get('activation_function', None)
            print(f"      ✅ Loaded Dense layer: {self.dense_in_features} → {self.dense_out_features}")
            if self.dense_activation:
                print(f"      📌 Activation: {self.dense_activation}")
        else:
            self.dense_weight = None
            self.dense_bias = None
            self.dense_activation = None

    def get_sentence_embedding_dimension(self):
        """Return embedding dimension (compatible with SentenceTransformer API)."""
        return self.embedding_dimension

    def _mean_pooling(self, hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Apply mean pooling to hidden states.

        Args:
            hidden_states: (batch_size, seq_length, hidden_dim)
            attention_mask: (batch_size, seq_length)

        Returns:
            pooled: (batch_size, hidden_dim)
        """
        # Expand attention mask to match hidden states shape
        attention_mask_expanded = np.expand_dims(attention_mask, -1)  # (batch, seq, 1)
        attention_mask_expanded = attention_mask_expanded.astype(hidden_states.dtype)

        # Apply mask and sum
        sum_embeddings = np.sum(hidden_states * attention_mask_expanded, axis=1)  # (batch, hidden)
        sum_mask = np.sum(attention_mask_expanded, axis=1)  # (batch, 1)

        # Avoid division by zero
        sum_mask = np.clip(sum_mask, 1e-9, None)

        # Mean pooling
        pooled = sum_embeddings / sum_mask

        return pooled

    def _cls_pooling(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Use CLS token (first token) as sentence embedding.

        Args:
            hidden_states: (batch_size, seq_length, hidden_dim)

        Returns:
            pooled: (batch_size, hidden_dim)
        """
        return hidden_states[:, 0, :]

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize embeddings to unit vectors.

        Args:
            embeddings: (batch_size, hidden_dim)

        Returns:
            normalized: (batch_size, hidden_dim)
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)  # Avoid division by zero
        return embeddings / norms

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = True,  # Always returns numpy, for API compatibility
        normalize_embeddings: bool = True,
        device: str = None  # Ignored, for API compatibility
    ) -> np.ndarray:
        """
        Encode sentences to embeddings.

        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_tensor: If True, return torch tensor (not implemented)
            normalize_embeddings: Whether to L2-normalize embeddings
            device: Device (ignored, ONNX uses providers set at init)

        Returns:
            embeddings: numpy array of shape (num_sentences, embedding_dim)
        """
        # Handle single sentence
        single_sentence = False
        if isinstance(sentences, str):
            sentences = [sentences]
            single_sentence = True

        all_embeddings = []

        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors='np'  # Return numpy arrays
            )

            # Prepare ONNX inputs
            onnx_inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
            }

            # Add token_type_ids if model expects it
            if 'token_type_ids' in self.input_names and 'token_type_ids' in inputs:
                onnx_inputs['token_type_ids'] = inputs['token_type_ids']

            # Run ONNX inference
            outputs = self.session.run(None, onnx_inputs)
            hidden_states = outputs[0]  # last_hidden_state

            # Apply pooling
            if self.pooling_config.get('pooling_mode_cls_token', False):
                # CLS pooling
                pooled = self._cls_pooling(hidden_states)
            elif self.pooling_config.get('pooling_mode_mean_tokens', False):
                # Mean pooling
                pooled = self._mean_pooling(hidden_states, inputs['attention_mask'])
            else:
                # Default to mean pooling
                pooled = self._mean_pooling(hidden_states, inputs['attention_mask'])

            # Apply Dense layer if present (may be for Matryoshka truncation or post-processing)
            if self.dense_weight is not None:
                # Apply: output = pooled @ dense_weight.T + dense_bias
                pooled = np.matmul(pooled, self.dense_weight.T)
                if self.dense_bias is not None:
                    pooled = pooled + self.dense_bias  # Broadcasting
                # Apply activation function if specified
                if self.dense_activation and 'Tanh' in str(self.dense_activation):
                    pooled = np.tanh(pooled)

            # Normalize if requested
            if normalize_embeddings:
                pooled = self._normalize_embeddings(pooled)

            all_embeddings.append(pooled)

        # Concatenate batches
        embeddings = np.vstack(all_embeddings)

        # Return single embedding if input was single sentence
        if single_sentence:
            embeddings = embeddings[0]

        # Convert to tensor if requested (not implemented, just return numpy)
        if convert_to_tensor:
            import torch
            embeddings = torch.from_numpy(embeddings)

        return embeddings


def load_onnx_model(model_path: str, providers: List[str] = None):
    """
    Load ONNX sentence-transformer model.

    Args:
        model_path: Path to directory containing model.onnx
        providers: ONNX Runtime providers (default: CPUExecutionProvider)

    Returns:
        ONNXSentenceTransformer instance
    """
    return ONNXSentenceTransformer(model_path, providers=providers)


if __name__ == '__main__':
    # Simple test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python onnx_sentence_transformer.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    print(f"Loading ONNX model from: {model_path}")
    model = load_onnx_model(model_path)

    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}d")
    print(f"Max sequence length: {model.max_seq_length}")

    # Test encoding
    test_texts = ["Toyota Motor Corporation", "トヨタ自動車", "Microsoft"]
    print(f"\nEncoding {len(test_texts)} test texts...")

    embeddings = model.encode(test_texts, normalize_embeddings=True)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding norms: {np.linalg.norm(embeddings, axis=1)}")
    print(f"\n✅ ONNX model working correctly!")
