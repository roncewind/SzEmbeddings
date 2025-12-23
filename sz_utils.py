# -----------------------------------------------------------------------------
# Shared utilities for Senzing embedding scripts.
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer


def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure and return a logger for the calling module."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    return logging.getLogger(__name__)


def format_seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def get_senzing_config() -> str:
    """
    Get Senzing configuration from environment variable.
    Exits with error if not set.
    """
    settings = os.getenv("SENZING_ENGINE_CONFIGURATION_JSON")
    if not settings:
        logging.error(
            "SENZING_ENGINE_CONFIGURATION_JSON environment variable not set. "
            "Please set this variable with your Senzing configuration."
        )
        sys.exit(1)
    return settings


def get_embedding(
    name: str, model: SentenceTransformer, truncate_dim: int | None = None
) -> npt.NDArray[np.float16]:
    """
    Create embedding for a single name.

    Args:
        name: Text to embed
        model: SentenceTransformer model
        truncate_dim: Optional Matryoshka truncation dimension (e.g., 512)

    Returns:
        Embedding vector as float16 numpy array
    """
    embedding = model.encode(
        [name],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Apply Matryoshka truncation if specified
    if truncate_dim is not None and truncate_dim < embedding.shape[1]:
        embedding = embedding[:, :truncate_dim]
        # Re-normalize to unit length after truncation
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    return embedding[0].astype(np.float16, copy=False)


def get_embeddings(
    names: list[str], model: SentenceTransformer, batch_size: int, truncate_dim: int | None = None
) -> npt.NDArray[np.float16]:
    """
    Create embeddings for all names in the list (batch mode).

    Args:
        names: List of texts to embed
        model: SentenceTransformer model
        batch_size: Batch size for encoding
        truncate_dim: Optional Matryoshka truncation dimension (e.g., 512)

    Returns:
        Embedding matrix as float16 numpy array
    """
    embeddings = model.encode(
        names,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Apply Matryoshka truncation if specified
    if truncate_dim is not None and truncate_dim < embeddings.shape[1]:
        embeddings = embeddings[:, :truncate_dim]
        # Re-normalize to unit length after truncation
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings.astype(np.float16, copy=False)
