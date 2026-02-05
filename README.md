# SzEmbeddings

Semantic vector embeddings integration with Senzing entity resolution engine. Enables cosine similarity searches on personal and business names using fine-tuned SentenceTransformer models stored in PostgreSQL with the pgvector extension.

## Quick Start

```bash
# 1. Source Senzing environment
source ~/senzingv4/setupEnv

# 2. Activate Python virtual environment
source venv/bin/activate

# 3. Set Senzing configuration (contains license and DB connection)
source secrets/DO_NOT_PUSH_sz_env_vars.sh

# 4. Verify setup
python test_setup.py --full
```

## Documentation

All detailed documentation is in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Setup guide, PostgreSQL configuration, Senzing config |
| [docs/TESTING.md](docs/TESTING.md) | Testing workflow, metrics, interpreting results |
| [docs/LOADING.md](docs/LOADING.md) | Loading data, troubleshooting, performance benchmarks |
| [docs/ONNX_MODELS.md](docs/ONNX_MODELS.md) | ONNX model evaluation (FP16, INT8) |
| [docs/POSTGRESQL_TUNING.md](docs/POSTGRESQL_TUNING.md) | PostgreSQL performance tuning |
| [docs/FILE_INVENTORY.md](docs/FILE_INVENTORY.md) | Complete file inventory and script reference |

### Results Documentation

| Document | Description |
|----------|-------------|
| [docs/results/GNR_ALIGNMENT_RESULTS.md](docs/results/GNR_ALIGNMENT_RESULTS.md) | GNR score alignment training results |
| [docs/results/E5_VALIDATION_RESULTS.md](docs/results/E5_VALIDATION_RESULTS.md) | E5 vs LaBSE model comparison |
| [docs/results/AAR_EXPERIMENT_RESULTS.md](docs/results/AAR_EXPERIMENT_RESULTS.md) | AAR experiment results (not recommended) |

## Model Formats

Use the packaged models from `~/999gz.git/name_model/` for production:

| Deployment | Format | Path | Notes |
|------------|--------|------|-------|
| **GPU Server** | ONNX FP16 | `~/999gz.git/name_model/*/onnx_fp16/` | 50% smaller, fast GPU inference |
| **CPU Server** | ONNX INT8 | `~/999gz.git/name_model/*/onnx_int8/` | 75% smaller, best CPU performance |
| **Development** | PyTorch FP32 | Dev repos | For debugging/fine-tuning |

**Important:** Each format produces embeddings in its own embedding space. Use the same model format for loading AND searching.

## Core Scripts

| Script | Purpose |
|--------|---------|
| `sz_load_embeddings.py` | Load records with PyTorch models |
| `sz_load_embeddings_onnx.py` | Load records with ONNX models |
| `sz_debug_search.py` | Interactive search with detailed analysis |
| `sz_validate_production.py` | Production recall validation |
| `sz_validate_with_variants.py` | Variant recall validation |
| `sz_evaluate_model.py` | Evaluate model on triplets |

See [CLAUDE.md](CLAUDE.md) for complete command reference and [docs/FILE_INVENTORY.md](docs/FILE_INVENTORY.md) for full file inventory.

## Related Projects

- **[name_model](~/999gz.git/name_model)** - Packaged ONNX models for production deployment
- **[PersonalNames](~/roncewind.git/PersonalNames)** - Personal names model training
- **[BizNames](~/roncewind.git/BizNames)** - Business names model training
