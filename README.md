
# Setting up Senzing database for vector operations

Look at [Postgres README](README-postgres.md) for information about using PostgreSQL
for vector operations and some details about which ANN method to use.

## Embedding Models

This project uses fine-tuned LaBSE models for semantic name matching. Models are available in multiple formats:

| Deployment | Format | Location | Notes |
|------------|--------|----------|-------|
| **GPU Server** | ONNX FP16 | `~/999gz.git/name_model/*/onnx_fp16/` | 50% smaller, optimized for GPU |
| **CPU Server** | ONNX INT8 | `~/999gz.git/name_model/*/onnx_int8/` | 75% smaller, best variant accuracy |
| **Development** | PyTorch FP32 | PersonalNames/BizNames repos | For debugging/fine-tuning |

**Important:** Use the same model format for loading AND searching. See [ONNX_EVALUATION.md](ONNX_EVALUATION.md) for benchmark results and [CLAUDE.md](CLAUDE.md) for usage examples.

## References:

- https://senzing.com/docs/quickstart/quickstart_api/
- https://senzing.zendesk.com/hc/en-us/articles/360041965973-Setup-with-PostgreSQL some names changes for v4

## Postgres vector database

Install a Postgres vector database and import training data embeddings into it using a fine-tuned model.
Further testing can be done using cosine distance and cosine similarity implemented in the Postgres
vector database.

### Install PostgreSQL

```
sudo apt update
sudo apt install postgresql postgresql-contrib
```

#### Switch to the postgres user and create a new db user/pwd if needed

dbuser is the user your programs will use to access the database.
Grant privs as appropriate, but the dbuser needs to be able to read the tables.

```
sudo -i -u postgres
psql

CREATE USER <dbuser> WITH PASSWORD '<dbpassword>';
CREATE DATABASE <database> OWNER <dbuser>;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO <dbuser>;

\q
```

#### Exit out to sudo user

```
exit
sudo apt install postgresql-server-dev-all
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### Launch psql again

```
sudo -u postgres psql -d <database>
CREATE EXTENSION vector;
\q
```

#### give linux user permissions... just an example, optional

```
sudo -u postgres psql
ALTER USER username WITH CREATEDB;
```

#### Postgresql performance tweaks, optional

Check how much shared memory is defined in the `postgresql.conf` file. Rule of thumb is
set it to 1/3 the amount of memory desired to be used by Postgresql.
Hint: by default it's located at `/etc/postgresql/16/main/postgresql.conf`, depending on Postgresql version.

#### Add Senzing schema

```
psql -U <dbuser> -d <database> -h <server> -W
\i <senzing_project_path>/resources/schema/szcore-schema-postgresql-create.sql
```
---

#### Create tables and indexes for embeddings:

The table name will be used as the attribute name in the Senzing configuration below.
The size of the `EMBEDDING VECTOR(??)` is determined by the chosen model. In this case,
the SentenceTransformer model used has a 512 embedding dimension.

Creating a new table and configuration for business embeddings:
```
CREATE TABLE BIZNAME_EMBEDDING (LIB_FEAT_ID BIGINT NOT NULL, LABEL VARCHAR(300) NOT NULL, EMBEDDING VECTOR(512), PRIMARY KEY(LIB_FEAT_ID));
CREATE INDEX ON BIZNAME_EMBEDDING USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);
SET hnsw.ef_search = 100;
 --or--
CREATE INDEX ON BIZNAME_EMBEDDING USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
SET ivfflat.probes = 10;
```

Creating a new table and configuration for personal name embeddings
```
CREATE TABLE NAME_EMBEDDING (LIB_FEAT_ID BIGINT NOT NULL, LABEL VARCHAR(300) NOT NULL, EMBEDDING VECTOR(512), PRIMARY KEY(LIB_FEAT_ID));
CREATE INDEX ON NAME_EMBEDDING USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);
SET hnsw.ef_search = 100;
 --or--
CREATE INDEX ON NAME_EMBEDDING USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
SET ivfflat.probes = 10;
```

#### Setup Senzing env

```
/opt/senzing/er/bin/sz_create_project <senzing_project_path>
cd <senzing_project_path>
source setupEnv
```

#### Senzing config: add artifacts for embeddings

- initial config file can be found at `resources/templates/g2config.json`
- add features and attributes for both Personal names and Business names.
- modify the scoring a bit

```
./bin/sz_configtool
addFeature {"feature": "NAME_EMBEDDING", "class": "ISSUED_ID", "behavior": "NAME", "anonymize": "No", "candidates": "Yes", "standardize": "", "expression": "", "comparison": "SEMANTIC_SIMILARITY_COMP", "matchKey": "Yes", "version": 1, "elementList": [{"element": "EMBEDDING", "expressed": "No", "compared": "Yes", "derived": "Yes", "display": "No"}, {"element": "ALGORITHM", "expressed": "No", "compared": "Yes", "derived": "No", "display": "No"}, {"element": "LABEL", "expressed": "No", "compared": "No", "derived": "No", "display": "Yes"}]}
addFeature {"feature": "BIZNAME_EMBEDDING", "class": "ISSUED_ID", "behavior": "NAME", "anonymize": "No", "candidates": "Yes", "standardize": "", "expression": "", "comparison": "SEMANTIC_SIMILARITY_COMP", "matchKey": "Yes", "version": 1, "elementList": [{"element": "EMBEDDING", "expressed": "No", "compared": "Yes", "derived": "Yes", "display": "No"}, {"element": "ALGORITHM", "expressed": "No", "compared": "Yes", "derived": "No", "display": "No"}, {"element": "LABEL", "expressed": "No", "compared": "No", "derived": "No", "display": "Yes"}]}

addAttribute {"attribute": "NAME_EMBEDDING", "class": "IDENTIFIER", "feature": "NAME_EMBEDDING", "element": "EMBEDDING", "required": "Yes", "default": null, "internal": "No"}
addAttribute {"attribute": "NAME_ALGORITHM", "class": "IDENTIFIER", "feature": "NAME_EMBEDDING", "element": "ALGORITHM", "required": "No", "default": null, "internal": "Yes"}
addAttribute {"attribute": "NAME_LABEL", "class": "IDENTIFIER", "feature": "NAME_EMBEDDING", "element": "LABEL", "required": "Yes", "default": null, "internal": "No"}

addAttribute {"attribute": "BIZNAME_EMBEDDING", "class": "IDENTIFIER", "feature": "BIZNAME_EMBEDDING", "element": "EMBEDDING", "required": "Yes", "default": null, "internal": "No"}
addAttribute {"attribute": "BIZNAME_ALGORITHM", "class": "IDENTIFIER", "feature": "BIZNAME_EMBEDDING", "element": "ALGORITHM", "required": "No", "default": null, "internal": "Yes"}
addAttribute {"attribute": "BIZNAME_LABEL", "class": "IDENTIFIER", "feature": "BIZNAME_EMBEDDING", "element": "LABEL", "required": "Yes", "default": null, "internal": "No"}


setFragment {"id": 11, "fragment": "SAME_NAME", "source": "./SUMMARY/BEHAVIOR/NAME[./SAME > 0] | ./FRAGMENT[./GNR_SAME_NAME>0]"}
setFragment {"id": 12, "fragment": "CLOSE_NAME", "source": "./SUMMARY/BEHAVIOR/NAME[sum(./SAME | ./CLOSE) > 0] | ./FRAGMENT[./GNR_CLOSE_NAME>0]"}

save

importFromFile <config filename>

```


#### Senzing config: add data sources as needed

- add the OPEN_SANCTIONS or ICIJ datasource as an example
- filename is the name of the config file edited above
- Note: in `./bin/sz_configtool` still

```
addDataSource OPEN_SANCTIONS
addDataSource ICIJ
listDataSources
save

```

#### Senzing config: set the Tau threshold for each embedding model

- Multiply Tau by 100 and set that as the "likelyScore", distribute the rest as
needed.
- Look at a similarity distribution histogram (or other method) and use it to inform score decisions.
- Note: scores are integers so tau will be need to be rounded to two decimal places.
- Note: in `./bin/sz_configtool` still

```

addComparisonThreshold {"function": "SEMANTIC_SIMILARITY_COMP", "returnOrder": 1, "scoreName": "FULL_SCORE", "feature": "BIZNAME_EMBEDDING", "sameScore": 80, "closeScore": 60, "likelyScore": 43, "plausibleScore": 30, "unlikelyScore": 20}

addComparisonThreshold {"function": "SEMANTIC_SIMILARITY_COMP", "returnOrder": 1, "scoreName": "FULL_SCORE", "feature": "NAME_EMBEDDING", "sameScore": 80, "closeScore": 50, "likelyScore": 30, "plausibleScore": 20, "unlikelyScore": 10}

listComparisonThresholds
save
quit

```

#### Data for Senzing

The loader here uses the Senzing JSONL format. In this data there are two
"RECORD_TYPES": ORGANIZATION and PERSON. Different models are used to create each
embedding. Senzing has been configured with two tables and separate features and
attribues to capture the data and embeddings.

As such, in Senzing JSON, there are fields based on our configuration above,
particularly the attribute names.

- BIZNAME_EMBEDDING: used for Business name embeddings
- BIZNAME_LABEL: used for Business name label
- NAME_EMBEDDING: used for Personal name embeddings
- NAME_LABEL: used for Personal name label


Organization record:
```
{
  "DATA_SOURCE": "TEST",
  "RECORD_ID": "1A",
  "RECORD_TYPE":"ORGANIZATION",
  "NAMES":[{"NAME_TYPE":"PRIMARY","NAME_ORG": "Alice Eng."},{{"NAME_TYPE":"ALIAS","NAME_ORG": "Alice Engineering Inc."}}]
  "PHONE_NUMBER": "+15551212",
  "BIZNAME_EMBEDDINGS": [{"BIZNAME_LABEL": "Alice Eng.", "BIZNAME_EMBEDDING": "[-0.021743419,...]"}, {"BIZNAME_LABEL": "Alice Engineering Inc.", "BIZNAME_EMBEDDING": "[0.521743123,...]"}, ...]
}
```

Person record:
```
{
  "DATA_SOURCE": "TEST",
  "RECORD_ID": "1A",
  "RECORD_TYPE":"PERSON",
  "NAMES":[{"NAME_TYPE":"PRIMARY","NAME_FULL": "Doctor Who"},{{"NAME_TYPE":"ALIAS","NAME_FULL": "The Doctor"}}]
  "PHONE_NUMBER": "+15551212",
  "NAME_EMBEDDINGS": [{"NAME_LABEL": "Doctor Who", "NAME_EMBEDDING": "[-0.021743419,...]"}, {"NAME_LABEL": "The Doctor", "NAME_EMBEDDING": "[0.521743123,...]"}, ...]
}
```

## Usage Examples

All programs require:
1. The `SENZING_ENGINE_CONFIGURATION_JSON` environment variable must be set with an appropriate license
2. Fine-tuned SentenceTransformer models for personal and business names
3. Senzing environment sourced: `source ~/senzingv4/setupEnv`
4. Python virtual environment activated: `source venv/bin/activate`

### Data Loading

**Load embeddings into Senzing:**
```bash
# Small test load (500 records)
python sz_load_embeddings.py \
  -i data/test_samples/opensanctions_test_500.jsonl \
  --name_model_path ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model \
  --biz_model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --truncate_dim 512 \
  --threads 24 \
  2> load_500_stderr.log
```

### Interactive Search & Debugging

**Search by name (interactive):**
```bash
# Search for a business
python sz_search_embeddings.py --type business \
  --model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  "Puget Sound Energy"

# Search for a person
python sz_search_embeddings.py --type personal \
  --model_path ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model \
  "José Luis Sanz Ruiz"
```

**Debug search with comprehensive analysis:**
```bash
# Full analysis showing all search modes
python sz_debug_search.py "Carlyle" --type business --top 20

# Test with abbreviation (shows embedding value)
python sz_debug_search.py "PSE" --type business

# Cross-script search (Latin query for Cyrillic/Arabic names)
python sz_debug_search.py "Alexander Makarov" --type personal

# Verbose output with full JSON
python sz_debug_search.py "Puget Sound Energy" --type business --verbose
```

The debug tool shows 6 sections:
- 0️⃣ Name-only search (baseline - traditional matching)
- 1️⃣ Embedding-only search (semantic similarity without name)
- 2️⃣ PostgreSQL cosine similarity (direct embedding search)
- 3️⃣ Candidate keys (what embeddings retrieved)
- 4️⃣ Final ranked entities (name + embedding search)
- 5️⃣ Analysis (comparison and diagnostics)

### Production Validation

**Extract test samples from loaded data:**
```bash
python sz_extract_validation_samples.py \
  --input data/test_samples/opensanctions_test_500.jsonl \
  --output data/test_samples/validation_samples_100.jsonl \
  --sample_size 100 \
  --filter both \
  --seed 42
```

**Run production validation:**
```bash
# Basic validation (Senzing only)
python sz_validate_production.py \
  --input data/test_samples/validation_samples_100.jsonl \
  --name_model_path ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model \
  --biz_model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --truncate_dim 512 \
  --output results/validation_results.json

# With PostgreSQL embedding validation
python sz_validate_production.py \
  --input data/test_samples/validation_samples_100.jsonl \
  --name_model_path ~/roncewind.git/PersonalNames/output/labse_finetuned/Epoch-000-fine_tuned_model \
  --biz_model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --validate_pg \
  --pg_database senzing \
  --output results/validation_results.json
```

**Compare validation runs:**
```bash
python sz_compare_validations.py results/validation_*.json
python sz_compare_validations.py -o results/comparison.txt results/validation_*.json
```

### Model Evaluation

**Evaluate model with test triplets:**
```bash
# Quick test (100 triplets)
python sz_evaluate_model.py \
  --type business \
  --model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --triplets ~/roncewind.git/BizNames/output/opensanctions_test_triplets.jsonl \
  --test_set opensanctions \
  --sample 100 \
  --output results/quick_test.json

# Full evaluation
python sz_evaluate_model.py \
  --type business \
  --model_path ~/roncewind.git/BizNames/output/phase9b_labse/Epoch-001-fine_tuned_model \
  --triplets ~/roncewind.git/BizNames/output/opensanctions_test_triplets.jsonl \
  --test_set opensanctions \
  --output results/opensanctions_business.json
```

**Compare model results:**
```bash
python sz_compare_models.py results/*.json
python sz_compare_models.py -o comparison_report.txt results/*.json
```

### Utility Tools

**Extract aliases from data:**
```bash
python sz_extract_aliases.py -i /data/OpenSanctions/senzing.json -o aliases.jsonl
```

**Sample data for testing:**
```bash
python sz_sample_data.py -i input_file.jsonl -o output_file.jsonl --sample_size 500
```

**Cross-validate Senzing vs PostgreSQL:**
```bash
python sz_cross_validate.py \
  --pg_biz_db embeddings_db \
  --pg_names_db personalnames_db \
  --output cross_validation_report.txt
```

### Test Data Examples

**Good test cases for embeddings:**

*Abbreviations (name-only fails, embeddings succeed):*
- "Carlyle" → finds "The Carlyle Group"
- "PSE" → finds "Puget Sound Energy"
- "J Sanz" → finds "José Luis Sanz Ruiz"

*Cross-script queries (semantic matching across scripts):*
- "Alexander Makarov" → finds "Александр Сергеевич Макаров" (Cyrillic)
- "Ucha Surmanidze" → finds "أوتشا سورمانيدزه" (Arabic)
- "China Power" → finds "中国电建集团" (Chinese)

## Troubleshooting

**Check database status:**
```bash
psql -d G2 -c "SELECT COUNT(*) as records,
  (SELECT COUNT(*) FROM bizname_embedding) as biz_embeddings,
  (SELECT COUNT(*) FROM name_embedding) as name_embeddings
FROM dsrc_record;"
```

**Check for database locks:**
```bash
psql -d G2 -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

**Monitor load progress:**
```bash
tail -f load.log
```

**Test environment setup:**
```bash
python test_setup.py         # Basic tests
python test_setup.py --full  # Includes embedding test
```

## Documentation

- **[TESTING.md](TESTING.md)** - Comprehensive testing workflow and metrics
- **[LOAD_INSTRUCTIONS.md](LOAD_INSTRUCTIONS.md)** - Detailed loading instructions
- **[POSTGRESQL_TUNING.md](POSTGRESQL_TUNING.md)** - PostgreSQL performance tuning
- **[FINDINGS_SUMMARY.md](FINDINGS_SUMMARY.md)** - Investigation findings
- **[CLAUDE.md](CLAUDE.md)** - Quick reference for common tasks

#### Helpful selects

```
SELECT COUNT(*) as records, (SELECT COUNT(*) FROM bizname_embedding) as biz_emb, (SELECT COUNT(*) FROM name_embedding) as name_emb FROM dsrc_record;
```

# TODO

- Select senzing uses for cosine similarity:
```
SELECT LIB_FEAT_ID FROM BIZNAME_EMBEDDING WHERE 1-(EMBEDDING <=> $1) > 0.43 ORDER BY EMBEDDING <=> $2,LIB_FEAT_ID ASC LIMIT 100
```
Curious about where the 0.43 comes from?