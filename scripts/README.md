# KG-R1 Data Setup Scripts

This directory contains scripts for setting up and processing knowledge graph datasets for KG-R1 training.

## Quick Start

The CWQ and WebQSP replication flow is:

```bash
cd ~/KG-R1
bash scripts/setup_data_kg.sh
wget -O freebase-rdf-latest.gz \
  http://commondatastorage.googleapis.com/freebase-public/rdf/freebase-rdf-latest.gz
bash scripts/process_entitites_freebase.sh
python scripts/convert_entities.py
python scripts/data_process_kg/cwq_search_augmented_initial_entities.py
python scripts/data_process_kg/webqsp_search_augmented_initial_entities.py
```

When prompted by `setup_data_kg.sh`, choose the option that downloads all data.

## Directory Structure

```
scripts/
├── setup_data_kg.sh                    # Master setup script (START HERE)
├── download_kg.py                      # Download Freebase KG data
├── convert_entities.py                 # Convert entity formats
├── process_entitites_freebase.sh       # Process Freebase entities
├── html_logger.py                      # Logging utilities
│
├── data_process_kg/                    # Core dataset processing
│   ├── cwq.py                          # Process ComplexWebQuestions
│   ├── cwq_search_augmented_initial_entities.py
│   ├── webqsp.py                       # Process WebQuestionsSP
│   └── webqsp_search_augmented_initial_entities.py
│
├── data_multitq_kg/                    # MultiTQ temporal QA dataset
│   ├── setup_multitq.sh                # Complete MultiTQ setup
│   ├── download_multitq.py             # Download MultiTQ data
│   ├── process_multitq.py              # Process for KG-R1
│   ├── multitq_search_augmented_initial_entities.py
│   ├── filter_multitq_2k.py
│   ├── integration_example.sh
│   └── README.md
│
└── webqsp_kg/                          # WebQSP specific scripts
    ├── setup_webqsp_kg.sh
    ├── data_process.sh
    ├── train_grpo.sh
    ├── train_ppo.sh
    ├── evaluate.sh
    └── README.md
```

## Available Datasets

### 1. ComplexWebQuestions (CWQ)

Multi-hop question answering over Freebase knowledge graph.

**Setup:**
```bash
bash scripts/setup_data_kg.sh
bash scripts/process_entitites_freebase.sh
python scripts/convert_entities.py
python scripts/data_process_kg/cwq_search_augmented_initial_entities.py
```

**Requirements:**
- Raw CWQ data in `data_kg/CWQ/`
- Freebase entities and relations

**Output:**
- `data_kg/cwq_search_augmented_initial_entities/train.parquet`
- `data_kg/cwq_search_augmented_initial_entities/test.parquet`

### 2. WebQuestionsSP (WebQSP)

Single-hop and simple multi-hop questions over Freebase.

**Setup:**
```bash
bash scripts/setup_data_kg.sh
bash scripts/process_entitites_freebase.sh
python scripts/convert_entities.py
python scripts/data_process_kg/webqsp_search_augmented_initial_entities.py
```

**Requirements:**
- Raw WebQSP data in `data_kg/webqsp/`
- Freebase entities and relations

**Output:**
- `data_kg/webqsp_search_augmented_initial_entities/train.parquet`
- `data_kg/webqsp_search_augmented_initial_entities/test.parquet`

### 3. MultiTQ (Temporal Multi-hop QA)

Multi-hop temporal reasoning questions with time constraints.

**Setup:**
```bash
# Option 1: Use master script
bash scripts/setup_data_kg.sh
# Then select option 3

# Option 2: Run MultiTQ setup directly
cd scripts/data_multitq_kg
bash setup_multitq.sh
```

**What it does:**
1. Downloads MultiTQ dataset from GitHub
2. Processes for KG-R1 compatibility
3. Creates search-augmented training data with initial entities

**Output:**
- `data_multitq_kg/MultiTQ/` - Raw data
- `data_kg/multitq/` - Processed data
- `data_kg/multitq_search_augmented_initial_entities/` - Training data

**Features:**
- Temporal reasoning with multi-granularity (year, month, day)
- Time-constrained multi-hop questions
- Temporal fact integration

## Freebase Knowledge Graph

CWQ and WebQSP use the Freebase RDF dump plus the entity-processing utilities in this directory.

**Recommended flow:**
```bash
bash scripts/setup_data_kg.sh
wget -O freebase-rdf-latest.gz \
  http://commondatastorage.googleapis.com/freebase-public/rdf/freebase-rdf-latest.gz
bash scripts/process_entitites_freebase.sh
python scripts/convert_entities.py
```

This produces the Freebase entity and relation files used by the augmentation scripts and KG server.

## Usage in Training

After setting up datasets, use them in your training scripts:

### CWQ Training Example
```bash
python -m verl.trainer.main_ppo \
    mode=kg-search \
    data.train_files="data_kg/cwq_search_augmented_initial_entities/train.parquet" \
    data.val_files="data_kg/cwq_search_augmented_initial_entities/test.parquet" \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct
```

### MultiTQ Training Example
```bash
python -m verl.trainer.main_ppo \
    mode=kg-search \
    data.train_files="data_kg/multitq_search_augmented_initial_entities/train.parquet" \
    data.val_files="data_kg/multitq_search_augmented_initial_entities/test.parquet" \
    data.prompt_augmentation.enable=true \
    data.prompt_augmentation.guideline_level=temporal_detailed \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct
```

## Data Format

All processed datasets use the same format (parquet files with these columns):

- `data_source`: "kgsearch"
- `prompt`: Full prompt with KG-search instructions
- `ability`: Task type (e.g., "kg_multihop_qa")
- `reward_model`: Reward specification
  - `style`: "rule" or "model"
  - `ground_truth`: Answer information
- `extra_info`: Metadata
  - `split`: train/dev/test
  - `index`: Sample index
  - `sample_id`: Unique identifier
  - `dataset_name`: Source dataset

## Troubleshooting

### Missing Dependencies
```bash
# Install required Python packages
pip install pandas pyarrow requests

# Install system tools
# Ubuntu/Debian: sudo apt-get install git wget
# CentOS/RHEL: sudo yum install git wget
```

### Path Issues
All scripts use relative paths from the project root. Make sure to:
1. Run scripts from the project root directory, OR
2. Use the provided setup scripts which handle paths automatically

### Dataset Not Found
If you get "dataset not found" errors:
1. Check that raw data exists in the expected location
2. Review the dataset-specific README files
3. Ensure Freebase KG data is downloaded

## Script Details

### Core Processing Scripts

- **`download_kg.py`**: Downloads Freebase knowledge graph data
- **`convert_entities.py`**: Converts entity ID formats
- **`cwq.py`**: Processes CWQ to parquet format
- **`webqsp.py`**: Processes WebQSP to parquet format
- **`*_search_augmented_initial_entities.py`**: Adds search hints and initial entities to prompts

### MultiTQ Scripts

See [data_multitq_kg/README.md](data_multitq_kg/README.md) for detailed MultiTQ setup instructions.

### WebQSP KG Scripts

See [webqsp_kg/README.md](webqsp_kg/README.md) for WebQSP-specific training scripts.

## Contributing

When adding new datasets:
1. Create a new subdirectory (e.g., `data_newdataset_kg/`)
2. Provide processing scripts and setup.sh
3. Update this README with setup instructions
4. Add option to `setup_data_kg.sh`
5. Ensure consistent output format (parquet with standard columns)

For issues or questions about data setup, refer to the main project README and the dataset-specific READMEs in each subdirectory.
