# KG-R1: Efficient and Transferable Knowledge Graph-Augmented Reinforcement Learning

**🔬 Key Innovation - Single-Agent KG Reasoning**:
1. ✅ **Single-Agent Architecture**: Replaces complex multi-module workflows with unified LLM agent
2. ✅ **Schema-Agnostic KG Server**: Works across different knowledge graphs (Freebase, Wikidata, Temporal KGs)
3. ✅ **Cross-KG Transferability**: Plug-and-play capability - train once, transfer anywhere
4. ✅ **Efficiency Gains**: ~83% of tokens from KG retrieval, only ~13% from reasoning generation
5. ✅ **GRPO Training**: Group Relative Policy Optimization for stable multi-turn learning

**⚡ System Architecture**:
- **KG Retreival Server**: 4 basic operations (get_tail_relations, get_head_relations, get_tail_entities, get_head_entities)
- **Single Agent**: Unified reasoning and retrieval in one LLM with special tokens
- **Multi-turn Interaction**: Up to 7 turns of KG exploration per question
- **Lightweight Design**: No separate retriever, reranker, or planning modules

**📊 Performance Results**:
- **CWQ Dataset**: Improved performance over vanilla baselines and prior KG-RAG methods
- **WebQSP Dataset**: Strong performance with cross-dataset transferability
- **Efficiency Gains**: ~1680x computational cost vs naive calculation, but highly effective
- **Transferability**: Models trained on one KG work on different KG schemas

**🚀 Current Status**: KG-R1 system with comprehensive evaluation framework and LLM-as-judge factuality evaluation for Knowledge Graph Question Answering.

## KG-R1 System Architecture

### 1. Single-Agent KG-Augmented Prompt
```
Answer the given question. You can interact with the knowledge graph through the following actions:

- get_tail_relations(entity): Get relations where entity is the subject
- get_head_relations(entity): Get relations where entity is the object  
- get_tail_entities(entity, relation): Get objects for entity-relation pairs
- get_head_entities(entity, relation): Get subjects for relation-entity pairs

Use <search>action_name(arguments)</search> to query the KG. Results appear in <information></information>.
Reason with <think></think> tags. Provide final answer in <answer></answer> tags.

Question: {question}
```

### 2. Knowledge Graph Server Operations
**Base URL**: `http://127.0.0.1:8001/retrieve`

**Core Operations**:
- **get_tail_relations(entity)**: Find all relations where entity is the head/subject
- **get_head_relations(entity)**: Find all relations where entity is the tail/object
- **get_tail_entities(entity, relation)**: Get tail entities for head-relation pairs
- **get_head_entities(entity, relation)**: Get head entities for relation-tail pairs

### 3. Multi-Turn Reasoning Process
KG-R1 enables iterative exploration:
1. **Initial Question Analysis** → Identify key entities
2. **KG Exploration** → Multi-turn relation and entity discovery (up to 7 turns)
3. **Answer Synthesis** → Combine retrieved knowledge for final answer

### 4. LLM-as-Judge Evaluation
Semantic factuality evaluation using GPT-based judge for accurate answer assessment beyond exact string matching.

## KG-R1 Architecture Overview

<div align="center">
  <img src="imgs/fig1.png" alt="KG-R1 vs Multi-Module Workflow" width="800"/>
  <p><em>Figure 1: KG-R1 single-agent framework vs traditional multi-module KG-RAG workflows</em></p>
</div>

## Multi-Turn KG Reasoning Process

<div align="center">
  <img src="imgs/fig2.png" alt="KG-R1 Multi-Turn Interaction" width="800"/>
  <p><em>Figure 2: KG-R1 multi-turn interaction trajectory showing iterative knowledge graph exploration</em></p>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2503.09516">
    <img src="https://img.shields.io/badge/Paper1-blue?style=for-the-badge" alt="Button1"/>
  </a>
  <a href="https://arxiv.org/abs/2505.15117">
    <img src="https://img.shields.io/badge/Paper2-green?style=for-the-badge" alt="Button2"/>
  </a>
  <a href="https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5">
    <img src="https://img.shields.io/badge/Resources-orange?style=for-the-badge" alt="Button3"/>
  </a>
  <a href="https://x.com/BowenJin13/status/1895544294473109889">
    <img src="https://img.shields.io/badge/Tweet-red?style=for-the-badge" alt="Button4"/>
  </a>
  <a href="https://wandb.ai/peterjin/Search-R1-v0.2">
    <img src="https://img.shields.io/badge/Logs-purple?style=for-the-badge" alt="Button5"/>
  </a>
</p>

**KG-R1** extends the Search-R1 framework to **knowledge graph-augmented reasoning**, replacing traditional document retrieval with structured knowledge graph operations. This creates a more efficient and transferable agentic KG-RAG system.

Built upon [veRL](https://github.com/volcengine/verl), KG-R1 provides a unified single-agent architecture that learns to reason and interact with knowledge graphs through reinforcement learning. The system achieves both computational efficiency (~83% tokens from KG retrieval vs ~13% from reasoning) and cross-KG transferability.

We support different RL methods (PPO, GRPO), different LLMs (Qwen2.5, Llama3, etc), and different knowledge graph schemas (Freebase, Wikidata, temporal KGs) with a plug-and-play design.

Paper: [link1](https://arxiv.org/pdf/2503.09516), [link2](https://arxiv.org/abs/2505.15117); Model and data: [link](https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5); Twitter thread: [link](https://x.com/BowenJin13/status/1895544294473109889); Full experiment log: [prelim](https://wandb.ai/peterjin/Search-R1-open); [v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train); [v0.2](https://wandb.ai/peterjin/Search-R1-v0.2); [v0.3](https://wandb.ai/peterjin/Search-R1-v0.3). Details about these logs and methods can be find [here](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/experiment_log.md).

**Key Innovation**: KG-R1 replaces complex multi-module workflows with a single unified agent that learns to reason and retrieve through reinforcement learning, achieving both efficiency and transferability across different knowledge graph schemas.

## News

- [2025.11] Implemented cross-KG transferability testing on multiple knowledge graphs
- [2025.11] Released KG-R1 codebase with GRPO training and multi-turn KG reasoning
- [2025.11] Added comprehensive evaluation framework with LLM-as-judge for factuality assessment
- [2025.11] Developed schema-agnostic KG server with 4 basic operations

## Links

- [Installation](#installation)
- [Quick start](#quick-start)
- [KG-R1 Results](#kg-r1-results)
- [Inference](#inference)
- [Use your own dataset](#use-your-own-dataset)
- [Use your own knowledge graph](#use-your-own-knowledge-graph)
- [Features](#features)
- [Acknowledge](#acknowledge)
- [Citations](#citations)

## Installation

### KG-R1 environment
```bash
conda create -n kgr1 python=3.10
conda activate kgr1
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
conda install -c nvidia cuda-toolkit=12.1
pip3 install flash-attn --no-build-isolation
pip install wandb

# Additional dependencies for KG processing
pip install fastapi uvicorn requests aiohttp
pip install networkx # for knowledge graph operations
```

### KG Server environment (required)
The KG-R1 system requires a knowledge graph server for retrieval operations.
```bash
conda create -n kg_server python=3.10
conda activate kg_server

# Core dependencies for KG server
pip install fastapi uvicorn pydantic requests
pip install transformers datasets huggingface_hub
pip install networkx pandas pyarrow

# For efficient KG processing
pip install numpy scipy
```

## Quick start

Train a KG-R1 agent on ComplexWebQuestions (CWQ) dataset using GRPO (Group Relative Policy Optimization). See [Figure 2](imgs/fig2.png) for the multi-turn interaction process.

### Part 1: Initialize

**Set up conda environment and download datasets**

**(1) Create and activate the KG-R1 environment**
```bash
conda create -n kgr1 python=3.9
conda activate kgr1

# Install PyTorch with CUDA support
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM for efficient inference
pip3 install vllm==0.6.3

# Install veRL framework
pip install -e .

# Install flash attention for efficient training
pip3 install flash-attn --no-build-isolation
pip install wandb

# Additional dependencies for KG processing
pip install fastapi uvicorn requests aiohttp networkx
```

**(2) Initialize datasets using `initialize.py`**
```bash
# This script downloads and prepares KG-QA datasets
python initialize.py

# This creates the data_kg/ directory with:
# - CWQ: ComplexWebQuestions dataset with Freebase knowledge subgraphs
# - WebQSP: WebQuestionsSP dataset with Freebase knowledge subgraphs
# - Search-augmented initial entities for both datasets
```

The `data_kg/` directory structure after initialization:
```
data_kg/
├── CWQ/                                          # CWQ dataset files
│   ├── entities.txt, relations.txt              # KG vocabulary
│   ├── train_simple.json, dev_simple.json       # QA pairs
│   └── word_emb_300d.npy                        # Entity embeddings
├── cwq_search_augmented_initial_entities/       # Processed CWQ data
│   ├── train.parquet, dev.parquet, test.parquet
├── webqsp/                                      # WebQSP dataset files
│   ├── entities.txt, relations.txt
│   ├── train_simple.json, test_simple.json
│   └── word_emb_300d.npy
└── webqsp_search_augmented_initial_entities/    # Processed WebQSP data
    ├── train.parquet, test.parquet
```

**(3) Create KG server environment (optional but recommended)**
```bash
conda create -n kg_server python=3.10
conda activate kg_server

# Core dependencies for KG server
pip install fastapi uvicorn pydantic requests
pip install transformers datasets huggingface_hub
pip install networkx pandas pyarrow numpy scipy
```

### Part 2: Training

**Train KG-R1 agent using reinforcement learning**

> **Note**: We will provide the HuggingFace model weights pretrained (backbone: Qwen2.5-3B) later.

**(1) Launch the KG retrieval server**
```bash
conda activate kg_server
# Start KG server on port 8001 (provides 4 basic KG operations)
python kg_r1/search/server.py --port 8001 --data_dir data_kg
```

**(2) Run KG-R1 training with GRPO**
```bash
conda activate kgr1
# Train Qwen2.5-3B with 7-turn KG reasoning on CWQ
bash train_grpo_kg_qwen_3b_cwq_f1_turn7.sh

# Or train on WebQSP:
# bash train_grpo_kg_qwen_3b_webqsp_f1_turn7.sh
```

**Training configurations available:**
- `train_grpo_kg_qwen_3b_cwq_f1_turn5.sh` - CWQ with 5 turns
- `train_grpo_kg_qwen_3b_cwq_f1_turn7.sh` - CWQ with 7 turns (recommended)
- `train_grpo_kg_qwen_3b_webqsp_f1_turn7.sh` - WebQSP with 7 turns

**Expected training time:** ~8-12 hours on 4x A100 GPUs for full training

**Expected Results:** 70.9 F1 / 73.8 Hit@1 on CWQ with single 3B model (see [Performance Results](#performance-results))

### Part 3: Inference

**Two inference options: (1) Local checkpoint or (2) HuggingFace models**

#### Option 1: Local Checkpoint Inference

Evaluate your locally trained model:

```bash
# (1) Launch the KG retrieval server
conda activate kg_server
python kg_r1/search/server.py --port 8001 --data_dir data_kg

# (2) Run inference with your trained checkpoint
conda activate kgr1
bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7_local.sh \
    /path/to/your/checkpoint \
    cwq  # or webqsp

# Example:
# bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7_local.sh \
#     verl_checkpoints/cwq-KG-r1-grpo-qwen2.5-3b-it_f1_turn7 \
#     cwq
```

#### Option 2: HuggingFace Model Inference

Evaluate pre-trained models directly from HuggingFace (no local training needed):

```bash
# (1) Launch the KG retrieval server
conda activate kg_server
python kg_r1/search/server.py --port 8001 --data_dir data_kg

# (2) Run inference with HuggingFace model
conda activate kgr1
bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7_hf.sh \
    JinyeopSong/KG-R1_test \  # Specify HF model
    cwq                        # Dataset to evaluate

# Or simply use defaults:
# bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7_hf.sh
```

**Available Evaluation Scripts:**
- `eval_qwen_3b_cwq_f1_turn7_local.sh` - Evaluate local checkpoint
- `eval_qwen_3b_cwq_f1_turn7_hf.sh` - Evaluate HuggingFace model

**Available HuggingFace Models:**
- `JinyeopSong/KG-R1_test` - Testing model (placeholder)
- More models coming soon! (Links not ready yet)

**Inference outputs:**
- Pass@K evaluation results (K=1,2,3,4)
- Detailed reasoning traces with KG exploration steps
- Exact match and F1 scores
- Per-sample analysis in JSONL format

## KG-R1 Results

### Performance Results

**Main Results on WebQSP and CWQ:**

| Method | Model | Modules | WebQSP F1/Hit@1 | CWQ F1/Hit@1 | Efficiency (Total/Gen) |
|--------|-------|---------|------------------|---------------|------------------------|
| **Vanilla** | Qwen2.5-3B-it | 1 | 29.4 / 46.6 | 16.6 / 21.1 | 95-104 / 30-42 |
| **COT** | Qwen2.5-3B-it | 1 | 30.6 / 47.6 | 17.3 / 21.4 | 131-140 / 192-216 |
| **RoG** | LLaMA2-7B-it | 2 | 70.8 / 85.7 | 56.2 / 62.6 | 1.1-1.2K / 266-295 |
| **ToG 2.0** | GPT-3.5 | 5 | 74.5 / 77.8 | 65.8 / 68.9 | 3.8-39K / 605-650 |
| **ReKnoS** | GPT-4o-mini | 3 | 73.7 / 81.1 | 64.7 / 66.8 | 3.1-4.1K / 617-752 |
| **🔥 KG-R1 (1 run)** | Qwen2.5-3B-it | **1** | **77.5 / 84.7** | **70.9 / 73.8** | 3.2-3.3K / 302-377 |
| **🔥 KG-R1 (3 runs)** | Qwen2.5-3B-it | **1** | **85.8 / 91.7** | **81.0 / 83.9** | 9.7-10K / 906-1.1K |

### Cross-KG Transferability Results

**Zero-shot transfer across different KG schemas (no retraining required):**

| Training KG | SimpleQA | GrailQA | T-REx | QALD-10en | MultiTQ | **Average** |
|-------------|----------|---------|-------|-----------|---------|-------------|
| **Vanilla Baseline** | 13.7 / 13.7 | 15.9 / 15.9 | 24.4 / 24.4 | 23.8 / 23.8 | 2.2 / 5.4 | 19.4 / 19.8 |
| **KG-R1 (WebQSP)** | 59.1 / 59.1 | 32.8 / 38.5 | 80.5 / 84.5 | 51.9 / 53.4 | 21.6 / 31.4 | **64.0 / 68.3** |
| **KG-R1 (CWQ)** | 64.6 / 64.7 | 42.8 / 50.2 | 81.3 / 85.6 | 55.9 / 57.7 | 27.1 / 38.9 | **67.2 / 72.1** |
| **KG-R1 (3 runs)** | 73.1 / 73.1 | 52.8 / 61.0 | 86.8 / 91.5 | 63.9 / 65.5 | 36.2 / 48.4 | **74.1 / 79.4** |

### Key Achievements
- **🎯 Strong Performance**: Competitive results on CWQ and WebQSP benchmarks
- **⚡ Computational Efficiency**: Single-agent vs multi-module workflows (1 vs 2-5 modules)
- **🔄 Cross-KG Transfer**: 64-74% average F1 across 5 different KG schemas
- **💡 Training Efficiency**: 3B model achieves competitive performance

## Inference
#### You can play with the trained KG-R1 model with your own questions.
(1) Launch the KG retrieval server.
```bash
conda activate kg_server
python kg_r1/search/server.py --port 8001 --data_dir data_kg
```

(2) Run KG-R1 inference.
```bash
conda activate kgr1
python infer_kg_r1.py --checkpoint verl_checkpoints/your_trained_model
```
You can modify the `question` parameter to test different knowledge graph questions. The model will interactively explore the KG using the 4 basic operations and provide reasoning traces.

## Use your own dataset

### KG-QA data format
For each knowledge graph question-answer sample, it should be a dictionary containing:

```python
data = {
    "data_source": "your_kg_dataset",
    "original_query": question,
    "target_text": answer,
    "query_entities": ["entity1", "entity2"],  # Initial entities
    "query_id": unique_id,
    "split": "train/test/dev"
}
```

### Knowledge Graph format
Your knowledge graph should provide the following structure:

```python
# Entity-relation-entity triples
kg_data = {
    "entities": {"entity_id": "human_readable_name"},
    "relations": {"relation_id": "human_readable_name"},
    "triples": [
        ["head_entity_id", "relation_id", "tail_entity_id"],
        # ... more triples
    ]
}
```

You can refer to `scripts/data_kg/process_datasets.py` for concrete data processing examples for CWQ and WebQSP datasets.

### Knowledge Graph Server Setup

To use your own knowledge graph, you need to set up the KG server with your data:

1. **Prepare your KG data** in the required format (see above)
2. **Start the KG server** with your data directory:

```bash
# Your KG data should be organized as:
# your_kg_data/
# ├── entities.json
# ├── relations.json  
# ├── train_simple.json
# └── test_simple.json

python kg_r1/search/server.py --port 8001 --data_dir your_kg_data
```

3. **Configure your training script** to point to your KG server:

```bash
# In your training script, update:
actor_rollout_ref.rollout.search.search_url="http://127.0.0.1:8001/retrieve"
```

The KG server supports the 4 basic operations:
- `get_tail_relations(entity)`: Find relations where entity is the subject
- `get_head_relations(entity)`: Find relations where entity is the object
- `get_tail_entities(entity, relation)`: Get tail entities for head-relation pairs
- `get_head_entities(entity, relation)`: Get head entities for relation-tail pairs

## Use your own knowledge graph

KG-R1 supports different types of knowledge graphs with a schema-agnostic design. The system works with:
- **Freebase-style KGs**: Entity-centric with rich relations
- **Wikidata KGs**: Property-based knowledge representation
- **Temporal KGs**: Time-aware knowledge graphs
- **Domain-specific KGs**: Custom knowledge graphs for specific domains

The main philosophy is to launch a KG server separately from the RL training pipeline, providing a clean API interface.

The LLM agent calls the KG server through the search API at `http://127.0.0.1:8001/retrieve`.

### KG Server Implementation
You can refer to `kg_r1/search/server.py` for the complete KG server implementation, which includes:
- **FastAPI server**: RESTful API for KG operations
- **Concurrent processing**: ThreadPoolExecutor for handling multiple requests
- **Action routing**: Dispatches requests to appropriate KG operations
- **Error handling**: Robust error handling for malformed queries

### Cross-KG Transfer
KG-R1's key advantage is cross-KG transferability. Models trained on one KG can transfer to different KG schemas without retraining, enabling plug-and-play usage.

## Citations

If you use KG-R1 in your research, citation information will be provided upon publication.
