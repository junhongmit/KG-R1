# Data KG Scripts Patch Summary

## Overview

Successfully patched data_kg setup scripts from [RL_KG](../RL_KG) to [KG-R1](../KG-R1). This adds comprehensive dataset processing capabilities for knowledge graph-based training.

**Date:** 2025-11-19
**Source:** `~/RL_KG/scripts`
**Target:** `~/KG-R1/scripts`

## What Was Added

### 1. Scripts Directory Structure

Created complete scripts directory with:
- Core KG data processing utilities
- Dataset-specific processing scripts
- Setup and automation scripts
- Comprehensive documentation

```
KG-R1/scripts/
├── setup_data_kg.sh                    # 🆕 Master setup script
├── README.md                           # 🆕 Complete documentation
├── download_kg.py                      # Freebase KG downloader
├── convert_entities.py                 # Entity format converter
├── process_entitites_freebase.sh       # Freebase processor
├── html_logger.py                      # Logging utilities
│
├── data_process_kg/                    # CWQ & WebQSP processing
│   ├── cwq.py
│   ├── cwq_search_augmented_initial_entities.py
│   ├── webqsp.py
│   └── webqsp_search_augmented_initial_entities.py
│
├── data_multitq_kg/                    # MultiTQ temporal QA
│   ├── setup_multitq.sh
│   ├── download_multitq.py
│   ├── process_multitq.py
│   ├── multitq_search_augmented_initial_entities.py
│   ├── filter_multitq_2k.py
│   ├── integration_example.sh
│   └── README.md
│
└── webqsp_kg/                          # WebQSP training scripts
    ├── setup_webqsp_kg.sh
    ├── data_process.sh
    ├── train_grpo.sh
    ├── train_ppo.sh
    ├── evaluate.sh
    └── README.md
```

### 2. Path Updates

All hardcoded paths from RL_KG have been updated to be relative to KG-R1:

**Before (RL_KG):**
```python
default='~/RL_KG/data_kg/CWQ/train_simple.json'
```

**After (KG-R1):**
```python
default='data_kg/CWQ/train_simple.json'  # Relative to project root
```

**Updated files:**
- `data_process_kg/cwq.py` - CWQ default paths
- `data_process_kg/webqsp.py` - WebQSP default paths
- `data_multitq_kg/setup_multitq.sh` - Shell script paths
- `data_multitq_kg/download_multitq.py` - Download paths
- `data_multitq_kg/process_multitq.py` - Processing paths
- `data_multitq_kg/filter_multitq_2k.py` - Filter script paths
- `data_multitq_kg/integration_example.sh` - Example paths
- `webqsp_kg/data_process.sh` - WebQSP processing paths

### 3. Master Setup Script

Created interactive setup script: [scripts/setup_data_kg.sh](scripts/setup_data_kg.sh)

**Features:**
- Interactive menu for dataset selection
- Dependency checking
- Automated download and processing
- Progress reporting and error handling
- Support for all datasets or individual setup

**Usage:**
```bash
cd ~/KG-R1
bash scripts/setup_data_kg.sh
```

**Options:**
1. Setup ComplexWebQuestions (CWQ)
2. Setup WebQuestionsSP (WebQSP)
3. Setup MultiTQ (Temporal QA)
4. Download Freebase KG data only
5. Setup all datasets at once
6. Exit

### 4. Documentation

Created comprehensive documentation:

- **[scripts/README.md](scripts/README.md)** - Complete guide covering:
  - Quick start instructions
  - Directory structure explanation
  - Dataset-specific setup guides
  - Usage examples for training
  - Troubleshooting tips
  - Data format specifications

- **Preserved existing READMEs:**
  - `scripts/data_multitq_kg/README.md`
  - `scripts/webqsp_kg/README.md`

## Supported Datasets

### 1. ComplexWebQuestions (CWQ)
- Multi-hop QA over Freebase
- Complex reasoning chains
- **Output:** `data_kg/cwq_search_augmented_initial_entities/`

### 2. WebQuestionsSP (WebQSP)
- Single and simple multi-hop QA
- Freebase-based questions
- **Output:** `data_kg/webqsp_search_augmented_initial_entities/`

### 3. MultiTQ (Temporal Multi-hop QA)
- Temporal reasoning questions
- Multi-granularity time constraints (year/month/day)
- **Output:** `data_kg/multitq_search_augmented_initial_entities/`

## Key Improvements

### 1. Project Portability
- All paths are now relative to project root
- No hardcoded absolute paths to RL_KG
- Works in any directory structure

### 2. User Experience
- Interactive setup wizard
- Clear progress indication
- Helpful error messages
- Automated dependency checking

### 3. Documentation Quality
- Comprehensive README with examples
- Quick start guide
- Troubleshooting section
- Training usage examples

### 4. Maintainability
- Modular script organization
- Consistent naming conventions
- Clear separation of concerns
- Easy to extend with new datasets

## Usage Examples

### Quick Setup (Recommended)
```bash
cd ~/KG-R1
bash scripts/setup_data_kg.sh
# Select option from interactive menu
```

### Manual Setup - CWQ
```bash
cd ~/KG-R1
python scripts/data_process_kg/cwq.py
python scripts/data_process_kg/cwq_search_augmented_initial_entities.py
```

### Manual Setup - MultiTQ
```bash
cd ~/KG-R1/scripts/data_multitq_kg
bash setup_multitq.sh
```

### Using in Training
```bash
# CWQ training
python -m verl.trainer.main_ppo \
    mode=kg-search \
    data.train_files="data_kg/cwq_search_augmented_initial_entities/train.parquet" \
    data.val_files="data_kg/cwq_search_augmented_initial_entities/test.parquet" \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct

# MultiTQ training (temporal reasoning)
python -m verl.trainer.main_ppo \
    mode=kg-search \
    data.train_files="data_kg/multitq_search_augmented_initial_entities/train.parquet" \
    data.val_files="data_kg/multitq_search_augmented_initial_entities/test.parquet" \
    data.prompt_augmentation.enable=true \
    data.prompt_augmentation.guideline_level=temporal_detailed \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct
```

## File Changes Summary

### New Files Created (2)
1. `scripts/setup_data_kg.sh` - Master setup script
2. `scripts/README.md` - Comprehensive documentation

### Files Copied from RL_KG (19)
Core utilities:
- `convert_entities.py`
- `download_kg.py`
- `html_logger.py`
- `process_entitites_freebase.sh`

Data processing:
- `data_process_kg/cwq.py`
- `data_process_kg/cwq_search_augmented_initial_entities.py`
- `data_process_kg/webqsp.py`
- `data_process_kg/webqsp_search_augmented_initial_entities.py`

MultiTQ scripts:
- `data_multitq_kg/download_multitq.py`
- `data_multitq_kg/filter_multitq_2k.py`
- `data_multitq_kg/integration_example.sh`
- `data_multitq_kg/multitq_search_augmented_initial_entities.py`
- `data_multitq_kg/process_multitq.py`
- `data_multitq_kg/setup_multitq.sh`
- `data_multitq_kg/README.md`

WebQSP scripts:
- `webqsp_kg/data_process.sh`
- `webqsp_kg/evaluate.sh`
- `webqsp_kg/setup_webqsp_kg.sh`
- `webqsp_kg/train_grpo.sh`
- `webqsp_kg/train_ppo.sh`
- `webqsp_kg/README.md`

### Files Modified (9)
Updated to use relative paths:
1. `data_process_kg/cwq.py` - Default paths
2. `data_process_kg/webqsp.py` - Default paths
3. `data_multitq_kg/setup_multitq.sh` - Directory paths
4. `data_multitq_kg/download_multitq.py` - Default output dir
5. `data_multitq_kg/process_multitq.py` - Input/output paths
6. `data_multitq_kg/filter_multitq_2k.py` - Data directory paths
7. `data_multitq_kg/integration_example.sh` - Example paths
8. `webqsp_kg/data_process.sh` - Work directory path
9. Made scripts executable (chmod +x)

## Verification

Run these commands to verify the setup:

```bash
# Check scripts directory structure
tree -L 2 ~/KG-R1/scripts/

# Verify executable permissions
ls -la ~/KG-R1/scripts/*.sh
ls -la ~/KG-R1/scripts/data_multitq_kg/*.sh

# Check for hardcoded paths (should return minimal results)
grep -r "~/RL_KG" ~/KG-R1/scripts/ || echo "No hardcoded RL_KG paths found!"
```

## Next Steps

1. **Try the setup script:**
   ```bash
   cd ~/KG-R1
   bash scripts/setup_data_kg.sh
   ```

2. **Review the documentation:**
   - Read [scripts/README.md](scripts/README.md)
   - Check dataset-specific READMEs

3. **Set up a dataset:**
   - Start with MultiTQ (includes download)
   - Or use CWQ/WebQSP if you have raw data

4. **Integrate with training:**
   - Update training scripts to use new data paths
   - Test with small dataset first
   - Scale to full datasets

## Notes

- All scripts use relative paths from KG-R1 project root
- Freebase KG data is shared across all datasets
- MultiTQ can be fully automated (includes download)
- CWQ and WebQSP require manual data download first
- Scripts are compatible with existing KG-R1 training pipeline

## References

- Source repository: `~/RL_KG`
- Target repository: `~/KG-R1`
- Main documentation: [scripts/README.md](scripts/README.md)
- RL_KG scripts: `~/RL_KG/scripts`