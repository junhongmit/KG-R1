# Data KG Setup - Quick Start Guide

## 🚀 Getting Started

### Step 1: Download the datasets
```bash
cd ~/KG-R1
bash scripts/setup_data_kg.sh
```

Choose the option that downloads all data in the interactive menu.

### Step 2: Prepare Freebase entities
```bash
wget -O freebase-rdf-latest.gz \
  http://commondatastorage.googleapis.com/freebase-public/rdf/freebase-rdf-latest.gz
bash scripts/process_entitites_freebase.sh
python scripts/convert_entities.py
```

### Step 3: Build the augmented CWQ and WebQSP files
```bash
python scripts/data_process_kg/cwq_search_augmented_initial_entities.py
python scripts/data_process_kg/webqsp_search_augmented_initial_entities.py
```

### Step 4: Launch the KG retrieval server
```bash
./kg_retrieval_launch_cwq.sh
```

### Step 5: Run evaluation
```bash
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh \
  your-org/KG-R1-model \
  cwq
```

## 🎯 Expected Output Locations

After setup, you should have:

```
KG-R1/
├── data_kg/
│   ├── cwq_search_augmented_initial_entities/
│   │   ├── train.parquet
│   │   └── test.parquet
│   │
│   ├── webqsp_search_augmented_initial_entities/
│   │   ├── train.parquet
│   │   └── test.parquet
```

---

## ❓ Common Questions

**Q: Where do I get CWQ/WebQSP raw data?**
A: Use `bash scripts/setup_data_kg.sh` and choose the option that downloads all data.

**Q: Do I need Freebase KG data?**
A: Yes. Download `freebase-rdf-latest.gz`, then run `bash scripts/process_entitites_freebase.sh` and `python scripts/convert_entities.py`.

---

## 📚 More Information

- **Full Documentation**: [scripts/README.md](README.md)
- **MultiTQ Details**: [data_multitq_kg/README.md](data_multitq_kg/README.md)
- **WebQSP Details**: [webqsp_kg/README.md](webqsp_kg/README.md)

---

## 🐛 Troubleshooting

**Setup script fails?**
```bash
# Check dependencies
python3 --version  # Should be 3.7+
git --version
wget --version

# Install missing dependencies
pip install pandas pyarrow requests
```

**Path errors?**
- Always run scripts from the KG-R1 project root
- Or use the provided setup scripts which handle paths

**Can't find data files?**
```bash
# Check what was created
ls -la data_kg/
tree data_kg/ -L 2
```

---

## 🎉 Success!

Once setup is complete, you should see:
```
✅ Dataset processing completed
✅ Training data created
📋 Available files for KG-R1 training
```

Now you're ready to train or evaluate. See the main README for end-to-end experiment commands.
