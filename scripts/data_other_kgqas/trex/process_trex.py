#!/usr/bin/env python3
"""
T-REX dataset processor for KG-R1 training pipeline

This script downloads T-REX from KGdata repository and processes it with 2-hop subgraphs
to create KG-R1 compatible training data following the README_BENCHMARK_INTEGRATION.md format.
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import argparse
from tqdm import tqdm

# Dataset URLs
TREX_URL = "https://github.com/lanjiuqing64/KGdata/raw/main/QAdata/T-REX.json"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUBGRAPH_ROOT = PROJECT_ROOT / "data_kg" / "Subgraph"

# Subgraph paths (from downloaded Subgraph repository)  
SUBGRAPH_PATHS = {
    "triples": str(DEFAULT_SUBGRAPH_ROOT / "hop_2_Wikidata" / "T-REX_2hop.json"),
    "labels": str(DEFAULT_SUBGRAPH_ROOT / "hop_2_Wikidata" / "T-REX_2hop_labels.json"),
}

def download_file(url: str, output_path: str) -> bool:
    """Download file with progress bar"""
    try:
        print(f"📥 Downloading {os.path.basename(output_path)}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Download") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        print(f"✅ Downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return False

def load_json_data(file_path: str) -> Optional[Dict]:
    """Load JSON data with error handling"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def process_trex_sample(sample: Dict, sample_id: str, subgraph_triples: Dict, subgraph_labels: Dict, global_entity_to_idx: Dict, global_relation_to_idx: Dict) -> Optional[Dict]:
    """
    Convert T-REX sample to KG-R1 compatible format with subgraph data
    """
    try:
        # Extract basic info - T-REX uses 'input' field
        question = sample.get('input', '').strip()
        if not question:
            return None
        
        # Process answers - T-REX format
        answers = []
        if 'answer' in sample:
            answer_text = sample['answer']
            if answer_text:
                answers.append({
                    'text': answer_text,
                    'kb_id': answer_text  # T-REX doesn't have KB IDs for answers
                })
        
        # Also check aliases
        if 'alias' in sample and isinstance(sample['alias'], list):
            for alias in sample['alias']:
                if alias and alias != answer_text:
                    answers.append({
                        'text': alias,
                        'kb_id': alias
                    })
        
        # Extract initial entities. ToG-style T-REx uses qid_topic_entity/topic_entity_ids;
        # some KGdata variants use topic_entity.
        initial_entities = []
        if 'topic_entity' in sample and isinstance(sample['topic_entity'], dict):
            initial_entities = list(sample['topic_entity'].keys())
        elif 'qid_topic_entity' in sample and isinstance(sample['qid_topic_entity'], dict):
            initial_entities = list(sample['qid_topic_entity'].keys())
        elif 'topic_entity_ids' in sample and isinstance(sample['topic_entity_ids'], dict):
            initial_entities = list(sample['topic_entity_ids'].keys())
        
        # Build subgraph from initial entities and subgraph data
        subgraph_entities = set()
        subgraph_relations = set()
        raw_tuples = []
        
        for entity_id in initial_entities:
            if isinstance(entity_id, str) and entity_id.startswith('Q'):
                subgraph_entities.add(entity_id)
            
            # Get triples for this entity from subgraph data
            if entity_id in subgraph_triples:
                triples = subgraph_triples[entity_id]
                if isinstance(triples, list):
                    for triple in triples:
                        if len(triple) >= 3:
                            subject, relation, obj = triple[0], triple[1], triple[2]
                            if isinstance(subject, str) and subject.startswith('Q'):
                                subgraph_entities.add(subject)
                            if isinstance(obj, str) and obj.startswith('Q'):
                                subgraph_entities.add(obj)
                            if isinstance(relation, str):
                                subgraph_relations.add(relation)
                            raw_tuples.append((subject, relation, obj))
        
        # Convert entities and relations to global indices
        entity_indices_global = []
        relation_indices_global = []
        subgraph_tuples_global = []
        
        # Map entities to global indices
        for entity in subgraph_entities:
            if entity in global_entity_to_idx:
                entity_indices_global.append(global_entity_to_idx[entity])
        
        # Map relations to global indices  
        for relation in subgraph_relations:
            if relation in global_relation_to_idx:
                relation_indices_global.append(global_relation_to_idx[relation])
        
        # Convert tuples to use global indices
        for subject, relation, obj in raw_tuples:
            subject_idx = global_entity_to_idx.get(subject)
            relation_idx = global_relation_to_idx.get(relation)
            obj_idx = global_entity_to_idx.get(obj)
            
            if subject_idx is not None and relation_idx is not None and obj_idx is not None:
                subgraph_tuples_global.append([subject_idx, relation_idx, obj_idx])
        
        # Map initial entities to global indices
        entity_indices = []
        for entity_id in initial_entities:
            if entity_id in global_entity_to_idx:
                entity_indices.append(global_entity_to_idx[entity_id])
        
        # Create standardized format for KG-R1
        processed_sample = {
            'id': sample_id,
            'question': question,
            'answers': answers,
            'entities': entity_indices,  # Global indices of initial entities
            'subgraph': {
                'entities': entity_indices_global,  # Global indices pointing to entities.txt
                'relations': relation_indices_global,  # Global indices pointing to relations.txt
                'tuples': subgraph_tuples_global  # Tuples using global indices
            }
        }
        
        return processed_sample
            
    except Exception as e:
        print(f"❌ Error processing sample {sample_id}: {e}")
        return None

def split_data(data: List[Dict], train_ratio: float = 0.8, dev_ratio: float = 0.1) -> Tuple[List, List, List]:
    """Split data into train/dev/test sets"""
    total = len(data)
    train_size = int(total * train_ratio)
    dev_size = int(total * dev_ratio)
    
    train_data = data[:train_size]
    dev_data = data[train_size:train_size + dev_size]
    test_data = data[train_size + dev_size:]
    
    return train_data, dev_data, test_data

def extract_entities_relations(processed_data: List[Dict]) -> Tuple[set, set]:
    """Extract all unique entities and relations for auxiliary files"""
    all_entities = set()
    all_relations = set()
    
    for sample in processed_data:
        subgraph = sample.get('subgraph', {})
        all_entities.update(subgraph.get('entities', []))
        all_relations.update(subgraph.get('relations', []))
    
    return all_entities, all_relations

def save_auxiliary_files(all_entities: set, all_relations: set, labels_data: Dict, output_dir: Path):
    """Save entities.txt, entities_text.txt, relations.txt files"""
    sorted_entities = sorted(all_entities)
    sorted_relations = sorted(all_relations)
    
    # Save entities.txt
    with open(output_dir / "entities.txt", 'w', encoding='utf-8') as f:
        for entity in sorted_entities:
            f.write(f"{entity}\n")
    
    # Save entities_text.txt with human-readable names or original IDs
    with open(output_dir / "entities_text.txt", 'w', encoding='utf-8') as f:
        for entity in sorted_entities:
            if labels_data:
                entity_text = labels_data.get(entity, entity)
                # Avoid unnamed_entity, keep original ID instead
                if entity_text == "unnamed_entity":
                    entity_text = entity
            else:
                entity_text = entity
            f.write(f"{entity_text}\n")
    
    # Save relations.txt
    with open(output_dir / "relations.txt", 'w', encoding='utf-8') as f:
        for relation in sorted_relations:
            # Map Wikidata property ID to human-readable text
            if labels_data:
                relation_text = labels_data.get(relation, relation)
                # Avoid unnamed_relation, keep original ID instead
                if relation_text == "unnamed_relation":
                    relation_text = relation
            else:
                relation_text = relation
            f.write(f"{relation_text}\n")
    
    print(f"✅ Generated auxiliary files:")
    print(f"   📄 entities.txt ({len(sorted_entities)} entities)")
    print(f"   📄 entities_text.txt ({len(sorted_entities)} entity names)")
    print(f"   📄 relations.txt ({len(sorted_relations)} relations)")

def save_json_splits(train_data: List, dev_data: List, test_data: List, output_dir: Path):
    """Save data splits as JSONL files"""
    splits = [
        ("train_simple.json", train_data),
        ("dev_simple.json", dev_data),
        ("test_simple.json", test_data)
    ]
    
    for filename, data in splits:
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved {filename} ({len(data)} samples)")

def save_outputs(processed_data: List[Dict], sorted_entities: List, sorted_relations: List, 
                labels_data: Dict, output_dir: Path):
    """Save all outputs for T-REX dataset"""
    # Save test data only (T-REX for KG-R1 evaluation)
    test_filename = "test_simple.json"
    with open(output_dir / test_filename, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ Saved {test_filename} ({len(processed_data)} samples)")
    
    # Save auxiliary files
    # entities.txt
    with open(output_dir / "entities.txt", 'w', encoding='utf-8') as f:
        for entity in sorted_entities:
            f.write(f"{entity}\n")
    
    # entities_text.txt (human-readable names or original IDs)
    with open(output_dir / "entities_text.txt", 'w', encoding='utf-8') as f:
        for entity in sorted_entities:
            if labels_data:
                entity_text = labels_data.get(entity, entity)
                # Avoid unnamed_entity, keep original ID instead
                if entity_text == "unnamed_entity":
                    entity_text = entity
            else:
                entity_text = entity
            f.write(f"{entity_text}\n")
    
    # relations.txt
    with open(output_dir / "relations.txt", 'w', encoding='utf-8') as f:
        for relation in sorted_relations:
            # Map Wikidata property ID to human-readable text
            if labels_data:
                relation_text = labels_data.get(relation, relation)
                # Avoid unnamed_relation, keep original ID instead
                if relation_text == "unnamed_relation":
                    relation_text = relation
            else:
                relation_text = relation
            f.write(f"{relation_text}\n")
    
    print(f"✅ Generated auxiliary files:")
    print(f"   📄 entities.txt ({len(sorted_entities)} entities)")
    print(f"   📄 entities_text.txt ({len(sorted_entities)} entity names)")
    print(f"   📄 relations.txt ({len(sorted_relations)} relations)")

def main():
    parser = argparse.ArgumentParser(description="Process T-REX dataset for KG-R1")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "data_kg" / "trex"), help="Output directory")
    parser.add_argument("--subgraph_triples", default=SUBGRAPH_PATHS["triples"], help="Path to T-REX 2-hop triples JSON")
    parser.add_argument("--subgraph_labels", default=SUBGRAPH_PATHS["labels"], help="Path to T-REX 2-hop labels JSON")
    parser.add_argument("--force_redownload", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Processing T-REX dataset for KG-R1")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    
    # Step 1: Check for T-REX dataset
    trex_file = output_dir / "trex_raw.json"
    if not trex_file.exists() or args.force_redownload:
        if not download_file(TREX_URL, str(trex_file)):
            print("❌ Failed to download T-REX dataset")
            return
    else:
        print(f"✅ Using existing T-REX file: {trex_file}")
    
    # Step 2: Load datasets
    print("\n📊 Loading datasets...")
    trex_data = load_json_data(str(trex_file))
    if not trex_data:
        return
    
    subgraph_data = load_json_data(args.subgraph_triples) or {}
    labels_data = load_json_data(args.subgraph_labels) or {}
    
    print(f"✅ Loaded {len(trex_data)} T-REX samples")
    print(f"✅ Loaded {len(subgraph_data)} subgraph entries")
    print(f"✅ Loaded {len(labels_data)} label entries")
    
    # Process all data as test set for KG-R1
    print("\n🔄 Processing T-REX samples...")
    processed_data = []
    all_entities = set()
    all_relations = set()
    
    for i, sample in enumerate(tqdm(trex_data, desc="Processing samples")):
        sample_id = f"trex_{i}"
        
        # Extract entities and relations first.
        initial_entities = []
        if 'topic_entity' in sample and isinstance(sample['topic_entity'], dict):
            initial_entities = list(sample['topic_entity'].keys())
        elif 'qid_topic_entity' in sample and isinstance(sample['qid_topic_entity'], dict):
            initial_entities = list(sample['qid_topic_entity'].keys())
        elif 'topic_entity_ids' in sample and isinstance(sample['topic_entity_ids'], dict):
            initial_entities = list(sample['topic_entity_ids'].keys())
        
        for entity_id in initial_entities:
            if isinstance(entity_id, str) and entity_id.startswith('Q'):
                all_entities.add(entity_id)
            
            # Get triples for this entity from subgraph data
            if entity_id in subgraph_data:
                triples = subgraph_data[entity_id]
                if isinstance(triples, list):
                    for triple in triples:
                        if len(triple) >= 3:
                            subject, relation, obj = triple[0], triple[1], triple[2]
                            if isinstance(subject, str) and subject.startswith('Q'):
                                all_entities.add(subject)
                            if isinstance(obj, str) and obj.startswith('Q'):
                                all_entities.add(obj)
                            if isinstance(relation, str):
                                all_relations.add(relation)
    
    print(f"✅ Successfully processed {len(trex_data)} samples")
    
    # Create global entity and relation mappings
    sorted_entities = sorted(list(all_entities))
    sorted_relations = sorted(list(all_relations))
    global_entity_to_idx = {entity: idx for idx, entity in enumerate(sorted_entities)}
    global_relation_to_idx = {relation: idx for idx, relation in enumerate(sorted_relations)}
    
    print(f"🗂️ Building global indices...")
    print(f"   📊 Entities: {len(sorted_entities)}")
    print(f"   📊 Relations: {len(sorted_relations)}")
    
    # Process samples with global indices
    processed_data = []
    for i, sample in enumerate(tqdm(trex_data, desc="Processing with global indices")):
        sample_id = f"trex_{i}"
        processed_sample = process_trex_sample(sample, sample_id, subgraph_data, labels_data, global_entity_to_idx, global_relation_to_idx)
        if processed_sample:
            processed_data.append(processed_sample)
    
    # Save outputs
    print("\n💾 Saving processed data...")
    save_outputs(processed_data, sorted_entities, sorted_relations, labels_data, output_dir)
    
    print(f"\n🎉 T-REX processing completed!")
    print(f"📁 Output location: {output_dir}")
    print(f"📊 Final statistics:")
    print(f"   - Total samples: {len(processed_data)}")
    print(f"   - Unique entities: {len(all_entities)}")
    print(f"   - Unique relations: {len(all_relations)}")
    
    print(f"\n📋 Next steps:")
    print(f"1. Create search augmented processor: data_process_kg/trex_search_augmented_initial_entities.py")
    print(f"2. Test data loading: Check {output_dir}/train_simple.json")
    print(f"3. Run training pipeline integration test")

if __name__ == "__main__":
    main()
