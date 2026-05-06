#!/usr/bin/env python3
"""
SimpleQA dataset processor for KG-R1 training pipeline

This script downloads SimpleQA from KGdata repository and processes it with 2-hop subgraphs
to create KG-R1 compatible training data following the README_BENCHMARK_INTEGRATION.md format.
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import argparse
from tqdm import tqdm
import pandas as pd

# Dataset URLs
SIMPLEQA_URL = "https://github.com/lanjiuqing64/KGdata/raw/main/QAdata/SimpleQA.json"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUBGRAPH_ROOT = PROJECT_ROOT / "data_kg" / "Subgraph"

# Subgraph paths (from downloaded Subgraph repository)  
SUBGRAPH_PATHS = {
    "triples": str(DEFAULT_SUBGRAPH_ROOT / "hop_2_Freebase" / "simpleqa_triples_2_hop_all.json"),
    "labels": str(DEFAULT_SUBGRAPH_ROOT / "hop_2_Freebase" / "simpleqa_labels_2_hop_all.json"),
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

def process_simpleqa_sample(sample: Dict, sample_id: str, subgraph_triples: Dict, subgraph_labels: Dict, global_entity_to_idx: Dict, global_relation_to_idx: Dict) -> Optional[Dict]:
    """
    Convert SimpleQA sample to KG-R1 compatible format
    Following README_BENCHMARK_INTEGRATION.md requirements
    """
    try:
        # Extract basic info
        question = sample.get('question', '').strip()
        if not question:
            return None
        
        # Process answers - SimpleQA format
        answers = []
        if 'answer' in sample:
            if isinstance(sample['answer'], list):
                for ans in sample['answer']:
                    if isinstance(ans, dict):
                        answers.append({
                            'kb_id': ans.get('kb_id', ans.get('id', '')),
                            'text': ans.get('text', str(ans))
                        })
                    else:
                        answers.append({
                            'kb_id': str(ans),
                            'text': str(ans)
                        })
            else:
                answers.append({
                    'kb_id': str(sample.get('answer_id', sample['answer'])),
                    'text': str(sample['answer'])
                })
        
        # Extract initial entities - SimpleQA uses 'topic_entity' field
        initial_entities = sample.get('topic_entity', [])
        if isinstance(initial_entities, dict):
            initial_entities = list(initial_entities.keys())
        elif not isinstance(initial_entities, list):
            initial_entities = []
        
        # Build subgraph from initial entities
        subgraph_entities = set()
        subgraph_relations = set()
        raw_tuples = []
        
        for entity_id in initial_entities:
            if isinstance(entity_id, str) and entity_id.startswith('m.'):
                subgraph_entities.add(entity_id)
            
            # Get triples for this entity from subgraph data
            if entity_id in subgraph_triples:
                for triple in subgraph_triples[entity_id]:
                    if len(triple) >= 3:
                        subject, relation, obj = triple[0], triple[1], triple[2]
                        if isinstance(subject, str) and subject.startswith('m.'):
                            subgraph_entities.add(subject)
                        if isinstance(obj, str) and obj.startswith('m.'):
                            subgraph_entities.add(obj)
                        if isinstance(relation, str) and ('.' in relation or '_' in relation):
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


def extract_entities_relations(processed_data: List[Dict]) -> Tuple[set, set]:
    """Extract all unique entities and relations for auxiliary files"""
    all_entities = set()
    all_relations = set()
    
    for sample in processed_data:
        subgraph = sample.get('subgraph', {})
        all_entities.update(subgraph.get('entities', []))
        all_relations.update(subgraph.get('relations', []))
    
    return all_entities, all_relations

def save_auxiliary_files(sorted_entities: list, sorted_relations: list, labels_data: Dict, output_dir: Path):
    """Save entities.txt, entities_text.txt, relations.txt files"""
    # Save entities.txt  
    with open(output_dir / 'entities.txt', 'w', encoding='utf-8') as f:
        for entity in sorted_entities:
            f.write(f"{entity}\n")
    
    # Save relations.txt
    with open(output_dir / 'relations.txt', 'w', encoding='utf-8') as f:
        for relation in sorted_relations:
            f.write(f"{relation}\n")
    
    # Save entities_text.txt (human-readable names or original IDs, one per line)
    with open(output_dir / 'entities_text.txt', 'w', encoding='utf-8') as f:
        for entity_id in sorted_entities:
            # Look for human-readable name in labels_data
            human_name = None
            
            # Check if this entity_id appears in any labels_data entry
            for main_entity_id, related_entities in labels_data.items():
                if entity_id == main_entity_id:
                    # Use the first available human name from related entities
                    for eid, label in related_entities.items():
                        if label and label != eid and not label.startswith('m.') and label != "unnamed_entity":
                            human_name = label
                            break
                    # If no human name found or it's unnamed_entity, keep original entity_id
                    if human_name is None or human_name == "unnamed_entity":
                        human_name = entity_id  # Keep original ID instead of "unnamed_entity"
                    break
                elif entity_id in related_entities:
                    # Found as related entity - use its human-readable label
                    label = related_entities[entity_id]
                    if label and not label.startswith('m.') and label != "unnamed_entity":
                        human_name = label
                    else:
                        human_name = entity_id  # Keep original ID instead of "unnamed_entity"
                    break
            
            # Fallback to entity_id if no human name found
            if human_name is None:
                human_name = entity_id
            
            # Only write the human name or original entity ID, no prefix
            f.write(f"{human_name}\n")
    
    print(f"✅ Saved auxiliary files:")
    print(f"   📄 entities.txt ({len(sorted_entities)} entities)")
    print(f"   📄 entities_text.txt ({len(sorted_entities)} entity names)")
    print(f"   📄 relations.txt ({len(sorted_relations)} relations)")

def save_test_data(test_data: List, output_dir: Path):
    """Save test data as JSONL file for KG-R1 compatibility"""
    filename = "test_simple.json"
    with open(output_dir / filename, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {filename} ({len(test_data)} samples)")

def main():
    parser = argparse.ArgumentParser(description="Process SimpleQA dataset for KG-R1")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "data_kg" / "simpleqa"), help="Output directory")
    parser.add_argument("--subgraph_triples", default=SUBGRAPH_PATHS["triples"], help="Path to SimpleQA 2-hop triples JSON")
    parser.add_argument("--subgraph_labels", default=SUBGRAPH_PATHS["labels"], help="Path to SimpleQA 2-hop labels JSON")
    parser.add_argument("--force_redownload", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Processing SimpleQA dataset for KG-R1")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    
    # Step 1: Download SimpleQA dataset
    simpleqa_file = output_dir / "simpleqa_raw.json"
    if not simpleqa_file.exists() or args.force_redownload:
        if not download_file(SIMPLEQA_URL, str(simpleqa_file)):
            print("❌ Failed to download SimpleQA dataset")
            return
    else:
        print(f"✅ Using existing SimpleQA file: {simpleqa_file}")
    
    # Step 2: Load datasets
    print("\n📊 Loading datasets...")
    simpleqa_data = load_json_data(str(simpleqa_file))
    if not simpleqa_data:
        return
    
    subgraph_data = load_json_data(args.subgraph_triples) or {}
    labels_data = load_json_data(args.subgraph_labels) or {}
    
    print(f"✅ Loaded {len(simpleqa_data)} SimpleQA samples")
    print(f"✅ Loaded {len(subgraph_data)} subgraph entries")
    print(f"✅ Loaded {len(labels_data)} label entries")
    
    # Step 3: First pass - collect ALL entities and relations
    print("\n🔍 First pass: collecting all entities and relations...")
    all_entities = set()
    all_relations = set()
    
    max_samples = len(simpleqa_data)  # Process all samples
    print(f"   📊 Processing first {max_samples} samples...")
    
    for i, sample in enumerate(simpleqa_data[:max_samples]):
        # Extract initial entities from topic_entity field
        initial_entities = sample.get('topic_entity', [])
        if isinstance(initial_entities, dict):
            initial_entities = list(initial_entities.keys())
        elif not isinstance(initial_entities, list):
            initial_entities = []
        
        for entity_id in initial_entities:
            if isinstance(entity_id, str) and entity_id.startswith('m.'):
                all_entities.add(entity_id)
            
            # Get triples for this entity from subgraph data
            if entity_id in subgraph_data:
                for triple in subgraph_data[entity_id]:
                    if len(triple) >= 3:
                        subject, relation, obj = triple[0], triple[1], triple[2]
                        if isinstance(subject, str) and subject.startswith('m.'):
                            all_entities.add(subject)
                        if isinstance(obj, str) and obj.startswith('m.'):
                            all_entities.add(obj)
                        if isinstance(relation, str) and ('.' in relation or '_' in relation):
                            all_relations.add(relation)
    
    # Create global entity and relation mappings
    sorted_entities = sorted(list(all_entities))
    sorted_relations = sorted(list(all_relations))
    global_entity_to_idx = {entity: idx for idx, entity in enumerate(sorted_entities)}
    global_relation_to_idx = {relation: idx for idx, relation in enumerate(sorted_relations)}
    
    print(f"   ✅ Collected {len(all_entities)} entities and {len(all_relations)} relations")
    
    # Step 4: Second pass - process samples with global indices
    print("\n🔄 Second pass: processing samples with global indices...")
    processed_data = []
    samples_with_subgraph = 0
    
    for i, sample in enumerate(simpleqa_data[:max_samples]):
        sample_id = f"simpleqa_{i}"
        processed_sample = process_simpleqa_sample(sample, sample_id, subgraph_data, labels_data, global_entity_to_idx, global_relation_to_idx)
        if processed_sample:
            processed_data.append(processed_sample)
            if processed_sample['subgraph']['entities']:
                samples_with_subgraph += 1
    
    print(f"✅ Successfully processed {len(processed_data)} samples")
    print(f"   📊 Samples with subgraph data: {samples_with_subgraph}")
    
    # Step 5: All data becomes test set for KG-R1 compatibility
    test_data = processed_data
    print(f"\n📊 Using all {len(test_data)} samples as test set (KG-R1 compatible)")
    
    print(f"   ✅ Final count: {len(all_entities)} entities and {len(all_relations)} relations")
    
    # Step 6: Save all outputs
    print("\n💾 Saving processed data...")
    save_test_data(test_data, output_dir)
    save_auxiliary_files(sorted_entities, sorted_relations, labels_data, output_dir)
    
    print(f"\n🎉 SimpleQA processing completed!")
    print(f"📁 Output location: {output_dir}")
    print(f"📊 Final statistics:")
    print(f"   - Total samples: {len(processed_data)}")
    print(f"   - Unique entities: {len(all_entities)}")
    print(f"   - Unique relations: {len(all_relations)}")
    
    print(f"\n📋 Next steps:")
    print(f"1. Create search augmented processor: data_process_kg/simpleqa_search_augmented_initial_entities.py")
    print(f"2. Test data loading: Check {output_dir}/test_simple.json")
    print(f"3. Run KG-R1 server with SimpleQA support")
    print(f"4. Test evaluation pipeline integration")

if __name__ == "__main__":
    main()
