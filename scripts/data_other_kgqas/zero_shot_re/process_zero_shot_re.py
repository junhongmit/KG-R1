#!/usr/bin/env python3
"""
Zero-shot RE dataset processor for KG-R1 training pipeline

This script downloads Zero-shot RE from KGdata repository and processes it with 2-hop subgraphs
to create KG-R1 compatible training data.
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import argparse
from tqdm import tqdm

# Dataset URLs
ZERO_SHOT_RE_URL = "https://github.com/lanjiuqing64/KGdata/raw/main/QAdata/zero_shot_re.json"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUBGRAPH_ROOT = PROJECT_ROOT / "data_kg" / "Subgraph"

# Subgraph paths (from downloaded Subgraph repository)  
SUBGRAPH_PATHS = {
    "triples": str(DEFAULT_SUBGRAPH_ROOT / "hop_2_Wikidata" / "Zero_Shot_RE_2hop.json"),
    "labels": str(DEFAULT_SUBGRAPH_ROOT / "hop_2_Wikidata" / "Zero_Shot_RE_2hop_labels.json"),
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

def load_json_data(file_path: str) -> Optional[Any]:
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

def process_zero_shot_re_sample(sample: Dict, sample_id: str, subgraph_triples: Dict, 
                                subgraph_labels: Dict, global_entity_to_idx: Dict, 
                                global_relation_to_idx: Dict) -> Optional[Dict]:
    """Process a single Zero-shot RE sample to KG-R1 compatible format"""
    try:
        # Extract basic info - Zero-shot RE uses 'template_questions'
        questions = sample.get('template_questions', [])
        if not questions:
            return None
        question = questions[0].strip()  # Use first template question
        
        # Process answers - Zero-shot RE format
        answers = []
        if 'answer' in sample:
            # Get topic entity ID as answer ID
            topic_entity = sample.get('topic_entity', {})
            answer_id = ''
            if topic_entity:
                # Find the entity ID that matches the answer
                for eid, ename in topic_entity.items():
                    if ename == sample['answer'] or sample['answer'] in sample.get('alias', []):
                        answer_id = eid
                        break
            
            answers.append({
                'kb_id': answer_id,
                'text': str(sample['answer'])
            })
            
            # Add aliases as additional answers
            for alias in sample.get('alias', []):
                if alias != sample['answer']:
                    answers.append({
                        'kb_id': answer_id,
                        'text': alias
                    })
        
        # Extract initial entities from topic_entity
        initial_entities = []
        if 'topic_entity' in sample and isinstance(sample['topic_entity'], dict):
            initial_entities = list(sample['topic_entity'].keys())
        
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

def save_outputs(processed_data: List[Dict], sorted_entities: List, sorted_relations: List, 
                labels_data: Dict, output_dir: Path):
    """Save all outputs for Zero-shot RE dataset"""
    # Save test data only
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
    parser = argparse.ArgumentParser(description="Process Zero-shot RE dataset for KG-R1")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "data_kg" / "zero_shot_re"), help="Output directory")
    parser.add_argument("--subgraph_triples", default=SUBGRAPH_PATHS["triples"], help="Path to Zero-shot RE 2-hop triples JSON")
    parser.add_argument("--subgraph_labels", default=SUBGRAPH_PATHS["labels"], help="Path to Zero-shot RE 2-hop labels JSON")
    parser.add_argument("--force_redownload", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Processing Zero-shot RE dataset for KG-R1")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    
    # Step 1: Check for Zero-shot RE dataset
    zero_shot_re_file = output_dir / "zero_shot_re_raw.json"
    if not zero_shot_re_file.exists() or args.force_redownload:
        if not download_file(ZERO_SHOT_RE_URL, str(zero_shot_re_file)):
            print("❌ Failed to download Zero-shot RE dataset")
            return
    else:
        print(f"✅ Using existing Zero-shot RE file: {zero_shot_re_file}")
    
    # Step 2: Load datasets
    print("\n📊 Loading datasets...")
    zero_shot_re_data = load_json_data(str(zero_shot_re_file))
    if not zero_shot_re_data:
        return
    
    subgraph_data = load_json_data(args.subgraph_triples) or {}
    labels_data = load_json_data(args.subgraph_labels) or {}
    
    print(f"✅ Loaded {len(zero_shot_re_data)} Zero-shot RE samples")
    print(f"✅ Loaded {len(subgraph_data)} subgraph entries")
    print(f"✅ Loaded {len(labels_data)} label entries")
    
    # Process all data as test set for KG-R1
    print("\n🔄 Processing Zero-shot RE samples...")
    processed_data = []
    all_entities = set()
    all_relations = set()
    
    for i, sample in enumerate(tqdm(zero_shot_re_data, desc="Processing samples")):
        sample_id = f"zero_shot_re_{i}"
        
        # Extract entities and relations first - Zero-shot RE uses 'topic_entity'
        initial_entities = []
        if 'topic_entity' in sample and isinstance(sample['topic_entity'], dict):
            initial_entities = list(sample['topic_entity'].keys())
            
        for entity_id in initial_entities:
            if isinstance(entity_id, str) and entity_id.startswith('Q'):
                all_entities.add(entity_id)
                
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
    
    # Create global mappings
    sorted_entities = sorted(list(all_entities))
    sorted_relations = sorted(list(all_relations))
    global_entity_to_idx = {entity: idx for idx, entity in enumerate(sorted_entities)}
    global_relation_to_idx = {relation: idx for idx, relation in enumerate(sorted_relations)}
    
    # Process samples with global mappings
    for i, sample in enumerate(tqdm(zero_shot_re_data, desc="Processing with global indices")):
        sample_id = f"zero_shot_re_{i}"
        processed_sample = process_zero_shot_re_sample(sample, sample_id, subgraph_data, 
                                                      labels_data, global_entity_to_idx, 
                                                      global_relation_to_idx)
        if processed_sample:
            processed_data.append(processed_sample)
    
    print(f"✅ Successfully processed {len(processed_data)} samples")
    
    # Save outputs
    print(f"\n💾 Saving processed data...")
    save_outputs(processed_data, sorted_entities, sorted_relations, labels_data, output_dir)
    
    print(f"\n🎉 Zero-shot RE processing completed!")
    print(f"📊 Final statistics:")
    print(f"   - Test samples: {len(processed_data)}")
    print(f"   - Unique entities: {len(all_entities)}")
    print(f"   - Unique relations: {len(all_relations)}")

if __name__ == "__main__":
    main()
