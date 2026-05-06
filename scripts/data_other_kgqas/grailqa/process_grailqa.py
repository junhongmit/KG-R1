#!/usr/bin/env python3
"""
GrailQA dataset processor for KG-R1 server compatibility

This script processes the GrailQA dataset and converts it to the KG-R1 compatible format:
- Generates test_simple.json (test-only for KG-R1)
- Creates entities.txt, entities_text.txt, relations.txt from subgraph data
- Uses subgraph structure from KG-R1/data_kg/Subgraph by default
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUBGRAPH_ROOT = PROJECT_ROOT / "data_kg" / "Subgraph"

def load_grailqa_data(file_path: Path) -> List[Dict]:
    """Load GrailQA dataset from JSON file"""
    print(f"📊 Loading GrailQA data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   ✅ Loaded {len(data)} samples")
    return data

def process_grailqa_sample(sample: Dict, sample_id: str, subgraph_triples: Dict, subgraph_labels: Dict, global_entity_to_idx: Dict, global_relation_to_idx: Dict) -> Dict:
    """
    Convert GrailQA sample to KG-R1 compatible format with subgraph data
    """
    try:
        # Extract basic info
        question = sample.get('question', '').strip()
        if not question:
            return None
        
        # Process answers - GrailQA format
        answers = []
        if 'answer' in sample:
            if isinstance(sample['answer'], list):
                for ans in sample['answer']:
                    if isinstance(ans, dict):
                        # GrailQA answer format: {answer_type, answer_argument, entity_name}
                        kb_id = ans.get('answer_argument', '')
                        entity_name = ans.get('entity_name', '')
                        
                        # Only add if we have valid kb_id and text
                        if kb_id and entity_name:
                            answers.append({
                                'kb_id': kb_id,
                                'text': entity_name
                            })
                    else:
                        # Fallback for non-dict answers
                        answers.append({
                            'kb_id': str(ans),
                            'text': str(ans)
                        })
            else:
                # Single answer (not a list)
                answers.append({
                    'kb_id': str(sample['answer']),
                    'text': str(sample.get('answer_text', sample['answer']))
                })
        
        # Extract initial entities from topic_entity field
        topic_entities = sample.get('topic_entity', {})
        initial_entities = list(topic_entities.keys())
        
        # Build subgraph from initial entities
        subgraph_entities = set()
        subgraph_relations = set()
        raw_tuples = []
        
        for entity_id in initial_entities:
            # Add the entity itself
            subgraph_entities.add(entity_id)
            
            # Get triples for this entity from subgraph data
            if entity_id in subgraph_triples:
                for triple in subgraph_triples[entity_id]:
                    if len(triple) >= 3:
                        subject, relation, obj = triple[0], triple[1], triple[2]
                        subgraph_entities.add(subject)
                        subgraph_entities.add(obj)
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

def extract_entities_relations_from_subgraph(triples_file: Path, labels_file: Path) -> Tuple[set, set, dict]:
    """Extract entities and relations from subgraph files with corrected JSON structure"""
    all_entities = set()
    all_relations = set()
    labels_data = {}
    
    try:
        # Load triples
        print(f"   📊 Loading triples from {triples_file}")
        with open(triples_file, 'r', encoding='utf-8') as f:
            triples_data = json.load(f)
            
        # Extract entities and relations from triples
        # Structure: entity_id -> [[subject, relation, object], ...]
        for entity_id, triples_list in triples_data.items():
            all_entities.add(entity_id)  # Add the key entity
            for triple in triples_list:
                if len(triple) >= 3:
                    subject, relation, obj = triple[0], triple[1], triple[2]
                    all_entities.add(subject)
                    all_entities.add(obj)
                    all_relations.add(relation)
        
        print(f"   ✅ Extracted {len(all_entities)} entities and {len(all_relations)} relations from triples")
        
        # Load labels
        print(f"   📊 Loading labels from {labels_file}")
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
            
        # Add entities from labels too (only entity IDs, not human names)
        # Structure: entity_id -> {related_entity_id -> human_readable_name}
        for entity_id, related_entities in labels_data.items():
            # Only add if it looks like an entity ID (starts with 'm.')
            if isinstance(entity_id, str) and entity_id.startswith('m.'):
                all_entities.add(entity_id)  # Add the key entity
            for related_entity_id in related_entities.keys():
                # Only add if it looks like an entity ID (starts with 'm.')
                if isinstance(related_entity_id, str) and related_entity_id.startswith('m.'):
                    all_entities.add(related_entity_id)
            
        print(f"   ✅ Loaded labels for {len(labels_data)} entity groups")
        print(f"   ✅ Final count: {len(all_entities)} entities and {len(all_relations)} relations")
        
    except Exception as e:
        print(f"   ❌ Error loading subgraph files: {e}")
        
    return all_entities, all_relations, labels_data

def save_auxiliary_files(sorted_entities: list, sorted_relations: list, labels_data: dict, output_dir: Path):
    """Save entities.txt, entities_text.txt, and relations.txt files"""
    # Save entities.txt  
    with open(output_dir / 'entities.txt', 'w', encoding='utf-8') as f:
        for entity in sorted_entities:
            f.write(f"{entity}\n")
    
    # Save relations.txt
    with open(output_dir / 'relations.txt', 'w', encoding='utf-8') as f:
        for relation in sorted_relations:
            f.write(f"{relation}\n")
    
    # Save entities_text.txt (human-readable names or original MIDs, one per line)
    with open(output_dir / 'entities_text.txt', 'w', encoding='utf-8') as f:
        for entity_id in sorted_entities:
            # Look for human-readable name in labels_data
            human_name = entity_id  # Default to original MID
            
            # Check if this entity_id appears in any labels_data entry
            for main_entity_id, related_entities in labels_data.items():
                if entity_id == main_entity_id:
                    # Check if entity has a good human-readable name
                    potential_name = related_entities.get(entity_id, None)
                    if (potential_name and 
                        potential_name != "unnamed_entity" and 
                        not potential_name.startswith('m.') and
                        potential_name.strip()):  # Ensure it's not empty/whitespace
                        human_name = potential_name
                    break
                elif entity_id in related_entities:
                    # Found as related entity, check for good human-readable name
                    potential_name = related_entities[entity_id]
                    if (potential_name and 
                        potential_name != "unnamed_entity" and 
                        not potential_name.startswith('m.') and
                        potential_name.strip()):  # Ensure it's not empty/whitespace
                        human_name = potential_name
                    break
            
            # Always write something - either human name or original MID
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
    parser = argparse.ArgumentParser(description="Process GrailQA dataset for KG-R1")
    parser.add_argument("--input_dir", type=str, default=str(PROJECT_ROOT / "data_kg" / "grailqa"),
                      help="Directory containing grailqa.json file")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "data_kg" / "grailqa"),
                      help="Output directory for processed data")
    parser.add_argument("--subgraph_dir", type=str, 
                      default=str(DEFAULT_SUBGRAPH_ROOT / "hop_2_Freebase"),
                      help="Directory containing subgraph files")
    parser.add_argument("--max_samples", type=int, default=0,
                      help="Limit samples for smoke tests; 0 means all")
    
    args = parser.parse_args()
    
    print("🚀 Processing GrailQA dataset for KG-R1")
    print("=" * 60)
    
    # Setup paths
    input_file = Path(args.input_dir) / "grailqa.json"
    output_dir = Path(args.output_dir)
    subgraph_triples_file = Path(args.subgraph_dir) / "grailqa_triples_2_hop_all.json"
    subgraph_labels_file = Path(args.subgraph_dir) / "grailqa_labels_2_hop_all.json"
    
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Subgraph triples: {subgraph_triples_file}")
    print(f"📁 Subgraph labels: {subgraph_labels_file}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load raw data
    raw_data = load_grailqa_data(input_file)
    
    # Step 2: Load subgraph data upfront
    print("\n🔍 Loading subgraph data...")
    print(f"   📊 Loading triples from {subgraph_triples_file}")
    with open(subgraph_triples_file, 'r', encoding='utf-8') as f:
        subgraph_triples = json.load(f)
    print(f"   ✅ Loaded triples for {len(subgraph_triples)} entities")
    
    print(f"   📊 Loading labels from {subgraph_labels_file}")
    with open(subgraph_labels_file, 'r', encoding='utf-8') as f:
        subgraph_labels = json.load(f)
    print(f"   ✅ Loaded labels for {len(subgraph_labels)} entities")
    
    # Step 3: First pass - collect ALL entities and relations
    print("\n🔍 First pass: collecting all entities and relations...")
    all_entities = set()
    all_relations = set()
    
    max_samples = args.max_samples if args.max_samples > 0 else len(raw_data)
    max_samples = min(max_samples, len(raw_data))
    print(f"   📊 Processing first {max_samples} samples...")
    
    for i, sample in enumerate(raw_data[:max_samples]):
        # Extract initial entities from topic_entity field
        topic_entities = sample.get('topic_entity', {})
        initial_entities = list(topic_entities.keys())
        
        for entity_id in initial_entities:
            if isinstance(entity_id, str) and entity_id.startswith('m.'):
                all_entities.add(entity_id)
            
            # Get triples for this entity from subgraph data
            if entity_id in subgraph_triples:
                for triple in subgraph_triples[entity_id]:
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
    
    for i, sample in enumerate(raw_data[:max_samples]):
        sample_id = f"grailqa_{i}"
        processed_sample = process_grailqa_sample(sample, sample_id, subgraph_triples, subgraph_labels, global_entity_to_idx, global_relation_to_idx)
        if processed_sample:
            processed_data.append(processed_sample)
            if processed_sample['subgraph']['entities']:
                samples_with_subgraph += 1
    
    print(f"✅ Successfully processed {len(processed_data)} samples")
    print(f"   📊 Samples with subgraph data: {samples_with_subgraph}")
    
    # Step 5: All data becomes test set for KG-R1 compatibility
    test_data = processed_data
    print(f"\n📊 Using all {len(test_data)} samples as test set (KG-R1 compatible)")
    
    # Step 6: Load labels data for entities_text.txt generation
    print(f"\n📊 Loading labels from {subgraph_labels_file}")
    with open(subgraph_labels_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    print(f"   ✅ Final count: {len(all_entities)} entities and {len(all_relations)} relations")
    
    # Step 7: Save all outputs
    print("\n💾 Saving processed data...")
    save_test_data(test_data, output_dir)
    save_auxiliary_files(sorted_entities, sorted_relations, labels_data, output_dir)
    
    print(f"\n🎉 GrailQA processing completed!")
    print(f"📁 Output saved to: {output_dir}")
    print(f"📊 Generated files:")
    print(f"   📄 test_simple.json ({len(test_data)} samples)")
    print(f"   📄 entities.txt ({len(all_entities)} entities)")
    print(f"   📄 entities_text.txt ({len(all_entities)} entity names)")
    print(f"   📄 relations.txt ({len(all_relations)} relations)")
    
    print(f"\n📋 Next steps:")
    print(f"1. Test data loading: Check {output_dir}/test_simple.json")
    print(f"2. Run KG-R1 server with GrailQA support")
    print(f"3. Test evaluation pipeline integration")

if __name__ == "__main__":
    main()
