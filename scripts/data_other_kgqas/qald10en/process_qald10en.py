#!/usr/bin/env python3
"""
QALD-10 dataset processor for KG-R1 training pipeline

This script processes the QALD-10 dataset with 2-hop Wikidata subgraphs
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
QALD10_URL = "https://github.com/KGQA/QALD_10/raw/main/data/qald_10_train.json"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUBGRAPH_ROOT = PROJECT_ROOT / "data_kg" / "Subgraph"

# Subgraph paths (from downloaded Subgraph repository)  
SUBGRAPH_PATHS = {
    "triples": str(DEFAULT_SUBGRAPH_ROOT / "hop_2_Wikidata" / "qald_10-en_2hop.json"),
    "labels": str(DEFAULT_SUBGRAPH_ROOT / "hop_2_Wikidata" / "qald_10-en_2hop_labels.json"),
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

def process_qald10_sample(sample: Dict, sample_id: str, subgraph_triples: Dict, subgraph_labels: Dict, global_entity_to_idx: Dict, global_relation_to_idx: Dict) -> Optional[Dict]:
    """
    Convert QALD-10 sample to KG-R1 compatible format
    Following README_BENCHMARK_INTEGRATION.md requirements
    """
    try:
        # Extract basic info. Official QALD uses a list of localized questions;
        # ToG-style data stores the English question directly as a string.
        raw_question = sample.get('question', '')
        if isinstance(raw_question, list):
            question = raw_question[0].get('string', '').strip() if raw_question else ''
        else:
            question = str(raw_question).strip()
        if not question:
            return None
        
        # Process answers - QALD format
        answers = []
        if 'answers' in sample:
            for ans in sample['answers'][0].get('results', {}).get('bindings', []):
                if 'uri' in ans:
                    entity_uri = ans['uri']['value']
                    # Extract Wikidata ID from URI
                    if 'wikidata.org' in entity_uri:
                        entity_id = entity_uri.split('/')[-1]
                        human_name = subgraph_labels.get(entity_id, entity_id)
                        answers.append({
                            'kb_id': entity_id,
                            'text': human_name
                        })
        elif 'answer' in sample and isinstance(sample['answer'], dict):
            for entity_id, entity_name in sample['answer'].items():
                answers.append({
                    'kb_id': entity_id,
                    'text': entity_name
                })
        
        # Extract initial entities from SPARQL query
        initial_entities = []
        if 'query' in sample:
            sparql = sample['query'].get('sparql', '')
        else:
            sparql = sample.get('sparql', '')

        # Prefer explicit linked entities when present, then add SPARQL entities.
        if 'qid_topic_entity' in sample and isinstance(sample['qid_topic_entity'], dict):
            initial_entities.extend(sample['qid_topic_entity'].keys())

        import re
        wikidata_entities = re.findall(r'wd:(Q\d+)', sparql)
        initial_entities.extend(wikidata_entities)
        
        # Remove duplicates
        initial_entities = list(set(initial_entities))
        
        # Build subgraph from initial entities
        subgraph_entities = set()
        subgraph_relations = set()
        raw_tuples = []
        
        for entity_id in initial_entities:
            if entity_id.startswith('Q') and entity_id in subgraph_triples:
                subgraph_entities.add(entity_id)
                
                # Get triples for this entity from subgraph data
                for triple in subgraph_triples[entity_id]:
                    if len(triple) >= 3:
                        subject, relation, obj = triple[0], triple[1], triple[2]
                        if isinstance(subject, str) and subject.startswith('Q'):
                            subgraph_entities.add(subject)
                        if isinstance(obj, str) and obj.startswith('Q'):
                            subgraph_entities.add(obj)
                        if isinstance(relation, str) and relation.startswith('P'):
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
    
    # Save entities_text.txt (only human-readable names, one per line)
    with open(output_dir / 'entities_text.txt', 'w', encoding='utf-8') as f:
        for entity_id in sorted_entities:
            # Look for human-readable name in labels_data
            human_name = labels_data.get(entity_id, entity_id)
            # Only write the human name, no entity_id prefix
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
    parser = argparse.ArgumentParser(description="Process QALD-10 dataset for KG-R1")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "data_kg" / "qald10en"), help="Output directory")
    parser.add_argument("--subgraph_triples", default=SUBGRAPH_PATHS["triples"], help="Path to QALD-10 2-hop triples JSON")
    parser.add_argument("--subgraph_labels", default=SUBGRAPH_PATHS["labels"], help="Path to QALD-10 2-hop labels JSON")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples for smoke tests; 0 means all")
    parser.add_argument("--force_redownload", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Processing QALD-10 dataset for KG-R1")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    
    # Step 1: Find existing QALD-10 dataset
    possible_paths = [
        output_dir / "qald10en_raw.json",
    ]
    
    qald10_file = None
    for path in possible_paths:
        if Path(path).exists():
            qald10_file = Path(path)
            print(f"✅ Using existing QALD-10 file: {qald10_file}")
            break
    
    if not qald10_file:
        qald10_file = output_dir / "qald10en_raw.json"
        if not download_file(QALD10_URL, str(qald10_file)):
            print("❌ Failed to download QALD-10 dataset")
            return
    elif args.force_redownload:
        qald10_file = output_dir / "qald10en_raw.json"
        if not download_file(QALD10_URL, str(qald10_file)):
            print("❌ Failed to download QALD-10 dataset")
            return
    
    # Step 2: Load datasets
    print("\n📊 Loading datasets...")
    qald10_data = load_json_data(str(qald10_file))
    if not qald10_data:
        return
    
    # Extract questions from either official QALD dict format or ToG-style list format.
    questions = qald10_data.get('questions', []) if isinstance(qald10_data, dict) else qald10_data
    
    subgraph_data = load_json_data(args.subgraph_triples) or {}
    labels_data = load_json_data(args.subgraph_labels) or {}
    
    print(f"✅ Loaded {len(questions)} QALD-10 questions")
    print(f"✅ Loaded {len(subgraph_data)} subgraph entries")
    print(f"✅ Loaded {len(labels_data)} label entries")
    
    # Step 3: First pass - collect ALL entities and relations (limit to 10 samples)
    print("\n🔍 First pass: collecting all entities and relations...")
    all_entities = set()
    all_relations = set()
    
    max_samples = args.max_samples if args.max_samples > 0 else len(questions)
    max_samples = min(max_samples, len(questions))
    print(f"   📊 Processing first {max_samples} samples...")
    
    for i, question in enumerate(questions[:max_samples]):
        # Extract initial entities from explicit qid_topic_entity and/or SPARQL query
        initial_entities = []
        if 'qid_topic_entity' in question and isinstance(question['qid_topic_entity'], dict):
            initial_entities.extend(question['qid_topic_entity'].keys())
        if 'query' in question:
            sparql = question['query'].get('sparql', '')
        else:
            sparql = question.get('sparql', '')
        import re
        wikidata_entities = re.findall(r'wd:(Q\d+)', sparql)
        initial_entities.extend(wikidata_entities)
        
        for entity_id in initial_entities:
            if entity_id.startswith('Q') and entity_id in subgraph_data:
                all_entities.add(entity_id)
                
                # Get triples for this entity from subgraph data
                for triple in subgraph_data[entity_id]:
                    if len(triple) >= 3:
                        subject, relation, obj = triple[0], triple[1], triple[2]
                        if isinstance(subject, str) and subject.startswith('Q'):
                            all_entities.add(subject)
                        if isinstance(obj, str) and obj.startswith('Q'):
                            all_entities.add(obj)
                        if isinstance(relation, str) and relation.startswith('P'):
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
    
    for i, question in enumerate(questions[:max_samples]):
        sample_id = f"qald10_{i}"
        processed_sample = process_qald10_sample(question, sample_id, subgraph_data, labels_data, global_entity_to_idx, global_relation_to_idx)
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
    
    print(f"\n🎉 QALD-10 processing completed!")
    print(f"📁 Output location: {output_dir}")
    print(f"📊 Final statistics:")
    print(f"   - Total samples: {len(processed_data)}")
    print(f"   - Unique entities: {len(all_entities)}")
    print(f"   - Unique relations: {len(all_relations)}")
    
    print(f"\n📋 Next steps:")
    print(f"1. Test data loading: Check {output_dir}/test_simple.json")
    print(f"2. Run KG-R1 server with QALD-10 support")
    print(f"3. Test evaluation pipeline integration")

if __name__ == "__main__":
    main()
