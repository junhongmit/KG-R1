import json
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def process_simpleqa_data():
    # Load the SimpleQA test dataset
    test_data_path = PROJECT_ROOT / "data_kg" / "simpleqa" / "test_simple.json"
    entities_path = PROJECT_ROOT / "data_kg" / "simpleqa" / "entities.txt"
    entities_text_path = PROJECT_ROOT / "data_kg" / "simpleqa" / "entities_text.txt"
    
    # Load entity ID and text mapping
    with open(entities_path, 'r') as f:
        entity_ids = [line.strip() for line in f.readlines()]
    
    with open(entities_text_path, 'r') as f:
        entity_texts = [line.strip() for line in f.readlines()]
    
    def load_and_process(data_path, split_name):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        processed_data = []
        
        for item in data:
            question = item['question']
            answers = item['answers']
            sample_id = item['id']
            entity_indices = item.get('entities', [])
            subgraph_entities = item['subgraph']['entities']
            
            # Extract initial entities - entity_indices are global entity indices
            initial_entities = []
            initial_entity_ids = []
            for global_entity_idx in entity_indices:
                # entity_indices contains global entity indices directly
                if global_entity_idx < len(entity_ids) and global_entity_idx < len(entity_texts):
                    entity_id = entity_ids[global_entity_idx]
                    entity_text = entity_texts[global_entity_idx]
                    # Only include if this entity exists in the subgraph
                    if global_entity_idx in subgraph_entities:
                        initial_entities.append(entity_text)
                        initial_entity_ids.append(entity_id)
            
            # Create augmented question with initial entities
            if initial_entities:
                entities_str = ", ".join([f'"{entity}"' for entity in initial_entities])
                augmented_question = f"{question}? (Initial entities: {entities_str})"
            else:
                augmented_question = question
            
            kg_prompt = (
                "Answer the given question. You must conduct reasoning inside <think> and </think> "
                "first every time you get new information. After reasoning, if you find you lack some "
                "knowledge, you can query the knowledge graph by using <kg-query> function_name(arguments) </kg-query>, and it will "
                "return the top query results between <information> and </information>. You "
                "can query as many times as you want. If you find no further external knowledge "
                "needed, you can directly provide the answer inside <answer> and </answer> without "
                f"detailed illustrations. For example, <answer> Beijing </answer>.\n\nQuestion: {augmented_question}"
            )
            
            # Extract answers and kb_ids
            target_text = [ans.get('text', '') for ans in answers]
            target_kb_id = [ans.get('kb_id', '') for ans in answers]
            
            processed_item = {
                "data_source": "kgR1_simpleqa",
                "prompt": [{"role": "user", "content": kg_prompt}],
                "ability": "kg-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target_text": target_text,
                        "target_kb_id": target_kb_id
                    }
                },
                "extra_info": {
                    "split": split_name,
                    "sample_id": sample_id,
                    "dataset_name": "simpleqa",
                    "initial_entities": initial_entities,
                    "initial_entity_ids": initial_entity_ids
                }
            }
            
            processed_data.append(processed_item)
        
        return processed_data
    
    # Create output directory
    output_dir = PROJECT_ROOT / "data_kg" / "simpleqa_search_augmented_initial_entities"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and save test data only
    if os.path.exists(test_data_path):
        test_data = load_and_process(test_data_path, "test")
        test_df = pd.DataFrame(test_data)
        test_output = os.path.join(output_dir, "test.parquet")
        test_df.to_parquet(test_output, index=False)
        print(f"Saved {len(test_data)} test samples to {test_output}")
        print(f"Sample test initial entities: {test_data[0]['extra_info']['initial_entities'] if test_data else 'None'}")
    else:
        print(f"Test data not found at {test_data_path}")


if __name__ == "__main__":
    process_simpleqa_data()
