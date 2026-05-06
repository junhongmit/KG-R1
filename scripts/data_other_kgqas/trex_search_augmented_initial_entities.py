import json
import os
from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def convert_sep_to_natural_language(question):
    """
    Convert T-REx [SEP] format to natural language questions with comprehensive relation coverage
    Example: "The Key to Theosophy [SEP] author" -> "Who is the author of The Key to Theosophy?"
    """
    if '[SEP]' not in question:
        return question
    
    try:
        parts = question.split(' [SEP] ')
        if len(parts) != 2:
            return question
            
        subject, relation = parts[0].strip(), parts[1].strip()
        relation_lower = relation.lower().strip()
        
        # Define comprehensive relation mappings to question templates
        relation_templates = {
            # People relations
            'author': f"Who is the author of {subject}?",
            'creator': f"Who created {subject}?",
            'director': f"Who directed {subject}?",
            'producer': f"Who produced {subject}?",
            'composer': f"Who composed {subject}?",
            'writer': f"Who wrote {subject}?",
            'painter': f"Who painted {subject}?",
            'sculptor': f"Who sculpted {subject}?",
            'architect': f"Who was the architect of {subject}?",
            'designer': f"Who designed {subject}?",
            'inventor': f"Who invented {subject}?",
            'founder': f"Who founded {subject}?",
            'developer': f"Who developed {subject}?",
            'editor': f"Who edited {subject}?",
            'translator': f"Who translated {subject}?",
            'illustrator': f"Who illustrated {subject}?",
            'performer': f"Who performed {subject}?",
            'actor': f"Who starred in {subject}?",
            'actress': f"Who starred in {subject}?",
            
            # Location relations
            'country': f"Which country is {subject} in?",
            'location': f"Where is {subject} located?",
            'place of birth': f"Where was {subject} born?",
            'place of death': f"Where did {subject} die?",
            'headquarters': f"Where is the headquarters of {subject}?",
            'capital': f"What is the capital of {subject}?",
            'continent': f"Which continent is {subject} in?",
            'city': f"Which city is {subject} in?",
            'state': f"Which state is {subject} in?",
            'region': f"Which region is {subject} in?",
            'birthplace': f"Where was {subject} born?",
            'origin': f"Where did {subject} originate?",
            'based in': f"Where is {subject} based?",
            'located in': f"Where is {subject} located?",
            
            # Descriptive relations
            'genre': f"What genre is {subject}?",
            'style': f"What style is {subject}?",
            'type': f"What type is {subject}?",
            'category': f"What category is {subject}?",
            'classification': f"What is the classification of {subject}?",
            'material': f"What material is {subject} made of?",
            'color': f"What color is {subject}?",
            'format': f"What format is {subject}?",
            'medium': f"What medium is {subject}?",
            'subject': f"What is the subject of {subject}?",
            'theme': f"What is the theme of {subject}?",
            'topic': f"What is the topic of {subject}?",
            
            # Time relations
            'birth date': f"When was {subject} born?",
            'death date': f"When did {subject} die?",
            'publication date': f"When was {subject} published?",
            'release date': f"When was {subject} released?",
            'date': f"When did {subject} occur?",
            'founded': f"When was {subject} founded?",
            'established': f"When was {subject} established?",
            'created': f"When was {subject} created?",
            'built': f"When was {subject} built?",
            'premiere': f"When did {subject} premiere?",
            'debut': f"When did {subject} debut?",
            'start date': f"When did {subject} start?",
            'end date': f"When did {subject} end?",
            'year': f"What year was {subject}?",
            
            # Professional relations
            'occupation': f"What is the occupation of {subject}?",
            'profession': f"What is the profession of {subject}?",
            'job': f"What job does {subject} have?",
            'position': f"What position does {subject} hold?",
            'role': f"What role does {subject} have?",
            'career': f"What is the career of {subject}?",
            'field': f"What field is {subject} in?",
            'industry': f"What industry is {subject} in?",
            'employer': f"Who employs {subject}?",
            'workplace': f"Where does {subject} work?",
            
            # Personal relations
            'nationality': f"What is the nationality of {subject}?",
            'citizenship': f"What citizenship does {subject} have?",
            'ethnicity': f"What is the ethnicity of {subject}?",
            'religion': f"What religion does {subject} follow?",
            'political party': f"Which political party does {subject} belong to?",
            'education': f"Where was {subject} educated?",
            'alma mater': f"Where did {subject} study?",
            'degree': f"What degree does {subject} have?",
            
            # Language relations
            'language': f"What language is {subject} in?",
            'original language': f"What is the original language of {subject}?",
            'spoken language': f"What language does {subject} speak?",
            'official language': f"What is the official language of {subject}?",
            
            # Family relations
            'spouse': f"Who is the spouse of {subject}?",
            'partner': f"Who is the partner of {subject}?",
            'child': f"Who is the child of {subject}?",
            'parent': f"Who is the parent of {subject}?",
            'father': f"Who is the father of {subject}?",
            'mother': f"Who is the mother of {subject}?",
            'sibling': f"Who is the sibling of {subject}?",
            'brother': f"Who is the brother of {subject}?",
            'sister': f"Who is the sister of {subject}?",
            'son': f"Who is the son of {subject}?",
            'daughter': f"Who is the daughter of {subject}?",
            
            # Organizational relations
            'member of': f"What is {subject} a member of?",
            'part of': f"What is {subject} part of?",
            'belongs to': f"What does {subject} belong to?",
            'affiliated with': f"What is {subject} affiliated with?",
            'subsidiary': f"What is {subject} a subsidiary of?",
            'division': f"What division is {subject} part of?",
            'department': f"What department is {subject} in?",
            'organization': f"What organization is {subject} part of?",
            'company': f"What company is {subject} part of?",
            'institution': f"What institution is {subject} part of?",
            
            # Measurement relations
            'currency': f"What is the currency of {subject}?",
            'population': f"What is the population of {subject}?",
            'area': f"What is the area of {subject}?",
            'length': f"What is the length of {subject}?",
            'height': f"What is the height of {subject}?",
            'weight': f"What is the weight of {subject}?",
            'size': f"What is the size of {subject}?",
            'volume': f"What is the volume of {subject}?",
            'capacity': f"What is the capacity of {subject}?",
            'duration': f"What is the duration of {subject}?",
            'speed': f"What is the speed of {subject}?",
            'temperature': f"What is the temperature of {subject}?",
            
            # Purpose/Function relations
            'cause': f"What caused {subject}?",
            'reason': f"What is the reason for {subject}?",
            'purpose': f"What is the purpose of {subject}?",
            'function': f"What is the function of {subject}?",
            'use': f"What is {subject} used for?",
            'application': f"What is the application of {subject}?",
            'effect': f"What is the effect of {subject}?",
            'result': f"What is the result of {subject}?",
            'outcome': f"What is the outcome of {subject}?",
            
            # Content relations
            'plot': f"What is the plot of {subject}?",
            'story': f"What is the story of {subject}?",
            'narrative': f"What is the narrative of {subject}?",
            'content': f"What is the content of {subject}?",
            'lyrics': f"What are the lyrics of {subject}?",
            'text': f"What is the text of {subject}?",
            'meaning': f"What is the meaning of {subject}?",
            'definition': f"What is the definition of {subject}?",
            
            # Awards/Recognition relations
            'award': f"What award did {subject} receive?",
            'prize': f"What prize did {subject} win?",
            'honor': f"What honor did {subject} receive?",
            'recognition': f"What recognition did {subject} get?",
            'achievement': f"What achievement did {subject} accomplish?",
            'nomination': f"What nomination did {subject} receive?",
            
            # Media relations
            'publisher': f"Who published {subject}?",
            'distributor': f"Who distributed {subject}?",
            'broadcaster': f"Who broadcast {subject}?",
            'network': f"What network aired {subject}?",
            'channel': f"What channel showed {subject}?",
            'platform': f"What platform features {subject}?",
            'label': f"What label released {subject}?",
            'studio': f"What studio produced {subject}?",
            
            # Scientific relations
            'species': f"What species is {subject}?",
            'genus': f"What genus is {subject}?",
            'family': f"What family does {subject} belong to?",
            'order': f"What order does {subject} belong to?",
            'class': f"What class does {subject} belong to?",
            'kingdom': f"What kingdom does {subject} belong to?",
            'domain': f"What domain does {subject} belong to?",
            'element': f"What element is {subject}?",
            'compound': f"What compound is {subject}?",
            'formula': f"What is the formula of {subject}?",
        }
        
        # Check for exact match first
        if relation_lower in relation_templates:
            return relation_templates[relation_lower]
        
        # Check for partial matches with better scoring
        best_match = None
        best_score = 0
        
        for template_relation, template in relation_templates.items():
            # Score based on word overlap
            relation_words = set(relation_lower.split())
            template_words = set(template_relation.split())
            
            # Exact substring match gets high score
            if template_relation in relation_lower or relation_lower in template_relation:
                score = 10
            # Word overlap scoring
            elif relation_words & template_words:
                score = len(relation_words & template_words) / max(len(relation_words), len(template_words))
            else:
                continue
                
            if score > best_score:
                best_score = score
                best_match = template
        
        if best_match and best_score > 0.3:  # Require reasonable confidence
            return best_match
        
        # Enhanced fallback logic based on relation patterns
        if any(word in relation_lower for word in ['who', 'person', 'people', 'individual']):
            return f"Who is the {relation} of {subject}?"
        elif any(word in relation_lower for word in ['where', 'place', 'location', 'city', 'country']):
            return f"Where is the {relation} of {subject}?"
        elif any(word in relation_lower for word in ['when', 'time', 'date', 'year']):
            return f"When was the {relation} of {subject}?"
        elif any(word in relation_lower for word in ['how many', 'count', 'number']):
            return f"How many {relation} does {subject} have?"
        else:
            return f"What is the {relation} of {subject}?"
        
    except Exception as e:
        # If conversion fails, return original question
        return question


def process_trex_data():
    # Load the T-REX test dataset
    test_data_path = PROJECT_ROOT / "data_kg" / "trex" / "test_simple.json"
    entities_path = PROJECT_ROOT / "data_kg" / "trex" / "entities.txt"
    entities_text_path = PROJECT_ROOT / "data_kg" / "trex" / "entities_text.txt"
    
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
            original_question = item['question']
            # Convert [SEP] format to natural language
            question = convert_sep_to_natural_language(original_question)
            answers = item['answers']
            sample_id = item['id']
            entity_indices = item.get('entities', [])
            subgraph_entities = item['subgraph']['entities']
            
            # Extract initial entities using entity indices (direct mapping to entity files)
            initial_entities = []
            initial_entity_ids = []
            for entity_idx in entity_indices:
                if entity_idx < len(entity_ids) and entity_idx < len(entity_texts):
                    entity_id = entity_ids[entity_idx]
                    entity_text = entity_texts[entity_idx]
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
                "data_source": "kgR1_trex",
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
                    "dataset_name": "trex",
                    "initial_entities": initial_entities,
                    "initial_entity_ids": initial_entity_ids,
                    "original_question": original_question,
                    "natural_language_question": question
                }
            }
            
            processed_data.append(processed_item)
        
        return processed_data
    
    # Create output directory
    output_dir = PROJECT_ROOT / "data_kg" / "trex_search_augmented_initial_entities"
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
    process_trex_data()
