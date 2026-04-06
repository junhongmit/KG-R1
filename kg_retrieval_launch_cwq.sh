#!/bin/bash
# Minimal KG Retrieval Server Launch Script

# Configuration
host="0.0.0.0"
port="8001"
workers="2"  # Two workers for balanced performance 
thread_workers="16"  # More ThreadPoolExecutor workers for concurrent request processing
base_data_path="./data_kg"
supported_kgs="webqsp,CWQ"
#supported_kgs="webqsp"  # Comma-separated list of supported KGs
# Use comprehensive action set including deprecated get_relations for backward compatibility
enabled_actions="get_head_relations,get_tail_relations,get_head_entities,get_tail_entities,get_conditional_relations"
use_entities_text=true  # Set to true to use entities_text.txt instead of entities.txt

# Relation format options - choose one:
# "flat" - Original format: rel1, rel2, rel3 (baseline)
# "full_indent" - Full hierarchy with each relation on separate line (24.4% token savings)
# "mixed" - Mixed hierarchy with properties on same line (34.8% token savings - most efficient)
# "compact" - Compact domain.type: props format (34.1% token savings)
relation_format="flat"  # Default: full indentation format

echo "Starting KG Retrieval Server..."
echo "Host: $host:$port | KGs: $supported_kgs"

# Change to the project directory
cd "$(dirname "$0")"


# Convert comma-separated values to space-separated for command line
kgs_args=$(echo "$supported_kgs" | tr ',' ' ')
actions_args=$(echo "$enabled_actions" | tr ',' ' ')

echo "Starting KG Retrieval Server..."
echo "Host: $host, Port: $port"
echo "KGs: $supported_kgs"
echo "Actions: $enabled_actions"
echo "Use entities text: $use_entities_text"
echo "Thread workers: $thread_workers"
echo "Relation format: $relation_format"

# Build command arguments
entities_flag=""
if [ "$use_entities_text" = "true" ]; then
    entities_flag="--use_entities_text"
fi

# Launch the server using module execution to fix relative imports
python3 -m kg_r1.search.kg_retrieval_server \
    --base_data_path "$base_data_path" \
    --host "$host" \
    --port "$port" \
    --workers "$workers" \
    --thread_workers "$thread_workers" \
    --kgs $kgs_args \
    --actions $actions_args \
    --relation_format "$relation_format" \
    $entities_flag
