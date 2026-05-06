#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBGRAPH_ROOT="${SUBGRAPH_ROOT:-$PROJECT_ROOT/data_kg/Subgraph}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

usage() {
  cat <<EOF
Usage:
  SUBGRAPH_ROOT=/path/to/Subgraph bash scripts/data_other_kgqas/setup_other_kgqas.sh [all|simpleqa|grailqa|trex|qald10en|zero_shot_re]

Environment:
  SUBGRAPH_ROOT  Root containing hop_2_Freebase/ and hop_2_Wikidata/. Default: $PROJECT_ROOT/data_kg/Subgraph
  MAX_SAMPLES    Optional smoke-test limit for GrailQA/QALD-10. Default 0 means all samples.

Notes:
  - SimpleQA, T-REx, QALD-10, and Zero-Shot RE raw files are downloaded if missing.
  - GrailQA raw data is not downloaded here; place grailqa.json in $PROJECT_ROOT/data_kg/grailqa/.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Missing required file: $path"
}

seed_raw_if_available() {
  local source_path="$1"
  local target_path="$2"
  if [[ ! -f "$target_path" && -f "$source_path" ]]; then
    mkdir -p "$(dirname "$target_path")"
    cp "$source_path" "$target_path"
    echo "Seeded raw data from $source_path"
  fi
}

run_simpleqa() {
  require_file "$SUBGRAPH_ROOT/hop_2_Freebase/simpleqa_triples_2_hop_all.json"
  require_file "$SUBGRAPH_ROOT/hop_2_Freebase/simpleqa_labels_2_hop_all.json"
  seed_raw_if_available "$PROJECT_ROOT/../ToG/data/SimpleQA.json" "$PROJECT_ROOT/data_kg/simpleqa/simpleqa_raw.json"
  python "$SCRIPT_DIR/simpleqa/process_simpleqa.py" \
    --output_dir "$PROJECT_ROOT/data_kg/simpleqa" \
    --subgraph_triples "$SUBGRAPH_ROOT/hop_2_Freebase/simpleqa_triples_2_hop_all.json" \
    --subgraph_labels "$SUBGRAPH_ROOT/hop_2_Freebase/simpleqa_labels_2_hop_all.json"
  python "$SCRIPT_DIR/simpleqa_search_augmented_initial_entities.py"
}

run_grailqa() {
  require_file "$PROJECT_ROOT/data_kg/grailqa/grailqa.json"
  require_file "$SUBGRAPH_ROOT/hop_2_Freebase/grailqa_triples_2_hop_all.json"
  require_file "$SUBGRAPH_ROOT/hop_2_Freebase/grailqa_labels_2_hop_all.json"
  python "$SCRIPT_DIR/grailqa/process_grailqa.py" \
    --input_dir "$PROJECT_ROOT/data_kg/grailqa" \
    --output_dir "$PROJECT_ROOT/data_kg/grailqa" \
    --subgraph_dir "$SUBGRAPH_ROOT/hop_2_Freebase" \
    --max_samples "$MAX_SAMPLES"
  python "$SCRIPT_DIR/grailqa_search_augmented_initial_entities.py"
}

run_trex() {
  require_file "$SUBGRAPH_ROOT/hop_2_Wikidata/T-REX_2hop.json"
  require_file "$SUBGRAPH_ROOT/hop_2_Wikidata/T-REX_2hop_labels.json"
  seed_raw_if_available "$PROJECT_ROOT/../ToG/data/T-REX.json" "$PROJECT_ROOT/data_kg/trex/trex_raw.json"
  python "$SCRIPT_DIR/trex/process_trex.py" \
    --output_dir "$PROJECT_ROOT/data_kg/trex" \
    --subgraph_triples "$SUBGRAPH_ROOT/hop_2_Wikidata/T-REX_2hop.json" \
    --subgraph_labels "$SUBGRAPH_ROOT/hop_2_Wikidata/T-REX_2hop_labels.json"
  python "$SCRIPT_DIR/trex_search_augmented_initial_entities.py"
}

run_qald10en() {
  require_file "$SUBGRAPH_ROOT/hop_2_Wikidata/qald_10-en_2hop.json"
  require_file "$SUBGRAPH_ROOT/hop_2_Wikidata/qald_10-en_2hop_labels.json"
  seed_raw_if_available "$PROJECT_ROOT/../ToG/data/qald_10-en.json" "$PROJECT_ROOT/data_kg/qald10en/qald10en_raw.json"
  python "$SCRIPT_DIR/qald10en/process_qald10en.py" \
    --output_dir "$PROJECT_ROOT/data_kg/qald10en" \
    --subgraph_triples "$SUBGRAPH_ROOT/hop_2_Wikidata/qald_10-en_2hop.json" \
    --subgraph_labels "$SUBGRAPH_ROOT/hop_2_Wikidata/qald_10-en_2hop_labels.json" \
    --max_samples "$MAX_SAMPLES"
  python "$SCRIPT_DIR/qald10en_search_augmented_initial_entities.py"
}

run_zero_shot_re() {
  require_file "$SUBGRAPH_ROOT/hop_2_Wikidata/Zero_Shot_RE_2hop.json"
  require_file "$SUBGRAPH_ROOT/hop_2_Wikidata/Zero_Shot_RE_2hop_labels.json"
  seed_raw_if_available "$PROJECT_ROOT/../ToG/data/zero_shot_re.json" "$PROJECT_ROOT/data_kg/zero_shot_re/zero_shot_re_raw.json"
  python "$SCRIPT_DIR/zero_shot_re/process_zero_shot_re.py" \
    --output_dir "$PROJECT_ROOT/data_kg/zero_shot_re" \
    --subgraph_triples "$SUBGRAPH_ROOT/hop_2_Wikidata/Zero_Shot_RE_2hop.json" \
    --subgraph_labels "$SUBGRAPH_ROOT/hop_2_Wikidata/Zero_Shot_RE_2hop_labels.json"
  python "$SCRIPT_DIR/zero_shot_re_search_augmented_initial_entities.py"
}

target="${1:-all}"
case "$target" in
  all)
    run_simpleqa
    run_grailqa
    run_trex
    run_qald10en
    ;;
  simpleqa) run_simpleqa ;;
  grailqa) run_grailqa ;;
  trex) run_trex ;;
  qald10en) run_qald10en ;;
  zero_shot_re) run_zero_shot_re ;;
  -h|--help|help) usage ;;
  *) usage; die "Unknown dataset target: $target" ;;
esac
