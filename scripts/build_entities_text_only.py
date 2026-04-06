#!/usr/bin/env python3
"""
Build data_kg/CWQ/entities_text.txt from existing entities.txt
using Freebase MID -> name mappings.

Inputs:
  - data_kg/CWQ/entities.txt
  - freebase_mid_gid_to_names.json

Output:
  - data_kg/CWQ/entities_text.txt

Format (tab-separated):
  <entity_id>\t<entity_name>

If a name is missing, the name field is left empty.
"""

import json
from pathlib import Path

CWQ_DIR = Path("data_kg/CWQ")
ENTITIES_TXT = CWQ_DIR / "entities.txt"
ENTITIES_TEXT_TXT = CWQ_DIR / "entities_text.txt"
NAME_MAP_JSON = Path("scripts/freebase_mid_gid_to_names.json")

def main():
    # --- sanity checks ---
    if not ENTITIES_TXT.exists():
        raise FileNotFoundError(f"Missing: {ENTITIES_TXT}")

    if not NAME_MAP_JSON.exists():
        raise FileNotFoundError(
            f"Missing: {NAME_MAP_JSON}\n"
            f"Run scripts/process_entities_freebase.sh first."
        )

    # --- load entity IDs (preserve order) ---
    with ENTITIES_TXT.open("r", encoding="utf-8") as f:
        entity_ids = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Loaded {len(entity_ids):,} entities from {ENTITIES_TXT}")

    # --- load Freebase name map ---
    with NAME_MAP_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    name_map = data.get("mappings", {})
    if not isinstance(name_map, dict):
        raise ValueError("Invalid format: 'mappings' field missing or not a dict")

    print(f"[INFO] Loaded {len(name_map):,} Freebase name mappings")

    # --- write entities_text.txt ---
    ENTITIES_TEXT_TXT.parent.mkdir(parents=True, exist_ok=True)

    missing = 0
    with ENTITIES_TEXT_TXT.open("w", encoding="utf-8") as f:
        for eid in entity_ids:
            name = name_map.get(eid, "")
            if not name:
                missing += 1
            f.write(f"{eid}\t{name}\n")

    print(f"[OK] Wrote: {ENTITIES_TEXT_TXT}")
    print(f"[INFO] Entities with names: {len(entity_ids) - missing:,}")
    print(f"[INFO] Entities without names: {missing:,}")

if __name__ == "__main__":
    main()
