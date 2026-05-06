import multiprocessing
import gzip
import re
import ujson as json
from collections import defaultdict
from datetime import datetime

pattern = re.compile(r'/ns/(m\.[^>]+|g\.[^>]+)>\s.*?/type\.object\.name>\s+"([^"]+)"@en')

def parse_chunk(lines):
    mappings = {}
    for line in lines:
        match = pattern.search(line)
        if match:
            mappings[match.group(1)] = match.group(2)
    return mappings

if __name__ == "__main__":
    CHUNK_SIZE = 1000000  # adjust chunk size if needed
    mappings = defaultdict(str)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    results = []
    chunk = []
    line_count = 0

    print(f"🚀 Started parsing: {datetime.now()}")

    with gzip.open("freebase-rdf-latest.gz", 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            chunk.append(line)
            line_count += 1

            if len(chunk) >= CHUNK_SIZE:
                results.append(pool.apply_async(parse_chunk, (chunk,)))
                chunk = []
                print(f"✅ Queued chunk, lines processed: {line_count:,}")

        # Final chunk
        if chunk:
            results.append(pool.apply_async(parse_chunk, (chunk,)))
            print(f"✅ Final chunk queued, total lines processed: {line_count:,}")

    pool.close()
    pool.join()

    print("🚧 Collecting results...")
    for res in results:
        mappings.update(res.get())

    print(f"✅ Finished parsing. Total mappings extracted: {len(mappings):,}")

    # Save JSON
    print("💾 Saving to JSON file...")
    with open("freebase_mid_gid_to_names.json", 'w', encoding='utf-8') as f:
        json.dump({
            'mappings': mappings,
            'metadata': {
                'total_mappings': len(mappings),
                'lines_processed': line_count,
                'created_at': datetime.now().isoformat()
            }
        }, f, indent=2, ensure_ascii=False)

    print("🎉 All done!")
