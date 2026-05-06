import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_COLUMNS = [
    "baseline",
    "dataset",
    "hits@1",
    "f1",
    "precision",
    "recall",
    "evaluated_samples",
    "total_samples",
    "skipped_samples",
    "total_tokens",
    "avg_total_tokens_per_question",
    "request_count",
    "output_file",
    "report_file",
]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _load_report(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as infile:
        payload = json.load(infile)
    metrics = payload.get("metrics") or {}
    counts = payload.get("counts") or {}
    tokens = payload.get("token_usage") or {}
    return {
        "baseline": payload.get("baseline"),
        "dataset": payload.get("dataset"),
        "hits@1": metrics.get("hits@1"),
        "f1": metrics.get("f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "evaluated_samples": counts.get("evaluated_samples"),
        "total_samples": counts.get("total_samples"),
        "skipped_samples": counts.get("skipped_samples"),
        "total_tokens": tokens.get("total_tokens"),
        "avg_total_tokens_per_question": tokens.get("avg_total_tokens_per_question"),
        "request_count": tokens.get("request_count"),
        "output_file": payload.get("output_file"),
        "report_file": str(path),
    }


def find_reports(paths: Iterable[str]) -> List[Path]:
    report_paths: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            report_paths.append(path)
        elif path.is_dir():
            report_paths.extend(path.rglob("*.unified_metrics.json"))
        else:
            report_paths.extend(Path().glob(raw_path))
    return sorted(set(report_paths))


def print_markdown(rows: List[Dict[str, Any]], columns: List[str]) -> None:
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect KG-R1 unified baseline metric reports into one table.")
    parser.add_argument("paths", nargs="+", help="Report files, directories, or glob patterns to scan.")
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    parser.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS, help="Columns to include.")
    args = parser.parse_args()

    reports = find_reports(args.paths)
    rows = [_load_report(path) for path in reports]
    rows.sort(key=lambda row: (str(row.get("dataset")), str(row.get("baseline")), str(row.get("output_file"))))
    print_markdown(rows, args.columns)
    if args.csv:
        write_csv(Path(args.csv), rows, args.columns)
        print(f"Wrote CSV to: {args.csv}")


if __name__ == "__main__":
    main()
