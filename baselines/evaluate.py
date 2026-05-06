import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from baselines.common import compute_pass_at_k, read_jsonl, score_prediction_against_ground_truths
from baselines.unified_report import write_unified_report


def _get_candidate_responses(record: Dict[str, Any]) -> List[str]:
    if record.get("candidate_responses"):
        return list(record["candidate_responses"])
    return list(record.get("responses", []))


def _get_primary_response(record: Dict[str, Any], candidate_responses: Sequence[str]) -> str:
    if record.get("aggregated_response"):
        return str(record["aggregated_response"])
    if candidate_responses:
        return str(candidate_responses[0])
    return ""


def evaluate_records(records: List[Dict[str, Any]], k_values: List[int]) -> Dict[str, Any]:
    exact_match_per_k = {k: [] for k in k_values}
    f1_per_k = {k: [] for k in k_values}
    precision_per_k = {k: [] for k in k_values}
    recall_per_k = {k: [] for k in k_values}
    response_lengths: List[int] = []
    generation_lengths: List[int] = []
    detailed_records: List[Dict[str, Any]] = []
    primary_exact_matches: List[float] = []
    primary_f1s: List[float] = []
    primary_precisions: List[float] = []
    primary_recalls: List[float] = []

    for record in records:
        ground_truths = list(record.get("ground_truths", []))
        candidate_responses = _get_candidate_responses(record)
        primary_response = _get_primary_response(record, candidate_responses)

        response_metrics: List[Dict[str, float]] = []
        for response in candidate_responses:
            metrics = score_prediction_against_ground_truths(response, ground_truths)
            response_metrics.append(metrics)
            response_lengths.append(len(str(response).split()))
            generation_lengths.append(len(str(response)))

        exact_values = [metrics["exact_match"] for metrics in response_metrics]
        f1_values = [metrics["f1"] for metrics in response_metrics]
        precision_values = [metrics["precision"] for metrics in response_metrics]
        recall_values = [metrics["recall"] for metrics in response_metrics]

        primary_metrics = score_prediction_against_ground_truths(primary_response, ground_truths)
        primary_exact_matches.append(primary_metrics["exact_match"])
        primary_f1s.append(primary_metrics["f1"])
        primary_precisions.append(primary_metrics["precision"])
        primary_recalls.append(primary_metrics["recall"])

        for k in k_values:
            exact_match_per_k[k].append(compute_pass_at_k(exact_values, k, reducer="binary"))
            f1_per_k[k].append(compute_pass_at_k(f1_values, k))
            precision_per_k[k].append(compute_pass_at_k(precision_values, k))
            recall_per_k[k].append(compute_pass_at_k(recall_values, k))

        detailed_records.append(
            {
                **record,
                "response_metrics": response_metrics,
                "primary_response": primary_response,
                "primary_metrics": primary_metrics,
                "best_exact_match": float(max(exact_values) if exact_values else 0.0),
                "best_f1": float(max(f1_values) if f1_values else 0.0),
                "best_precision": float(max(precision_values) if precision_values else 0.0),
                "best_recall": float(max(recall_values) if recall_values else 0.0),
            }
        )

    summary: Dict[str, Any] = {}
    for k in k_values:
        summary[f"exact_match_pass@{k}/mean"] = float(np.mean(exact_match_per_k[k])) if exact_match_per_k[k] else 0.0
        summary[f"exact_match_pass@{k}/std"] = float(np.std(exact_match_per_k[k])) if exact_match_per_k[k] else 0.0
        summary[f"f1_pass@{k}/mean"] = float(np.mean(f1_per_k[k])) if f1_per_k[k] else 0.0
        summary[f"f1_pass@{k}/std"] = float(np.std(f1_per_k[k])) if f1_per_k[k] else 0.0
        summary[f"precision_pass@{k}/mean"] = float(np.mean(precision_per_k[k])) if precision_per_k[k] else 0.0
        summary[f"precision_pass@{k}/std"] = float(np.std(precision_per_k[k])) if precision_per_k[k] else 0.0
        summary[f"recall_pass@{k}/mean"] = float(np.mean(recall_per_k[k])) if recall_per_k[k] else 0.0
        summary[f"recall_pass@{k}/std"] = float(np.std(recall_per_k[k])) if recall_per_k[k] else 0.0

    summary["exact_match/mean"] = float(np.mean(primary_exact_matches)) if primary_exact_matches else 0.0
    summary["exact_match/std"] = float(np.std(primary_exact_matches)) if primary_exact_matches else 0.0
    summary["f1/mean"] = float(np.mean(primary_f1s)) if primary_f1s else 0.0
    summary["f1/std"] = float(np.std(primary_f1s)) if primary_f1s else 0.0
    summary["precision/mean"] = float(np.mean(primary_precisions)) if primary_precisions else 0.0
    summary["precision/std"] = float(np.std(primary_precisions)) if primary_precisions else 0.0
    summary["recall/mean"] = float(np.mean(primary_recalls)) if primary_recalls else 0.0
    summary["recall/std"] = float(np.std(primary_recalls)) if primary_recalls else 0.0
    summary["response_length/mean"] = float(np.mean(response_lengths)) if response_lengths else 0.0
    summary["generation_length/mean"] = float(np.mean(generation_lengths)) if generation_lengths else 0.0
    summary["num_questions"] = len(records)
    summary["num_failed"] = int(sum(1 for record in records if record.get("error")))

    return {"summary": summary, "detailed_records": detailed_records}


def write_evaluation_outputs(
    *,
    input_path: Path,
    summary: Dict[str, Any],
    detailed_records: List[Dict[str, Any]],
    summary_path: Path = None,
    detailed_path: Path = None,
    write_summary: bool = True,
) -> None:
    if write_summary and summary_path is None:
        summary_path = input_path.with_name(input_path.stem + "_metrics.json")
    if detailed_path is None:
        detailed_path = input_path.with_name(input_path.stem + "_evaluated.jsonl")

    if write_summary:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fout:
            json.dump(summary, fout, indent=2, ensure_ascii=False)

    detailed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(detailed_path, "w", encoding="utf-8") as fout:
        for record in detailed_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KG-R1 baseline predictions JSONL.")
    parser.add_argument("--input_file", required=True, help="Path to prediction JSONL file.")
    parser.add_argument("--k_values", nargs="+", type=int, default=[1], help="K values for Pass@K metrics.")
    parser.add_argument("--summary_file", default=None, help="Optional summary JSON output path.")
    parser.add_argument("--detailed_file", default=None, help="Optional evaluated JSONL output path.")
    parser.add_argument("--baseline", default="llm_baseline", help="Baseline name for unified report.")
    parser.add_argument("--dataset", default=None, help="Dataset name for unified report.")
    parser.add_argument("--unified_report_file", default=None, help="Optional unified metrics JSON output path.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    records = read_jsonl(input_path)
    result = evaluate_records(records, args.k_values)
    write_evaluation_outputs(
        input_path=input_path,
        summary=result["summary"],
        detailed_records=result["detailed_records"],
        summary_path=Path(args.summary_file) if args.summary_file else None,
        detailed_path=Path(args.detailed_file) if args.detailed_file else None,
    )

    print(f"Evaluated {result['summary']['num_questions']} questions from {input_path}")
    for k in args.k_values:
        print(f"Pass@{k} exact match: {result['summary'][f'exact_match_pass@{k}/mean']:.4f}")
    print(f"Primary F1 mean: {result['summary']['f1/mean']:.4f}")
    write_unified_report(
        baseline=args.baseline,
        dataset=args.dataset or (records[0].get("dataset") if records else "unknown"),
        output_file=str(input_path),
        metrics={
            "hits@1": result["summary"].get("exact_match_pass@1/mean", result["summary"].get("exact_match/mean")),
            "exact_match": result["summary"].get("exact_match/mean"),
            "f1": result["summary"].get("f1/mean"),
            "precision": result["summary"].get("precision/mean"),
            "recall": result["summary"].get("recall/mean"),
        },
        counts={
            "total_samples": result["summary"].get("num_questions", len(records)),
            "evaluated_samples": result["summary"].get("num_questions", len(records)),
            "error": result["summary"].get("num_failed", 0),
        },
        records=records,
        report_file=args.unified_report_file,
    )


if __name__ == "__main__":
    main()
