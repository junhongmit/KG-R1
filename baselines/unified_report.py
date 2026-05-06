import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SCHEMA_VERSION = "kg-r1-baseline-report-v1"


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as infile:
        payload = json.load(infile)
    return payload if isinstance(payload, dict) else {}


def _default_report_path(output_file: Optional[str], baseline: str, dataset: str) -> Path:
    if output_file:
        return Path(output_file).with_name(Path(output_file).name + ".unified_metrics.json")
    return Path(f"{baseline}_{dataset}_unified_metrics.json")


def normalize_token_usage(
    token_usage: Optional[Dict[str, Any]] = None,
    *,
    token_file: Optional[str] = None,
    records: Optional[Iterable[Dict[str, Any]]] = None,
    num_questions: Optional[int] = None,
) -> Dict[str, Any]:
    usage: Dict[str, Any] = {}
    source = "none"

    if token_file:
        token_path = Path(token_file)
        if token_path.exists():
            usage = _read_json(token_path)
            source = str(token_path)

    if not usage and token_usage:
        usage = dict(token_usage)
        source = str(usage.get("source") or "provided")

    if not usage and records is not None:
        aggregate = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
            "missing_usage_count": 0,
        }
        for record in records:
            if all(key in record for key in ("input_token", "output_token", "total_token")):
                aggregate["prompt_tokens"] += _as_int(record.get("input_token"))
                aggregate["completion_tokens"] += _as_int(record.get("output_token"))
                aggregate["total_tokens"] += _as_int(record.get("total_token"))
                aggregate["request_count"] += _as_int(record.get("call_num"), 1)
                continue

            for call in record.get("api_calls") or []:
                aggregate["prompt_tokens"] += _as_int(call.get("prompt_tokens"))
                aggregate["completion_tokens"] += _as_int(call.get("completion_tokens"))
                aggregate["total_tokens"] += _as_int(call.get("total_tokens"))
                aggregate["request_count"] += 1
                if (
                    call.get("prompt_tokens") is None
                    and call.get("completion_tokens") is None
                    and call.get("total_tokens") is None
                ):
                    aggregate["missing_usage_count"] += 1

        if aggregate["request_count"] or aggregate["total_tokens"]:
            usage = aggregate
            source = "records"

    prompt_tokens = _as_int(usage.get("prompt_tokens", usage.get("input_token")))
    completion_tokens = _as_int(usage.get("completion_tokens", usage.get("output_token")))
    total_tokens = _as_int(usage.get("total_tokens", usage.get("total_token")), prompt_tokens + completion_tokens)
    request_count = _as_int(usage.get("request_count", usage.get("call_num")))
    missing_usage_count = _as_int(usage.get("missing_usage_count"))
    denominator = num_questions or 0

    return {
        "source": source,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "request_count": request_count,
        "missing_usage_count": missing_usage_count,
        "avg_prompt_tokens_per_question": float(prompt_tokens / denominator) if denominator else None,
        "avg_completion_tokens_per_question": float(completion_tokens / denominator) if denominator else None,
        "avg_total_tokens_per_question": float(total_tokens / denominator) if denominator else None,
    }


def write_unified_report(
    *,
    baseline: str,
    dataset: str,
    output_file: Optional[str],
    metrics: Dict[str, Any],
    counts: Dict[str, Any],
    token_usage: Optional[Dict[str, Any]] = None,
    token_file: Optional[str] = None,
    records: Optional[Iterable[Dict[str, Any]]] = None,
    report_file: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    evaluated_samples = _as_int(counts.get("evaluated_samples", counts.get("total_samples")))
    total_samples = _as_int(counts.get("total_samples"), evaluated_samples)
    skipped_samples = _as_int(counts.get("skipped_samples"), max(total_samples - evaluated_samples, 0))

    exact_match = _as_float(metrics.get("exact_match", metrics.get("Exact Match")))
    hits_at_1 = _as_float(metrics.get("hits@1", metrics.get("Hits@1")), exact_match)
    if exact_match is None:
        exact_match = hits_at_1
    normalized_metrics = {
        "hits@1": hits_at_1,
        "exact_match": exact_match,
        "f1": _as_float(metrics.get("f1", metrics.get("F1"))),
        "precision": _as_float(metrics.get("precision", metrics.get("Precision"))),
        "recall": _as_float(metrics.get("recall", metrics.get("Recall"))),
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": baseline,
        "dataset": dataset,
        "output_file": output_file,
        "metrics": normalized_metrics,
        "counts": {
            "right": _as_int(counts.get("right", counts.get("Right Samples"))),
            "error": _as_int(counts.get("error", counts.get("Error Sampels"))),
            "evaluated_samples": evaluated_samples,
            "total_samples": total_samples,
            "skipped_samples": skipped_samples,
        },
        "token_usage": normalize_token_usage(
            token_usage,
            token_file=token_file,
            records=records,
            num_questions=evaluated_samples or total_samples,
        ),
    }
    if extra:
        report["extra"] = extra

    path = Path(report_file) if report_file else _default_report_path(output_file, baseline, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as outfile:
        json.dump(report, outfile, ensure_ascii=False, indent=2)
    tmp_path.replace(path)
    print(f"Unified metrics saved to: {path}")
    return path
