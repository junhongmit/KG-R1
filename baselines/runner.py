import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from tqdm import tqdm

from baselines.common import (
    DEFAULT_API_BASE,
    DEFAULT_API_KEY,
    append_jsonl,
    atomic_write_json,
    discover_model_id,
    load_samples,
    make_progress_payload,
    openai_chat_completion,
    read_jsonl,
)
from baselines.evaluate import evaluate_records, write_evaluation_outputs
from baselines.unified_report import write_unified_report

PromptBuilder = Callable[[str], str]
ResponseParser = Callable[[str], str]
Aggregator = Callable[[Sequence[str]], str]


def add_common_args(parser: argparse.ArgumentParser, strategy_name: str, default_n_rollouts: int) -> None:
    parser.add_argument("--dataset", required=True, choices=["cwq", "webqsp", "simpleqa", "grailqa", "trex", "qald"], help="Dataset to evaluate.")
    parser.add_argument("--data_file", default=None, help="Override parquet input path.")
    parser.add_argument("--api_base", default=DEFAULT_API_BASE, help="OpenAI-compatible API base.")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY, help="API key for the OpenAI-compatible server.")
    parser.add_argument("--model", default=None, help="Model name. Defaults to auto-discovering the first served model.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel API workers.")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit number of evaluated samples.")
    parser.add_argument("--start_index", type=int, default=0, help="Start offset into the dataset.")
    parser.add_argument("--n_rollouts", type=int, default=default_n_rollouts, help="Number of responses per question.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for rollout indices > 0.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling value.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum generation tokens.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries per sample.")
    parser.add_argument("--retry_errors_on_resume", action="store_true", default=True, help="Retry saved error rows on resume.")
    parser.add_argument("--no_retry_errors_on_resume", action="store_false", dest="retry_errors_on_resume")
    parser.add_argument("--output_dir", default=None, help=f"Output directory. Defaults to baselines/results/{strategy_name}/<dataset>.")
    parser.add_argument("--run_name", default=None, help="Full run name. Defaults to a timestamped name.")
    parser.add_argument("--experiment_name", default=None, help="Optional stable experiment prefix used in the run directory name.")
    parser.add_argument("--run_postfix", default=None, help="Optional postfix appended to the generated run directory name, e.g. _qwen3_235B_0.")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation after generation.")
    parser.add_argument("--k_values", nargs="+", type=int, default=None, help="K values for Pass@K evaluation.")


def default_data_file(dataset: str) -> Path:
    if dataset in {"cwq", "webqsp"}:
        return Path(__file__).resolve().parents[1] / "data_kg" / f"{dataset}_search_augmented_initial_entities" / "test.parquet"
    root = Path(__file__).resolve().parents[2]
    files = {
        "simpleqa": root / "data" / "other_kgqa" / "simpleqa.json",
        "grailqa": root / "data" / "other_kgqa" / "grailqa.json",
        "trex": root / "data" / "other_kgqa" / "trex.json",
        "qald": root / "data" / "other_kgqa" / "qald_10-en.json",
    }
    return files[dataset]


def get_output_paths(args: argparse.Namespace, strategy_name: str) -> Dict[str, Path]:
    root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else root / "results" / strategy_name / args.dataset
    if args.run_name:
        run_name = args.run_name
    else:
        name_prefix = args.experiment_name or f"{args.dataset}-{strategy_name}"
        run_name = f"{name_prefix}-{time.strftime('%m%d_%H%M%S')}"
        if args.run_postfix:
            run_name = f"{run_name}{args.run_postfix}"
    run_dir = output_dir / run_name
    return {
        "run_dir": run_dir,
        "predictions": run_dir / "predictions.jsonl",
        "progress": run_dir / "predictions.jsonl.progress.json",
        "evaluated": run_dir / "predictions_evaluated.jsonl",
    }


def load_completed_indices(predictions_path: Path, retry_errors_on_resume: bool) -> Set[int]:
    completed: Set[int] = set()
    for record in read_jsonl(predictions_path):
        idx = record.get("_idx")
        if idx is None:
            continue
        if retry_errors_on_resume and record.get("error"):
            continue
        responses = record.get("responses") or record.get("candidate_responses") or []
        if not responses and not record.get("aggregated_response"):
            continue
        completed.add(int(idx))
    return completed


def majority_vote(responses: Sequence[str]) -> str:
    cleaned = [response.strip() for response in responses if str(response).strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    return max(cleaned, key=lambda value: (counts[value], -cleaned.index(value)))


def _run_one_sample(
    sample,
    args: argparse.Namespace,
    model_name: str,
    prompt_builder: PromptBuilder,
    response_parser: ResponseParser,
    aggregate_fn: Optional[Aggregator],
) -> Dict[str, Any]:
    prompt = prompt_builder(sample.question)
    raw_responses: List[str] = []
    parsed_responses: List[str] = []
    api_calls: List[Dict[str, float]] = []

    for rollout_idx in range(args.n_rollouts):
        temperature = 0.0 if rollout_idx == 0 else args.temperature
        retries = 0
        while True:
            try:
                texts, latency, usage = openai_chat_completion(
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=model_name,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=temperature,
                    top_p=args.top_p,
                    timeout=args.timeout,
                    n=1,
                )
                raw_response = texts[0] if texts else ""
                parsed_response = response_parser(raw_response)
                raw_responses.append(raw_response)
                parsed_responses.append(parsed_response)
                api_calls.append(
                    {
                        "rollout_idx": rollout_idx,
                        "temperature": temperature,
                        "latency_sec": latency,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    }
                )
                break
            except Exception as exc:
                retries += 1
                if retries >= args.max_retries:
                    raise RuntimeError(f"API generation failed after {args.max_retries} attempts: {exc}") from exc
                time.sleep(min(2 ** (retries - 1), 8) + random.random())

    aggregated_response = aggregate_fn(parsed_responses) if aggregate_fn else None
    responses = list(parsed_responses)
    if aggregated_response:
        responses = [aggregated_response] + responses

    return {
        "_idx": sample.idx,
        "sample_id": sample.sample_id,
        "dataset": sample.dataset_name,
        "question": sample.question,
        "prompt": prompt,
        "ground_truths": sample.ground_truths,
        "responses": responses,
        "candidate_responses": parsed_responses,
        "aggregated_response": aggregated_response,
        "raw_responses": raw_responses,
        "api_calls": api_calls,
        "error": None,
    }


def run_baseline(
    *,
    args: argparse.Namespace,
    strategy_name: str,
    prompt_builder: PromptBuilder,
    response_parser: ResponseParser,
    aggregate_fn: Optional[Aggregator] = None,
) -> None:
    model_name = args.model or discover_model_id(args.api_base, args.api_key, args.timeout)
    paths = get_output_paths(args, strategy_name)
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)

    data_file = Path(args.data_file) if args.data_file else default_data_file(args.dataset)
    samples = load_samples(data_file, max_samples=args.max_samples, start_index=args.start_index, dataset_name=args.dataset)
    completed_indices = load_completed_indices(paths["predictions"], args.retry_errors_on_resume)
    pending_samples = [sample for sample in samples if sample.idx not in completed_indices]

    print(f"Running {strategy_name} baseline on {args.dataset}")
    print(f"Model: {model_name}")
    print(f"Data file: {data_file}")
    print(f"Output dir: {paths['run_dir']}")
    print(f"Workers: {args.num_workers}")
    print(f"Rollouts/question: {args.n_rollouts}")
    print(f"Resuming {len(completed_indices)}/{len(samples)} completed")

    progress_payload = make_progress_payload(
        total=len(samples),
        completed_indices=completed_indices,
        meta={"dataset": args.dataset, "model": model_name, "data_file": str(data_file), "strategy": strategy_name},
    )
    atomic_write_json(paths["progress"], progress_payload)

    if pending_samples:
        progress_bar = tqdm(total=len(samples), initial=len(completed_indices))
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_sample = {
                executor.submit(_run_one_sample, sample, args, model_name, prompt_builder, response_parser, aggregate_fn): sample
                for sample in pending_samples
            }
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "_idx": sample.idx,
                        "sample_id": sample.sample_id,
                        "dataset": sample.dataset_name,
                        "question": sample.question,
                        "prompt": prompt_builder(sample.question),
                        "ground_truths": sample.ground_truths,
                        "responses": [],
                        "candidate_responses": [],
                        "aggregated_response": None,
                        "raw_responses": [],
                        "api_calls": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                append_jsonl(paths["predictions"], record)
                responses = record.get("responses") or record.get("candidate_responses") or []
                if responses and not record.get("error"):
                    completed_indices.add(int(record["_idx"]))
                progress_payload = make_progress_payload(
                    total=len(samples),
                    completed_indices=completed_indices,
                    meta={"dataset": args.dataset, "model": model_name, "data_file": str(data_file), "strategy": strategy_name},
                )
                atomic_write_json(paths["progress"], progress_payload)
                progress_bar.update(1)
        progress_bar.close()

    ordered_records = sorted(read_jsonl(paths["predictions"]), key=lambda row: int(row.get("_idx", 10**12)))
    with open(paths["predictions"], "w", encoding="utf-8") as fout:
        for record in ordered_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.skip_evaluation:
        print(f"Generation complete: {paths['predictions']}")
        return

    k_values = args.k_values or list(range(1, args.n_rollouts + 1))
    evaluation = evaluate_records(ordered_records, k_values)
    write_evaluation_outputs(
        input_path=paths["predictions"],
        summary=evaluation["summary"],
        detailed_records=evaluation["detailed_records"],
        detailed_path=paths["evaluated"],
        write_summary=False,
    )
    write_unified_report(
        baseline=strategy_name,
        dataset=args.dataset,
        output_file=str(paths["predictions"]),
        metrics={
            "hits@1": evaluation["summary"].get("exact_match_pass@1/mean", evaluation["summary"].get("exact_match/mean")),
            "exact_match": evaluation["summary"].get("exact_match/mean"),
            "f1": evaluation["summary"].get("f1/mean"),
            "precision": evaluation["summary"].get("precision/mean"),
            "recall": evaluation["summary"].get("recall/mean"),
        },
        counts={
            "total_samples": evaluation["summary"].get("num_questions", len(ordered_records)),
            "evaluated_samples": evaluation["summary"].get("num_questions", len(ordered_records)),
            "error": evaluation["summary"].get("num_failed", 0),
        },
        records=ordered_records,
    )

    print(f"Generation complete: {paths['predictions']}")
    print(f"Unified metrics saved next to predictions")
    for k in k_values:
        print(f"Pass@{k} exact match: {evaluation['summary'][f'exact_match_pass@{k}/mean']:.4f}")
    print(f"Primary F1 mean: {evaluation['summary']['f1/mean']:.4f}")
