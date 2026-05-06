import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import request

import numpy as np
import pandas as pd


DEFAULT_API_BASE = os.environ.get("KGR1_BASELINE_OPENAI_API_BASE", "http://localhost:7878/v1")
DEFAULT_API_KEY = os.environ.get("KGR1_BASELINE_OPENAI_API_KEY", "EMPTY")
DEFAULT_MODEL = os.environ.get("KGR1_BASELINE_OPENAI_MODEL")


VANILLA_PROMPT_TEMPLATE = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Answer the given question directly and concisely based on your knowledge.
Provide only the factual answer without explanation or reasoning.

Examples:
Question: What is the capital of France?
Answer: Paris

Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare

Question: What year did World War II end?
Answer: 1945

Question: What is the largest planet in our solar system?
Answer: Jupiter

{question}
Answer:"""


COT_PROMPT_TEMPLATE = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Answer the given question based on your knowledge.
Think through the problem step by step, then end your response with a single line in the format:
Final answer: <answer>

Examples:
Question: What is the capital of France?
Reasoning: France's capital city is Paris.
Final answer: Paris

Question: Who wrote Romeo and Juliet?
Reasoning: Romeo and Juliet is a famous play by William Shakespeare.
Final answer: William Shakespeare

Question: What year did World War II end?
Reasoning: World War II ended in 1945.
Final answer: 1945

Question: What is the largest planet in our solar system?
Reasoning: Jupiter is the largest planet in our solar system.
Final answer: Jupiter

Question: {question}
Reasoning:"""


@dataclass
class BaselineSample:
    idx: int
    dataset_name: str
    sample_id: str
    question: str
    prompt: str
    ground_truths: List[str]
    raw_prompt: Any
    extra_info: Dict[str, Any]


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def compute_precision_recall_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0

    pred_counts: Dict[str, int] = {}
    gt_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gt_tokens:
        gt_counts[token] = gt_counts.get(token, 0) + 1

    overlap = 0
    for token, pred_count in pred_counts.items():
        overlap += min(pred_count, gt_counts.get(token, 0))

    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_prediction_against_ground_truths(prediction: str, ground_truths: Sequence[str]) -> Dict[str, float]:
    if not ground_truths:
        return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    best = {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for ground_truth in ground_truths:
        exact_match = compute_exact_match(prediction, ground_truth)
        precision, recall, f1 = compute_precision_recall_f1(prediction, ground_truth)
        candidate = {
            "exact_match": exact_match,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if (candidate["exact_match"], candidate["f1"], candidate["precision"], candidate["recall"]) > (
            best["exact_match"],
            best["f1"],
            best["precision"],
            best["recall"],
        ):
            best = candidate
    return best


def compute_pass_at_k(values: Sequence[float], k: int, reducer: str = "max") -> float:
    if not values:
        return 0.0
    topk = list(values[:k])
    if reducer == "binary":
        return float(any(v > 0 for v in topk))
    return float(max(topk))


def extract_question_from_prompt_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    if "Question:" in text:
        question_part = text.split("Question:", 1)[1].strip()
        if "(Initial entities:" in question_part:
            question_part = question_part.split("(Initial entities:", 1)[0].strip()
        for marker in ("Reasoning:", "Answers:", "Answer:"):
            if marker in question_part:
                question_part = question_part.split(marker, 1)[0].strip()
        return question_part.split("\n")[0].strip()

    return text.strip()


def _coerce_prompt_value(prompt_value: Any) -> str:
    if isinstance(prompt_value, np.ndarray):
        prompt_value = prompt_value.tolist()
    if isinstance(prompt_value, list) and prompt_value:
        first = prompt_value[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    if isinstance(prompt_value, dict):
        return str(prompt_value.get("content", ""))
    return str(prompt_value)


def _to_string_list(value: Any) -> List[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _extract_answers_from_json_row(row: Dict[str, Any]) -> List[str]:
    answer = row.get("answer", row.get("answers", []))
    if isinstance(answer, dict):
        return [str(value) for value in answer.values() if value is not None]
    if isinstance(answer, list):
        answers: List[str] = []
        for item in answer:
            if isinstance(item, dict):
                value = item.get("entity_name") or item.get("answer") or item.get("answer_argument")
                if value is not None:
                    answers.append(str(value))
            elif item is not None:
                answers.append(str(item))
        return answers
    if answer is None:
        return []
    return [str(answer)]


def _extract_question_from_json_row(row: Dict[str, Any]) -> str:
    for key in ("question", "RawQuestion", "input", "sentence", "claim"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _load_json_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    with open(path, "r", encoding="utf-8") as fin:
        payload = json.load(fin)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return payload


def load_json_samples(json_path: Path, dataset_name: str, max_samples: int = 0, start_index: int = 0) -> List[BaselineSample]:
    rows = _load_json_rows(json_path)
    if start_index > 0:
        rows = rows[start_index:]
    if max_samples > 0:
        rows = rows[:max_samples]

    samples: List[BaselineSample] = []
    for row_offset, row in enumerate(rows):
        idx = start_index + row_offset
        question = _extract_question_from_json_row(row)
        sample_id = str(row.get("id", row.get("qid", row.get("uid", f"{dataset_name}-{idx}"))))
        samples.append(
            BaselineSample(
                idx=idx,
                dataset_name=dataset_name,
                sample_id=sample_id,
                question=question,
                prompt=question,
                ground_truths=_extract_answers_from_json_row(row),
                raw_prompt=question,
                extra_info={key: value for key, value in row.items() if key not in {"answer", "answers"}},
            )
        )
    return samples


def load_samples(data_path: Path, max_samples: int = 0, start_index: int = 0, dataset_name: Optional[str] = None) -> List[BaselineSample]:
    if data_path.suffix in {".json", ".jsonl"}:
        return load_json_samples(data_path, dataset_name or data_path.stem, max_samples=max_samples, start_index=start_index)

    dataframe = pd.read_parquet(data_path)
    if start_index > 0:
        dataframe = dataframe.iloc[start_index:]
    if max_samples > 0:
        dataframe = dataframe.head(max_samples)

    samples: List[BaselineSample] = []
    for row_offset, (_, row) in enumerate(dataframe.iterrows()):
        prompt_value = row.get("prompt", "")
        prompt_text = _coerce_prompt_value(prompt_value)
        question = extract_question_from_prompt_text(prompt_text)

        reward_model = row.get("reward_model", {}) or {}
        ground_truth = reward_model.get("ground_truth", {}) if isinstance(reward_model, dict) else {}
        ground_truths = _to_string_list(ground_truth.get("target_text", [])) if isinstance(ground_truth, dict) else []

        extra_info = row.get("extra_info", {}) or {}
        if isinstance(extra_info, np.ndarray):
            extra_info = extra_info.tolist()
        if not isinstance(extra_info, dict):
            extra_info = {}

        dataset_name = str(extra_info.get("dataset_name", row.get("data_source", "unknown")))
        sample_id = str(extra_info.get("sample_id", f"sample-{start_index + row_offset}"))
        idx = start_index + row_offset

        samples.append(
            BaselineSample(
                idx=idx,
                dataset_name=dataset_name,
                sample_id=sample_id,
                question=question,
                prompt=question,
                ground_truths=ground_truths,
                raw_prompt=prompt_value,
                extra_info=extra_info,
            )
        )
    return samples


def discover_model_id(api_base: str = DEFAULT_API_BASE, api_key: str = DEFAULT_API_KEY, timeout: int = 60) -> str:
    if DEFAULT_MODEL:
        return DEFAULT_MODEL

    url = api_base.rstrip("/") + "/models"
    req = request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data") or []
    if not data:
        raise RuntimeError(f"No models returned by {url}")
    model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError(f"Could not determine model id from {url}: {payload}")
    return str(model_id)


def openai_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    n: int = 1,
) -> Tuple[List[str], float, Dict[str, Optional[int]]]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    start_time = time.time()
    with request.urlopen(req, timeout=timeout) as resp:
        response_payload = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - start_time
    choices = response_payload.get("choices") or []
    texts = []
    for choice in choices:
        message = choice.get("message", {})
        texts.append(str(message.get("content", "")).strip())
    usage_payload = response_payload.get("usage") or {}
    usage = {
        "prompt_tokens": usage_payload.get("prompt_tokens"),
        "completion_tokens": usage_payload.get("completion_tokens"),
        "total_tokens": usage_payload.get("total_tokens"),
    }
    return texts, elapsed, usage


def build_vanilla_prompt(question: str) -> str:
    return VANILLA_PROMPT_TEMPLATE.format(question=question)


def build_cot_prompt(question: str) -> str:
    return COT_PROMPT_TEMPLATE.format(question=question)


def extract_final_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    patterns = [
        r"final answer\s*:\s*(.+)",
        r"answer\s*:\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, stripped, flags=re.IGNORECASE)
        if matches:
            candidate = str(matches[-1]).strip()
            if candidate:
                return candidate
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def make_progress_payload(
    *,
    total: int,
    completed_indices: Iterable[int],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    completed_list = sorted(set(int(idx) for idx in completed_indices))
    payload = {
        "total": int(total),
        "completed": len(completed_list),
        "remaining": max(int(total) - len(completed_list), 0),
        "completed_indices": completed_list,
        "updated_at": utc_now_iso(),
    }
    if meta:
        payload.update(meta)
    return payload
