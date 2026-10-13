from __future__ import annotations

import json
import os
import re
import string
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

_ANSWER_PAIR_RE = re.compile(r"<answer\b[^>]*>\s*([^<>]*?)\s*</answer\s*>", re.DOTALL | re.IGNORECASE)
_MATH_ANSWER_PAIR_RE = re.compile(
    r"<answer\b[^>]*>\s*((?:(?!<answer\b).)*?)\s*</answer\s*>", re.DOTALL | re.IGNORECASE
)
_MATH_FINAL_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CODE_BLOCK_RE = re.compile(r"```(?:python)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)

_MATH_SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

_MATH_REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km", "units",
    "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents", "degrees", "cm",
    "gm", "pounds", "meters", "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}",
    "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"',
    "\\dots",
]

def normalize_answer(value: str) -> str:
    value = str(value or "").lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split()).strip()

def extract_answer(text: str) -> str:
    text = str(text or "")
    lower = text.lower()
    last_think_end = lower.rfind("</think>")
    primary = text[last_think_end + len("</think>"):] if last_think_end != -1 else text
    matches = _ANSWER_PAIR_RE.findall(primary) or _ANSWER_PAIR_RE.findall(text)
    if matches:
        return matches[-1].strip()
    answer_idx = primary.lower().rfind("answer:")
    if answer_idx >= 0:
        return primary[answer_idx + len("answer:"):].strip().splitlines()[0].strip()
    lines = [line.strip() for line in primary.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()

def extract_code(text: str) -> str:
    matches = _CODE_BLOCK_RE.findall(str(text or ""))
    if matches:
        return matches[-1].strip("\n\r")
    return str(text or "").strip()

def exact_match(prediction: str, gold_answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(gold) == pred for gold in gold_answers))

def sub_em(prediction: str, gold_answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    pred_tokens = pred.split()
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens or len(gold_tokens) > len(pred_tokens):
            continue
        for start in range(0, len(pred_tokens) - len(gold_tokens) + 1):
            if pred_tokens[start:start + len(gold_tokens)] == gold_tokens:
                return 1.0
    return 0.0

def f1_score(prediction: str, gold_answers: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return float(any(not normalize_answer(gold) for gold in gold_answers))
    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best

def evaluate_open_answer(response: str, gold_answers: list[str]) -> dict:
    prediction = extract_answer(response)
    em = exact_match(prediction, gold_answers)
    substring = sub_em(prediction, gold_answers)
    f1 = f1_score(prediction, gold_answers)
    return {
        "em": em,
        "f1": f1,
        "sub_em": substring,
        "hard": int(max(em, substring)),
        "soft": max(f1, substring),
        "predicted_answer": prediction,
        "gold_answers": gold_answers,
    }

def normalize_math_answer(value: str) -> str:
    value = str(value or "").split("=")[-1]
    for before, after in _MATH_SUBSTITUTIONS:
        value = value.replace(before, after)
    for expression in _MATH_REMOVED_EXPRESSIONS:
        value = value.replace(expression, "")
    value = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", value)
    value = re.sub(r"(\\text\{)(.*?)(\})", "\\2", value)
    value = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", value)
    value = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", value)
    value = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", value)
    value = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", value)
    value = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", value)
    value = value.replace("$", "")
    if value.replace(",", "").isdigit():
        value = value.replace(",", "")
    return value.strip()

def evaluate_dapo_math(
    response: str,
    gold_answers: list[str],
    require_completed_thinking: bool = False,
    *,
    native_gemma: bool = False,
) -> dict:
    submitted = str(response or "")
    if native_gemma:
        from evaluation.model import split_gemma4_response

        submitted, _, _ = split_gemma4_response(submitted)
    else:
        think_end = submitted.lower().rfind("</think>")
        if think_end >= 0:
            submitted = submitted[think_end + len("</think>"):]
        elif require_completed_thinking:
            submitted = ""
    matches = _MATH_ANSWER_PAIR_RE.findall(submitted)
    final_matches = _MATH_FINAL_RE.findall(submitted)
    prediction = matches[-1].strip() if matches else final_matches[-1].strip().rstrip(".;") if final_matches else ""
    normalized_prediction = normalize_math_answer(prediction)
    normalized_golds = [normalize_math_answer(gold) for gold in gold_answers]
    has_submission = bool(matches or final_matches)
    hard = int(has_submission and normalized_prediction in normalized_golds)
    return {
        "em": float(hard),
        "f1": float(hard),
        "sub_em": float(hard),
        "hard": hard,
        "soft": float(hard),
        "predicted_answer": prediction,
        "normalized_prediction": normalized_prediction,
        "gold_answers": gold_answers,
        "has_valid_answer_pair": bool(matches),
        "has_valid_submission": has_submission,
    }

def _load_json_text(value: Any, fallback: Any) -> Any:
    if isinstance(value, str) and value.strip():
        return json.loads(value)
    return fallback

def _build_candidate_code(response: str, item: dict) -> str:
    code = extract_code(response)
    tests_info = _load_json_text(item.get("tests_json"), {})
    entry_point = str(item.get("entry_point") or tests_info.get("entry_point") or "").strip()
    code_prompt = str(item.get("code_prompt") or "").rstrip()
    if entry_point and f"def {entry_point}" not in code and code_prompt:
        if code.startswith("    ") or code.startswith("\t") or "\n    " in code[:200]:
            return f"{code_prompt}\n{code}\n"
    return code

def evaluate_bigcodebench(response: str, item: dict, timeout: int = 20) -> dict:
    code = _build_candidate_code(response, item)
    tests = _load_json_text(item.get("tests_json"), {}).get("test") or ""
    if not tests.strip():
        raise ValueError(f"Missing private tests for {item.get('id', 'task')}")
    script = f"{code.rstrip()}\n\n{tests}\n\nif __name__ == '__main__':\n    import unittest\n    unittest.main(verbosity=0)\n"
    with tempfile.TemporaryDirectory(prefix="skillopt_bigcode_") as tmp:
        script_path = Path(tmp) / "candidate_test.py"
        script_path.write_text(script, encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or exc.output or "")[-2000:]
            stderr = (exc.stderr or "")[-2000:]
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return {
                "hard": 0,
                "soft": 0.0,
                "predicted_answer": code[:1000],
                "code": code,
                "stdout": stdout,
                "stderr": stderr,
                "fail_reason": f"tests timed out after {timeout}s",
            }
    ok = completed.returncode == 0
    stderr = completed.stderr[-2000:]
    stdout = completed.stdout[-2000:]
    return {
        "hard": int(ok),
        "soft": 1.0 if ok else 0.0,
        "predicted_answer": code[:1000],
        "code": code,
        "stdout": stdout,
        "stderr": stderr,
        "fail_reason": "" if ok else f"tests failed rc={completed.returncode}: {stderr or stdout}",
    }
