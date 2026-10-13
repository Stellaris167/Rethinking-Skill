"""OpenAI-compatible inference for Qwen and Gemma."""
from dataclasses import asdict, dataclass
import json
import os
import re
from types import SimpleNamespace
import urllib.request

@dataclass
class ModelConfig:
    base_url: str
    deployment: str
    api_key: str = ""
    max_tokens: int = 16384
    timeout_seconds: float = 900
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    enable_thinking: bool = True
    seed: int = 42

_config: ModelConfig

def configure(config: ModelConfig) -> None:
    global _config
    _config = config

def get_target_config_snapshot(max_tokens: int | None = None) -> dict:
    return {**asdict(_config), "max_tokens": max_tokens or _config.max_tokens,
            "truncate_prompt_tokens": None, "use_raw_api": True}

def is_gemma4_model(model: str) -> bool:
    return bool(re.search(r"gemma-?4(?:[-_/]|$)", model.lower()))

def split_gemma4_response(content: str) -> tuple[str, str, bool]:
    start = re.match(r"^\s*<\|channel>thought(?:\r?\n|$)", content)
    if start is None:
        return content, "", bool(content.strip())
    thought, end, visible = content[start.end():].partition("<channel|>")
    return visible if end else "", thought, bool(end)

class ContextWindowExceeded(RuntimeError):
    pass

def post_chat(base_url: str, api_key: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key or 'EMPTY'}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def chat_target_messages(messages: list[dict], max_completion_tokens: int = 16384,
                         timeout: float | None = None,
                         seed: int | None = None, tools: list[dict] | None = None,
                         tool_choice: str | None = None, return_message: bool = False):
    payload = {
        "model": _config.deployment, "messages": messages,
        "max_tokens": min(max_completion_tokens, _config.max_tokens),
        "temperature": _config.temperature, "top_p": _config.top_p,
        "top_k": _config.top_k, "min_p": _config.min_p,
        "seed": _config.seed if seed is None else seed,
        "chat_template_kwargs": {"enable_thinking": _config.enable_thinking},
    }
    gemma = is_gemma4_model(_config.deployment)
    if gemma:
        payload.update(skip_special_tokens=False, return_token_ids=True)
    if tools:
        payload.update(tools=tools, tool_choice=tool_choice or "auto")
        if gemma:
            payload["parallel_tool_calls"] = False
    body = post_chat(_config.base_url, _config.api_key, payload, timeout or _config.timeout_seconds)
    choice = body["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    usage = {**body.get("usage", {}), "finish_reason": choice.get("finish_reason"),
             "max_tokens": payload["max_tokens"], "raw_response": content,
             "reasoning_content": message.get("reasoning_content") or message.get("reasoning") or ""}
    if gemma:
        content, thought, complete = split_gemma4_response(content)
        usage.update(reasoning_content=usage["reasoning_content"] or thought,
                     thinking_complete=complete or bool(message.get("tool_calls")),
                     tool_calls=message.get("tool_calls", []))
    if return_message:
        calls = [SimpleNamespace(function=SimpleNamespace(**call["function"]))
                 for call in message.get("tool_calls", [])]
        return SimpleNamespace(content=content, tool_calls=calls, metadata={"choice0": choice}), usage
    return content, usage

def chat_target(system: str, user: str, **kwargs):
    return chat_target_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)

def judge_answer(question: str, prediction: str, answers: list[str], *, base_url: str,
                 model: str, benchmark: str, timeout: float = 300) -> dict:
    system = (
        "Judge whether the prediction and reference denote the same complete answer. "
        "Do not solve the question or supply a new answer. Accept aliases, number-word variants, "
        "equivalent mathematical expressions, harmless punctuation and optional geographic qualifiers. "
        "Reject related but different entities and answers missing required content. "
        "Return only same_answer, different_answer, or insufficient."
    )
    body = post_chat(base_url, os.environ.get("JUDGE_API_KEY", ""), {
        "model": model, "temperature": 0, "max_tokens": 2048,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content":
            json.dumps({"benchmark": benchmark, "question": question,
                        "prediction": prediction, "reference": answers})}],
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout)
    label = body["choices"][0]["message"]["content"].strip()
    if label not in {"same_answer", "different_answer", "insufficient"}:
        raise ValueError(f"Invalid judge label: {label}")
    return {"accepted": label == "same_answer", "label": label, "response": body}
