import json
from pathlib import Path

from evaluation.agent import CodeToolContext, QAContext, run_qwen_agent_react
from evaluation.model import chat_target, get_target_config_snapshot, is_gemma4_model, judge_answer
from evaluation.scoring import evaluate_bigcodebench, evaluate_dapo_math, evaluate_open_answer
from evaluation.skills import render_skill

def _answers(item: dict) -> list[str]:
    raw = item.get("answers_json")
    if isinstance(raw, str) and raw.strip():
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(answer) for answer in parsed]
    answer = str(item.get("answer") or item.get("ground_truth") or "").strip()
    return [answer] if answer else []

def _system(benchmark: str, skill_content: str) -> str:
    skill = f"\n\n## Skill\n{skill_content.strip()}" if str(skill_content or "").strip() else ""
    if benchmark == "bigcodebench":
        return (
            "You are a careful Python engineer solving BigCodeBench tasks. "
            "Write self-contained executable Python that satisfies the requested function signature. "
            "Return exactly one ```python code block and no extra prose."
            f"{skill}"
        )
    if benchmark == "hotpotqa":
        return (
            "You are a careful HotpotQA solver. Use the provided context evidence and the skill when it helps. "
            "Answer with the shortest entity or phrase supported by the context."
            f"{skill}\n\n"
            "## Output Contract\n"
            "Reason internally if needed. The visible reply must be exactly one line: "
            "<answer>final answer</answer>. Do not write reasoning, explanations, or text after </answer>."
        )
    return (
        "You are a careful benchmark solver. Use the provided skill when it helps. "
        "Ground the answer in the given problem data."
        f"{skill}\n\n"
        "## Output Contract\n"
        "Reason internally if needed. The visible reply must be exactly one line: "
        "<answer>final answer</answer>. Do not write reasoning, explanations, or text after </answer>."
    )

def _user(benchmark: str, item: dict) -> str:
    if benchmark == "dapo_math":
        return f"## Math Problem\n{item['question']}\n\nReturn the final answer inside <answer>...</answer>."
    if benchmark == "hotpotqa":
        return (
            f"## Context\n{item.get('context', '')}\n\n"
            f"## Question\n{item['question']}\n\n"
            "Answer with the shortest context-supported answer inside <answer>...</answer>."
        )
    if benchmark == "bigcodebench":
        public_tests = _public_tests(item)
        tests = f"\n\n## Public tests\n```python\n{public_tests.rstrip()}\n```" if public_tests else ""
        return (
            f"## Task\n{item.get('instruct_prompt') or item.get('question')}\n\n"
            f"## Required starter code\n```python\n{item.get('code_prompt', '').rstrip()}\n```\n\n"
            f"{tests}\n\n"
            "Return a single Python code block containing the complete solution."
        )
    raise ValueError(f"Unsupported skill experiment benchmark: {benchmark}")

def _react_user(benchmark: str, item: dict) -> str:
    if benchmark == "hotpotqa":
        return (
            f"## Question\n{item['question']}\n\n"
            "Use the context tools to inspect the HotpotQA documents. Return the final short answer inside "
            "<answer>...</answer>."
        )
    return _user(benchmark, item)

def _public_tests(item: dict) -> str:
    for key in ("public_tests", "public_test", "sample_tests", "example_tests"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raw = item.get("public_tests_json")
    if isinstance(raw, str) and raw.strip():
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return str(parsed.get("test") or parsed.get("tests") or "").strip()
        if isinstance(parsed, list):
            return "\n".join(str(part) for part in parsed if str(part).strip()).strip()
    return ""

def evaluate_item(item: dict, skill, *, benchmark: str, mode: str, output: Path,
                  max_tokens: int, max_steps: int, seed: int, timeout: int,
                  judge_url: str = "", judge_model: str = "") -> dict:
    directory = output / "predictions" / str(item["id"]).replace("/", "_")
    directory.mkdir(parents=True, exist_ok=True)
    config = get_target_config_snapshot(max_tokens=max_tokens)
    result = {"id": item["id"], "question": item.get("question", ""), "seed": seed,
              "hard": 0, "soft": 0.0, "agent_ok": True, "fail_reason": ""}
    if mode == "react":
        system = (_system(benchmark, "") + "\n\n## Skill Access\n"
                  "Use the ReAct tools to inspect the available skill files before answering. "
                  "Reading at least one skill file is mandatory.")
        user = _react_user(benchmark, item)
        code = CodeToolContext(item["code_prompt"], item["entry_point"], _public_tests(item),
                               str(directory), timeout) if benchmark == "bigcodebench" else None
        qa = QAContext(item["question"], item["context"], 6000) if benchmark == "hotpotqa" else None
        agent = run_qwen_agent_react(
            system=system, user=user, skill_resource=skill, max_output_tokens=max_tokens,
            code_context=code, qa_context=qa, max_agent_steps=max_steps, seed=seed,
            max_solution_repair_retries=2 if code else 1, target_config=config)
        response = agent.final_response or agent.response
        if code and agent.solution_code:
            response = f"```python\n{agent.solution_code.rstrip()}\n```"
        conversation = agent.conversation
        result.update(n_turns=agent.n_turns, skill_routing=agent.metadata, raw_response=agent.response)
        calls = [call for message in conversation
                 for call in (message.get("extra") or {}).get("model_responses", [])]
        result["length_truncated"] = any((c.get("extra") or {}).get("finish_reason") == "length" for c in calls)
        checks = [(agent.metadata["required_skill_file_read"], "no skill file read"),
                  (agent.metadata["final_response_ready"], "no final response")]
        if qa:
            checks.append((agent.metadata["context_tool_used"], "no context tool used"))
        if code:
            checks.append((bool(agent.solution_code), "no solution written"))
            if agent.metadata["public_tests_available"]:
                checks.append((bool(agent.metadata["public_tests_attempts"]), "no public tests run"))
        errors = [reason for ok, reason in checks if not ok]
        result.update(agent_ok=not errors, fail_reason="; ".join(errors))
    else:
        system, user = _system(benchmark, render_skill(skill)), _user(benchmark, item)
        response, usage = chat_target(system, user, max_completion_tokens=max_tokens, seed=seed)
        conversation = [{"role": "assistant", "content": response, **usage}]
        result.update(n_turns=1, length_truncated=usage["finish_reason"] == "length", **usage)
    (directory / "conversation.json").write_text(json.dumps(conversation, ensure_ascii=False, indent=2))
    (directory / "prompt.json").write_text(json.dumps({"system": system, "user": user}, ensure_ascii=False, indent=2))
    result["response"] = response
    result["gold_answers"] = _answers(item)
    if not result["agent_ok"]:
        return result
    if benchmark == "bigcodebench":
        score = evaluate_bigcodebench(response, item, timeout)
        (directory / "candidate.py").write_text(score.pop("code"))
    elif benchmark == "dapo_math":
        score = evaluate_dapo_math(response, _answers(item),
            require_completed_thinking=(mode == "direct" and config["enable_thinking"]
                                        and not result.get("reasoning_content")),
            native_gemma=mode == "direct" and is_gemma4_model(config["deployment"]))
    else:
        score = evaluate_open_answer(response, _answers(item))
    if (benchmark != "bigcodebench" and not score["hard"] and judge_model
            and score["predicted_answer"].strip()):
        judgment = judge_answer(item["question"], score["predicted_answer"], _answers(item),
                               base_url=judge_url, model=judge_model, benchmark=benchmark)
        score.update(rule_hard=score["hard"], judge=judgment)
        if judgment["accepted"]:
            score.update(hard=1, soft=1.0)
    result.update(score)
    return result
