"""ALFWorld episodes and trajectory recording."""
from __future__ import annotations

import concurrent.futures
import json
import os
import re

from evaluation.model import chat_target, chat_target_messages
from evaluation.model import get_target_config_snapshot, is_gemma4_model, split_gemma4_response

# ── Constants ─────────────────────────────────────────────────────────────────

TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

class AlfworldRolloutError(RuntimeError):
    """Raised when ALFWorld rollout cannot continue because model calls failed."""

def _get_task_type(gamefile: str) -> str:
    for task in TASKS:
        if task in gamefile:
            return task
    return "other"

def _extract_action(model_response: str) -> str | None:
    match = re.search(r"<action>(.*?)</action>", model_response, re.DOTALL)
    return match.group(1).strip() if match else None

def _gemma4_action_projection(responses: list[str], action_pools: list[list[str]]) -> tuple[list[str], list[int]]:
    del action_pools
    actions = [_extract_action(split_gemma4_response(text)[0]) for text in responses]
    return [action.lower() if action is not None else "" for action in actions], [
        int(action is not None) for action in actions
    ]

def _extract_think(model_response: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", model_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    closing = model_response.find("</think>")
    if closing >= 0:
        return model_response[:closing].strip()
    return None

def _is_no_effect(feedback: str) -> bool:
    return str(feedback or "").strip().lower() in {"nothing happens.", "nothing happens"}

def _build_skill_prompt(skill_content: str) -> str:
    """Build the skill section to inject into the agent's system prompt."""
    if not skill_content or not skill_content.strip():
        return ""
    return (
        "\n\n## Skill Knowledge\n"
        "Below is a skill document with learned strategies. "
        "Use these guidelines to inform your decisions:\n\n"
        f"{skill_content}\n"
    )

def _build_react_system_prompt(skill_content: str, *, native_gemma: bool = False) -> str:
    prompt = (
        "You are an expert agent operating in the ALFRED Embodied Environment.\n"
        "At each turn, read the current observation and admissible actions, then "
        "respond with exactly this format:\n"
        "<think>brief reasoning about the next useful step</think>\n"
        "<action>one admissible action</action>\n"
        "Do not output anything after the closing </action> tag."
    )
    if native_gemma:
        prompt = (
            "You are an expert agent operating in the ALFRED Embodied Environment.\n"
            "At each turn, read the current observation and admissible actions. "
            "Keep reasoning in the model's native thought channel. The visible response must be "
            "exactly <action>one admissible action</action>. "
            "This response format overrides any reasoning-markup instructions in the skill document. "
            "Do not output anything after the closing </action> tag."
        )
    if skill_content and skill_content.strip():
        prompt += (
            "\n\n## Skill Knowledge\n"
            "Use the following learned strategy document when choosing actions:\n\n"
            f"{skill_content if native_gemma else skill_content.strip()}"
        )
    return prompt

def _trim_chat_history(
    messages: list[dict],
    *,
    max_turns: int,
    max_chars: int,
) -> list[dict]:
    if max_turns > 0:
        messages = messages[-2 * max_turns:]
    if max_chars <= 0:
        return list(messages)
    trimmed = list(messages)
    while len(trimmed) > 2 and sum(len(str(m.get("content", ""))) for m in trimmed) > max_chars:
        trimmed = trimmed[2:]
    return trimmed

def _resolve_alfworld_gamefile(gamefile: str) -> str:
    path = os.path.expanduser(os.path.expandvars(str(gamefile)))
    if os.path.isabs(path):
        return path

    data_root = os.environ.get("ALFWORLD_DATA", "").strip()
    if not data_root:
        return path

    root = os.path.expanduser(os.path.expandvars(data_root))
    return os.path.abspath(os.path.join(root, path))

def _resolve_alfworld_gamefiles(gamefiles: list[str] | None) -> list[str] | None:
    if gamefiles is None:
        return None
    return [_resolve_alfworld_gamefile(gamefile) for gamefile in gamefiles]

# ── Environment builder ──────────────────────────────────────────────────────

def build_alfworld_env(
    env_num: int,
    eval_dataset: str = "eval_out_of_distribution",
    seed: int = 42,
    is_train: bool = False,
    specific_gamefiles: list[str] | None = None,
):
    """Build ALFWorld environment manager.

    Args:
        env_num: number of parallel environments
        eval_dataset: 'eval_in_distribution' or 'eval_out_of_distribution' or train
        seed: random seed
        is_train: whether to use training set

    Returns:
        env_manager: AlfWorldEnvironmentManager instance
    """
    from functools import partial

    from omegaconf import OmegaConf

    from evaluation.alfworld_env.alfworld_envs import build_alfworld_envs
    from evaluation.alfworld_env.alfworld_projection import alfworld_projection
    from evaluation.alfworld_env.env_manager import AlfWorldEnvironmentManager

    HERE = os.path.dirname(os.path.abspath(__file__))

    alf_config_path = os.path.join(HERE, "alfworld_env", "config_tw.yaml")
    env_kwargs = {"eval_dataset": eval_dataset}
    resolved_gamefiles = _resolve_alfworld_gamefiles(specific_gamefiles)

    envs = build_alfworld_envs(
        alf_config_path,
        seed=seed,
        env_num=env_num,
        group_n=1,
        is_train=is_train,
        env_kwargs=env_kwargs,
        resources_per_worker=None,
        gamefiles=resolved_gamefiles,
    )

    config = OmegaConf.create(
        {
            "env": {
                "history_length": 2,
                "env_name": "alfworld/AlfredTWEnv",
            }
        }
    )

    projection_f = partial(alfworld_projection)
    if is_gemma4_model(get_target_config_snapshot().get("deployment", "")):
        projection_f = _gemma4_action_projection
    env_manager = AlfWorldEnvironmentManager(envs, projection_f, config)
    return env_manager

# ── Batch rollout ─────────────────────────────────────────────────────────────

def run_alfworld_batch(
    env_manager,
    skill_content: str,
    max_steps: int = 50,
    out_root: str = "",
    max_api_workers: int = 8,
    temperature: float = 0.4,
    max_completion_tokens: int = 16384,
    result_ids: list[str] | None = None,
    react_history: bool = False,
    react_history_max_turns: int = 12,
    react_history_max_chars: int = 30000,
    llm_timeout: int | None = 900,
) -> list[dict]:
    """Run a batch of ALFWorld episodes.

    Returns per-episode scores and action statistics:
    [
        {
            "id": "<task_id>",
            "hard": 0 or 1,
            "soft": 0.0 or 1.0,
            "n_turns": <int>,
            "fail_reason": "<str>",
            "agent_ok": True,
            "task_type": "<str>",
            "gamefile": "<str>",
            "task_description": "<str>",
        },
        ...
    ]

    Also saves conversation.json per environment in out_root/predictions/<task_id>/
    """
    skill_prompt = _build_skill_prompt(skill_content)
    native_gemma = is_gemma4_model(
        get_target_config_snapshot().get("deployment", "")
    )
    react_system_prompt = _build_react_system_prompt(skill_content, native_gemma=native_gemma)

    obs, infos = env_manager.reset({})
    env_num = len(obs["text"])
    env_dones = [False] * env_num
    overall_success = [False] * env_num

    # Build per-env metadata
    env_meta: list[dict] = []
    for i in range(env_num):
        gamefile = infos[i].get("extra.gamefile", "") if isinstance(infos[i], dict) else ""
        task_type = _get_task_type(gamefile)
        # Extract task description from initial observation
        task_desc = ""
        anchor_text = obs["anchor"][i] if "anchor" in obs else ""
        task_start = anchor_text.find("Your task is to: ")
        if task_start != -1:
            task_desc = anchor_text[task_start + len("Your task is to: "):].strip()

        env_meta.append({
            "gamefile": gamefile,
            "task_type": task_type,
            "task_description": task_desc,
        })

    # Per-env conversation records
    conversations: list[list[dict]] = [[] for _ in range(env_num)]
    chat_histories: list[list[dict]] = [[] for _ in range(env_num)]

    for step_idx in range(max_steps):
        if all(env_dones):
            break

        active_indices = [i for i in range(env_num) if not env_dones[i]]

        # Build prompts with skill injection
        prompts: dict[int, str] = {}
        for i in active_indices:
            prompt = obs["text"][i]
            if native_gemma:
                # Replace only the vendored runtime instruction, never the skill document.
                prompt = prompt.replace(
                    "This reasoning process MUST be enclosed within <think> </think> tags.",
                    "Keep this reasoning in the model's native thought channel.",
                )
            if skill_prompt and not react_history:
                # Inject skill before the action instruction
                prompt = skill_prompt + "\n" + prompt
            prompts[i] = prompt

        # Call API in parallel
        actions = ["None"] * env_num
        model_reasoning = [""] * env_num
        finish_reasons = [""] * env_num
        raw_responses = [""] * env_num

        def call_api(idx):
            if react_history:
                history = _trim_chat_history(
                    chat_histories[idx],
                    max_turns=react_history_max_turns,
                    max_chars=react_history_max_chars,
                )
                messages = (
                    [{"role": "system", "content": react_system_prompt}]
                    + history
                    + [{"role": "user", "content": prompts[idx]}]
                )
                response, usage = chat_target_messages(
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    timeout=llm_timeout,
                )
            else:
                response, usage = chat_target(
                    system=(
                        _build_react_system_prompt("", native_gemma=True) if native_gemma else
                        "You are an expert agent operating in the ALFRED Embodied Environment."
                    ),
                    user=prompts[idx],
                    max_completion_tokens=max_completion_tokens,
                    timeout=llm_timeout,
                )
            raw_response = usage.get("raw_response", response)
            response = (response or "").strip()
            reasoning = str(usage.get("reasoning_content") or "")
            if not native_gemma:
                reasoning = reasoning.strip()
            finish_reason = str(usage.get("finish_reason") or "").strip()
            return idx, response, reasoning, finish_reason, raw_response

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_api_workers)
        try:
            futures = {executor.submit(call_api, i): i for i in active_indices}
            pending_futs = set(futures)
            while pending_futs:
                done, _ = concurrent.futures.wait(
                    pending_futs,
                    timeout=5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    pending_futs.remove(future)
                    try:
                        idx, response, reasoning, finish_reason, raw_response = future.result()
                    except Exception as exc:  # noqa: BLE001
                        idx = futures[future]
                        raise AlfworldRolloutError(
                            f"ALFWorld LLM call failed at step {step_idx}, env {idx}: {exc}"
                        ) from exc
                    actions[idx] = response
                    model_reasoning[idx] = reasoning
                    finish_reasons[idx] = finish_reason
                    raw_responses[idx] = raw_response
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        # Save model responses before stepping
        model_responses = {i: actions[i] for i in active_indices}
        if react_history:
            for i in active_indices:
                chat_histories[i].append({"role": "user", "content": prompts[i]})
                chat_histories[i].append({"role": "assistant", "content": model_responses[i]})
                chat_histories[i] = _trim_chat_history(
                    chat_histories[i],
                    max_turns=react_history_max_turns,
                    max_chars=react_history_max_chars,
                )

        admissible_actions = [
            {str(action).strip().lower() for action in commands}
            for commands in env_manager.envs.get_admissible_commands
        ]

        # Step environment
        obs, rewards, dones, infos = env_manager.step(actions)

        # Record trajectory
        for i in active_indices:
            action = _extract_action(model_responses[i])
            reasoning = model_reasoning[i] or _extract_think(model_responses[i]) or ""
            previous_action = conversations[i][-1]["action"] if conversations[i] else None
            feedback = obs["anchor"][i] if "anchor" in obs else ""
            step_record = {
                "step": step_idx,
                "action": action,
                "reasoning": reasoning,
                "model_response": model_responses[i],
                "finish_reason": finish_reasons[i],
                "env_feedback": feedback,
                "reward": float(rewards[i]),
                "done": bool(dones[i]),
                "action_format_valid": action is not None,
                "action_admissible": bool(
                    action and action.strip().lower() in admissible_actions[i]
                ),
                "repeated_action": bool(action and action == previous_action),
                "no_effect": _is_no_effect(feedback),
            }
            if native_gemma:
                step_record["raw_response"] = raw_responses[i]
            conversations[i].append(step_record)

        # Update done status
        for i in range(env_num):
            if env_dones[i]:
                continue
            if dones[i]:
                env_dones[i] = True
                won = bool(infos[i].get("won", False))
                overall_success[i] = won

    # Build results and save conversations
    results: list[dict] = []
    pred_dir = os.path.join(out_root, "predictions") if out_root else ""

    for i in range(env_num):
        gamefile = env_meta[i]["gamefile"]
        task_type = env_meta[i]["task_type"]
        task_desc = env_meta[i]["task_description"]
        n_turns = len(conversations[i])
        won = overall_success[i]
        n_format_valid = sum(step["action_format_valid"] for step in conversations[i])
        n_admissible = sum(step["action_admissible"] for step in conversations[i])
        n_repeated = sum(step["repeated_action"] for step in conversations[i])
        n_no_effect = sum(step["no_effect"] for step in conversations[i])
        n_length_truncated = sum(
            step["finish_reason"] == "length" for step in conversations[i]
        )

        # Generate stable task ID from env index and gamefile
        task_id = str(result_ids[i]) if result_ids and i < len(result_ids) else f"env_{i:03d}"

        fail_reason = ""
        if not won:
            if not env_dones[i]:
                fail_reason = f"Timeout after {max_steps} steps"
            else:
                fail_reason = "Episode ended without completing the task"

        result = {
            "id": task_id,
            "hard": 1 if won else 0,
            "soft": 1.0 if won else 0.0,
            "n_turns": n_turns,
            "fail_reason": fail_reason,
            "agent_ok": True,  # ALFWorld agent always runs OK (no crash)
            "n_format_valid": n_format_valid,
            "n_admissible": n_admissible,
            "n_repeated": n_repeated,
            "n_no_effect": n_no_effect,
            "n_length_truncated": n_length_truncated,
            "format_valid_rate": n_format_valid / n_turns if n_turns else 0.0,
            "admissible_action_rate": n_admissible / n_turns if n_turns else 0.0,
            "repeated_action_rate": n_repeated / n_turns if n_turns else 0.0,
            "no_effect_rate": n_no_effect / n_turns if n_turns else 0.0,
            "length_truncated_rate": n_length_truncated / n_turns if n_turns else 0.0,
            "task_type": task_type,
            "gamefile": gamefile,
            "task_description": task_desc,
        }
        results.append(result)

        # Save conversation
        if pred_dir:
            conv_dir = os.path.join(pred_dir, task_id)
            os.makedirs(conv_dir, exist_ok=True)
            with open(os.path.join(conv_dir, "conversation.json"), "w") as f:
                json.dump(conversations[i], f, ensure_ascii=False, indent=2)

    return results
