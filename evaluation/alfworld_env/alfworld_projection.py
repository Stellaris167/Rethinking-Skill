# Vendored from SkillRL (Apache-2.0 License)
# Original: agent_system/environments/env_package/alfworld/projection.py

from typing import List
import re

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """Process raw model outputs into valid ALFWorld actions.

    Extracts text from ``<action>...</action>`` tags and validates that
    the response also contains ``<think>...</think>`` tags.

    Parameters
    ----------
    actions : list[str]
        Raw model outputs, one per environment.
    action_pools : list[list[str]]
        Admissible action lists per environment (unused but kept for API compat).

    Returns
    -------
    actions : list[str]
        Cleaned action strings.
    valids : list[int]
        1 if the action was successfully parsed, 0 otherwise.
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        lowered = original_str.lower()
        match = re.search(r"<action>(.*?)</action>", lowered, re.DOTALL)
        if match is None:
            actions[i] = lowered[-30:]
            continue

        actions[i] = match.group(1).strip()
        valids[i] = 1

        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        has_complete_think = think_start_idx != -1 and think_end_idx > think_start_idx
        has_prompt_opened_think = think_start_idx == -1 and think_end_idx > 0
        if not has_complete_think and not has_prompt_opened_think:
            valids[i] = 0

        # Reject responses containing Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids
