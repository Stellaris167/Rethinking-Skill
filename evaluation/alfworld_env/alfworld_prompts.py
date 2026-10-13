# Vendored from SkillRL (Apache-2.0 License)
# Original: agent_system/environments/prompts/alfworld.py

from pathlib import Path

ROOT = Path(__file__).parent
ALFWORLD_TEMPLATE_NO_HIS = (ROOT / "rollout_no_history.md").read_text()
ALFWORLD_TEMPLATE = (ROOT / "rollout_with_history.md").read_text()
