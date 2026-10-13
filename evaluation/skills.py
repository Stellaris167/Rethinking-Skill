from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SkillFile:
    path: str
    content: str
    title: str
    description: str = ""

@dataclass
class SkillBundle:
    root: Path
    files: dict[str, SkillFile]

def load_skill(path: Path) -> str | SkillBundle:
    if path.is_file():
        return path.read_text()
    files = {}
    for entry in sorted(path.rglob("*.md")):
        relative = entry.relative_to(path)
        if any(part.startswith(".") for part in relative.parts):
            continue
        content = entry.read_text()
        title = next((line.lstrip("# ") for line in content.splitlines() if line.startswith("#")), entry.stem)
        description = next((line[12:].strip().strip("\"'") for line in content.splitlines()
                            if line.lower().startswith("description:")), "")
        files[relative.as_posix()] = SkillFile(relative.as_posix(), content, title, description)
    if not files:
        raise ValueError(f"No skill markdown files in {path}")
    return SkillBundle(path, files)

def render_skill(skill: str | SkillBundle) -> str:
    if isinstance(skill, str):
        return skill
    return "\n\n".join(f"### File: {item.path}\n{item.content}" for item in skill.files.values())
