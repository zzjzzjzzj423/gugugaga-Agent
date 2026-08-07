from __future__ import annotations

import yaml

from . import config


SKILL_REGISTRY: dict[str, dict] = {}


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        metadata = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        metadata = {}
    return metadata, parts[2].strip()


def scan_skills():
    SKILL_REGISTRY.clear()
    if not config.SKILLS_DIR.exists():
        return
    for directory in sorted(config.SKILLS_DIR.iterdir()):
        if not directory.is_dir():
            continue
        manifest = directory / "SKILL.md"
        if not manifest.exists():
            continue
        raw = manifest.read_text()
        metadata, _ = _parse_frontmatter(raw)
        name = metadata.get("name", directory.name)
        description = metadata.get(
            "description", raw.split("\n")[0].lstrip("#").strip()
        )
        SKILL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "content": raw,
        }


scan_skills()


def list_skills() -> str:
    if not SKILL_REGISTRY:
        return "(no skills found)"
    return "\n".join(
        f"- {skill['name']}: {skill['description']}"
        for skill in SKILL_REGISTRY.values()
    )


def load_skill(name: str) -> str:
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        available = ", ".join(SKILL_REGISTRY.keys()) or "(none)"
        return f"Skill not found: {name}. Available: {available}"
    return skill["content"]
