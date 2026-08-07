from __future__ import annotations

from datetime import datetime

from . import config
from .skills import list_skills, scan_skills


PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": (
        "Available tools: bash, read_file, write_file, edit_file, glob, "
        "todo_write, task, load_skill, compact, "
        "create_task, list_tasks, get_task, claim_task, complete_task."
    ),
    "memory": "Relevant memories are injected below when available.",
}

SUB_SYSTEM = (
    f"You are a coding subagent at {config.WORKDIR}. "
    "Complete the task, then return a concise final summary. "
    "Do not spawn more agents."
)


def subagent_system_prompt() -> str:
    return (
        f"You are a coding subagent at {config.WORKDIR}. "
        "Complete the task, then return a concise final summary. "
        "Do not spawn more agents."
    )


def assemble_system_prompt(context: dict) -> str:
    scan_skills()
    sections = [
        PROMPT_SECTIONS["identity"],
        PROMPT_SECTIONS["tools"],
        f"Working directory: {config.WORKDIR}",
    ]
    sections.append(
        f"Current time: {datetime.now().isoformat(timespec='seconds')}"
    )
    sections.append(
        "Skills catalog:\n"
        + list_skills()
        + "\nUse load_skill(name) when a skill is relevant."
    )
    if context.get("memories"):
        sections.append(f"Relevant memories:\n{context['memories']}")
    return "\n\n".join(sections)


class PromptAssembler:
    """Compatibility wrapper for the pre-migration runtime."""

    def build(self, state: dict) -> str:
        sections = [
            (
                "Identity",
                f"You are {state.get('identity', 'Simple CC')}, a pragmatic coding agent. Use tools to inspect and modify the selected workspace.",
            ),
            ("Workspace", str(state.get("workspace", ""))),
            ("Tools", str(state.get("tools", ""))),
            ("Skills", str(state.get("skills", "No skills discovered."))),
            ("Memory", str(state.get("memory", "No memories."))),
            ("Tasks", str(state.get("tasks", "No tasks."))),
            ("Team", str(state.get("team", "No teammates."))),
            (
                "Safety",
                "Stay inside the workspace. Respect permission denials. Verify changes before claiming success.",
            ),
        ]
        return "\n\n".join(f"## {title}\n{body}" for title, body in sections)
