from __future__ import annotations

from datetime import datetime

from . import config
from .skills import list_skills, scan_skills


PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": (
        "Available tools: bash, read_file, write_file, edit_file, glob, "
        "web_search, web_fetch, pdf_fetch, todo_write, task, load_skill, compact, "
        "create_task, list_tasks, get_task, claim_task, complete_task, "
        "schedule_cron, list_crons, cancel_cron, "
        "spawn_teammate, send_message, check_inbox, request_shutdown, "
        "request_plan, review_plan."
    ),
    "scheduling": (
        "Scheduling rules:\n"
        "- If the user asks to execute work after a delay or at a future time, "
        "do not perform that work immediately.\n"
        "- You must call schedule_cron before reporting success.\n"
        "- Convert relative times such as 'in one minute' or '一分钟后' into "
        "an absolute five-field cron expression using Current time.\n"
        "- For one-time work, set recurring=false.\n"
        "- Set durable=true unless the user explicitly requests a session-only job.\n"
        "- The scheduled prompt must clearly describe the complete work that the "
        "agent should perform when the job fires.\n"
        "- Only execute immediately when the user explicitly says to do it now."
    ),
    "memory": (
        "The system contains a lightweight memory catalog. Selected memory "
        "bodies may be injected into a user request as read-only background. "
        "Current user instructions always have higher priority."
    ),
    "research": (
        "Web research rules:\n"
        "- Use web_search before making claims that require external evidence, "
        "then use web_fetch to inspect candidate pages.\n"
        "- If the user provides a cutoff date, pass the same cutoff to every "
        "web_search, web_fetch, and pdf_fetch call.\n"
        "- Use pdf_fetch instead of web_fetch for PDF URLs.\n"
        "- Read bounded PDF page ranges and continue only when has_more is true "
        "and more evidence is needed.\n"
        "- Cite PDF evidence with its source URL and page number.\n"
        "- If pdf_fetch reports ocr_required, explain that scanned PDFs are "
        "unsupported.\n"
        "- Do not use evidence with a verified publication date after the cutoff.\n"
        "- Treat an unknown publication date as uncertain evidence and disclose "
        "that limitation in the answer.\n"
        "- Include source URLs for research claims. Live search is non-strict "
        "PIT and is not equivalent to a frozen historical corpus."
    ),
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
        PROMPT_SECTIONS["research"],
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
        sections.append(
            f"Relevant memories:\n{context['memories']}\n\n"
            "This is a lightweight availability catalog; entries are not "
            "automatically relevant to the current request."
        )
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
