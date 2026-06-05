"""Console output helpers powered by Rich."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Prompt

STDOUT_CONSOLE = Console(highlight=False, soft_wrap=True)
STDERR_CONSOLE = Console(stderr=True, highlight=False, soft_wrap=True)

STYLE_BY_ROLE = {
    "divider": "grey50",
    "observation": "cyan",
    "command": "bold green",
    "result": "white",
    "goal": "magenta",
    "summary_success": "bold green",
    "summary_warning": "bold yellow",
    "debug": "yellow",
    "agent_status": "bold blue",
    "agent_debug": "magenta",
    "hint_request": "bold bright_cyan",
    "error": "bold red",
}


def ui_print(text: str, *, role: str = "result", stderr: bool = False) -> None:
    console = STDERR_CONSOLE if stderr else STDOUT_CONSOLE
    console.print(text, style=STYLE_BY_ROLE.get(role), markup=False)


def ui_prompt(prompt_text: str, *, role: str = "hint_request") -> str:
    style = STYLE_BY_ROLE.get(role)
    if style:
        return Prompt.ask(f"[{style}]{prompt_text}[/{style}]", console=STDOUT_CONSOLE)
    return Prompt.ask(prompt_text, console=STDOUT_CONSOLE)
