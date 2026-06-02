#!/usr/bin/env python3
"""
Claude Code ``PreToolUse`` gate for ``git commit`` (Definition-of-Done enforcement).

Wired into ``.claude/settings.json`` on the ``Bash`` tool.  It reads the hook
payload on stdin; for any Bash command that is *not* a ``git commit`` it exits
immediately (allow).  For a ``git commit`` it enforces two Definition-of-Done
rules and **blocks** the commit (exit code 2) if either fails:

1. **Documentation standard** — staged ``*.py`` files must satisfy
   ``tools/check_docs.py`` (every public class / function / method documented).
   Uses only the standard library, so it runs without the conda environment.
2. **Tests pass** — the full ``pytest`` suite must be green, run inside the
   ``rama`` conda environment.

Blocking is done with exit code 2 and a reason on stderr, which Claude Code
surfaces back to the agent so it can fix the issue before retrying the commit.
"""

import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMMIT_RE = re.compile(r"\bgit\b[^\n|&;]*\bcommit\b")


def _is_git_commit(command: str) -> bool:
    """
    Decide whether a shell command invokes ``git commit``.

    :param command: The Bash command line from the hook payload.
    :return: True if the command runs a ``git commit`` (ignores ``--help``).
    """
    if "--help" in command:
        return False
    return bool(_COMMIT_RE.search(command))


def _staged_python_files() -> list:
    """
    List staged ``*.py`` files (added / copied / modified).

    :return: Repo-relative paths of staged Python files.
    """
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout
    return [line for line in out.splitlines() if line.endswith(".py")]


def _block(reason: str) -> None:
    """
    Emit a blocking decision and exit.

    :param reason: Human-readable explanation shown to the agent.
    """
    print(reason, file=sys.stderr)
    sys.exit(2)


def main() -> int:
    """
    Hook entry point: parse the payload and run the gate for git commits.

    :return: ``0`` to allow the tool call (non-commit or all checks passed).
    """
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # No / malformed payload — do not interfere.

    command = (payload.get("tool_input") or {}).get("command", "")
    if not _is_git_commit(command):
        return 0

    # --- 1. Documentation standard on staged files (stdlib only) ---
    staged = _staged_python_files()
    if staged:
        check_docs = os.path.join(REPO_ROOT, "tools", "check_docs.py")
        doc = subprocess.run(
            [sys.executable, check_docs, *staged],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if doc.returncode != 0:
            _block(
                "Commit blocked — documentation standard not met:\n"
                + doc.stdout + doc.stderr
            )

    # --- 2. Full test suite (rama conda env) ---
    pytest_cmd = (
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate rama && "
        f"cd '{REPO_ROOT}' && python -m pytest -q"
    )
    tests = subprocess.run(["bash", "-lc", pytest_cmd], capture_output=True, text=True)
    if tests.returncode != 0:
        tail = "\n".join((tests.stdout + tests.stderr).splitlines()[-25:])
        _block("Commit blocked — pytest is not green:\n" + tail)

    return 0


if __name__ == "__main__":
    sys.exit(main())
