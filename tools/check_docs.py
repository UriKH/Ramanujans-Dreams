#!/usr/bin/env python3
"""
Documentation-standard checker for the Ramanujan's Dreams codebase.

Enforces the docstring rules from ``context/background/CODE_QUALITY.md`` and
``context/DEFINITION_OF_DONE.md`` §2: every public class, function and method
(plus ``__init__``) must carry a docstring, and a public *function* that takes
parameters should document them with ``:param`` lines.

Two severities:

* **error** (blocks a commit): a public definition has *no* docstring at all.
* **warning** (reported, never blocks): a public function documents fewer
  ``:param`` entries than it has documentable parameters, or omits ``:return:``
  while clearly returning a value.  Warnings flag drift without making the gate
  noisy enough to be bypassed.

Usage::

    python tools/check_docs.py path/to/file.py [more.py ...]
    python tools/check_docs.py --staged          # check git-staged *.py files

Exit code is ``1`` iff at least one *error* was found; warnings alone exit ``0``.
"""

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Finding:
    """A single documentation issue located in a source file.

    :param path: File the issue was found in.
    :param line: 1-based line number of the offending definition.
    :param severity: ``"error"`` (blocks) or ``"warning"`` (advisory).
    :param message: Human-readable description of the issue.
    """

    path: str
    line: int
    severity: str
    message: str


def _is_public(name: str) -> bool:
    """
    Decide whether a definition name is part of the public surface to document.

    Public means it does not start with an underscore; ``__init__`` is treated
    as public because it is the documented constructor of a class.

    :param name: The class / function / method name.
    :return: True if the definition should carry a docstring.
    """
    if name == "__init__":
        return True
    return not name.startswith("_")


def _documentable_params(func: ast.AST) -> List[str]:
    """
    Collect the parameter names of a function that ought to be documented.

    Excludes ``self`` / ``cls`` and the ``*args`` / ``**kwargs`` catch-alls,
    which the standard does not require an individual ``:param`` line for.

    :param func: A ``FunctionDef`` / ``AsyncFunctionDef`` AST node.
    :return: The list of parameter names expected in the docstring.
    """
    args = func.args
    names: List[str] = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    return [n for n in names if n not in ("self", "cls")]


def _returns_value(func: ast.AST) -> bool:
    """
    Heuristically decide whether a function returns a meaningful value.

    True if any ``return <expr>`` (with a non-``None`` expression) or any
    ``yield`` appears in the body.

    :param func: A function AST node.
    :return: True if the function appears to produce a value.
    """
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                return True
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
    return False


def _check_def(node: ast.AST, path: str, findings: List[Finding]) -> None:
    """
    Check a single class / function definition and append any findings.

    Only the definition itself is examined; nested functions inside a function
    body are *not* public surface and are skipped by the caller.

    :param node: A ``ClassDef`` / ``FunctionDef`` / ``AsyncFunctionDef`` node.
    :param path: Path label used in the findings.
    :param findings: Mutable list that findings are appended to.
    """
    if not _is_public(node.name):
        return

    doc = ast.get_docstring(node)
    if not doc or not doc.strip():
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        findings.append(Finding(
            path, node.lineno, "error",
            f"public {kind} '{node.name}' has no docstring.",
        ))
        return

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        params = _documentable_params(node)
        documented = doc.count(":param ")
        if params and documented < len(params):
            findings.append(Finding(
                path, node.lineno, "warning",
                f"function '{node.name}' documents {documented} of "
                f"{len(params)} parameters with ':param'.",
            ))
        if _returns_value(node) and ":return" not in doc:
            findings.append(Finding(
                path, node.lineno, "warning",
                f"function '{node.name}' returns a value but has no ':return:'.",
            ))


def check_source(path: str, source: str) -> List[Finding]:
    """
    Check one Python source string against the documentation standard.

    Only module-level definitions and class methods are inspected — functions
    nested inside other functions (closures, test helpers) are private by
    construction and exempt.

    :param path: Path label used in the findings (not read from disk here).
    :param source: The Python source code to analyse.
    :raises SyntaxError: If *source* cannot be parsed.
    :return: All findings (errors and warnings) for this file.
    """
    tree = ast.parse(source, filename=path)
    findings: List[Finding] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_def(node, path, findings)
        elif isinstance(node, ast.ClassDef):
            _check_def(node, path, findings)
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _check_def(member, path, findings)

    return findings


def _staged_python_files() -> List[str]:
    """
    List the git-staged Python files (added / copied / modified).

    :return: Paths of staged ``*.py`` files, relative to the repo root.
    """
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if line.endswith(".py")]


def _is_exempt(path: str) -> bool:
    """
    Decide whether a file is exempt from the hard documentation gate.

    Test files and conftest are exempt: their intent is carried by descriptive
    names and validated by *running* pytest, and a blocking docstring gate on
    every ``test_*`` would be noisy enough to invite ``--no-verify`` bypass.
    Test quality is enforced by review per the Definition of Done.

    :param path: The file path to classify.
    :return: True if the file should be skipped by the checker.
    """
    norm = path.replace("\\", "/")
    base = norm.rsplit("/", 1)[-1]
    return (
        "/tests/" in norm
        or norm.startswith("tests/")
        or base.startswith("test_")
        or base == "conftest.py"
    )


def _check_paths(paths: List[str]) -> Tuple[List[Finding], int]:
    """
    Run the checker over a list of file paths, reading each from disk.

    :param paths: Python file paths to check.
    :return: ``(findings, error_count)`` across all files; files that fail to
        parse are reported as a single error finding each.
    """
    findings: List[Finding] = []
    for path in paths:
        if _is_exempt(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                source = fh.read()
        except OSError:
            continue  # File staged for deletion / unreadable — nothing to check.
        try:
            findings.extend(check_source(path, source))
        except SyntaxError as exc:
            findings.append(Finding(path, exc.lineno or 0, "error", f"syntax error: {exc.msg}"))

    errors = sum(1 for f in findings if f.severity == "error")
    return findings, errors


def main(argv: List[str]) -> int:
    """
    Command-line entry point.

    :param argv: Argument vector (excluding the program name).
    :return: Process exit code (1 if any error-severity finding, else 0).
    """
    parser = argparse.ArgumentParser(description="Check docstring standards.")
    parser.add_argument("files", nargs="*", help="Python files to check.")
    parser.add_argument("--staged", action="store_true",
                        help="Check git-staged *.py files instead of positional paths.")
    args = parser.parse_args(argv)

    paths = _staged_python_files() if args.staged else args.files
    if not paths:
        return 0

    findings, errors = _check_paths(paths)

    for f in sorted(findings, key=lambda x: (x.severity != "error", x.path, x.line)):
        print(f"{f.severity.upper()}: {f.path}:{f.line}: {f.message}")

    if errors:
        print(f"\nDocumentation check failed: {errors} error(s). "
              f"Add docstrings (see context/background/CODE_QUALITY.md).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
