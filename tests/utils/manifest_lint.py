# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for linting shipped Kubernetes/DGD manifests without a cluster.

Only the standard library and ``yaml`` are imported, so tests built on these
helpers run on any CPU runner. Containers are found by shape rather than by a
fixed DGD path, so the same code reads v1alpha1
(``spec.services.*.extraPodSpec.mainContainer``), v1beta1
(``spec.components[].podTemplate.spec.containers[]``) and plain pod specs.
"""

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


def iter_yaml_documents(
    root: Path, globs: Iterable[str] = ("*.yaml", "*.yml")
) -> Iterator[tuple[Path, Any]]:
    """Yield ``(path, document)`` for every non-empty YAML document under ``root``."""
    paths = sorted({p for pattern in globs for p in Path(root).rglob(pattern)})
    for path in paths:
        with path.open() as f:
            for doc in yaml.safe_load_all(f):
                if doc is not None:
                    yield path, doc


def iter_containers(node: Any, location: str = "") -> Iterator[tuple[str, dict]]:
    """Yield ``(location, mapping)`` for every mapping with a list or string
    ``command`` or ``args``.

    ``location`` is the dotted key path to the mapping, with list items named
    by their ``name`` field when they have one, so a failure can point at the
    service and container.
    """
    if isinstance(node, Mapping):
        if any(isinstance(node.get(key), (list, str)) for key in ("command", "args")):
            yield location, node
        for key, value in node.items():
            yield from iter_containers(
                value, f"{location}.{key}" if location else str(key)
            )
    elif isinstance(node, list):
        for index, item in enumerate(node):
            name = item.get("name") if isinstance(item, Mapping) else None
            yield from iter_containers(item, f"{location}[{name or index}]")


# Characters that end a simple command outside quotes. ``&`` right after a
# redirection (``2>&1``) belongs to the redirection instead.
_COMMAND_SEPARATORS = ";&|()\n"


def _split_simple_commands(text: str) -> list[list[str]]:
    """Split shell text into simple commands, each a list of unquoted words.

    Handles single and double quotes, backslash escapes and line continuations,
    ``#`` comments that start a word, ``$(...)`` substitutions (kept inside
    their word), and the ``;``, ``&``, ``|``, ``(``, ``)`` and newline
    separators. Expansions are not performed; an unterminated quote runs to
    the end of the text.
    """
    commands: list[list[str]] = []
    words: list[str] = []
    word: list[str] = []
    in_word = False
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        i += 1
        if quote:
            if ch == quote:
                quote = ""
            elif ch == "\\" and quote == '"' and i < len(text):
                nxt = text[i]
                i += 1
                if nxt not in '"\\$`\n':
                    word.append(ch)
                if nxt != "\n":
                    word.append(nxt)
            else:
                word.append(ch)
        elif ch == "\\":
            if i < len(text) and text[i] != "\n":
                word.append(text[i])
                in_word = True
            i += 1
        elif ch in "'\"":
            quote = ch
            in_word = True
        elif ch == "$" and text[i : i + 1] == "(":
            depth = 0
            start = i - 1
            while i < len(text):
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                i += 1
                if depth == 0:
                    break
            word.append(text[start:i])
            in_word = True
        elif ch == "&" and word and word[-1] in "<>":
            word.append(ch)
        elif ch == "#" and not in_word:
            newline = text.find("\n", i)
            i = len(text) if newline < 0 else newline
        elif ch in " \t\r" or ch in _COMMAND_SEPARATORS:
            if in_word:
                words.append("".join(word))
                word = []
                in_word = False
            if ch in _COMMAND_SEPARATORS and words:
                commands.append(words)
                words = []
        else:
            word.append(ch)
            in_word = True
    if in_word:
        words.append("".join(word))
    if words:
        commands.append(words)
    return commands


def _is_script_option(word: str) -> bool:
    """Return whether ``word`` is a short option group containing ``c``, as in
    ``sh -c``, ``bash -lc`` and ``sh -ec``, so the next word is a script."""
    return word.startswith("-") and not word.startswith("--") and "c" in word


def argv_commands(argv: list[str]) -> list[list[str]]:
    """Return ``argv`` followed by the commands of every script operand in it.

    A script operand is the word after a ``-c``-style option group; its
    commands are found with :func:`shell_commands`, so nested wrappers are
    unwrapped too.
    """
    commands = [argv]
    for option, operand in zip(argv, argv[1:]):
        if _is_script_option(option):
            commands.extend(shell_commands(operand))
    return commands


def shell_commands(text: str) -> list[list[str]]:
    """Return every simple command in shell text as a list of words, including
    the commands inside ``sh -c``/``bash -lc`` script operands.

    Each command keeps its own words, so a flag in one command (or in a
    comment) never counts for another.
    """
    commands: list[list[str]] = []
    for argv in _split_simple_commands(text):
        commands.extend(argv_commands(argv))
    return commands


def container_commands(container: Mapping) -> list[list[str]]:
    """Return the commands a container runs, each as a list of words.

    A list ``command`` + ``args`` is one argv, followed by the commands of any
    script operand in it (the script may sit in ``command`` or ``args``). A
    string ``command`` or ``args`` is read as shell text.
    """
    argv: list[str] = []
    scripts: list[list[str]] = []
    for key in ("command", "args"):
        value = container.get(key)
        if isinstance(value, str):
            scripts.extend(shell_commands(value))
        elif isinstance(value, list):
            argv.extend(str(item) for item in value)
    return (argv_commands(argv) if argv else []) + scripts


def has_flag(argv: list[str], flag: str, value: str | None = None) -> bool:
    """Return whether ``argv`` passes ``flag`` (with ``value`` when given), as
    either ``--flag value`` or ``--flag=value``."""
    for index, token in enumerate(argv):
        if token == flag:
            if value is None or argv[index + 1 : index + 2] == [value]:
                return True
        elif token.startswith(f"{flag}="):
            if value is None or token[len(flag) + 1 :] == value:
                return True
    return False
