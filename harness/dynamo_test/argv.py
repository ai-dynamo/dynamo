# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a worker's command line is written, read, and edited.

A Dynamo worker is launched one of two ways, and a test that wants to know what
model it serves — or to change it — has to handle both:

``ARGV``
    ``command: [python3, -m, dynamo.vllm]`` with ``args`` as a token list.
``SHELL``
    ``command: [/bin/bash, -lc]`` with ``args`` as a **single string** that a
    shell parses. Everything after that string binds to ``$0``/``$1``, so it is
    the only element there will ever be.

Measured across every ``DynamoGraphDeployment`` in ``recipes/`` and
``examples/``: 184 containers are shell-invoked, and **100 of them use ``-lc``**
rather than ``-c``. ``-lc`` is the majority spelling, not an exception — which is
why this module classifies on "a short-option cluster ending in ``c``" and keeps
the shell's own argv verbatim, instead of enumerating the spellings it expects.

## Why writes splice instead of rebuilding

The obvious implementation of an edit is ``shlex.split`` → change a token →
``" ".join(quote(t))``. That is what the existing test helpers do, and it is
lossy in three independent ways that were each measured on the live corpus:

* **47 of 184** containers contain a shell control operator. ``shlex`` hands back
  ``&&`` as an ordinary word, so a quoting re-join emits ``'&&'`` and
  ``ulimit -l unlimited && exec python3 …`` collapses into a single ``ulimit``
  call with a literal ``&&`` argument. The worker never starts.
* **17 of 184** contain whole-line ``#`` comments. ``shlex`` has no notion of
  comments, so the apostrophe in ``# Dynamo's forward-pass metrics adapter``
  opens a quote that never closes; tokenising raised ``No closing quotation`` on
  four recipes outright.
* Most of the 184 are ``\\``-continued multi-line scripts. A re-join flattens
  them to one line and deletes the comments explaining each flag.

So this module never rebuilds a shell command. It tokenises **with source
spans**, and an edit replaces exactly the bytes of the token it is changing.
Operators, comments, line continuations, and the author's original quoting come
out the far side untouched, because they are never re-emitted at all.

## Why reads return :class:`Fact`

A flag that could not be *looked for* is not a flag that is absent. Four recipes
cannot be tokenised at all; answering "no ``--model``" for those is the
false-green this harness exists to prevent. Unparseable input yields
``UNKNOWN``, never ``ABSENT``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Iterator, Sequence

from .facts import Fact

__all__ = [
    "ArgForm",
    "ArgV",
    "Token",
    "TokenKind",
    "ArgVError",
    "UnparseableCommand",
    "AmbiguousInsertion",
    "is_shell_command_flag",
]


class ArgVError(Exception):
    """Base class for command-line editing failures."""


class UnparseableCommand(ArgVError):
    """The shell command string could not be tokenised, so it cannot be edited."""


class AmbiguousInsertion(ArgVError):
    """A new flag has no unambiguous insertion point in this shell command.

    Raised rather than guessed. Appending to the end of a command that finishes
    with ``&& something-else`` or a redirection would attach the flag to the
    wrong program, and doing that silently is how a test ends up measuring a
    configuration it did not set.
    """


class ArgForm(str, Enum):
    """How the container's ``args`` are interpreted.

    Deliberately two values, not one per shell spelling. ``-c`` and ``-lc``
    differ in whether the shell reads login profiles; they do not differ in how
    the command string is parsed or edited, and treating them as separate forms
    invites an enumeration that ``-ec`` and ``-euxc`` immediately escape.
    """

    ARGV = "argv"
    SHELL = "shell"


class TokenKind(str, Enum):
    WORD = "word"
    OPERATOR = "operator"


@dataclass(frozen=True)
class Token:
    """One token, and the exact bytes of the source it came from.

    ``start``/``end`` index the original command string. ``text`` is the token
    after quote removal, so ``text`` and ``source[start:end]`` differ whenever
    the author quoted something — which is precisely why edits address the span
    and comparisons use the text.
    """

    text: str
    start: int
    end: int
    kind: TokenKind = TokenKind.WORD
    quoted: bool = False

    @property
    def is_option(self) -> bool:
        return (
            self.kind is TokenKind.WORD
            and len(self.text) > 1
            and self.text.startswith("-")
        )


# A short-option cluster that makes a shell read its command from the next word:
# -c, and the -lc / -ec / -euxc variants that are common in the recipe corpus.
# Long options are excluded so --config is never mistaken for one.
_SHELL_FLAG = re.compile(r"^-[A-Za-z]*c$")

# Characters that begin a shell control operator when unquoted.
_OPERATOR_START = frozenset("|&;<>")

# Tokens that survive unquoted in a rewritten command. ``$`` is included so
# ``$MODEL_PATH`` still expands at pod start; ``shlex.quote`` would make it
# literal.
_BARE_SAFE = re.compile(r"^[A-Za-z0-9_./:=,@%+\-$]+$")


def is_shell_command_flag(flag: object) -> bool:
    """True for a flag that makes a shell read its command from the next word."""
    return isinstance(flag, str) and bool(_SHELL_FLAG.match(flag))


def _quote(text: str) -> str:
    """Quote ``text`` for a shell command, preserving ``$VAR`` expansion."""
    if not text:
        return "''"
    if _BARE_SAFE.match(text):
        return text
    return "'" + text.replace("'", "'\\''") + "'"


def tokenize(command: str) -> tuple[Token, ...]:
    """Tokenise a shell command string, recording each token's source span.

    Handles the constructs the corpus actually contains: single and double
    quotes, backslash escapes, ``\\``-newline continuations, whole-line ``#``
    comments, and control operators.

    A ``#`` is only a comment when it starts a line (ignoring leading blanks).
    Mid-line it is left in the token, because it may sit inside a quoted JSON
    value or a URL fragment, where a shell would not treat it as a comment
    either.

    Raises:
        UnparseableCommand: on an unterminated quote or a trailing backslash.
    """
    tokens: list[Token] = []
    i, n = 0, len(command)
    at_line_start = True

    while i < n:
        ch = command[i]

        if ch == "\n":
            at_line_start = True
            i += 1
            continue
        if ch in " \t":
            i += 1
            continue
        if ch == "\\" and i + 1 < n and command[i + 1] == "\n":
            i += 2  # line continuation: not a token boundary, not a new line
            continue
        if ch == "#" and at_line_start:
            nl = command.find("\n", i)
            i = n if nl == -1 else nl
            continue

        at_line_start = False

        if ch in _OPERATOR_START:
            start = i
            while i < n and command[i] in _OPERATOR_START:
                i += 1
            tokens.append(Token(command[start:i], start, i, kind=TokenKind.OPERATOR))
            continue

        start = i
        text: list[str] = []
        quoted = False
        while i < n:
            c = command[i]
            if c in " \t\n" or c in _OPERATOR_START:
                break
            if c == "\\":
                if i + 1 >= n:
                    raise UnparseableCommand(
                        f"trailing backslash at offset {i} in shell command"
                    )
                if command[i + 1] == "\n":
                    i += 2
                    continue
                text.append(command[i + 1])
                i += 2
                continue
            if c == "'":
                quoted = True
                close = command.find("'", i + 1)
                if close == -1:
                    raise UnparseableCommand(
                        f"unterminated single quote opened at offset {i}"
                    )
                text.append(command[i + 1 : close])
                i = close + 1
                continue
            if c == '"':
                quoted = True
                i += 1
                while i < n and command[i] != '"':
                    if command[i] == "\\" and i + 1 < n:
                        if command[i + 1] == "\n":
                            i += 2
                            continue
                        text.append(command[i + 1])
                        i += 2
                        continue
                    text.append(command[i])
                    i += 1
                if i >= n:
                    raise UnparseableCommand(
                        "unterminated double quote in shell command"
                    )
                i += 1
                continue
            text.append(c)
            i += 1

        tokens.append(Token("".join(text), start, i, quoted=quoted))

    return tuple(tokens)


@dataclass(frozen=True)
class ArgV:
    """A worker's command line, readable and editable in either form.

    Construct with :meth:`from_container`; the form is decided once, there, and
    carried in the type so no caller re-derives it from ``command[-1]``.

    Instances are immutable: every edit returns a new :class:`ArgV`.
    """

    form: ArgForm
    command: tuple[str, ...]
    source: str
    _argv: tuple[str, ...] = ()
    _shell: str = ""
    _parse_error: str | None = None

    # ---------------------------------------------------------------- build

    @classmethod
    def from_container(cls, container: dict, source: str = "container") -> "ArgV":
        """Read a Kubernetes container spec into an :class:`ArgV`.

        Shell form requires all three of: a ``command`` list, a trailing
        short-option cluster ending in ``c``, and ``args`` holding exactly one
        string. Anything else is argv form — including a single-element ``args``
        under a non-shell command, which is one argument, not a script.
        """
        command = container.get("command")
        command_tuple = (
            tuple(str(c) for c in command) if isinstance(command, list) else ()
        )
        args = container.get("args", [])

        shell_invoked = len(command_tuple) >= 2 and is_shell_command_flag(
            command_tuple[-1]
        )
        single_string = (
            isinstance(args, list) and len(args) == 1 and isinstance(args[0], str)
        )

        if shell_invoked and single_string:
            return cls.shell(args[0], command=command_tuple, source=source)

        if isinstance(args, str):
            tokens = tuple(args.split())
        elif isinstance(args, list):
            tokens = tuple(str(a) for a in args)
        else:
            tokens = ()
        return cls.argv(tokens, command=command_tuple, source=source)

    @classmethod
    def argv(
        cls,
        tokens: Iterable[str],
        command: Sequence[str] = (),
        source: str = "argv",
    ) -> "ArgV":
        return cls(
            form=ArgForm.ARGV,
            command=tuple(command),
            source=source,
            _argv=tuple(tokens),
        )

    @classmethod
    def shell(
        cls,
        command_string: str,
        command: Sequence[str] = ("/bin/sh", "-c"),
        source: str = "shell",
    ) -> "ArgV":
        parse_error: str | None = None
        try:
            tokenize(command_string)
        except UnparseableCommand as exc:
            parse_error = str(exc)
        return cls(
            form=ArgForm.SHELL,
            command=tuple(command),
            source=source,
            _shell=command_string,
            _parse_error=parse_error,
        )

    # ----------------------------------------------------------------- read

    @property
    def is_parseable(self) -> bool:
        return self._parse_error is None

    @property
    def parse_error(self) -> str | None:
        return self._parse_error

    def tokens(self) -> tuple[Token, ...]:
        """Every token with its source span. Empty tuple if unparseable.

        Prefer :meth:`get` — it reports unparseable input as ``UNKNOWN`` rather
        than as nothing, which is the distinction that matters.
        """
        if self.form is ArgForm.ARGV:
            out, pos = [], 0
            for tok in self._argv:
                out.append(Token(tok, pos, pos + len(tok)))
                pos += len(tok) + 1
            return tuple(out)
        if self._parse_error is not None:
            return ()
        return tokenize(self._shell)

    def _words(self) -> tuple[Token, ...]:
        return tuple(t for t in self.tokens() if t.kind is TokenKind.WORD)

    def _matches(self, token: Token, flag: str) -> bool:
        return token.text == flag or token.text.startswith(f"{flag}=")

    def _value_of(
        self, words: Sequence[Token], index: int, flag: str
    ) -> tuple[str, Token | None] | None:
        """Value for the flag at ``words[index]``, plus the token holding it.

        Returns ``None`` when the flag is a bare switch. The value token is
        ``None`` for the ``--flag=value`` spelling, where the value lives inside
        the flag token itself.
        """
        token = words[index]
        if token.text.startswith(f"{flag}="):
            return token.text[len(flag) + 1 :], None
        nxt = words[index + 1] if index + 1 < len(words) else None
        if nxt is None or nxt.is_option:
            return None
        return nxt.text, nxt

    def has(self, flag: str) -> Fact[bool]:
        """Whether ``flag`` appears at all, in either spelling."""
        if self._parse_error is not None:
            return Fact.unknown(
                self.source,
                f"shell command could not be tokenised: {self._parse_error}",
            )
        words = self._words()
        present = any(self._matches(t, flag) for t in words)
        return Fact.known(
            present,
            self.source,
            f"scanned {len(words)} tokens in {self.form.value} form",
        )

    def get(self, flag: str) -> Fact[str]:
        """The value of ``flag``, understanding ``--f v`` and ``--f=v``.

        ``ABSENT`` covers two justified cases: the flag is not present, or it is
        present as a bare switch and so genuinely has no value. Use :meth:`has`
        to tell those apart. ``UNKNOWN`` means the command could not be read.
        """
        if self._parse_error is not None:
            return Fact.unknown(
                self.source,
                f"shell command could not be tokenised: {self._parse_error}",
            )
        words = self._words()
        for i, token in enumerate(words):
            if not self._matches(token, flag):
                continue
            found = self._value_of(words, i, flag)
            if found is None:
                return Fact.absent(
                    self.source,
                    f"'{flag}' is present as a switch and carries no value",
                )
            return Fact.known(found[0], self.source, f"{self.form.value} form")
        return Fact.absent(
            self.source,
            f"'{flag}' is not among the {len(words)} tokens of this "
            f"{self.form.value}-form command",
        )

    def get_all(self, flag: str) -> Fact[tuple[str, ...]]:
        """Every value given for ``flag``, in order.

        A repeated flag is usually a bug — and it is the bug a naive editor
        creates, by appending ``--max-model-len 1024`` to a command that already
        sets it. Being able to *see* the repetition is what makes that testable.
        """
        if self._parse_error is not None:
            return Fact.unknown(
                self.source,
                f"shell command could not be tokenised: {self._parse_error}",
            )
        words = self._words()
        values = [
            found[0]
            for i, token in enumerate(words)
            if self._matches(token, flag)
            and (found := self._value_of(words, i, flag)) is not None
        ]
        if not values:
            return Fact.absent(
                self.source, f"'{flag}' carries no value in this command"
            )
        return Fact.known(tuple(values), self.source, f"{len(values)} occurrence(s)")

    def model(self) -> Fact[str]:
        """The served model, however this engine spells the flag."""
        for flag in ("--model-path", "--model", "--served-model-name"):
            fact = self.get(flag)
            if fact.is_known or fact.is_unknown:
                return fact
        return Fact.absent(
            self.source, "none of --model-path/--model/--served-model-name is set"
        )

    # ---------------------------------------------------------------- write

    def set(self, flag: str, value: str | bool | None) -> "ArgV":
        """Set ``flag``, replacing any existing occurrence rather than appending.

        * ``str`` — ``--flag value``, or ``--flag=value`` if that is how the flag
          is already written.
        * ``True`` — a bare switch. Never ``--flag True``: engines that declare
          the flag ``store_true`` reject a value, and emitting one is a measured
          bug in the existing helpers.
        * ``False`` / ``None`` — remove the flag and its value.

        Raises:
            UnparseableCommand: the shell command could not be tokenised, so
                editing it would be guesswork.
            AmbiguousInsertion: the flag is new and the command has no
                unambiguous place to put it.
        """
        if value is False or value is None:
            return self.unset(flag)
        if self.form is ArgForm.ARGV:
            return self._set_argv(flag, value)
        return self._set_shell(flag, value)

    def unset(self, flag: str) -> "ArgV":
        """Remove ``flag`` and its value. A no-op if it is not present."""
        if self.form is ArgForm.ARGV:
            return self._unset_argv(flag)
        return self._unset_shell(flag)

    def _require_parseable(self) -> None:
        if self._parse_error is not None:
            raise UnparseableCommand(
                f"cannot edit {self.source}: {self._parse_error}. Editing an "
                "un-tokenisable command would corrupt it silently."
            )

    # -- argv form ------------------------------------------------------

    def _set_argv(self, flag: str, value: str | bool) -> "ArgV":
        out = list(self._argv)
        for i, tok in enumerate(out):
            if tok.startswith(f"{flag}="):
                out[i] = flag if value is True else f"{flag}={value}"
                return replace(self, _argv=tuple(out))
            if tok != flag:
                continue
            has_value = i + 1 < len(out) and not _looks_like_option(out[i + 1])
            if value is True:
                if has_value:
                    del out[i + 1]
            elif has_value:
                out[i + 1] = str(value)
            else:
                out.insert(i + 1, str(value))
            return replace(self, _argv=tuple(out))
        out.append(flag)
        if value is not True:
            out.append(str(value))
        return replace(self, _argv=tuple(out))

    def _unset_argv(self, flag: str) -> "ArgV":
        out = list(self._argv)
        i = 0
        while i < len(out):
            if out[i].startswith(f"{flag}="):
                del out[i]
                continue
            if out[i] == flag:
                drop = (
                    2 if i + 1 < len(out) and not _looks_like_option(out[i + 1]) else 1
                )
                del out[i : i + drop]
                continue
            i += 1
        return replace(self, _argv=tuple(out))

    # -- shell form: every edit is a splice into the original string -----

    def _splice(self, start: int, end: int, text: str) -> "ArgV":
        return replace(self, _shell=self._shell[:start] + text + self._shell[end:])

    def _set_shell(self, flag: str, value: str | bool) -> "ArgV":
        self._require_parseable()
        words = self._words()
        for i, token in enumerate(words):
            if not self._matches(token, flag):
                continue
            if token.text.startswith(f"{flag}="):
                text = flag if value is True else f"{flag}={_quote(str(value))}"
                return self._splice(token.start, token.end, text)
            found = self._value_of(words, i, flag)
            if value is True:
                # Collapse to a bare switch, dropping any value it had.
                end = found[1].end if found and found[1] else token.end
                return self._splice(token.start, end, flag)
            if found and found[1] is not None:
                return self._splice(found[1].start, found[1].end, _quote(str(value)))
            # Present as a switch: insert the value directly after it.
            return self._splice(token.end, token.end, " " + _quote(str(value)))

        return self._insert_shell(flag, value)

    def _insert_shell(self, flag: str, value: str | bool) -> "ArgV":
        """Insert a flag that is not yet present, at a defensible offset.

        Placed immediately after the last existing option (and its value), so it
        joins the flag list of the same program. Falling back to "end of string"
        is only safe when the command has no trailing operator; a command ending
        in ``&& something-else`` would otherwise silently hand the flag to the
        wrong program.
        """
        text = flag if value is True else f"{flag} {_quote(str(value))}"
        words = self._words()

        for i in range(len(words) - 1, -1, -1):
            if not words[i].is_option:
                continue
            end = words[i].end
            nxt = words[i + 1] if i + 1 < len(words) else None
            if nxt is not None and not nxt.is_option:
                end = nxt.end
            return self._splice(end, end, " " + text)

        all_tokens = self.tokens()
        if any(t.kind is TokenKind.OPERATOR for t in all_tokens):
            raise AmbiguousInsertion(
                f"{self.source}: cannot place '{flag}' — the command declares no "
                "flags to append beside and contains shell operators, so the end "
                "of the string may belong to a different program. Add the flag to "
                "the manifest instead."
            )
        if not all_tokens:
            return replace(self, _shell=text)
        end = all_tokens[-1].end
        return self._splice(end, end, " " + text)

    def _unset_shell(self, flag: str) -> "ArgV":
        self._require_parseable()
        words = self._words()
        for i, token in enumerate(words):
            if not self._matches(token, flag):
                continue
            end = token.end
            if not token.text.startswith(f"{flag}="):
                found = self._value_of(words, i, flag)
                if found and found[1] is not None:
                    end = found[1].end
            start = token.start
            # Absorb one leading separator so removal does not leave a double
            # space or a dangling line continuation.
            while start > 0 and self._shell[start - 1] in " \t":
                start -= 1
            if start >= 2 and self._shell[start - 2 : start] == "\\\n":
                start -= 2
                while start > 0 and self._shell[start - 1] in " \t":
                    start -= 1
            return self._splice(start, end, "").unset(flag)
        return self

    # ---------------------------------------------------------------- emit

    def as_container_args(self) -> list[str]:
        """The value to write back into the container's ``args``.

        Shell form always yields exactly one element. That is not a convention:
        a shell binds everything after its command string to ``$0``/``$1``, so a
        second element is a flag the worker will never see.
        """
        if self.form is ArgForm.ARGV:
            return list(self._argv)
        return [self._shell]

    def as_shell_string(self) -> str:
        """The shell command string. Only meaningful in shell form."""
        if self.form is not ArgForm.SHELL:
            raise ArgVError(f"{self.source} is in {self.form.value} form, not shell")
        return self._shell

    def apply_to(self, container: dict) -> None:
        """Write these args back into a container spec, in place."""
        container["args"] = self.as_container_args()

    def __iter__(self) -> Iterator[str]:
        return iter(t.text for t in self._words())

    def __repr__(self) -> str:
        state = "unparseable" if self._parse_error else f"{len(self._words())} tokens"
        return f"ArgV({self.form.value}, {self.source!r}, {state})"


def _looks_like_option(token: str) -> bool:
    return len(token) > 1 and token.startswith("-")
