# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

from tests.serve.common import _cleanup_prepared_deployment, _prepare_deployment
from tests.utils.constants import DynamoPortRange
from tests.utils.engine_process import EngineConfig
from tests.utils.port_utils import ServicePorts, reserved_ports

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


class _RequestNode:
    def get_closest_marker(self, _name: str) -> None:
        return None


class _Request:
    node = _RequestNode()


def _config(directory: str) -> EngineConfig:
    return EngineConfig(
        name="port-contract",
        directory=directory,
        script_name="agg_multimodal_router.sh",
        model="test-model",
        marks=[],
        request_payloads=[],
    )


def _prepare(ports: ServicePorts, directory: str):
    """Build and clean up a deployment while returning its environment."""
    prepared = _prepare_deployment(
        _config(directory), _Request(), ports=ports, extra_env=None
    )
    try:
        return dict(prepared.merged_env)
    finally:
        _cleanup_prepared_deployment(prepared)


def test_prepared_environment_exports_complete_worker_port_vectors(
    tmp_path: Path,
) -> None:
    """Export one managed system, KV-event, and NIXL port per worker."""
    with reserved_ports(10, DynamoPortRange.SERVE.value) as allocated:
        frontend = allocated[0]
        system_ports = allocated[1:4]
        kv_event_ports = allocated[4:7]
        nixl_ports = allocated[7:10]
        ports = ServicePorts(
            frontend_port=frontend,
            system_ports=system_ports,
            kv_event_ports=kv_event_ports,
            nixl_side_channel_ports=nixl_ports,
        )
        env = _prepare(ports, str(tmp_path))

    assert env["DYN_MANAGED_PORTS"] == "1"
    assert env["DYN_SYSTEM_PORT"] == str(system_ports[0])
    assert "DYN_VLLM_KV_EVENT_PORT" not in env
    assert [env[f"DYN_SYSTEM_PORT{i}"] for i in range(1, 4)] == [
        str(port) for port in system_ports
    ]
    assert [env[f"DYN_VLLM_KV_EVENT_PORT{i}"] for i in range(1, 4)] == [
        str(port) for port in kv_event_ports
    ]
    assert [env[f"DYN_VLLM_NIXL_SIDE_CHANNEL_PORT{i}"] for i in range(1, 4)] == [
        str(port) for port in nixl_ports
    ]


def test_dyn_port_accepts_managed_ephemeral_system_port() -> None:
    """Allow zero for a managed system port so the runtime can bind ephemerally."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_MANAGED_PORTS": "1", "DYN_SYSTEM_PORT1": "0"},
        )

    assert result.returncode == 0
    assert result.stdout.strip() == "0"


def test_dyn_port_requires_indexed_values_in_managed_mode() -> None:
    """Reject a managed launch when a required indexed port is absent."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(2, DynamoPortRange.SERVE.value) as allocated:
        command = (
            f"source {launch_utils}; "
            "DYN_MANAGED_PORTS=1; "
            f"dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}; "
            f"dyn_port DYN_SYSTEM_PORT 2 {allocated[1]}"
        )
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_SYSTEM_PORT1": str(allocated[0])},
        )

    assert result.returncode != 0
    assert result.stdout.splitlines() == [str(allocated[0])]
    assert "DYN_SYSTEM_PORT2" in result.stderr


def test_dyn_port_keeps_standalone_fallback() -> None:
    """Use the supplied fallback when a standalone launch has no override."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={},
        )

    assert result.returncode == 0
    assert result.stdout.strip() == str(allocated[0])


def test_dyn_port_keeps_explicit_standalone_system_port() -> None:
    """Preserve an explicit system-port override for standalone launches."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_SYSTEM_PORT1": str(allocated[0])},
        )

    assert result.returncode == 0
    assert result.stdout.strip() == str(allocated[0])


def test_dyn_port_rejects_invalid_explicit_value() -> None:
    """Reject malformed standalone port overrides."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_SYSTEM_PORT1": "not-a-port"},
        )

    assert result.returncode != 0
    assert "DYN_SYSTEM_PORT1" in result.stderr


def test_dyn_port_rejects_invalid_managed_value() -> None:
    """Reject malformed managed port overrides."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_MANAGED_PORTS": "1", "DYN_SYSTEM_PORT1": "not-a-port"},
        )

    assert result.returncode != 0
    assert "DYN_SYSTEM_PORT1" in result.stderr


def test_dyn_port_rejects_out_of_range_value() -> None:
    """Reject numeric port overrides outside the runtime range."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_SYSTEM_PORT1": "65536"},
        )

    assert result.returncode != 0
    assert "DYN_SYSTEM_PORT1" in result.stderr


def test_dyn_port_rejects_out_of_range_fallback() -> None:
    """Reject a standalone fallback outside the runtime port range."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 65536",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={},
    )

    assert result.returncode != 0
    assert "DYN_SYSTEM_PORT1" in result.stderr


def test_dyn_port_accepts_high_non_system_port() -> None:
    """Allow valid TCP ports above the system-status server's signed range."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {launch_utils}; dyn_port DYN_VLLM_KV_EVENT_PORT 1 20000",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"DYN_VLLM_KV_EVENT_PORT1": "40000"},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "40000"


# ``wait_any_exit`` (examples/common/launch_utils.sh) waits on background jobs
# only; a foreground service hides other failures and delays the TERM/INT trap.

_EXAMPLES_DIR = Path(__file__).parents[2] / "examples"
_HEREDOC = re.compile(r"<<(-?)[ \t]*")
# `python -m dynamo.vllm`, `python3 -m dynamo.frontend`, `python -m "$WORKER_MODULE"`.
_SERVICE = re.compile(
    r"(?<![\w./-])python3?\s+(?:-\S+\s+)*-m\s+(?:dynamo\.\S*|\$\{?\w+\}?)"
)
# What bash allows before a segment's command word: variable assignments, and an
# expansion, which is how the sglang scripts pass a GPU pin.
_ASSIGNMENT = re.compile(r"\w+=\S*|\$\{?\S*")
# A prefix command runs the rest of the segment, so the service is still launched.
# `env` carries its own flags and their arguments, as in `env -u DYN_SYSTEM_PORT`.
_PREFIX_COMMAND = re.compile(
    r"env|exec|nohup|setsid|stdbuf|time|sudo|timeout|srun|mpirun|numactl|taskset|uv"
)
# These reserved words introduce commands in a compound statement or pipeline.
_COMMAND_RESERVED_WORD = re.compile(r"if|then|elif|else|while|until|do|!")
_FUNCTION_HEADER = re.compile(
    r"(?:function\s+([A-Za-z_][\w-]*)(?:\s*\(\s*\))?|"
    r"([A-Za-z_][\w-]*)\s*\(\s*\))\s*"
)


class _Command(NamedTuple):
    line: int
    text: str  # quotes dropped, continuations joined, pipelines kept together
    quoted: str  # per-character mask of text: "q" came from inside quotes
    terminator: str  # operator that ended the command; "&" backgrounds it
    segments: tuple[int, ...]  # offsets in text where each pipeline segment starts
    body: tuple["_Command", ...] | None = None  # function definitions are not calls


def _read_quoted(script: str, index: int, *, ansi_c: bool = False) -> tuple[int, str]:
    """Return the index past a quoted span and the text inside it."""
    if script[index] == "'":
        end = index + 1
        while end < len(script) and script[end] != "'":
            if ansi_c and script[end] == "\\":
                end += 2  # escaped apostrophes do not close an ANSI-C string
            else:
                end += 1
        end = min(end, len(script))
        return end + 1, script[index + 1 : end]
    chunk: list[str] = []
    cursor = index + 1
    while cursor < len(script) and script[cursor] != '"':
        if script[cursor] == "\\" and cursor + 1 < len(script):
            following = script[cursor + 1]
            if following in '$`"\\\n':
                if following != "\n":
                    chunk.append(following)
                cursor += 2
                continue
        chunk.append(script[cursor])
        cursor += 1
    return cursor + 1, "".join(chunk)


def _read_heredoc_word(script: str, index: int) -> tuple[int, str]:
    """Read a delimiter with shell quote removal, without expanding its contents."""
    chunks: list[str] = []
    while index < len(script) and script[index] not in " \t\n;&|()<>":
        char = script[index]
        if char in "'\"":
            index, chunk = _read_quoted(script, index)
            chunks.append(chunk)
        elif char == "\\" and index + 1 < len(script):
            if script[index + 1] != "\n":
                chunks.append(script[index + 1])
            index += 2
        else:
            chunks.append(char)
            index += 1
    return index, "".join(chunks)


def _split_commands(script: str) -> list[_Command]:
    """Split a bash script on the separators bash itself honours."""
    commands: list[_Command] = []
    parts: list[str] = []
    mask: list[str] = []
    heredocs: list[tuple[str, bool]] = []  # (delimiter, "<<-" drops leading tabs)
    segments: list[int] = [0]
    word_started = False  # quotes, including empty ones, start a shell word
    prev_code = ""  # last non-blank character added, for redirection detection
    pending = False  # an operator still needs its next command
    line = 1
    start = 1
    group = 0  # index in commands of the first member of the open AND-OR list
    substitutions = 0  # command/process substitution parentheses are not group closers
    # closer, first member, outer AND-OR list, function name, definition line
    scopes: list[tuple[str, int, int, str, int]] = []
    scoped: set[int] = set()  # preserve inner terminators unless the group gets &
    index = 0
    size = len(script)

    def add(chunk: str, quoted: bool) -> None:
        nonlocal word_started, prev_code, pending
        if quoted:
            word_started = True
            pending = False
        elif chunk:
            word_started = chunk[-1] not in " \t\n"
        if not chunk:
            return
        parts.append(chunk)
        mask.append(("q" if quoted else ".") * len(chunk))
        trimmed = chunk.rstrip()
        if trimmed:
            prev_code = trimmed[-1]
            pending = False

    def flush(terminator: str) -> None:
        nonlocal word_started, prev_code, start, group
        text = "".join(parts)
        lead = len(text) - len(text.lstrip())
        stripped = text.strip()
        if stripped:
            commands.append(
                _Command(
                    start,
                    stripped,
                    "".join(mask)[lead : lead + len(stripped)],
                    terminator,
                    tuple(
                        sorted(
                            {
                                min(max(offset - lead, 0), len(stripped))
                                for offset in segments
                            }
                        )
                    ),
                )
            )
        segments[:] = [0]
        if terminator not in ("&&", "||"):
            # Bash runs a whole AND-OR list in the background, so the terminator
            # that closes the list applies to every member, not just the last.
            for position in range(group, len(commands)):
                if position in scoped and terminator != "&":
                    continue
                commands[position] = commands[position]._replace(terminator=terminator)
            group = len(commands)
        parts.clear()
        mask.clear()
        word_started = False
        prev_code = ""
        start = line

    while index < size:
        char = script[index]
        if char == "\\":
            if script.startswith("\\\n", index):
                line += 1
                index += 2
                continue
            add(script[index + 1 : index + 2], True)
            index += 2
            continue
        ansi_c = script.startswith("$'", index)
        if char in "'\"" or ansi_c:
            end, chunk = _read_quoted(script, index + int(ansi_c), ansi_c=ansi_c)
            line += script[index:end].count("\n")
            add(chunk, True)
            index = end
            continue
        if char == "#" and not word_started:
            end = script.find("\n", index)
            index = size if end < 0 else end
            continue
        if script.startswith("<<<", index):
            add("<<<", False)
            index += 3
            continue
        if script.startswith("<<", index):
            match = _HEREDOC.match(script, index)
            if match is not None:
                end, delimiter = _read_heredoc_word(script, match.end())
                if end > match.end():
                    heredocs.append((delimiter, match.group(1) == "-"))
                    line += script[index:end].count("\n")
                    index = end
                    continue
        if script.startswith(("$(", "<(", ">("), index):
            add(script[index : index + 2], False)
            substitutions += 1
            index += 2
            continue
        if substitutions and char in "()":
            add(char, False)
            substitutions += 1 if char == "(" else -1
            index += 1
            continue
        header = "".join(parts).strip()
        function = (
            _FUNCTION_HEADER.fullmatch(header)
            if char in "{\n" and "q" not in "".join(mask)
            else None
        )
        if function and char == "\n":
            add(" ", False)  # the opening brace may follow the header on a new line
            line += 1
            index += 1
            continue
        brace_opener = char == "{" and script[index + 1 : index + 2].isspace()
        opens_scope = (char == "(" or brace_opener) and (function or not header)
        if not substitutions and opens_scope:
            name = (function.group(1) or function.group(2)) if function else ""
            scopes.append(
                (")" if char == "(" else "}", len(commands), group, name, start)
            )
            group = len(commands)
            parts.clear()
            mask.clear()
            segments[:] = [0]
            word_started = False
            pending = False
            start = line
            index += 1
            continue
        closes_scope = not substitutions and scopes and char == scopes[-1][0]
        if closes_scope and (char == ")" or not "".join(parts).strip()):
            flush("\n")
            _, first, group, name, definition_line = scopes.pop()
            if name:
                body = tuple(commands[first:])
                del commands[first:]
                scoped.intersection_update(range(first))
                commands.append(
                    _Command(definition_line, name, "." * len(name), "\n", (0,), body)
                )
            else:
                scoped.update(range(first, len(commands)))
            prev_code = char
            pending = False
            index += 1
            continue
        if char == "\n":
            continued = pending
            if not continued:
                flush("\n")
            line += 1
            index += 1
            while heredocs:
                delimiter, drop_tabs = heredocs.pop(0)
                while index < size:
                    end = script.find("\n", index)
                    end = size if end < 0 else end
                    body = script[index:end]
                    # Bash ends the body on an exact delimiter line, and `<<-`
                    # removes leading tabs only. A looser test ends the heredoc
                    # early and reads the rest of the body as commands.
                    done = (body.lstrip("\t") if drop_tabs else body) == delimiter
                    index = end + 1
                    line += 1
                    if done:
                        break
            if not continued or not "".join(parts).strip():
                start = line
            continue
        if char == ";":
            flush(";")
            index += 1
            continue
        if script.startswith("&&", index) or script.startswith("||", index):
            flush(script[index : index + 2])
            pending = True
            index += 2
            continue
        if char == "&":
            if prev_code in "<>" or script.startswith("&>", index):
                add(char, False)
                index += 1
                continue
            flush("&")
            index += 1
            continue
        if char == "|":
            add(" ", False)  # a pipeline is backgrounded as a whole
            segments.append(sum(map(len, parts)))
            pending = True
            index += 2 if script.startswith("|&", index) else 1
            continue
        add(char, False)
        index += 1
    flush("\n")
    return commands


def _calls_wait_any_exit(script: str) -> bool:
    """Report whether the script runs ``wait_any_exit``, whatever follows the call."""
    return any(
        command.text.split()[:1] == ["wait_any_exit"] and command.quoted[0] == "."
        for command in _executed_commands(script)
    )


def _in_command_position(command: _Command, start: int) -> bool:
    """Report whether a match is the command word of its pipeline segment."""
    segment = max(offset for offset in command.segments if offset <= start)
    for match in re.finditer(r"\S+", command.text[segment:start]):
        word = match.group()
        word_mask = command.quoted[segment + match.start() : segment + match.end()]
        if _COMMAND_RESERVED_WORD.fullmatch(word) and "q" not in word_mask:
            continue
        if _ASSIGNMENT.fullmatch(word):
            continue
        # A prefix command consumes the rest of the segment as its own command
        # line, so the service still runs. Anything else, `echo` or `grep`, takes
        # the match as an argument and launches nothing.
        return bool(_PREFIX_COMMAND.fullmatch(word))
    return True


def _executed_commands(script: str) -> list[_Command]:
    """Expand function calls, retaining body launches and invocation backgrounding."""
    functions: dict[str, tuple[_Command, ...]] = {}
    executed: list[_Command] = []

    def visit(commands: tuple[_Command, ...], active: frozenset[str]) -> None:
        for command in commands:
            if command.body is not None:
                functions[command.text] = command.body
                continue
            executed.append(command)
            for match in re.finditer(r"\S+", command.text):
                name = match.group()
                if name not in functions or name in active:
                    continue
                if "q" in command.quoted[match.start() : match.end()]:
                    continue
                if not _in_command_position(command, match.start()):
                    continue
                body = functions[name]
                if command.terminator == "&":
                    body = tuple(part._replace(terminator="&") for part in body)
                visit(body, active | {name})

    visit(tuple(_split_commands(script)), frozenset())
    return executed


def _service_launches(script: str) -> list[tuple[int, str, bool]]:
    """Return (line, command, is_background) for each Dynamo service launched."""
    launches = []
    for command in _executed_commands(script):
        # Every match, not just the first: a pipeline holds several commands, and
        # a quoted match early in one would otherwise hide a real launch after it.
        for match in _SERVICE.finditer(command.text):
            if command.quoted[match.start()] == "q":
                continue  # a quoted match builds a command string, it does not run one
            if not _in_command_position(command, match.start()):
                continue  # named as an argument, as in `echo python -m dynamo.frontend`
            launches.append((command.line, match.group(0), command.terminator == "&"))
    return launches


_FOREGROUND_SAMPLE = """\
#!/bin/bash
source launch_utils.sh
DYN_LOG=debug python -m dynamo.frontend 2>&1 | tee frontend.log &
CUDA_VISIBLE_DEVICES=0 \\
    python -m "$WORKER_MODULE" --model "$MODEL" \\
    --config '{"speculative": true}'
wait_any_exit
"""


def test_foreground_service_detector_reads_bash_separators() -> None:
    """Flag a service that is not a background job, whatever its command shape."""
    backgrounded = _FOREGROUND_SAMPLE.replace(
        "--config '{\"speculative\": true}'",
        "--config '{\"speculative\": true}' &",
    )

    assert _service_launches(_FOREGROUND_SAMPLE) == [
        (3, "python -m dynamo.frontend", True),
        (4, "python -m $WORKER_MODULE", False),
    ]
    assert [
        line
        for line, _, is_background in _service_launches(backgrounded)
        if not is_background
    ] == []


def test_line_continuation_preserves_shell_words() -> None:
    """Backslash-newline joins words without inserting a separator."""
    script = "py\\\nthon -\\\nm dynamo.frontend\nwait_any_exit\n"
    assert _service_launches(script) == [(1, "python -m dynamo.frontend", False)]


@pytest.mark.parametrize("prefix", ['""', '"prefix "'])
def test_hash_after_quoted_word_is_not_a_comment(prefix: str) -> None:
    """Only unquoted whitespace separates a quoted word from a comment."""
    script = f"echo {prefix}#literal; python -m dynamo.frontend\nwait_any_exit\n"
    assert _service_launches(script) == [(1, "python -m dynamo.frontend", False)]
    comment = f"echo {prefix} # comment; python -m dynamo.frontend\nwait_any_exit\n"
    assert _service_launches(comment) == []


_AND_OR_SAMPLE = """\
#!/bin/bash
python -m dynamo.frontend && echo ready &
python -m dynamo.vllm --model "$MODEL" || exit 1
wait_any_exit
"""

_HEREDOC_SAMPLE = """\
#!/bin/bash
cat > config.yaml <<EOF
model: $MODEL
  EOF
python -m dynamo.vllm --model "$MODEL"
EOF
cat > notes.txt <<-END
\tpython -m dynamo.planner
\tEND
python -m dynamo.frontend &
wait_any_exit
"""


_MASKED_SAMPLE = """\
#!/bin/bash
echo "python -m dynamo.fake" | python -m dynamo.frontend &
CMD="python -m dynamo.vllm"
eval "$CMD" &
python -m dynamo.planner
wait_any_exit # watch the children
"""


_ARGUMENT_SAMPLE = """\
#!/bin/bash
echo python -m dynamo.frontend
echo "starting" | grep python -m dynamo.vllm
env ${GPU_PIN:+"$GPU_PIN"} python3 -m dynamo.sglang &
CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm &
timeout 300 python -m dynamo.vllm
wait_any_exit
"""


def test_only_a_segment_command_word_counts_as_a_launch() -> None:
    """A `python` named as an argument is not a service the script runs."""
    assert _service_launches(_ARGUMENT_SAMPLE) == [
        (4, "python3 -m dynamo.sglang", True),
        (5, "python -m dynamo.vllm", True),
        (6, "python -m dynamo.vllm", False),
    ]


def test_compound_command_keywords_expose_service_launches() -> None:
    """Reserved words introduce commands; quoted words remain command names."""
    script = (
        "if condition; then python -m dynamo.vllm; "
        "else python -m dynamo.frontend; fi\n"
        "'then' python -m dynamo.fake\n"
        "wait_any_exit\n"
    )
    assert _service_launches(script) == [
        (1, "python -m dynamo.vllm", False),
        (1, "python -m dynamo.frontend", False),
    ]


def test_function_launches_inherit_each_invocation_background_status() -> None:
    """A definition launches nothing; each call applies its own background status."""
    script = (
        "run_worker() {\n"
        "    python -m dynamo.vllm\n"
        "}\n"
        "unused() {\n"
        "    python -m dynamo.fake\n"
        "}\n"
        "run_worker &\n"
        "run_worker\n"
        "wait_any_exit\n"
    )
    assert _service_launches(script) == [
        (2, "python -m dynamo.vllm", True),
        (2, "python -m dynamo.vllm", False),
    ]


def test_nested_function_calls_keep_inner_jobs_and_wait_detection() -> None:
    """Expand nested calls without losing inner jobs or recursing indefinitely."""
    script = (
        "function worker\n{\n"
        "    python -m dynamo.vllm &\n"
        "    worker\n"
        "}\n"
        "launch() {\n"
        "    worker\n"
        "    wait_any_exit\n"
        "}\n"
        "launch\n"
    )
    assert _service_launches(script) == [(3, "python -m dynamo.vllm", True)]
    assert _calls_wait_any_exit(script)
    assert not _calls_wait_any_exit("unused() { wait_any_exit; }\n")


def test_quoted_match_does_not_hide_a_later_launch() -> None:
    """A quoted match masks only itself, not an executable one in the same command."""
    assert _service_launches(_MASKED_SAMPLE) == [
        (2, "python -m dynamo.frontend", True),
        (5, "python -m dynamo.planner", False),
    ]


@pytest.mark.parametrize(
    "quoted",
    [
        r"$'can\'t # literal'",
        r"$'backslash \\'",
        r"$'python -m dynamo.fake; \' # still quoted'",
        r"'ordinary backslash \'",
        r"\$'ordinary backslash \'",
    ],
)
def test_escaped_ansi_c_quote_does_not_hide_a_later_launch(quoted: str) -> None:
    """Only ANSI-C quoting allows a backslash to escape a closing apostrophe."""
    script = f"echo {quoted}; python -m dynamo.frontend\nwait_any_exit\n"
    assert _service_launches(script) == [(1, "python -m dynamo.frontend", False)]


def test_wait_any_exit_is_found_whatever_follows_the_call() -> None:
    """Keep a script in scope when its `wait_any_exit` carries trailing syntax."""
    assert _calls_wait_any_exit(_MASKED_SAMPLE)
    assert _calls_wait_any_exit("wait_any_exit || true\n")
    assert not _calls_wait_any_exit('echo "wait_any_exit"\n')
    assert not _calls_wait_any_exit("# wait_any_exit\n")


def test_and_or_list_inherits_its_trailing_ampersand() -> None:
    """Bash backgrounds a whole `&&`/`||` list, so every member of it is one job."""
    assert _service_launches(_AND_OR_SAMPLE) == [
        (2, "python -m dynamo.frontend", True),
        (3, "python -m dynamo.vllm", False),
    ]


def test_heredoc_body_ends_only_at_the_bash_delimiter() -> None:
    """An indented `EOF` is body text; only `<<-END` ignores the leading tabs."""
    assert _service_launches(_HEREDOC_SAMPLE) == [
        (10, "python -m dynamo.frontend", True),
    ]


@pytest.mark.parametrize(
    "word,delimiter",
    [
        ("'END-SCRIPT'", "END-SCRIPT"),
        ("END-SCRIPT", "END-SCRIPT"),
        (r"\EOF", "EOF"),
        ('"END SCRIPT"', "END SCRIPT"),
        ("END'-SCRIPT'", "END-SCRIPT"),
        (r'"\EOF"', r"\EOF"),
        (r'"\$EOF"', "$EOF"),
        ("''", ""),
    ],
)
def test_heredoc_delimiter_is_a_shell_word(word: str, delimiter: str) -> None:
    """Punctuation and quote removal must not expose heredoc text as commands."""
    script = (
        f"cat <<{word}\n"
        "python -m dynamo.fake\n"
        f"{delimiter}\n"
        "python -m dynamo.frontend &\n"
        "wait_any_exit\n"
    )
    assert _service_launches(script) == [(4, "python -m dynamo.frontend", True)]


def test_here_string_does_not_start_a_heredoc() -> None:
    """Consuming <<< must not reinterpret its final two characters as <<."""
    script = 'cat <<<"configuration"\npython -m dynamo.frontend &\nwait_any_exit\n'
    assert _service_launches(script) == [(2, "python -m dynamo.frontend", True)]


@pytest.mark.parametrize("operator", ["&&", "||", "|"])
@pytest.mark.parametrize("gap", ["\n", "  \n\n    # keep waiting\n"])
@pytest.mark.parametrize("terminator", ["&", ";"])
def test_operator_continues_across_newlines(
    operator: str, gap: str, terminator: str
) -> None:
    """Continuation lines preserve the whole list's background status."""
    script = (
        f"python -m dynamo.frontend {operator}{gap}    echo ready {terminator}\n"
        "python -m dynamo.vllm\n"
        "wait_any_exit\n"
    )
    assert _service_launches(script) == [
        (1, "python -m dynamo.frontend", terminator == "&"),
        (gap.count("\n") + 2, "python -m dynamo.vllm", False),
    ]


def test_continued_pipeline_still_skips_heredoc_body() -> None:
    """A continuation newline also starts any pending heredoc body."""
    script = (
        "cat <<EOF |\n"
        "python -m dynamo.fake\n"
        "EOF\n"
        "    python -m dynamo.frontend &\n"
        "wait_any_exit\n"
    )
    assert _service_launches(script) == [(1, "python -m dynamo.frontend", True)]


def test_stderr_pipeline_requires_a_separate_background_marker() -> None:
    """The ampersand in |& redirects stderr; it does not background the pipeline."""
    script = "python -m dynamo.frontend |& tee frontend.log"
    assert _service_launches(script) == [(1, "python -m dynamo.frontend", False)]


@pytest.mark.parametrize("grouped", ["{ %s; }", "(%s)", "( { %s; } )"])
@pytest.mark.parametrize("terminator", ["&", ";"])
def test_grouped_service_launches(grouped: str, terminator: str) -> None:
    """Groups expose their launches and apply an outer ampersand to every member."""
    script = grouped % "python -m dynamo.frontend & python -m dynamo.vllm"
    script += f" {terminator}\npython -m dynamo.planner\nwait_any_exit\n"
    assert _service_launches(script) == [
        (1, "python -m dynamo.frontend", True),
        (1, "python -m dynamo.vllm", terminator == "&"),
        (2, "python -m dynamo.planner", False),
    ]


def test_substitutions_do_not_close_enclosing_group() -> None:
    """Substitution parentheses, including nested subshells, preserve the outer job."""
    script = (
        "(\n"
        '    MODEL=$( (basename "$MODEL_PATH") )\n'
        "    cat <(echo input) > >(cat)\n"
        "    python -m dynamo.vllm\n"
        ") &\n"
        "wait_any_exit\n"
    )
    assert _service_launches(script) == [(4, "python -m dynamo.vllm", True)]


def test_grouped_and_or_list_inherits_background_status() -> None:
    """An outer AND-OR list backgrounds both a group and its sibling command."""
    script = "{ python -m dynamo.frontend; } &&\npython -m dynamo.vllm &\n"
    assert _service_launches(script) == [
        (1, "python -m dynamo.frontend", True),
        (2, "python -m dynamo.vllm", True),
    ]


def test_launch_scripts_background_the_services_wait_any_exit_watches() -> None:
    """Run every Dynamo service as a background job in wait_any_exit scripts."""
    scripts = sorted(
        path
        for path in _EXAMPLES_DIR.rglob("*.sh")
        if _calls_wait_any_exit(path.read_text())
    )
    launched = 0
    foreground = []
    for path in scripts:
        for line, command, is_background in _service_launches(path.read_text()):
            launched += 1
            if not is_background:
                relative = path.relative_to(_EXAMPLES_DIR.parent)
                foreground.append(f"{relative}:{line}: {command}")

    assert scripts, "no launch script calls wait_any_exit"
    assert launched, "no service launch parsed; the scan lost its subject"
    assert foreground == [], (
        "wait_any_exit waits on background jobs, so a foreground service hides "
        "every other service's failure until it exits:\n" + "\n".join(foreground)
    )
