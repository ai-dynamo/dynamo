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
_WAIT_ANY_EXIT = re.compile(r"^[ \t]*wait_any_exit[ \t]*$", re.MULTILINE)
_HEREDOC = re.compile(r"<<(-?)[ \t]*([\"']?)(\w+)\2")
# `python -m dynamo.vllm`, `python3 -m dynamo.frontend`, `python -m "$WORKER_MODULE"`.
_SERVICE = re.compile(
    r"(?<![\w./-])python3?\s+(?:-\S+\s+)*-m\s+(?:dynamo\.\S*|\$\{?\w+\}?)"
)


class _Command(NamedTuple):
    line: int
    text: str  # quotes dropped, continuations joined, pipelines kept together
    quoted: str  # per-character mask of text: "q" came from inside quotes
    terminator: str  # operator that ended the command; "&" backgrounds it


def _read_quoted(script: str, index: int) -> tuple[int, str]:
    """Return the index past a quoted span and the text inside it."""
    if script[index] == "'":
        end = script.find("'", index + 1)
        end = len(script) if end < 0 else end
        return end + 1, script[index + 1 : end]
    chunk: list[str] = []
    cursor = index + 1
    while cursor < len(script) and script[cursor] != '"':
        if script[cursor] == "\\" and cursor + 1 < len(script):
            chunk.append(script[cursor + 1])
            cursor += 2
            continue
        chunk.append(script[cursor])
        cursor += 1
    return cursor + 1, "".join(chunk)


def _split_commands(script: str) -> list[_Command]:
    """Split a bash script on the separators bash itself honours."""
    commands: list[_Command] = []
    parts: list[str] = []
    mask: list[str] = []
    heredocs: list[tuple[str, bool]] = []  # (delimiter, "<<-" drops leading tabs)
    prev = ""  # last character added, for comment detection
    prev_code = ""  # last non-blank character added, for redirection detection
    line = 1
    start = 1
    group = 0  # index in commands of the first member of the open AND-OR list
    index = 0
    size = len(script)

    def add(chunk: str, quoted: bool) -> None:
        nonlocal prev, prev_code
        if not chunk:
            return
        parts.append(chunk)
        mask.append(("q" if quoted else ".") * len(chunk))
        prev = chunk[-1]
        trimmed = chunk.rstrip()
        if trimmed:
            prev_code = trimmed[-1]

    def flush(terminator: str) -> None:
        nonlocal prev, prev_code, start, group
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
                )
            )
        if terminator not in ("&&", "||"):
            # Bash runs a whole AND-OR list in the background, so the terminator
            # that closes the list applies to every member, not just the last.
            for position in range(group, len(commands)):
                commands[position] = commands[position]._replace(terminator=terminator)
            group = len(commands)
        parts.clear()
        mask.clear()
        prev = ""
        prev_code = ""
        start = line

    while index < size:
        char = script[index]
        if char == "\\":
            if script.startswith("\\\n", index):
                add(" ", False)
                line += 1
                index += 2
                continue
            add(script[index + 1 : index + 2], True)
            index += 2
            continue
        if char in "'\"":
            end, chunk = _read_quoted(script, index)
            line += script[index:end].count("\n")
            add(chunk, True)
            index = end
            continue
        if char == "#" and prev in ("", " ", "\t"):
            end = script.find("\n", index)
            index = size if end < 0 else end
            continue
        if script.startswith("<<", index) and not script.startswith("<<<", index):
            match = _HEREDOC.match(script, index)
            if match is not None:
                heredocs.append((match.group(3), match.group(1) == "-"))
                index = match.end()
                continue
        if char == "\n":
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
            start = line
            continue
        if char == ";":
            flush(";")
            index += 1
            continue
        if script.startswith("&&", index) or script.startswith("||", index):
            flush(script[index : index + 2])
            index += 2
            continue
        if char == "&":
            if prev_code in "<>" or script.startswith("&>", index):  # a redirection
                add(char, False)
                index += 1
                continue
            flush("&")
            index += 1
            continue
        if char == "|":
            add(" ", False)  # a pipeline is backgrounded as a whole
            index += 1
            continue
        add(char, False)
        index += 1
    flush("\n")
    return commands


def _service_launches(script: str) -> list[tuple[int, str, bool]]:
    """Return (line, command, is_background) for each Dynamo service launched."""
    launches = []
    for command in _split_commands(script):
        match = _SERVICE.search(command.text)
        if match is None or command.quoted[match.start()] == "q":
            continue  # a quoted match builds a command string, it does not run one
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


def test_launch_scripts_background_the_services_wait_any_exit_watches() -> None:
    """Run every Dynamo service as a background job in wait_any_exit scripts."""
    scripts = sorted(
        path
        for path in _EXAMPLES_DIR.rglob("*.sh")
        if _WAIT_ANY_EXIT.search(path.read_text())
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
