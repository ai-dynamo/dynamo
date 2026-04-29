# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend API-surface compliance suite against a live Dynamo frontend.

Subject under test is Dynamo's HTTP surface (`/v1/responses` and
`/v1/messages` wire shapes, tool-call routing through both); sglang is
just the backend vehicle for producing real traffic. Runs three suites
sequentially against one server:

1. Upstream OpenResponses compliance-test.ts harness (bun/TypeScript
   validator against zod schemas generated from the OpenAPI spec).
2. `codex exec` smoke — forces the shell tool-call path through
   `/v1/responses`.
3. `claude -p` smoke — forces the Bash tool-call path through
   `/v1/messages` (Anthropic Messages API).

All external tooling (bun, node, the OpenResponses suite, and the codex /
claude CLIs) is installed lazily at test time by session-scoped fixtures
into a session-shared cache directory. Versions and the OpenResponses
SHA are pinned as module-level constants. FileLock coordination makes
concurrent xdist workers share a single install.
"""

import logging
import os
import platform
import shlex
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

import pytest
import requests
from filelock import FileLock

from tests.serve.common import WORKSPACE_DIR
from tests.utils.engine_process import EngineConfig, EngineProcess

logger = logging.getLogger(__name__)

sglang_dir = os.environ.get("SGLANG_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/sglang"
)

COMPLIANCE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

# Pinned external-tool versions. Bun and node are pinned for reproducibility.
# The agent CLIs (@openai/codex, @anthropic-ai/claude-code) float to @latest
# so we automatically pick up protocol fixes — they're client-side harnesses,
# not Dynamo surface.
BUN_VERSION = "1.3.12"
NODE_VERSION = "20.19.0"
OPENRESPONSES_REPO = "https://github.com/openresponses/openresponses.git"
OPENRESPONSES_SHA = "fa29df5"

# Retry budget for network-touching installs. Exponential backoff starting
# at 2s; 3 attempts caps the worst-case wait at ~6s before we surface a
# clear "upstream unavailable" error.
_RETRY_COUNT = 3
_RETRY_BACKOFF_INITIAL_S = 2.0

# Env keys forwarded into codex/claude subprocesses. These agents run with tool
# permissions (`--dangerously-bypass-approvals-and-sandbox`, `--dangerously-skip-permissions`),
# and even against a local model they may emit telemetry; inheriting the whole
# CI environment would expose `GITHUB_TOKEN`, AWS creds, registry credentials,
# etc. Keep to a minimal allowlist covering only what the runtime needs:
# PATH to resolve the binaries, locale/TLS/proxy for HTTPS, HOME so Node/bun
# finds per-user caches, and NVIDIA/CUDA vars so any GPU-touching side effects
# see the same device the test was given.
_SUBPROCESS_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TZ",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
    }
)


def _agent_subprocess_env(
    extra_env: dict[str, str], path_prepend: list[Path] | None = None
) -> dict[str, str]:
    """Build a minimal env for codex/claude subprocesses: allowlist from
    `os.environ` merged with explicit test-scoped vars. Optional
    `path_prepend` prepends directories to PATH so the fixture-installed
    node/codex/claude binaries resolve without contaminating the
    inherited PATH."""
    base = {
        k: v for k in _SUBPROCESS_ENV_ALLOWLIST if (v := os.environ.get(k)) is not None
    }
    if path_prepend:
        existing = base.get("PATH", "")
        prefix = os.pathsep.join(str(p) for p in path_prepend)
        base["PATH"] = f"{prefix}{os.pathsep}{existing}" if existing else prefix
    base.update(extra_env)
    return base


# ---------------------------------------------------------------------------
# Tool-install fixtures
# ---------------------------------------------------------------------------


def _retry_network_op(fn, description: str):
    """Run `fn()` with a small exponential-backoff retry budget so that
    transient github/npm/nodejs.org blips don't flake the test.
    Captures subprocess stderr into the final error message so post-mortem
    doesn't require digging through logs."""
    last_err: BaseException | None = None
    for attempt in range(_RETRY_COUNT):
        try:
            return fn()
        except (OSError, requests.RequestException, subprocess.CalledProcessError) as e:
            last_err = e
            if attempt + 1 < _RETRY_COUNT:
                wait = _RETRY_BACKOFF_INITIAL_S * (2**attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    description,
                    attempt + 1,
                    _RETRY_COUNT,
                    e,
                    wait,
                )
                time.sleep(wait)
    detail = ""
    if isinstance(last_err, subprocess.CalledProcessError):
        detail = f"\nstdout:\n{last_err.stdout or ''}\nstderr:\n{last_err.stderr or ''}"
    raise RuntimeError(
        f"{description} failed after {_RETRY_COUNT} attempts: {last_err}{detail}"
    ) from last_err


def _download_url(url: str, dest: Path) -> None:
    """Stream GET `url` into `dest` atomically via a `.part` sibling."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.rename(dest)


def _bun_arch() -> str:
    m = platform.machine()
    if m == "x86_64":
        return "x64"
    if m == "aarch64":
        return "aarch64"
    raise RuntimeError(f"Unsupported machine architecture for bun: {m}")


def _node_arch() -> str:
    m = platform.machine()
    if m == "x86_64":
        return "x64"
    if m == "aarch64":
        return "arm64"
    raise RuntimeError(f"Unsupported machine architecture for node: {m}")


@pytest.fixture(scope="session")
def _tools_cache(tmp_path_factory) -> Path:
    """Session-shared cache directory for downloaded compliance tooling.
    Lives under the pytest basetemp so it's reused across xdist workers
    in the same session and cleaned up automatically when the session
    ends."""
    base = Path(tmp_path_factory.getbasetemp()) / "_frontend_api_surface_tools"
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(scope="session")
def _bun_binary(_tools_cache) -> Path:
    """Pinned-version bun executable. FileLock-coordinated so concurrent
    xdist workers share a single download."""
    install_dir = _tools_cache / f"bun-{BUN_VERSION}"
    bun_bin = install_dir / "bun"
    with FileLock(str(_tools_cache / "bun.lock")):
        if bun_bin.exists():
            return bun_bin
        install_dir.mkdir(parents=True, exist_ok=True)
        arch = _bun_arch()
        url = (
            f"https://github.com/oven-sh/bun/releases/download/"
            f"bun-v{BUN_VERSION}/bun-linux-{arch}.zip"
        )
        zip_path = install_dir / "bun.zip"
        _retry_network_op(
            lambda: _download_url(url, zip_path),
            description=f"download bun v{BUN_VERSION} ({arch})",
        )
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(install_dir)
        extracted = install_dir / f"bun-linux-{arch}" / "bun"
        shutil.copy(extracted, bun_bin)
        bun_bin.chmod(0o755)
        zip_path.unlink(missing_ok=True)
    return bun_bin


@pytest.fixture(scope="session")
def _node_bin(_tools_cache) -> Path:
    """Pinned-version node runtime root `bin/` directory containing
    `node` and `npm`. FileLock-coordinated."""
    install_dir = _tools_cache / f"node-v{NODE_VERSION}"
    bin_dir = install_dir / "bin"
    with FileLock(str(_tools_cache / "node.lock")):
        if (bin_dir / "node").exists() and (bin_dir / "npm").exists():
            return bin_dir
        install_dir.mkdir(parents=True, exist_ok=True)
        arch = _node_arch()
        tarball_name = f"node-v{NODE_VERSION}-linux-{arch}.tar.xz"
        url = f"https://nodejs.org/dist/v{NODE_VERSION}/{tarball_name}"
        tar_path = install_dir / tarball_name
        _retry_network_op(
            lambda: _download_url(url, tar_path),
            description=f"download node v{NODE_VERSION} ({arch})",
        )
        with tarfile.open(tar_path) as tf:
            # `filter="data"` is the safe extraction filter added in 3.12 and
            # required in 3.14; passing it explicitly silences the pytest
            # filterwarnings=error escalation of the DeprecationWarning.
            tf.extractall(install_dir, filter="data")
        extracted = install_dir / f"node-v{NODE_VERSION}-linux-{arch}"
        for item in extracted.iterdir():
            shutil.move(str(item), str(install_dir / item.name))
        extracted.rmdir()
        tar_path.unlink(missing_ok=True)
    return bin_dir


@pytest.fixture(scope="session")
def _openresponses_suite(_tools_cache, _bun_binary) -> Path:
    """Pinned-SHA clone of the OpenResponses compliance suite with bun
    deps installed. A `.installed` sentinel file marks a completed setup
    so an interrupted prior install forces a clean redo."""
    install_dir = _tools_cache / f"openresponses-{OPENRESPONSES_SHA}"
    sentinel = install_dir / ".installed"
    with FileLock(str(_tools_cache / "openresponses.lock")):
        if sentinel.exists():
            return install_dir
        if install_dir.exists():
            shutil.rmtree(install_dir)
        _retry_network_op(
            lambda: subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    OPENRESPONSES_REPO,
                    str(install_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            ),
            description="clone openresponses",
        )
        subprocess.run(
            ["git", "-C", str(install_dir), "checkout", OPENRESPONSES_SHA],
            check=True,
            capture_output=True,
            text=True,
        )
        _retry_network_op(
            lambda: subprocess.run(
                [str(_bun_binary), "install", "--frozen-lockfile"],
                cwd=str(install_dir),
                check=True,
                capture_output=True,
                text=True,
            ),
            description="bun install openresponses deps",
        )
        sentinel.touch()
    return install_dir


def _install_npm_cli(
    tools_cache: Path,
    node_bin: Path,
    package: str,
    binary_name: str,
    slot: str,
) -> Path:
    """Install `package` into `{tools_cache}/{slot}` via npm and return
    the path to the CLI entry point. Shared helper for codex + claude."""
    install_dir = tools_cache / slot
    cli_bin = install_dir / "node_modules" / ".bin" / binary_name
    with FileLock(str(tools_cache / f"{slot}.lock")):
        if cli_bin.exists():
            return cli_bin
        install_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "PATH": f"{node_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        }
        _retry_network_op(
            lambda: subprocess.run(
                [
                    str(node_bin / "npm"),
                    "install",
                    "--prefix",
                    str(install_dir),
                    package,
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            ),
            description=f"npm install {package}",
        )
    return cli_bin


@pytest.fixture(scope="session")
def _codex_cli(_tools_cache, _node_bin) -> Path:
    return _install_npm_cli(
        _tools_cache,
        _node_bin,
        package="@openai/codex",
        binary_name="codex",
        slot="codex",
    )


@pytest.fixture(scope="session")
def _claude_cli(_tools_cache, _node_bin) -> Path:
    return _install_npm_cli(
        _tools_cache,
        _node_bin,
        package="@anthropic-ai/claude-code",
        binary_name="claude",
        slot="claude",
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.sglang
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(COMPLIANCE_MODEL)
@pytest.mark.profiled_vram_gib(6.0)
@pytest.mark.requested_sglang_kv_tokens(512)
# Budget: tool-install fixtures (~30-60s first session run, near-zero on
# cache hit) + sglang cold start (30-60s) + bun compliance (up to 180s) +
# codex exec (up to 180s) + claude exec (up to 180s) + two inter-suite
# health checks + teardown. 750s leaves headroom for CI variance without
# masking real hangs.
@pytest.mark.timeout(750)
@pytest.mark.frontend_api_surface_compliance
@pytest.mark.post_merge
@pytest.mark.xfail(reason="DYN-2924 assigned to Ishan")
def test_frontend_api_surface_compliance(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
    tmp_path,
    _bun_binary,
    _node_bin,
    _openresponses_suite,
    _codex_cli,
    _claude_cli,
):
    """Assert the frontend passes the upstream OpenResponses compliance suite."""

    frontend_port = int(dynamo_dynamic_ports.frontend_port)
    system_port = int(dynamo_dynamic_ports.system_ports[0])

    config = EngineConfig(
        name="responses_compliance",
        directory=sglang_dir,
        marks=[],
        request_payloads=[],
        model=COMPLIANCE_MODEL,
        script_name="agg.sh",
        # Qwen3-VL-2B-specific flags: vision-model CUDA graph workaround +
        # model-aware reasoning/tool-call parsers. Forwarded verbatim to
        # `dynamo.sglang` by agg.sh's pass-through loop.
        #
        # Tool-call parser is `hermes`, not `qwen3_coder`: Qwen3-VL-Instruct
        # emits `<tool_call>{"name":..., "arguments":...}</tool_call>` (JSON
        # inside the tags — Hermes-style), while `qwen3_coder` expects the
        # XML-structured `<tool_call><function=name><parameter=k>v</parameter>
        # </function></tool_call>` that Qwen3-Coder models emit. Using the
        # wrong parser leaves tool calls as raw text in the response and
        # breaks end-to-end agent flows (codex exec, etc.).
        script_args=[
            "--model-path",
            COMPLIANCE_MODEL,
            "--disable-piecewise-cuda-graph",
            "--dyn-reasoning-parser",
            "qwen3",
            "--dyn-tool-call-parser",
            "hermes",
        ],
        timeout=360,
        env={},
        frontend_port=frontend_port,
    )

    merged_env = {
        "DYN_HTTP_PORT": str(frontend_port),
        "DYN_SYSTEM_PORT": str(system_port),
        # agg.sh doesn't forward frontend args, but the frontend reads this
        # env var directly. Enables /v1/messages for the claude smoke step.
        "DYN_ENABLE_ANTHROPIC_API": "1",
    }

    codex_home = tmp_path / "codex_home"
    _write_codex_config(codex_home, frontend_port)

    # Marker file that the agents can only "see" by invoking their shell/Bash
    # tool; if a model answers from its prior without actually running `ls`,
    # the marker won't appear in stdout and the assertion fails. Proves the
    # tool-call paths through the frontend end-to-end (both /v1/responses
    # for codex and /v1/messages for claude), not just text generation.
    agent_cwd = tmp_path / "agent_cwd"
    agent_cwd.mkdir()
    marker_filename = "dynamo_compliance_marker.txt"
    (agent_cwd / marker_filename).write_text("compliance-smoke")

    # Isolated HOME so claude doesn't write session state into the runner's
    # ~/.claude during CI / local invocation.
    claude_home = tmp_path / "claude_home"
    claude_home.mkdir()

    with EngineProcess.from_script(config, request, extra_env=merged_env):
        _run_bun_compliance(_bun_binary, _openresponses_suite, frontend_port)
        _wait_for_frontend_healthy(frontend_port)
        _run_codex_exec_smoke(
            _codex_cli, _node_bin, codex_home, agent_cwd, marker_filename
        )
        _wait_for_frontend_healthy(frontend_port)
        _run_claude_exec_smoke(
            _claude_cli,
            _node_bin,
            claude_home,
            agent_cwd,
            marker_filename,
            frontend_port,
        )


def _attach_subprocess_log(
    name: str,
    cmd: list[str],
    result: subprocess.CompletedProcess,
    extra_env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> None:
    """Attach a reproducible transcript of `cmd` to the Allure report.

    Lands in `test-results/allure-results/<uuid>-attachment.txt`, which the
    CI workflow uploads as an artifact on every run (pass or fail). Contents
    are a cut-and-paste-able shell invocation plus the raw stdout + stderr
    so a failing CI run can be reproduced locally from the artifact alone.

    Only explicitly listed env vars (`extra_env`) are recorded — not the
    inherited `os.environ` — to avoid leaking runner secrets into the
    artifact. CI runners keep HF tokens and cloud creds in env vars the
    subprocess inherits; we don't need those in the log to reproduce.
    """
    # Local import: `allure` is only available inside the test image (via
    # allure-pytest). Pre-commit's collection-only pytest runs in a clean
    # uvx env without it, so a module-level import would fail collection.
    import allure

    lines: list[str] = []
    if cwd:
        lines.append(f"$ cd {shlex.quote(cwd)}")
    if extra_env:
        for k, v in sorted(extra_env.items()):
            lines.append(f"$ export {k}={shlex.quote(v)}")
    lines.append("$ " + " ".join(shlex.quote(c) for c in cmd))
    lines.append("")
    lines.append(f"exit: {result.returncode}")
    lines.append("")
    lines.append("=== stdout ===")
    lines.append(result.stdout or "(empty)")
    lines.append("")
    lines.append("=== stderr ===")
    lines.append(result.stderr or "(empty)")

    allure.attach(
        "\n".join(lines),
        name=name,
        attachment_type=allure.attachment_type.TEXT,
    )


def _wait_for_frontend_healthy(
    frontend_port: int, timeout_s: float = 15.0, model: str = COMPLIANCE_MODEL
) -> None:
    """Confirm the frontend is still serving before the next subprocess fires.

    Without this check, if bun compliance accidentally destabilized the
    server (e.g. a hang that the bun timeout cut short) a codex exec
    failure looks identical to "codex is broken" in CI logs. The health
    probe collapses that ambiguity: if the frontend has crashed or the
    worker has deregistered, fail here with a clear message rather than
    letting codex run and time out.
    """
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(
                f"http://localhost:{frontend_port}/v1/models", timeout=2
            )
            if resp.ok and any(
                m.get("id") == model for m in resp.json().get("data", [])
            ):
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(0.5)
    pytest.fail(
        f"frontend unhealthy after bun compliance — /v1/models did not list "
        f"{model!r} within {timeout_s}s (last error: {last_err})"
    )


def _run_bun_compliance(
    bun_binary: Path, openresponses_dir: Path, frontend_port: int
) -> None:
    """Invoke compliance-test.ts against the running frontend."""
    base_url = f"http://localhost:{frontend_port}/v1"
    logger.info("Running OpenResponses compliance suite against %s", base_url)

    cmd = [
        str(bun_binary),
        "run",
        "bin/compliance-test.ts",
        "--base-url",
        base_url,
        "--api-key",
        "sk-compliance-dummy",
        "--model",
        COMPLIANCE_MODEL,
        "--verbose",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(openresponses_dir),
        capture_output=True,
        text=True,
        timeout=180,
    )

    _attach_subprocess_log(
        name="bun_compliance_suite.log",
        cmd=cmd,
        result=result,
        cwd=str(openresponses_dir),
    )
    if result.stdout:
        logger.info("compliance stdout:\n%s", result.stdout)
    if result.stderr:
        logger.info("compliance stderr:\n%s", result.stderr)

    if result.returncode != 0:
        pytest.fail(
            f"OpenResponses compliance suite failed (exit={result.returncode}).\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


def _write_codex_config(codex_home, frontend_port: int) -> None:
    """Emit a minimal ~/.codex/config.toml pointing Codex at Dynamo.

    Using a per-test CODEX_HOME keeps the runner's global Codex state
    (if any) untouched.
    """
    codex_home.mkdir(parents=True, exist_ok=True)
    config_path = codex_home / "config.toml"
    config_path.write_text(
        f"""
[model_providers.local]
name = "local-dynamo"
base_url = "http://localhost:{frontend_port}/v1"
wire_api = "responses"
env_key = "LOCAL_API_KEY"
""".lstrip()
    )


def _run_codex_exec_smoke(
    codex_cli: Path, node_bin: Path, codex_home, cwd, marker_filename: str
) -> None:
    """Run `codex exec` against the Dynamo Responses endpoint and assert the
    tool-call path actually fires.

    We prompt codex to list `cwd`; `cwd` contains `marker_filename` and nothing
    else the model could pattern-match from prior knowledge. If codex answers
    without invoking its shell tool, the marker won't appear in stdout and the
    assertion fails — which proves we're testing the full Responses API
    tool-calling chain, not just text generation.
    """
    logger.info("Running codex exec smoke test against CODEX_HOME=%s", codex_home)

    # Isolate HOME for codex the same way we do for claude below. CODEX_HOME
    # scopes codex's own state, but the agent still invokes a shell tool under
    # `--dangerously-bypass-approvals-and-sandbox`, which inherits HOME for
    # any shell/helper reads and writes. Point it at `codex_home` so nothing
    # escapes `tmp_path`.
    extra_env = {
        "CODEX_HOME": str(codex_home),
        "HOME": str(codex_home),
        "LOCAL_API_KEY": "sk-none",
    }
    # codex is a node script (`#!/usr/bin/env node`); prepend the fixture-
    # installed node runtime to PATH so the shebang resolves without pulling
    # in the runner's system node (if any).
    env = _agent_subprocess_env(extra_env, path_prepend=[node_bin])

    cmd = [
        str(codex_cli),
        "-m",
        COMPLIANCE_MODEL,
        "-c",
        "model_provider=local",
        "exec",
        "What files exist in the current working directory? Use your shell tool to run ls and report each filename verbatim from the output.",
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    _attach_subprocess_log(
        name="codex_exec_smoke.log",
        cmd=cmd,
        result=result,
        extra_env=extra_env,
        cwd=str(cwd),
    )
    if result.stdout:
        logger.info("codex stdout:\n%s", result.stdout)
    if result.stderr:
        logger.info("codex stderr:\n%s", result.stderr)

    if result.returncode != 0:
        pytest.fail(
            f"codex exec failed (exit={result.returncode}).\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    if marker_filename not in result.stdout:
        pytest.fail(
            "codex exec did not report the marker file — expected stdout to "
            f"contain {marker_filename!r} (implies the shell tool was invoked "
            f"and actually ran `ls` in {cwd}). Got:\n{result.stdout}"
        )


def _run_claude_exec_smoke(
    claude_cli: Path,
    node_bin: Path,
    claude_home,
    cwd,
    marker_filename: str,
    frontend_port: int,
) -> None:
    """Run `claude -p` against the Dynamo Anthropic Messages endpoint and
    assert the Bash tool-call path actually fires.

    Same marker-file pattern as the codex step but hitting /v1/messages:
    if claude answers without invoking its Bash tool, the marker won't
    appear in stdout and the assertion fails — which proves the full
    Anthropic Messages + tool-calling chain, not just text generation.

    Isolated HOME so claude doesn't write session state into the runner's
    `~/.claude`. An `ANTHROPIC_AUTH_TOKEN` is required even though Dynamo
    ignores the value: on a fresh HOME with no cached OAuth, the CLI
    aborts with "Not logged in" unless a bearer is supplied.
    """
    base_url = f"http://localhost:{frontend_port}"
    logger.info("Running claude exec smoke test against %s", base_url)

    extra_env = {
        "HOME": str(claude_home),
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": "sk-none",
    }
    # claude shells out to `node` internally; make sure the fixture-installed
    # runtime resolves on PATH without inheriting the runner's node.
    env = _agent_subprocess_env(extra_env, path_prepend=[node_bin])

    cmd = [
        str(claude_cli),
        "--model",
        COMPLIANCE_MODEL,
        "--dangerously-skip-permissions",
        "-p",
        "What files exist in the current working directory? Use your shell tool to run ls and report each filename verbatim from the output.",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    _attach_subprocess_log(
        name="claude_exec_smoke.log",
        cmd=cmd,
        result=result,
        extra_env=extra_env,
        cwd=str(cwd),
    )
    if result.stdout:
        logger.info("claude stdout:\n%s", result.stdout)
    if result.stderr:
        logger.info("claude stderr:\n%s", result.stderr)

    if result.returncode != 0:
        pytest.fail(
            f"claude -p failed (exit={result.returncode}).\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    if marker_filename not in result.stdout:
        pytest.fail(
            "claude -p did not report the marker file — expected stdout to "
            f"contain {marker_filename!r} (implies the Bash tool was invoked "
            f"and actually ran `ls` in {cwd}). Got:\n{result.stdout}"
        )
