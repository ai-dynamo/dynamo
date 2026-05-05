# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess boot helper for `vllm serve` / `python -m sglang.launch_server`.

Used by the e2e parity harness to bring up an HTTP server for one
(impl, parser_family) combination, yield its base URL, and tear it
down cleanly. Pytest fixtures in `conftest.py` wrap this for
session-scoped reuse.

Smoke-tested against vLLM 0.20.0 with `Qwen/Qwen3-0.6B
--load-format dummy --enforce-eager`. Boot ~30 s on a GPU runner;
`--gpu-memory-utilization 0.05` keeps the per-server footprint
small (~2.5 GiB on a 47 GiB device) so multiple session-cached
servers coexist on one GPU.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import requests

# Maps `parser_family` (our internal name) → vLLM `--tool-call-parser` key.
_FAMILY_TO_VLLM_KEY: dict[str, str] = {
    "kimi_k2": "kimi_k2",
    "qwen3_coder": "qwen3_coder",
    "glm47": "glm47",
    "deepseek_v3_1": "deepseek_v31",
    "minimax_m2": "minimax_m2",
    "harmony": "openai",
    # "nemotron_deci": no vLLM parser today
}


# Maps `parser_family` → SGLang `--tool-call-parser` key. SGLang uses
# slightly different naming (no underscore in `deepseekv31`, hyphen in
# `minimax-m2`, `gpt-oss` for harmony).
_FAMILY_TO_SGLANG_KEY: dict[str, str] = {
    "kimi_k2": "kimi_k2",
    "qwen3_coder": "qwen3_coder",
    "glm47": "glm47",
    "deepseek_v3_1": "deepseekv31",
    "minimax_m2": "minimax-m2",
    "harmony": "gpt-oss",
    # "nemotron_deci": no SGLang detector today
}

# Per-family (model, tokenizer) selection. None tokenizer = use the
# model's bundled tokenizer. Weights are always loaded with
# `--load-format dummy`; only architectures + tokenizer files matter.
_FAMILY_MODELS: dict[str, tuple[str, str | None]] = {
    # Default: Qwen3-0.6B's tokenizer is generic enough for text-based parsers.
    "kimi_k2": ("Qwen/Qwen3-0.6B", None),
    "qwen3_coder": ("Qwen/Qwen3-0.6B", None),
    "glm47": ("Qwen/Qwen3-0.6B", None),
    # Token-ID-aware parsers need the matching family's tokenizer.
    # Use Qwen3 model architecture (cheap to dummy-load) + override tokenizer.
    "deepseek_v3_1": ("Qwen/Qwen3-0.6B", "deepseek-ai/DeepSeek-V3.1"),
    "minimax_m2": ("Qwen/Qwen3-0.6B", "MiniMaxAI/MiniMax-M2"),
    # Harmony's vLLM parser is token-stream-only; a plain text round-trip
    # like the other families is not supported. Slot reserved for a
    # future streaming-with-token-ids variant.
    "harmony": ("openai/gpt-oss-20b", None),
}


def resolve_model(parser_family: str) -> str:
    """Return the (dummy-loaded) model name to use for `parser_family`.

    Single source of truth for client-side request shaping
    (`client.parse`) and server-side boot (`_build_*_cmd`) so the
    `model` field in the chat-completion request always matches what
    the server has loaded.
    """
    model, _ = _FAMILY_MODELS[parser_family]
    return model


def _free_port() -> int:
    """Pick an ephemeral port the OS reports as free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class ServerSpec:
    """Inputs to `serve()`. One spec → one server instance."""

    impl: str  # "vllm" or "sglang"
    parser_family: str  # one of the keys in _FAMILY_MODELS
    port: int = field(default_factory=_free_port)

    # Override for tests that want a specific model/tokenizer
    model: str | None = None
    tokenizer: str | None = None

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        m, _ = _FAMILY_MODELS[self.parser_family]
        return m

    @property
    def resolved_tokenizer(self) -> str | None:
        if self.tokenizer:
            return self.tokenizer
        _, t = _FAMILY_MODELS[self.parser_family]
        return t


def _build_vllm_cmd(spec: ServerSpec) -> list[str]:
    parser_key = _FAMILY_TO_VLLM_KEY.get(spec.parser_family)
    if parser_key is None:
        raise ValueError(f"vLLM has no parser for family={spec.parser_family!r}")
    cmd = [
        "vllm",
        "serve",
        spec.resolved_model,
        "--load-format",
        "dummy",
        "--enforce-eager",
        "--tool-call-parser",
        parser_key,
        "--enable-auto-tool-choice",
        # Cap memory per server: pytest fixtures keep all booted
        # (impl, family) servers alive across the session, so 7 vLLM
        # instances need to coexist on one GPU. 0.05 ≈ 2.5 GiB on a
        # 47 GiB device — plenty for a 0.6B-param model.
        "--gpu-memory-utilization",
        "0.05",
        "--max-model-len",
        "2048",
        "--port",
        str(spec.port),
        "--host",
        "127.0.0.1",
    ]
    if spec.resolved_tokenizer:
        cmd += ["--tokenizer", spec.resolved_tokenizer]
    return cmd


def _build_sglang_cmd(spec: ServerSpec) -> list[str]:
    parser_key = _FAMILY_TO_SGLANG_KEY.get(spec.parser_family)
    if parser_key is None:
        raise ValueError(f"SGLang has no parser for family={spec.parser_family!r}")
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        spec.resolved_model,
        "--load-format",
        "dummy",
        "--tool-call-parser",
        parser_key,
        # Same memory ceiling as the vLLM path so multiple session-cached
        # servers coexist on one GPU.
        "--mem-fraction-static",
        "0.05",
        "--port",
        str(spec.port),
        "--host",
        "127.0.0.1",
    ]
    if spec.resolved_tokenizer:
        cmd += ["--tokenizer-path", spec.resolved_tokenizer]
    return cmd


def _build_cmd(spec: ServerSpec) -> list[str]:
    if spec.impl == "vllm":
        return _build_vllm_cmd(spec)
    if spec.impl == "sglang":
        return _build_sglang_cmd(spec)
    raise ValueError(f"unknown impl: {spec.impl!r}")


def _wait_for_health(
    base_url: str, proc: subprocess.Popen, timeout: float = 180.0
) -> None:
    """Poll /health until the server reports OK, or raise on timeout / exit."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited early with code {proc.returncode}; "
                f"check the server log"
            )
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"server at {base_url} did not become healthy in {timeout}s")


def _kill(proc: subprocess.Popen, port: int, timeout: float = 30.0) -> None:
    """Terminate the process group; wait for the port to release."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)
    # Wait briefly for the OS to release the port.
    deadline = time.time() + 10
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return  # port released
            except OSError:
                time.sleep(0.5)


@contextmanager
def serve(spec: ServerSpec, log_path: Path | None = None) -> Iterator[str]:
    """Boot the server, yield its base URL, kill on exit.

    `log_path` controls where stdout+stderr are tee'd; defaults to
    `/tmp/parity (parser)-<impl>-<family>.log`.
    """
    if log_path is None:
        log_path = Path(f"/tmp/parity (parser)-{spec.impl}-{spec.parser_family}.log")
    cmd = _build_cmd(spec)
    base_url = f"http://127.0.0.1:{spec.port}"
    log_handle = open(log_path, "w")  # noqa: SIM115 — closed in finally
    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    try:
        _wait_for_health(base_url, proc, timeout=180.0)
        yield base_url
    finally:
        _kill(proc, spec.port)
        log_handle.close()


def get_or_boot_server(
    impl: str,
    parser_family: str,
    cache: dict,
    lifecycles: list,
) -> str:
    """Return a cached base_url for (impl, family), booting on first miss.

    Used by e2e pytest tests via the `e2e_server_cache` and
    `e2e_server_lifecycles` session fixtures.
    """
    key = (impl, parser_family)
    if key in cache:
        return cache[key]
    spec = ServerSpec(impl=impl, parser_family=parser_family)
    cm = serve(spec)
    base_url = cm.__enter__()
    lifecycles.append(cm)
    cache[key] = base_url
    return base_url


def smoke(impl: str = "vllm", family: str = "kimi_k2") -> None:
    """One-shot manual probe: boot, hit /v1/models, kill. For sanity checks."""
    spec = ServerSpec(impl=impl, parser_family=family)
    with serve(spec) as base_url:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        data = r.json()
        print(f"[{impl}/{family}] /v1/models OK — model={data['data'][0]['id']!r}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Probe an e2e parity server.")
    p.add_argument("--impl", default="vllm", choices=["vllm", "sglang"])
    p.add_argument("--family", default="kimi_k2")
    args = p.parse_args()
    smoke(args.impl, args.family)
