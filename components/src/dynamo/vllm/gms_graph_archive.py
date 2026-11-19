# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--gms-graph-archive``: GMS shadow engines save and load CUDA graphs with Foundry.

The policy (which engine saves, which loads, what is set aside) is
``gpu_memory_service.integrations.vllm.graph_archive``. This module applies its
decision to one engine: Foundry's environment for the engine's processes, its
per-engine settings file, and the GMS loader options for a fixed-address,
load-diff read-only attach. Imported only when GMS is in use.
"""

import contextlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Optional

from gpu_memory_service.integrations.common.utils import (
    FIXED_VA_WINDOWS_OPTION,
    get_gms_fixed_va_window,
)
from gpu_memory_service.integrations.vllm.graph_archive import (
    LOADER_OPTION,
    ArchiveMode,
    GraphArchive,
    engine_identity,
)
from vllm.engine.arg_utils import AsyncEngineArgs

logger = logging.getLogger(__name__)

FLAG = "--gms-graph-archive"
CONFIG_FLAG = "--gms-graph-archive-config"

# Graphs hold raw device addresses, so the weights and the KV cache have to be
# mapped at the same addresses in the engine that saves and in every engine
# that loads. Address space only: GMS maps memory into it as it attaches.
# Why these addresses (override with the gms_fixed_va_windows loader option):
# - 66-75 TiB was free in a fresh process wherever this ran (GB200): the
#   driver granted both windows and placed nothing of its own there;
# - both windows end below Foundry's region (default base 0x600000000000) and
#   its peer-import zone at 0x700000000000; the overlap with the configured
#   region is checked at start-up;
# - 512 GiB and 256 GiB exceed any current GPU's memory, so one engine's
#   weights or KV cache always fit;
# - the driver honors a hint only before the process's first kernel, so the GMS
#   worker reserves both windows in init_device.
DEFAULT_FIXED_VA_WINDOWS = {
    "weights": (0x4200_0000_0000, 512 << 30),
    "kv_cache": (0x4A00_0000_0000, 256 << 30),
}

# Foundry's own defaults, for a settings file that leaves them unset.
_FOUNDRY_DEFAULT_BASE_ADDR = 0x6000_0000_0000
_FOUNDRY_DEFAULT_REGION_SIZE = "64GB"
_SIZE_UNITS = {"B": 0, "KB": 10, "MB": 20, "GB": 30, "TB": 40}

# The archive has only been validated with load-diff read-only attach.
RECONSTRUCTION_OPTION = "gms_reconstruction_mode"

# Settings the policy chooses per engine; the user's Foundry file must not set them.
_PER_ENGINE_FOUNDRY_KEYS = ("mode", "workspace_root")

# Foundry's vLLM plugin: activated in each engine process by FOUNDRY_VLLM_CONFIG.
_PLUGIN_GROUP = "vllm.general_plugins"
_PLUGIN_NAME = "foundry"

ARCHIVE_PACKAGES = ("vllm", "torch", "foundry-core", "gpu-memory-service")


def _foundry_hook_library() -> str:
    """Foundry's hook library, from a Foundry that has the vLLM plugin."""
    spec = importlib.util.find_spec("foundry")
    if spec is None or spec.origin is None:
        raise ValueError(f"{FLAG} requires the foundry package in the runtime image")
    plugins = importlib.metadata.entry_points(group=_PLUGIN_GROUP)
    if _PLUGIN_NAME not in plugins.names:
        raise ValueError(
            f"{FLAG} requires Foundry's vLLM plugin (entry point {_PLUGIN_NAME!r} in "
            f"{_PLUGIN_GROUP}), which the installed foundry package does not provide"
        )
    hook = Path(spec.origin).parent / "libcuda_hook.so"
    if not hook.is_file():
        raise ValueError(f"{FLAG}: Foundry's hook library {hook} is missing")
    return str(hook)


def _foundry_settings(settings_file: Optional[str]) -> tuple[str, dict]:
    """The user's Foundry settings: the file's text and its parsed content."""
    if not settings_file:
        return "", {}
    try:
        text = Path(settings_file).read_text(encoding="utf-8")
        parsed = tomllib.loads(text)
    except (OSError, ValueError) as e:
        raise ValueError(f"{CONFIG_FLAG} {settings_file}: {e}") from e
    reserved = [k for k in _PER_ENGINE_FOUNDRY_KEYS if k in parsed]
    if reserved:
        raise ValueError(
            f"{CONFIG_FLAG} must not set {reserved}: {FLAG} sets them per engine"
        )
    return text, parsed


def _foundry_size(value: int | str) -> int:
    """A Foundry size setting in bytes (``"64GB"``: binary units, as Foundry reads it)."""
    if isinstance(value, int):
        return value
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)?\s*", str(value).upper()
    )
    if match is None:
        raise ValueError(f"{FLAG}: cannot read Foundry size {value!r}")
    return int(float(match.group(1)) * (1 << _SIZE_UNITS[match.group(2) or "B"]))


def _foundry_address(value: int | str) -> int:
    """A Foundry address setting (an integer, or a string such as ``"0x6000..."``)."""
    try:
        if isinstance(value, str):
            return int(value, 0)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    except ValueError:
        pass
    raise ValueError(f"{FLAG}: cannot read Foundry address {value!r}")


def fixed_va_windows(loader_options: dict) -> dict[str, tuple[int, int]]:
    """The fixed-VA window of each tag: the defaults, replaced per tag by the
    ``gms_fixed_va_windows`` loader option."""
    user = loader_options.get(FIXED_VA_WINDOWS_OPTION) or {}
    unknown = sorted(set(user) - set(DEFAULT_FIXED_VA_WINDOWS))
    if unknown:
        raise ValueError(
            f"{FLAG}: {FIXED_VA_WINDOWS_OPTION} has no tag {unknown}; the tags are "
            f"{sorted(DEFAULT_FIXED_VA_WINDOWS)}"
        )
    windows = dict(DEFAULT_FIXED_VA_WINDOWS)
    for tag in user:
        try:
            window = get_gms_fixed_va_window(loader_options, tag)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{FLAG}: {FIXED_VA_WINDOWS_OPTION}[{tag!r}]: {e}") from e
        if window is None or window[0] < 0 or window[1] <= 0:
            raise ValueError(
                f"{FLAG}: {FIXED_VA_WINDOWS_OPTION}[{tag!r}] must be [base, size] with "
                f"a positive size, got {user[tag]!r}"
            )
        windows[tag] = window
    return windows


def check_fixed_va_windows(
    windows: dict[str, tuple[int, int]], foundry_settings: dict
) -> None:
    """Refuse fixed-VA windows that overlap each other or Foundry's region.

    Both are reserved in every archive engine; an overlap would make one of the
    reservations fail or land elsewhere, and the archived addresses with it.
    """
    region = (
        _foundry_address(foundry_settings.get("base_addr", _FOUNDRY_DEFAULT_BASE_ADDR)),
        _foundry_size(
            foundry_settings.get("region_size", _FOUNDRY_DEFAULT_REGION_SIZE)
        ),
    )
    ranges = {"Foundry region": region}
    for tag, window in windows.items():
        ranges[f"{FIXED_VA_WINDOWS_OPTION}[{tag!r}]"] = window
    names = list(ranges)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            (a, a_size), (b, b_size) = ranges[first], ranges[second]
            if a < b + b_size and b < a + a_size:
                raise ValueError(
                    f"{FLAG}: {first} [{a:#x}, {a + a_size:#x}) overlaps {second} "
                    f"[{b:#x}, {b + b_size:#x}); move the window with the "
                    f"{FIXED_VA_WINDOWS_OPTION} loader option or Foundry's base_addr"
                )


def _write_foundry_config(mode: ArchiveMode, workspace: Path, settings: str) -> str:
    fd, path = tempfile.mkstemp(prefix="foundry-", suffix=".toml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        # A JSON string is a valid TOML basic string when it is not escaped to ASCII.
        f.write(f"mode = {json.dumps(mode.value)}\n")
        f.write(f"workspace_root = {json.dumps(str(workspace), ensure_ascii=False)}\n")
        f.write(settings)
    return path


class GraphArchiveStart(contextlib.AbstractContextManager):
    """Start the engine inside this context.

    While it is open, the process environment carries what Foundry needs in
    the engine processes the start spawns: its settings file, its mode and its
    hook library. On exit the environment is restored, so nothing else this
    process starts runs with Foundry's hook, and the policy is told the
    outcome: ``commit()`` if the engine started, ``abort()`` with the error if
    it did not.
    """

    def __init__(
        self,
        archive: GraphArchive,
        foundry_settings: str = "",
        hook: Optional[str] = None,
        windows: Optional[dict[str, tuple[int, int]]] = None,
    ):
        self.archive = archive
        self._foundry_settings = foundry_settings
        self._hook = hook
        self._windows = windows or {}
        self._settings_file: Optional[str] = None
        self._saved: dict[str, Optional[str]] = {}

    def __enter__(self) -> "GraphArchiveStart":
        mode, workspace, hook = self.archive.mode, self.archive.workspace, self._hook
        if mode == ArchiveMode.NATIVE:
            return self
        if mode is None or workspace is None or hook is None:
            raise RuntimeError(
                f"{FLAG}: an engine that saves or loads starts from a planned "
                "archive, with Foundry's hook library"
            )
        try:
            self._settings_file = _write_foundry_config(
                mode, workspace, self._foundry_settings
            )
        except OSError as e:
            self.archive.abort(f"{type(e).__name__}: {e}")
            raise
        preload = [p for p in os.environ.get("LD_PRELOAD", "").split(":") if p]
        if hook not in preload:
            preload.insert(0, hook)
        environment = {
            "FOUNDRY_VLLM_CONFIG": self._settings_file,
            "FOUNDRY_MODE": mode.value,
            "LD_PRELOAD": ":".join(preload),
        }
        self._saved = {name: os.environ.get(name) for name in environment}
        os.environ.update(environment)
        logger.info(
            "[GMS] Graph archive: Foundry %s with %s, fixed-VA windows %s",
            mode.value,
            self._settings_file,
            {
                tag: [hex(base), hex(size)]
                for tag, (base, size) in self._windows.items()
            },
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc is None:
                self.archive.commit()
            else:
                self.archive.abort(f"{exc_type.__name__}: {exc}")
        finally:
            for name, value in self._saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            # Every engine process has read it by now.
            if self._settings_file is not None:
                Path(self._settings_file).unlink(missing_ok=True)


def start_graph_archive(
    directory: Optional[str],
    settings_file: Optional[str],
    engine_args: AsyncEngineArgs,
) -> contextlib.AbstractContextManager:
    """Choose this engine's archive mode and return the context to start it in.

    Call after the GMS lock mode is set and before the engine config is
    created: it sets loader options on ``engine_args``. ``directory`` and
    ``settings_file`` are ``--gms-graph-archive`` and
    ``--gms-graph-archive-config``; without a directory nothing happens.
    """
    if not directory:
        return contextlib.nullcontext()
    settings, parsed = _foundry_settings(settings_file)
    if "FOUNDRY_VLLM_CONFIG" in os.environ:
        raise ValueError(f"FOUNDRY_VLLM_CONFIG is set; {FLAG} configures Foundry")
    plugins = os.environ.get("VLLM_PLUGINS")
    if plugins is not None and _PLUGIN_NAME not in plugins.split(","):
        raise ValueError(f"{FLAG} needs the foundry plugin, but VLLM_PLUGINS={plugins}")
    hook = _foundry_hook_library()

    options = dict(engine_args.model_loader_extra_config or {})
    options.setdefault(RECONSTRUCTION_OPTION, "load_diff")
    windows = fixed_va_windows(options)
    check_fixed_va_windows(windows, parsed)
    engine_args.model_loader_extra_config = options

    # The windows are in the key: archived graphs hold addresses inside them.
    identity = engine_identity(
        engine_args,
        {"foundry": parsed, FIXED_VA_WINDOWS_OPTION: windows},
        ARCHIVE_PACKAGES,
    )
    archive = GraphArchive(directory, identity, range(engine_args.tensor_parallel_size))
    mode = archive.plan(read_only=bool(options.get("gms_read_only", False)))
    if mode == ArchiveMode.NATIVE:
        return GraphArchiveStart(archive)

    options[LOADER_OPTION] = mode.value
    options[FIXED_VA_WINDOWS_OPTION] = {tag: list(w) for tag, w in windows.items()}
    return GraphArchiveStart(archive, settings, hook, windows)
