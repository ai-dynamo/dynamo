# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one AIS perf model input, shared by every Dynamo component that takes it.

A value is a file path, inline JSON, one perf model config (a mapping) or a
list of configs with one ``worker_type`` each. AISimulate reads and resolves
it; this module is the only place in Dynamo that calls that code.

AISimulate is an optional dependency: importing this module does not import
it. The functions that read or resolve a value import it on first use and
raise ``ImportError`` when it is not installed.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from aisimulate_core.ais_perf_model import AisPerfModelDocument

ENV_VAR = "DYN_AIS_PERF_MODEL"
WORKER_TYPE_ENV_VAR = "DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE"
# The release that removes the old inputs. Every warning names it.
REMOVE_IN = "v1.9"
# The worker types a router can estimate prefill load for.
ROUTER_WORKER_TYPES = ("prefill", "aggregated")
# The keys every resolved config must set to a non-empty value.
_REQUIRED_KEYS = ("backend", "system", "model", "worker_type")

# The folder each process prepares datasets in, by process id. A forked child
# adds a folder of its own and keeps the entry it inherited: dropping that
# entry would delete the folder while the parent still uses it.
_dataset_workspaces: dict[int, tempfile.TemporaryDirectory[str]] = {}


def _loader():
    # Imported here: Dynamo must import without AISimulate installed.
    from aisimulate_core import ais_perf_model

    return ais_perf_model


def _dataset_workspace() -> str:
    """Return the folder this process prepares datasets in.

    The folder is created on first use and removed when the interpreter exits.
    """
    pid = os.getpid()
    if pid not in _dataset_workspaces:
        _dataset_workspaces[pid] = tempfile.TemporaryDirectory(
            prefix="dynamo-ais-perf-model-"
        )
    return _dataset_workspaces[pid].name


def env_value(name: str = ENV_VAR) -> str | None:
    """Return the variable ``name``, or ``None`` when it is unset or empty."""
    return os.environ.get(name) or None


def load_document(
    value: Any, *, base_dir: str | os.PathLike[str] | None = None
) -> AisPerfModelDocument:
    """Read an AIS perf model value without following ``from``.

    ``value`` is a file path, inline JSON (text that starts with ``{`` or
    ``[``), one config or a list of configs. A relative file path, and
    relative paths inside inline JSON, a mapping or a list, start at
    ``base_dir`` (the working directory when it is ``None``). Relative paths
    inside a file start at that file's folder.

    No template file, dataset or state snapshot is opened here;
    ``resolve_entry`` does that. A wrong value raises the loader's error, a
    ``ValueError``.
    """
    return _loader().load(value, base_dir=base_dir)


def select_router_entry(
    document: AisPerfModelDocument, worker_type: str | None = None
) -> dict[str, Any]:
    """Return the entry a router uses, as written.

    ``worker_type`` is ``"prefill"``, ``"aggregated"`` or ``None`` when the
    user did not choose one. Without a choice the document must name exactly
    one of the two. Pass the same ``worker_type`` to ``resolve_entry``: it
    fills a missing ``worker_type`` and rejects a different written one.
    """
    if worker_type is not None and worker_type not in ROUTER_WORKER_TYPES:
        raise ValueError(
            f"worker_type must be 'prefill' or 'aggregated', got {worker_type!r}"
        )
    name = document.source or "ais_perf_model"
    written = document.worker_types
    if document.bare:
        if worker_type is None:
            if not written:
                raise ValueError(
                    f"{name}: the config has no worker_type; write worker_type or "
                    "set --router-prefill-load-worker-type"
                )
            if written[0] not in ROUTER_WORKER_TYPES:
                raise ValueError(
                    f"{name}: worker_type is {written[0]!r}; the router needs "
                    "prefill or aggregated"
                )
        # One config: to_data() returns that config, not a list.
        return cast("dict[str, Any]", document.to_data())
    if worker_type is None:
        usable = [
            candidate for candidate in ROUTER_WORKER_TYPES if candidate in written
        ]
        if len(usable) == 2:
            raise ValueError(
                f"{name}: it has entries for prefill and aggregated; set "
                "--router-prefill-load-worker-type to choose one"
            )
        if not usable:
            raise ValueError(
                f"{name}: no entry for worker type prefill or aggregated; it has: "
                f"{', '.join(written)}"
            )
        worker_type = usable[0]
    return document.select(worker_type)


def resolve_entry(
    entry: Mapping[str, Any],
    *,
    worker_type: str | None = None,
    fill: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the config one entry stands for, ready for ``ForwardPassPerfModelConfig``.

    This is the step that reads template files, state snapshots and datasets.
    ``worker_type`` is the worker type the caller runs as: it fills a missing
    ``worker_type`` and must equal a written one. ``fill`` holds values the
    caller requires: a missing key is added and a written key must be equal.
    ``defaults`` are used only for keys the entry does not write.
    """
    loader = _loader()
    # An entry that is not a mapping goes on to the loader, which reports it.
    source = entry.get("from") if isinstance(entry, Mapping) else None
    # Only a dataset is unpacked on disk, so only a dataset entry gets a folder.
    workspace = (
        _dataset_workspace()
        if isinstance(source, Mapping) and "dataset" in source
        else None
    )
    return loader.resolve(
        entry,
        worker_type=worker_type,
        fill=fill,
        defaults=defaults,
        workspace=workspace,
    )


def router_config(
    value: Any,
    worker_type: str | None = None,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return the ``AisPerfConfig(config=...)`` mapping for an AIS perf model value.

    The value is read, the router's entry is selected and resolved, and the
    result is normalized as the old router inputs are. Raises the errors of
    ``normalized_config``.
    """
    document = load_document(value, base_dir=base_dir)
    resolved = resolve_entry(
        select_router_entry(document, worker_type), worker_type=worker_type
    )
    return normalized_config(resolved)


def normalized_config(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Return a resolved config as ``ForwardPassPerfModelConfig`` normalizes it.

    A missing or empty ``backend``, ``system``, ``model`` or ``worker_type``
    and a key that class does not know are a ``ValueError``.
    """
    missing = [key for key in _REQUIRED_KEYS if not resolved.get(key)]
    if missing:
        raise ValueError(f"ais_perf_model requires {', '.join(missing)}")
    # Imported here: Dynamo must import without AISimulate installed.
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    try:
        config = ForwardPassPerfModelConfig(**resolved)
    except TypeError as error:
        # A key the class does not take is a wrong value in the user's config.
        raise ValueError(f"ais_perf_model: {error}") from error
    return config.to_dict()


def uses_relative_paths(value: Any) -> bool:
    """Return whether a mapping or list value holds a relative path.

    Not meant for a string value: a file path is opened.
    """
    # The loader makes relative paths absolute from base_dir, so two folders
    # give different data exactly when a relative path is written.
    return (
        load_document(value, base_dir="/__a__").to_data()
        != load_document(value, base_dir="/__b__").to_data()
    )


def timing_model(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``engine.timing_model`` value for AIS with a copy of ``config``."""
    return {
        "type": "external",
        "provider": "ais",
        "config": copy.deepcopy(dict(config)),
    }


def engine_args_as_written(
    engine_args: str | Mapping[str, Any] | None,
    *,
    engine_args_name: str,
    perf_model_name: str,
) -> dict[str, Any]:
    """Return a copy of the engine args the user wrote, as a ``dict``.

    ``engine_args`` is JSON text, a mapping or ``None`` (an empty ``dict``).
    Engine args that already write ``engine.timing_model`` are an error: the
    AIS perf model input ``perf_model_name`` would replace it. Both names
    appear in the error as the user wrote them.
    """
    not_an_object = f"{engine_args_name} must be a JSON object"
    if engine_args is None:
        return {}
    if isinstance(engine_args, str):
        try:
            written = json.loads(engine_args)
        except json.JSONDecodeError as error:
            raise ValueError(not_an_object) from error
        if not isinstance(written, dict):
            raise ValueError(not_an_object)
    elif isinstance(engine_args, Mapping):
        written = copy.deepcopy(dict(engine_args))
    else:
        raise ValueError(not_an_object)
    if "engine" in written:
        if not isinstance(written["engine"], Mapping):
            raise ValueError(not_an_object)
        if "timing_model" in written["engine"]:
            raise ValueError(
                f"{engine_args_name} already sets engine.timing_model; it cannot be "
                f"combined with {perf_model_name}"
            )
    return written


def set_timing_model(
    engine_args: str | Mapping[str, Any] | None,
    config: Mapping[str, Any],
    *,
    engine_args_name: str,
    perf_model_name: str,
) -> dict[str, Any]:
    """Return a copy of the engine args with ``config`` as the AIS timing model.

    Raises the errors of ``engine_args_as_written``.
    """
    result = engine_args_as_written(
        engine_args,
        engine_args_name=engine_args_name,
        perf_model_name=perf_model_name,
    )
    result.setdefault("engine", {})["timing_model"] = timing_model(config)
    return result


def join_names(names: Sequence[str]) -> str:
    """Return ``a``, ``a and b`` or ``a, b and c``."""
    if len(names) <= 2:
        return " and ".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def old_inputs_message(old_names: Sequence[str], replacement: str) -> str:
    """Return the sentence that says ``old_names`` are deprecated."""
    if not old_names:
        raise ValueError("old_names is empty")
    verb = "is" if len(old_names) == 1 else "are"
    return (
        f"{join_names(old_names)} {verb} deprecated and will be removed in "
        f"{REMOVE_IN}; use {replacement}."
    )


def warn_old_inputs(
    old_names: Sequence[str], replacement: str, *, stacklevel: int
) -> None:
    """Emit one ``FutureWarning`` that names every old input in ``old_names``.

    ``stacklevel=1`` is the line that calls this function, ``2`` the caller of
    that function, and so on. Earlier calls are not remembered: a component
    collects its old inputs and calls this once.
    """
    warnings.warn(
        old_inputs_message(old_names, replacement),
        FutureWarning,
        stacklevel=stacklevel + 1,
    )
