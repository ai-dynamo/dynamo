# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one AIS perf model input, shared by every Dynamo component that takes it.

A value is a file path, inline JSON, one perf model config (a mapping) or a
list of configs with one ``worker_type`` each.

AISimulate is an optional dependency: importing this module does not import
it. The functions that read or resolve a value import it on first use and
raise ``ImportError`` when it is not installed.
"""

from __future__ import annotations

import atexit
import copy
import importlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from aisim_core.ais_perf_model import AisPerfModelDocument

AIS_PERF_MODEL_ENV = "DYN_AIS_PERF_MODEL"
ROUTER_WORKER_TYPE_ENV = "DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE"
# The release that removes the old inputs. Every warning names it.
AIS_OLD_INPUTS_REMOVE_IN = "v1.9"
# The worker types a router can estimate prefill load for.
ROUTER_WORKER_TYPES = ("prefill", "aggregated")
# The keys every resolved config must set to a non-empty value.
_REQUIRED_KEYS = ("backend", "system", "model", "worker_type")

# The folder each process prepares datasets in, by process id. A forked child
# makes a folder of its own. It also inherits the parent's exit hook, so the
# hook removes a folder only in the process that made it: the parent may
# still use its folder when the child exits.
_dataset_workspaces: dict[int, str] = {}


def _loader() -> Any:
    # Loaded on first use: Dynamo must import without AISimulate installed.
    return importlib.import_module("aisim_core.ais_perf_model")


def _sdk() -> Any:
    # Loaded on first use, as the loader is.
    return importlib.import_module("aisim_core.sdk")


def _remove_dataset_workspace(pid: int, folder: str) -> None:
    """Remove ``folder`` if this process is ``pid``, the process that made it."""
    if os.getpid() == pid:
        shutil.rmtree(folder, ignore_errors=True)


def _dataset_workspace() -> str:
    """Return this process's dataset folder, made on first use and removed at exit."""
    pid = os.getpid()
    if pid not in _dataset_workspaces:
        folder = tempfile.mkdtemp(prefix="dynamo-ais-perf-model-")
        atexit.register(_remove_dataset_workspace, pid, folder)
        _dataset_workspaces[pid] = folder
    return _dataset_workspaces[pid]


def env_value(name: str = AIS_PERF_MODEL_ENV) -> str | None:
    """Return the variable ``name``, or ``None`` when it is unset or empty."""
    return os.environ.get(name) or None


def load_document(
    value: Any, *, base_dir: str | os.PathLike[str] | None = None
) -> AisPerfModelDocument:
    """Read an AIS perf model value without following ``from``.

    ``value`` is a file path, inline JSON (text that starts with ``{`` or
    ``[``), one config or a list of configs. Relative paths start at
    ``base_dir`` (the working directory when ``None``); those inside a file
    start at that file's folder. A wrong value raises ``ValueError``.
    """
    return _loader().load(value, base_dir=base_dir)


def select_router_entry(
    document: AisPerfModelDocument, worker_type: str | None = None
) -> dict[str, Any]:
    """Return the entry a router uses, as written.

    ``worker_type`` is ``"prefill"``, ``"aggregated"`` or ``None``; with
    ``None`` the document must name exactly one of the two. Pass the same
    ``worker_type`` to ``resolve_entry``.
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

    Reads template files, state snapshots and datasets. ``worker_type`` and
    each ``fill`` key are added when missing and must equal a written value.
    ``defaults`` are used only for keys the entry does not write. With a
    dataset or state snapshot ``from``, ``worker_type`` and ``fill`` are only
    compared and ``defaults`` are ignored.
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


def router_perf_config(
    value: Any,
    worker_type: str | None = None,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return the ``AisPerfConfig(config=...)`` mapping for an AIS perf model value.

    Raises the errors of ``normalized_config``.
    """
    document = load_document(value, base_dir=base_dir)
    resolved = resolve_entry(
        select_router_entry(document, worker_type), worker_type=worker_type
    )
    return normalized_config(resolved)


def engine_perf_config(
    value: Any, worker_type: str, *, fill: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the config of an engine that runs as ``worker_type``.

    ``fill`` is passed to ``resolve_entry``. No default is added for a key the
    value does not give.
    """
    return resolve_entry(
        load_document(value).select(worker_type), worker_type=worker_type, fill=fill
    )


def normalized_config(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Return a resolved config as ``ForwardPassPerfModelConfig`` normalizes it.

    A missing or empty ``backend``, ``system``, ``model`` or ``worker_type``
    and a key that class does not know are a ``ValueError``.
    """
    missing = [key for key in _REQUIRED_KEYS if not resolved.get(key)]
    if missing:
        raise ValueError(f"ais_perf_model requires {', '.join(missing)}")
    try:
        config = _sdk().ForwardPassPerfModelConfig(**resolved)
    except TypeError as error:
        # A key the class does not take is a wrong value in the user's config.
        raise ValueError(f"ais_perf_model: {error}") from error
    return config.to_dict()


def save_tuned_state_snapshot(
    config: Mapping[str, Any],
    iterations: list[list[dict[str, Any]]],
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Tune a new perf model with observed forward passes and save its state.

    ``config`` is a resolved config. ``iterations`` holds one list per forward
    pass, with one forward pass metrics mapping per attention-DP rank. The
    state goes to the new folder ``path``, which a value restores with
    ``from: {state_snapshot: {path}}``. Returns the saved config.

    A workload that has observations but no usable fit, and a model without
    any observation, are a ``ValueError``; nothing is saved then.
    """
    model = _sdk().RustForwardPassPerfModel.best_available(dict(config))
    model.tune_with_fpms(iterations)
    stores = model.readiness_report()["stores"]
    unready = [
        f"{store['workload_kind']} ({store['retained_observations']} observations, "
        f"{store['not_ready_reason']})"
        for store in stores
        if store["retained_observations"] and not store["ready"]
    ]
    if unready or not any(store["ready"] for store in stores):
        raise ValueError(
            "ais_perf_model: the model is not ready after tuning: "
            + (", ".join(unready) or "no observation was kept")
        )
    return model.save_state_snapshot(path)


def has_relative_paths(value: Any) -> bool:
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
    Engine args that already write ``engine.timing_model`` are an error that
    names ``engine_args_name`` and ``perf_model_name`` as the user wrote them.
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
