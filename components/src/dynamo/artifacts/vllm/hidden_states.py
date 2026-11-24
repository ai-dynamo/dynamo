# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hidden-state capture through vLLM's ``extract_hidden_states`` connector contract.

vLLM extracts target-model activations when it runs the ``extract_hidden_states``
speculative method with ``ExampleHiddenStatesConnector``. When a request finishes,
the connector writes ``{"hidden_states": [T, L, H], "token_ids": [T]}`` as a
safetensors file and returns its path in the request's
``kv_transfer_params["hidden_states_path"]``. It holds an exclusive lock on
``<path>.lock`` until the write completes.

This module derives the worker's capture profile from the engine config, and reads,
checks and deletes one such file. It imports no heavy dependencies at module load.
"""

from __future__ import annotations

import fcntl
import os
import threading
from dataclasses import dataclass
from typing import Any

HIDDEN_STATES_CONNECTOR = "ExampleHiddenStatesConnector"
EXTRACT_HIDDEN_STATES_METHOD = "extract_hidden_states"
# Default `shared_storage_path` of ExampleHiddenStatesConnector.
_DEFAULT_STORAGE_PATH = "/tmp"
_DTYPE_NAMES = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "bf16": "bf16",
    "fp16": "fp16",
    "fp32": "fp32",
}
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


class HiddenStateFileError(ValueError):
    """The connector output is missing, outside the storage root, or malformed."""


@dataclass(frozen=True)
class HiddenStateProfile:
    """What a worker captures: fixed at launch by the engine config."""

    layer_ids: tuple[int, ...]
    hidden_size: int
    dtype: str
    storage_root: str

    @property
    def row_bytes(self) -> int:
        return len(self.layer_ids) * self.hidden_size * DTYPE_BYTES[self.dtype]

    def runtime_data(self) -> dict[str, Any]:
        """Capability published for discovery. Storage paths stay worker-local."""
        return {
            "layer_ids": list(self.layer_ids),
            "hidden_size": self.hidden_size,
            "dtype": self.dtype,
        }


def _get(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _connector_extra_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Extra config of the hidden-states connector, directly or inside MultiConnector."""
    connector = _get(kv_transfer_config, "kv_connector")
    extra = _get(kv_transfer_config, "kv_connector_extra_config") or {}
    if connector == HIDDEN_STATES_CONNECTOR:
        return dict(extra)
    if connector == "MultiConnector":
        for child in _get(extra, "connectors") or []:
            if _get(child, "kv_connector") == HIDDEN_STATES_CONNECTOR:
                return dict(_get(child, "kv_connector_extra_config") or {})
    return None


def _dtype_name(dtype: Any) -> str | None:
    return _DTYPE_NAMES.get(str(dtype))


def hidden_state_profile_from_vllm_config(
    vllm_config: Any,
) -> HiddenStateProfile | None:
    """Return the capture profile, or None when this worker cannot capture.

    Raises ValueError for a configuration that enables extraction but cannot be
    read safely, so a misconfigured worker fails at launch instead of per request.
    """
    speculative = _get(vllm_config, "speculative_config")
    if _get(speculative, "method") != EXTRACT_HIDDEN_STATES_METHOD:
        return None
    extra = _connector_extra_config(_get(vllm_config, "kv_transfer_config"))
    if extra is None:
        return None
    if extra.get("use_synchronization_lock", True) is not True:
        raise ValueError(
            "hidden-state capture requires use_synchronization_lock: the worker "
            "reads each file only after the connector releases its lock"
        )
    if extra.get("allow_custom_save_path", False):
        raise ValueError(
            "hidden-state capture does not support allow_custom_save_path: files "
            "must stay under the connector's shared_storage_path"
        )
    hf_config = _get(_get(speculative, "draft_model_config"), "hf_config")
    layer_ids = _get(hf_config, "eagle_aux_hidden_state_layer_ids")
    if (
        not isinstance(layer_ids, (list, tuple))
        or not layer_ids
        or any(type(layer) is not int or layer < 0 for layer in layer_ids)
        or len(set(layer_ids)) != len(layer_ids)
    ):
        raise ValueError(
            "hidden-state capture requires eagle_aux_hidden_state_layer_ids in "
            "the extract_hidden_states draft config"
        )
    model_config = _get(vllm_config, "model_config")
    get_hidden_size = _get(model_config, "get_hidden_size")
    hidden_size = (
        get_hidden_size()
        if callable(get_hidden_size)
        else _get(model_config, "hidden_size")
    )
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("hidden-state capture requires the model hidden size")
    dtype = _dtype_name(_get(model_config, "dtype"))
    if dtype is None:
        raise ValueError(
            "hidden-state capture supports bfloat16, float16 and float32 models"
        )
    storage_root = os.path.realpath(
        str(extra.get("shared_storage_path") or _DEFAULT_STORAGE_PATH)
    )
    return HiddenStateProfile(tuple(layer_ids), hidden_size, dtype, storage_root)


def hidden_state_path(
    kv_transfer_params: Any, profile: HiddenStateProfile
) -> str | None:
    """The connector's file path for this request, confined to the storage root."""
    if not isinstance(kv_transfer_params, dict):
        return None
    path = kv_transfer_params.get("hidden_states_path")
    if path is None:
        return None
    if not isinstance(path, str) or not path:
        raise HiddenStateFileError("invalid hidden-state path")
    resolved = os.path.realpath(path)
    if os.path.commonpath((resolved, profile.storage_root)) != profile.storage_root:
        raise HiddenStateFileError("hidden-state path is outside the storage root")
    return resolved


def remove_hidden_state_file(path: str) -> None:
    for name in (path, path + ".lock"):
        try:
            os.remove(name)
        except FileNotFoundError:
            pass


def _wait_and_remove(path: str) -> None:
    try:
        lock_fd = os.open(path + ".lock", os.O_RDONLY)
    except FileNotFoundError:
        lock_fd = None
    try:
        if lock_fd is not None:
            # Removing before the writer creates the file would leak it.
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        remove_hidden_state_file(path)


def discard_hidden_state_file(path: str) -> None:
    """Delete an unread file once its writer finishes, without blocking the caller."""
    threading.Thread(
        target=_wait_and_remove,
        args=(path,),
        name="dynamo-hidden-state-discard",
        daemon=True,
    ).start()


@dataclass(frozen=True)
class LoadedHiddenStates:
    token_ids: tuple[int, ...]
    dtype: str
    shape: tuple[int, int, int]
    data: bytes


def load_hidden_state_file(path: str) -> LoadedHiddenStates:
    """Wait for the writer, read the file, and delete it. Blocking: run off-loop."""
    try:
        import torch
        from safetensors import SafetensorError
        from safetensors.torch import load_file
    except ImportError as exc:
        raise HiddenStateFileError("hidden-state capture requires torch") from exc

    try:
        lock_fd = os.open(path + ".lock", os.O_RDONLY)
    except FileNotFoundError:
        lock_fd = None
    try:
        if lock_fd is not None:
            # Sleeps until the connector's writer thread releases LOCK_EX.
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
        try:
            tensors = load_file(path, device="cpu")
        except (OSError, RuntimeError, ValueError, SafetensorError) as exc:
            raise HiddenStateFileError("hidden-state file is unreadable") from exc
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        remove_hidden_state_file(path)

    hidden = tensors.get("hidden_states")
    token_ids = tensors.get("token_ids")
    if (
        hidden is None
        or token_ids is None
        or set(tensors)
        - {
            "hidden_states",
            "token_ids",
        }
    ):
        raise HiddenStateFileError("unexpected hidden-state file contents")
    if hidden.ndim != 3 or token_ids.ndim != 1 or hidden.shape[0] != token_ids.shape[0]:
        raise HiddenStateFileError("unexpected hidden-state file geometry")
    dtype = _dtype_name(hidden.dtype)
    if dtype is None:
        raise HiddenStateFileError("unsupported hidden-state dtype")
    hidden = hidden.contiguous()
    if dtype == "bf16":
        # NumPy has no bfloat16; the artifact format stores its raw bit patterns.
        data = hidden.view(torch.int16).numpy().tobytes()
    else:
        data = hidden.numpy().tobytes()
    return LoadedHiddenStates(
        tuple(int(token) for token in token_ids.tolist()),
        dtype,
        (int(hidden.shape[0]), int(hidden.shape[1]), int(hidden.shape[2])),
        data,
    )
