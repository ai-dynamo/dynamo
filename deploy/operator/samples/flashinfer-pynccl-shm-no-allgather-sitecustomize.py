# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail if the TP1/PP1/DP8/EP8 experiment reaches model all-gather."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType
from typing import Any

_TARGET = "vllm.distributed.device_communicators.cuda_communicator"


def _patch(module: ModuleType) -> None:
    communicator = module.CudaCommunicator
    if getattr(communicator, "_no_allgather_evidence_installed", False):
        return

    original_init = communicator.__init__
    original_all_gather = communicator.all_gather
    original_all_gatherv = communicator.all_gatherv
    original_checkpoint_prepare = communicator.checkpoint_prepare

    def instrumented_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self._model_allgather_calls = 0
        original_init(self, *args, **kwargs)

    def reject_all_gather(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self.world_size > 1:
            self._model_allgather_calls += 1
            raise RuntimeError(
                "Unexpected multi-rank model all_gather in TP1/PP1/DP8/EP8 "
                f"group {self.unique_name}"
            )
        return original_all_gather(self, *args, **kwargs)

    def reject_all_gatherv(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self.world_size > 1:
            self._model_allgather_calls += 1
            raise RuntimeError(
                "Unexpected multi-rank model all_gatherv in TP1/PP1/DP8/EP8 "
                f"group {self.unique_name}"
            )
        return original_all_gatherv(self, *args, **kwargs)

    def checkpoint_prepare(self: Any) -> None:
        calls = self._model_allgather_calls
        workspace_count = len(getattr(self, "fi_ag_workspaces", {}))
        print(
            "FLASHINFER_PYNCCL_NO_MODEL_ALLGATHER "
            f"group={self.unique_name} world_size={self.world_size} "
            f"calls={calls} fi_workspace_count={workspace_count}",
            flush=True,
        )
        if calls or workspace_count:
            raise RuntimeError(
                "Model all-gather evidence failed: "
                f"calls={calls}, fi_workspace_count={workspace_count}"
            )
        original_checkpoint_prepare(self)

    communicator.__init__ = instrumented_init
    communicator.all_gather = reject_all_gather
    communicator.all_gatherv = reject_all_gatherv
    communicator.checkpoint_prepare = checkpoint_prepare
    communicator._no_allgather_evidence_installed = True
    print("FLASHINFER_PYNCCL_NO_MODEL_ALLGATHER_GUARD_INSTALLED", flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: Any) -> ModuleType | None:
        create_module = getattr(self._wrapped, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: list[str] | None, target: ModuleType | None = None
    ) -> Any:
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _Loader(spec.loader)
        return spec


def _install() -> None:
    module = sys.modules.get(_TARGET)
    if module is not None:
        _patch(module)
    else:
        sys.meta_path.insert(0, _Finder())


if os.environ.get("DYN_SNAPSHOT_RESTORE_STANDBY") == "1":
    print("FLASHINFER_PYNCCL_NO_MODEL_ALLGATHER_GUARD_STANDBY_SKIP", flush=True)
else:
    _install()
