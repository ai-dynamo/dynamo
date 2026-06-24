# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types
from types import SimpleNamespace

import pytest

from dynamo.common.snapshot.constants import SNAPSHOT_CONTROL_DIR_ENV
from dynamo.vllm.snapshot_worker_config import (
    DEFAULT_NO_NCCL_ALL2ALL_BACKEND,
    DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES,
    DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER,
    DYN_VLLM_NO_NCCL_SNAPSHOT,
    GMS_WORKER_CLASS,
    SNAPSHOT_WORKER_CLASS,
    configure_flashinfer_snapshot_worker,
    configure_flashinfer_snapshot_worker_before_engine_config,
    configure_no_nccl_snapshot_before_engine_config,
    configure_gms_worker_cls,
    validate_no_nccl_snapshot_config,
    validate_flashinfer_snapshot_worker_config,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _config(worker_cls=None, load_format="auto"):
    return SimpleNamespace(
        engine_args=SimpleNamespace(worker_cls=worker_cls, load_format=load_format)
    )


def _install_fake_snapshot_worker(monkeypatch):
    base_cls = type("Worker", (), {})
    snapshot_cls = type("SnapshotWorker", (base_cls,), {})
    gpu_worker_module = types.ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker_module.Worker = base_cls
    snapshot_worker_module = types.ModuleType("dynamo.vllm.snapshot_worker")
    snapshot_worker_module.SnapshotWorker = snapshot_cls
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", gpu_worker_module)
    monkeypatch.setitem(
        sys.modules, "dynamo.vllm.snapshot_worker", snapshot_worker_module
    )


def test_flashinfer_snapshot_worker_env_unset_is_noop(monkeypatch):
    monkeypatch.delenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, raising=False)
    config = _config()

    assert configure_flashinfer_snapshot_worker(config) is False
    assert config.engine_args.worker_cls is None


def test_flashinfer_snapshot_worker_env_set_configures_worker(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    _install_fake_snapshot_worker(monkeypatch)
    config = _config(worker_cls="auto")

    assert configure_flashinfer_snapshot_worker(config) is True
    assert config.engine_args.worker_cls == SNAPSHOT_WORKER_CLASS


def test_flashinfer_snapshot_worker_configure_is_idempotent(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    _install_fake_snapshot_worker(monkeypatch)
    config = _config(worker_cls=SNAPSHOT_WORKER_CLASS)

    assert configure_flashinfer_snapshot_worker(config) is True
    assert config.engine_args.worker_cls == SNAPSHOT_WORKER_CLASS


def test_before_engine_config_legacy_alias_installs_snapshot_worker(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    monkeypatch.delenv(SNAPSHOT_CONTROL_DIR_ENV, raising=False)
    _install_fake_snapshot_worker(monkeypatch)

    class EngineArgs:
        worker_cls = "auto"
        load_format = "auto"
        create_engine_config_worker_cls = None

        def create_engine_config(self, usage_context):
            del usage_context
            self.create_engine_config_worker_cls = self.worker_cls
            return SimpleNamespace(
                parallel_config=SimpleNamespace(worker_cls=self.worker_cls)
            )

    config = SimpleNamespace(engine_args=EngineArgs())

    assert configure_flashinfer_snapshot_worker_before_engine_config(config) is True
    vllm_config = config.engine_args.create_engine_config(usage_context=object())

    assert config.engine_args.create_engine_config_worker_cls == SNAPSHOT_WORKER_CLASS
    assert vllm_config.parallel_config.worker_cls == SNAPSHOT_WORKER_CLASS


def test_no_nccl_snapshot_preconfigures_engine_args_before_config(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.setenv(DYN_VLLM_NO_NCCL_SNAPSHOT, "1")
    monkeypatch.delenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, raising=False)
    monkeypatch.delenv(SNAPSHOT_CONTROL_DIR_ENV, raising=False)
    for name in (
        "VLLM_DISABLE_PYNCCL",
        "VLLM_ALLREDUCE_USE_SYMM_MEM",
        "VLLM_USE_NCCL_SYMM_MEM",
        "VLLM_DISTRIBUTED_USE_SPLIT_GROUP",
        "VLLM_ALLREDUCE_USE_FLASHINFER",
        "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB",
    ):
        monkeypatch.delenv(name, raising=False)
    _install_fake_snapshot_worker(monkeypatch)
    monkeypatch.setattr(
        "dynamo.vllm.flashinfer_collectives.patch_vllm_distributed_backend_for_snapshot",
        lambda: True,
    )

    class EngineArgs:
        worker_cls = "auto"
        load_format = "auto"
        disable_custom_all_reduce = False
        disable_nccl_for_dp_synchronization = False
        tensor_parallel_size = 1
        all2all_backend = "allgather_reducescatter"
        create_engine_config_observed = None

        def create_engine_config(self, usage_context):
            del usage_context
            self.create_engine_config_observed = {
                "worker_cls": self.worker_cls,
                "disable_custom_all_reduce": self.disable_custom_all_reduce,
                "disable_nccl_for_dp_synchronization": (
                    self.disable_nccl_for_dp_synchronization
                ),
                "all2all_backend": self.all2all_backend,
            }
            return SimpleNamespace(
                parallel_config=SimpleNamespace(
                    worker_cls=self.worker_cls,
                    disable_custom_all_reduce=self.disable_custom_all_reduce,
                    disable_nccl_for_dp_synchronization=(
                        self.disable_nccl_for_dp_synchronization
                    ),
                    all2all_backend=self.all2all_backend,
                )
            )

    config = SimpleNamespace(engine_args=EngineArgs())

    assert configure_no_nccl_snapshot_before_engine_config(config) is True
    vllm_config = config.engine_args.create_engine_config(usage_context=object())
    validate_no_nccl_snapshot_config(config.engine_args, vllm_config)

    assert config.engine_args.create_engine_config_observed == {
        "worker_cls": SNAPSHOT_WORKER_CLASS,
        "disable_custom_all_reduce": True,
        "disable_nccl_for_dp_synchronization": True,
        "all2all_backend": DEFAULT_NO_NCCL_ALL2ALL_BACKEND,
    }
    assert config.engine_args.all2all_backend == DEFAULT_NO_NCCL_ALL2ALL_BACKEND
    assert config.engine_args.worker_cls == SNAPSHOT_WORKER_CLASS
    assert os.environ["VLLM_DISABLE_PYNCCL"] == "1"
    assert os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] == "0"
    assert os.environ["VLLM_USE_NCCL_SYMM_MEM"] == "0"
    assert os.environ["VLLM_DISTRIBUTED_USE_SPLIT_GROUP"] == "0"
    assert "VLLM_ALLREDUCE_USE_FLASHINFER" not in os.environ
    assert "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB" not in os.environ


@pytest.mark.parametrize(
    "all2all_backend",
    (
        "allgather_reducescatter",
        "deepep_high_throughput",
        "deepep_low_latency",
        "deepep_v2",
        "flashinfer_nvlink_two_sided",
    ),
)
def test_validate_no_nccl_snapshot_rejects_non_poc_all2all_backend(
    monkeypatch, all2all_backend
):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.setenv(DYN_VLLM_NO_NCCL_SNAPSHOT, "1")
    monkeypatch.delenv(SNAPSHOT_CONTROL_DIR_ENV, raising=False)
    engine_args = SimpleNamespace(worker_cls=SNAPSHOT_WORKER_CLASS)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            worker_cls=SNAPSHOT_WORKER_CLASS,
            disable_custom_all_reduce=True,
            disable_nccl_for_dp_synchronization=True,
            all2all_backend=all2all_backend,
        )
    )

    with pytest.raises(ValueError, match="all2all_backend"):
        validate_no_nccl_snapshot_config(engine_args, vllm_config)


def test_before_engine_config_inherited_env_non_strict_no_snapshot_is_noop(
    monkeypatch,
):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.delenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, raising=False)
    monkeypatch.delenv(DYN_VLLM_NO_NCCL_SNAPSHOT, raising=False)
    monkeypatch.delenv(SNAPSHOT_CONTROL_DIR_ENV, raising=False)
    config = _config(worker_cls="auto")

    assert configure_flashinfer_snapshot_worker_before_engine_config(config) is False
    assert config.engine_args.worker_cls == "auto"


def test_flashinfer_snapshot_worker_rejects_existing_worker_cls(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    config = _config(worker_cls="custom.Worker")

    with pytest.raises(ValueError, match="cannot override existing"):
        configure_flashinfer_snapshot_worker(config)


def test_flashinfer_snapshot_worker_rejects_gms_load_format(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    config = _config(load_format="gms")

    with pytest.raises(ValueError, match="incompatible with --load-format gms"):
        configure_flashinfer_snapshot_worker(config)


def test_validate_flashinfer_snapshot_worker_config_passes(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, "/snapshot-control")
    engine_args = SimpleNamespace(worker_cls=SNAPSHOT_WORKER_CLASS)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(worker_cls=SNAPSHOT_WORKER_CLASS)
    )

    validate_flashinfer_snapshot_worker_config(engine_args, vllm_config)


def test_validate_flashinfer_snapshot_worker_config_skips_without_control_dir(
    monkeypatch,
):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.delenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, raising=False)
    monkeypatch.delenv(DYN_VLLM_NO_NCCL_SNAPSHOT, raising=False)
    monkeypatch.delenv(SNAPSHOT_CONTROL_DIR_ENV, raising=False)
    engine_args = SimpleNamespace(worker_cls="auto")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(worker_cls="vllm.v1.worker.gpu_worker.Worker")
    )

    validate_flashinfer_snapshot_worker_config(engine_args, vllm_config)


def test_validate_snapshot_worker_config_no_nccl_fails_lost_worker(
    monkeypatch,
):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.setenv(DYN_VLLM_NO_NCCL_SNAPSHOT, "1")
    monkeypatch.delenv(SNAPSHOT_CONTROL_DIR_ENV, raising=False)
    engine_args = SimpleNamespace(worker_cls="auto")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(worker_cls="vllm.v1.worker.gpu_worker.Worker")
    )

    with pytest.raises(ValueError, match="parallel_config.worker_cls"):
        validate_flashinfer_snapshot_worker_config(engine_args, vllm_config)


def test_validate_flashinfer_snapshot_worker_config_fails_lost_engine_worker(
    monkeypatch,
):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, "/snapshot-control")
    engine_args = SimpleNamespace(worker_cls="auto")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(worker_cls=SNAPSHOT_WORKER_CLASS)
    )

    with pytest.raises(ValueError, match="engine_args.worker_cls='auto'"):
        validate_flashinfer_snapshot_worker_config(engine_args, vllm_config)


def test_validate_flashinfer_snapshot_worker_config_fails_lost_parallel_worker(
    monkeypatch,
):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.setenv(SNAPSHOT_CONTROL_DIR_ENV, "/snapshot-control")
    engine_args = SimpleNamespace(worker_cls=SNAPSHOT_WORKER_CLASS)
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(worker_cls="auto"))

    with pytest.raises(ValueError, match="parallel_config.worker_cls='auto'"):
        validate_flashinfer_snapshot_worker_config(engine_args, vllm_config)


def test_configure_gms_worker_cls_sets_empty_worker_cls():
    engine_args = SimpleNamespace(worker_cls="auto")

    configure_gms_worker_cls(engine_args)

    assert engine_args.worker_cls == GMS_WORKER_CLASS


def test_configure_gms_worker_cls_rejects_existing_custom_worker_cls():
    engine_args = SimpleNamespace(worker_cls="custom.Worker")

    with pytest.raises(ValueError, match="cannot override existing vLLM worker_cls"):
        configure_gms_worker_cls(engine_args)
