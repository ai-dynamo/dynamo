# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install a stub for ``dynamo._core`` when the native binding is unavailable.

The α-class testbed deliberately avoids the Rust runtime (no NATS, no etcd,
no Rust scheduler). The only reason it would fail to import without this
shim is that ``dynamo.planner/__init__.py`` eagerly pulls in Kubernetes
connectors → ``dynamo.runtime.logging`` → ``dynamo._core.log_message``.

This module installs a minimal in-memory stand-in for ``dynamo._core`` so
the planner package imports cleanly on a developer laptop without the
maturin-built extension. If the real ``dynamo._core`` is available (CI,
production), the stub is **not** installed.

Hardware-safety remains intact: nothing in the stub can talk to a real
NATS/etcd cluster or NVML; calls into any stubbed symbol log and return
sentinel values.

Idempotent — safe to import / call multiple times.
"""
from __future__ import annotations

import sys
import types


def _ensure_dynamo_runtime_on_path() -> None:
    """Add ``lib/bindings/python/src`` to ``sys.path`` so the pure-Python
    ``dynamo.runtime`` package is importable.

    ``dynamo.runtime`` is not installed as a distribution package on dev
    boxes — it lives under ``lib/bindings/python/src`` in the repo.  Its
    only native dependency is ``dynamo._core.log_message``, which is already
    stubbed to a no-op, so the module imports cleanly once the path is
    present.

    Idempotent — skips if ``dynamo.runtime`` is already importable or if the
    path is already in ``sys.path``.
    """
    if "dynamo.runtime" in sys.modules:
        return
    # Walk up from this file's location to the repo root, then locate the
    # pure-Python bindings.  Path: <repo>/components/src/dynamo/planner/tests/testbed/
    # → up 6 levels → <repo>/ → lib/bindings/python/src
    import pathlib
    this_file = pathlib.Path(__file__).resolve()
    # Ascend: testbed → tests → planner → dynamo → src → components → <repo>
    repo_root = this_file.parents[6]
    bindings_src = repo_root / "lib" / "bindings" / "python" / "src"
    if bindings_src.is_dir():
        bindings_str = str(bindings_src)
        if bindings_str not in sys.path:
            sys.path.insert(0, bindings_str)


def install_stub_if_needed() -> bool:
    """Install the stub iff the real ``dynamo._core`` cannot be imported.

    Returns True if a stub was installed (or already in place); False if the
    real binding is loaded.
    """
    if "dynamo._core" in sys.modules:
        # Some other code path beat us to it. Honor whatever is there.
        return getattr(sys.modules["dynamo._core"], "_TESTBED_STUB", False)

    try:
        import dynamo._core  # noqa: F401  — real binding present
        return False
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # Build the stub module.
    # ------------------------------------------------------------------
    stub = types.ModuleType("dynamo._core")
    stub._TESTBED_STUB = True
    stub.__doc__ = (
        "Testbed stand-in for dynamo._core. Installed by "
        "dynamo.planner.tests.testbed._runtime_stub when the native binding "
        "is unavailable. Any call to a stubbed symbol raises immediately — "
        "the testbed must never reach the runtime layer."
    )

    # --- Functions ---
    def log_message(*args, **kwargs):
        # Rust's env_logger sink; the Python LogHandler forwards records here.
        # Silently drop — tests don't care about runtime log output.
        return None

    stub.log_message = log_message

    def _unimplemented(name):
        def _raise(*_args, **_kwargs):
            raise NotImplementedError(
                f"dynamo._core.{name} is stubbed for the testbed — "
                f"the α-class harness must not reach the runtime layer. "
                f"If you need this in a γ-class scenario, build the real "
                f"binding (maturin develop in lib/bindings/python)."
            )
        return _raise

    # --- Classes consumed at module-import time by dynamo.runtime ---
    # These are *imported* but never *instantiated* during testbed runs;
    # making them stub classes satisfies the import-time binding check.
    for _name in (
        "Client", "Context", "DistributedRuntime", "Endpoint",
        # dynamo.planner.connectors.virtual imports this at top level
        "VirtualConnectorCoordinator",
        # dynamo.llm imports (only needed for γ; harmless to stub)
        "AicPerfConfig", "EngineType", "EntrypointArgs",
        "FpmDirectPublisher", "FpmEventRelay", "FpmEventSubscriber",
        "HttpAsyncEngine", "HttpService", "KserveGrpcService",
        "KvEventPublisher", "KvRouter", "KvRouterConfig",
        "LoRADownloader", "MediaDecoder", "MediaFetcher",
        "MockEngineArgs", "ModelCardInstanceId", "ModelInput",
        "ModelRuntimeConfig", "ModelType", "OverlapScores",
        "PlannerReplayBridge", "PythonAsyncEngine", "RadixTree",
        "ReasoningConfig", "RouterConfig", "RouterMode", "SglangArgs",
        "WorkerMetricsPublisher", "ModelDeploymentCard",
        # Exceptions used in dynamo.llm.exceptions
        "Cancelled", "CannotConnect", "ConnectionTimeout",
        "Disconnected", "DynamoException", "EngineShutdown",
        "InvalidArgument", "StreamIncomplete", "Unknown",
    ):
        setattr(stub, _name, type(_name, (), {"__init__": _unimplemented(_name)}))

    # --- Free functions used at module-import time ---
    for _name in (
        "compute_block_hash_for_seq", "fetch_model", "lora_name_to_id",
        "make_engine", "register_model", "run_input", "run_kv_indexer",
        "run_mocker_trace_replay", "unregister_model",
    ):
        setattr(stub, _name, _unimplemented(_name))

    sys.modules["dynamo._core"] = stub

    # With dynamo._core now stubbed, the pure-Python dynamo.runtime package
    # can import cleanly — but only if its source directory is on sys.path.
    # Add it now (idempotent; no-ops if real binding already installed it).
    _ensure_dynamo_runtime_on_path()

    return True


def install_predictor_deps_stub_if_needed() -> None:
    """Stub heavy ML predictor deps (``pmdarima``/``filterpy``/``prophet``).

    ``dynamo.planner.core.load.predictors`` imports them at module load,
    even though the testbed only ever uses the ``constant`` predictor.
    These deps are large and not part of a standard developer install,
    so we provide do-nothing stubs to unblock import.

    Idempotent. If the real package is available, nothing happens.
    """
    if "pmdarima" not in sys.modules:
        try:
            import pmdarima  # noqa: F401
        except ImportError:
            pmd = types.ModuleType("pmdarima")
            pmd._TESTBED_STUB = True

            def _no_predictor(*_a, **_kw):
                raise NotImplementedError(
                    "pmdarima is stubbed for the testbed (constant predictor only)."
                )

            pmd.auto_arima = _no_predictor
            pmd.ARIMA = type("ARIMA", (), {"__init__": _no_predictor})
            sys.modules["pmdarima"] = pmd

    if "filterpy" not in sys.modules:
        try:
            import filterpy.kalman  # noqa: F401
        except ImportError:
            fp = types.ModuleType("filterpy")
            fp.__path__ = []  # mark as package
            fp_kal = types.ModuleType("filterpy.kalman")
            fp_kal._TESTBED_STUB = True
            fp_kal.KalmanFilter = type(
                "KalmanFilter",
                (),
                {"__init__": lambda *a, **kw: (_ for _ in ()).throw(
                    NotImplementedError(
                        "filterpy is stubbed for the testbed (constant predictor only)."
                    )
                )},
            )
            sys.modules["filterpy"] = fp
            sys.modules["filterpy.kalman"] = fp_kal
            fp.kalman = fp_kal

    if "prophet" not in sys.modules:
        try:
            import prophet  # noqa: F401
        except ImportError:
            pr = types.ModuleType("prophet")
            pr._TESTBED_STUB = True
            pr.Prophet = type(
                "Prophet",
                (),
                {"__init__": lambda *a, **kw: (_ for _ in ()).throw(
                    NotImplementedError(
                        "prophet is stubbed for the testbed (constant predictor only)."
                    )
                )},
            )
            sys.modules["prophet"] = pr
