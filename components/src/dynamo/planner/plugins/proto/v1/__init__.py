# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated protobuf stubs for the planner plugin protocol v1.

The ``plugin_pb2.py``, ``plugin_pb2_grpc.py``, and ``plugin_pb2.pyi``
modules in this directory are generated from ``plugin.proto`` and
**checked into git** so test/build images don't need ``grpcio-tools``
just to import the package. When you edit ``plugin.proto`` regenerate
with the protoc command in ``README.md`` (and re-prepend the SPDX header).
A drift-catching CI check is deferred to a follow-up build infra PR.
"""

# ---------------------------------------------------------------------------
# Collection-time compatibility shim.
#
# CI's pre-commit ``pytest-marker-report`` collects test modules without
# running their bodies. Several tests import ``plugin_pb2`` /
# ``plugin_pb2_grpc`` at module top via ``from
# dynamo.planner.plugins.proto.v1 import plugin_pb2``. When the generated
# stubs are *not on disk* (which is the case in a fresh pre-commit virtualenv
# that hasn't yet executed the container build's ``protoc`` step), that
# import raises ``ImportError`` and the test fails *collection*, taking the
# whole hook down with it even though no test body would have run.
#
# Provide empty module-shaped placeholders in ``sys.modules`` so the
# ``from . import plugin_pb2`` attribute lookup succeeds at collection time.
# These shims are NEVER reached at runtime in any normal deployment because:
#   1. Production containers regenerate plugin_pb2 at install time before
#      Python starts → ``importlib.import_module`` succeeds → the real
#      modules are placed in sys.modules first → our shim block is a no-op.
#   2. Developer local runs require the same regen via ``README.md`` protoc
#      command before tests can pass; the shim only patches the
#      *collection* import-path for ``--collect-only`` invocations.
import importlib
import sys
import types as _types


class _PlaceholderModule(_types.ModuleType):
    """Module placeholder whose attribute lookups synthesize a dummy
    class on demand. Lets ``_proto_bridge`` 's module-top lookup table
    (``pb.RegisterRequest`` etc.) survive collection-time import even
    when the generated proto stubs aren't on disk. Attempting to
    *use* one of the dummy classes at runtime gives a useful error
    message pointing at the missing ``protoc`` step.
    """

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        msg = (
            f"{self.__name__}.{name} not available — generated proto stub "
            "missing. Run protoc per components/src/dynamo/planner/plugins/"
            "proto/v1/README.md to generate plugin_pb2.py / plugin_pb2_grpc.py."
        )
        dummy = type(name, (), {"__init__": lambda self, *a, **kw: (_ for _ in ()).throw(RuntimeError(msg))})  # type: ignore[arg-type]
        dummy.__module__ = self.__name__
        # Cache so identity holds for repeated attribute access.
        setattr(self, name, dummy)
        return dummy


for _stub_name in ("plugin_pb2", "plugin_pb2_grpc"):
    _fq = f"{__name__}.{_stub_name}"
    if _fq in sys.modules:
        continue
    try:
        importlib.import_module(_fq)
    except (ImportError, AttributeError):
        # Two failure modes both land us here:
        #   - ImportError: generated stub not on disk yet (pre-protoc env)
        #   - AttributeError: stub IS on disk but its module-top code
        #     (e.g. ``GRPC_VERSION = grpc.__version__`` in plugin_pb2_grpc.py)
        #     crashes because grpc/protobuf are themselves stubbed by
        #     pytest-marker-report's ``--collect-only`` runner, which
        #     strips dunder attributes from stubbed modules.
        # In either case install a placeholder so ``from <pkg> import
        # plugin_pb2`` succeeds at collection time AND the
        # ``_proto_bridge`` module-top lookup table can resolve attributes
        # like ``plugin_pb2.RegisterRequest`` (synthesised on demand by
        # ``_PlaceholderModule.__getattr__``).
        _placeholder = _PlaceholderModule(_fq)
        _placeholder.__doc__ = (
            f"Pre-generation placeholder for {_fq}. Run protoc per "
            "``components/src/dynamo/planner/plugins/proto/v1/README.md`` "
            "to generate the real module."
        )
        sys.modules[_fq] = _placeholder

del importlib, sys, _types, _stub_name, _fq
