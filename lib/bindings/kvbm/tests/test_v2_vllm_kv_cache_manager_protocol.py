# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Drift test: assert that ``VllmKvCacheManagerProtocol`` still matches the
real ``vllm.v1.core.kv_cache_manager.KVCacheManager`` exactly.

Compares full signatures (parameter names, order, defaults, type
annotations, and return types) for every method on the Protocol against
the corresponding method on the real class, and also asserts that every
public method on the real class is present on the Protocol. Skipped if
vLLM is not installed.

When this test fails, it means vLLM's upstream cache manager API moved:

1. Read the diff reported by the test.
2. Decide whether the change is a rename, a retype, a new method, a
   removed method, etc.
3. Update ``kvbm/v2/vllm/kv_cache_manager_protocol.py`` and the
   implementing Rust/Python shim (``kv_cache_manager.py`` + the Rust
   ``RustKvCacheManager``) to match.
4. Bump ``PINNED_VLLM_VERSION`` in the Protocol module if the supported
   version floor moved.
"""

from __future__ import annotations

import importlib.util
import inspect
from typing import Any, get_type_hints

import pytest

VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

pytestmark = pytest.mark.skipif(
    not VLLM_AVAILABLE,
    reason="vLLM is not installed; drift test requires the real KVCacheManager",
)


# Methods that are intentionally not mirrored on the Protocol because
# they are private (leading underscore) or are dunder methods inherited
# from ``object``. The drift test uses this as the "reverse check" —
# anything on vLLM's class that isn't here or on the Protocol is a
# drift failure.
_IGNORED_ATTRIBUTE_NAMES = frozenset(
    {
        # All dunders come from object and do not need to be mirrored.
    }
)


def _public_method_names(cls: type) -> set[str]:
    """Names of all non-dunder public attributes that are callables or
    properties on ``cls``."""
    result: set[str] = set()
    for name, value in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if name in _IGNORED_ATTRIBUTE_NAMES:
            continue
        if inspect.isfunction(value) or inspect.ismethod(value) or isinstance(
            value, property
        ):
            result.add(name)
    return result


def _protocol_method_names(proto: type) -> set[str]:
    """Names of all explicitly declared methods/properties on a
    ``typing.Protocol`` subclass."""
    result: set[str] = set()
    for name, value in vars(proto).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(value) or isinstance(value, property):
            result.add(name)
    return result


def _normalize_hints(
    obj: Any, include_extras: bool = False
) -> dict[str, Any]:
    """Resolve a function's type annotations to real types."""
    try:
        return get_type_hints(obj, include_extras=include_extras)
    except Exception as exc:  # pragma: no cover - best effort
        return {"__unresolved__": repr(exc)}


def _compare_signature(
    method_name: str, proto_fn: Any, real_fn: Any
) -> list[str]:
    """Return a list of human-readable drift descriptions, empty on match."""
    errors: list[str] = []

    proto_sig = inspect.signature(proto_fn)
    real_sig = inspect.signature(real_fn)

    proto_params = list(proto_sig.parameters.values())
    real_params = list(real_sig.parameters.values())

    # Strip `self` from both sides.
    if proto_params and proto_params[0].name == "self":
        proto_params = proto_params[1:]
    if real_params and real_params[0].name == "self":
        real_params = real_params[1:]

    proto_hints = _normalize_hints(proto_fn)
    real_hints = _normalize_hints(real_fn)

    if len(proto_params) != len(real_params):
        errors.append(
            f"{method_name}: parameter count mismatch — "
            f"protocol has {len(proto_params)} "
            f"({[p.name for p in proto_params]}), "
            f"vLLM has {len(real_params)} "
            f"({[p.name for p in real_params]})"
        )
        return errors

    for idx, (p, r) in enumerate(zip(proto_params, real_params)):
        if p.name != r.name:
            errors.append(
                f"{method_name}: param #{idx} name mismatch — "
                f"protocol={p.name!r} vs vLLM={r.name!r}"
            )
            continue
        if p.default != r.default:
            errors.append(
                f"{method_name}: param {p.name!r} default mismatch — "
                f"protocol={p.default!r} vs vLLM={r.default!r}"
            )
        proto_hint = proto_hints.get(p.name, inspect.Parameter.empty)
        real_hint = real_hints.get(p.name, inspect.Parameter.empty)
        if proto_hint != real_hint:
            errors.append(
                f"{method_name}: param {p.name!r} type mismatch — "
                f"protocol={proto_hint!r} vs vLLM={real_hint!r}"
            )

    proto_ret = proto_hints.get("return", inspect.Parameter.empty)
    real_ret = real_hints.get("return", inspect.Parameter.empty)
    if proto_ret != real_ret:
        errors.append(
            f"{method_name}: return type mismatch — "
            f"protocol={proto_ret!r} vs vLLM={real_ret!r}"
        )

    return errors


def _property_errors(name: str, proto_prop: property, real_prop: property) -> list[str]:
    errors: list[str] = []
    if not isinstance(real_prop, property):
        errors.append(
            f"{name}: protocol declares a property but vLLM does not "
            f"(got {type(real_prop).__name__})"
        )
        return errors
    if proto_prop.fget is None or real_prop.fget is None:
        return errors
    return _compare_signature(name, proto_prop.fget, real_prop.fget)


def test_protocol_matches_vllm_kv_cache_manager() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    from kvbm.v2.vllm.kv_cache_manager_protocol import (
        PINNED_VLLM_VERSION,
        VllmKvCacheManagerProtocol,
    )

    all_errors: list[str] = []

    # Forward direction: every method on the protocol must match.
    for name in sorted(_protocol_method_names(VllmKvCacheManagerProtocol)):
        proto_attr = vars(VllmKvCacheManagerProtocol)[name]
        real_attr = inspect.getattr_static(KVCacheManager, name, None)

        if real_attr is None:
            all_errors.append(
                f"{name}: declared on protocol but missing from "
                f"vllm.v1.core.kv_cache_manager.KVCacheManager"
            )
            continue

        if isinstance(proto_attr, property):
            all_errors.extend(_property_errors(name, proto_attr, real_attr))
            continue

        if isinstance(real_attr, property):
            all_errors.append(
                f"{name}: vLLM declares a property but protocol declares "
                f"a plain method"
            )
            continue

        all_errors.extend(_compare_signature(name, proto_attr, real_attr))

    # Reverse direction: every public method on vLLM's class must exist
    # on the protocol — catches upstream additions.
    proto_names = _protocol_method_names(VllmKvCacheManagerProtocol)
    for name in sorted(_public_method_names(KVCacheManager)):
        if name not in proto_names:
            all_errors.append(
                f"{name}: present on vLLM KVCacheManager but missing from "
                f"VllmKvCacheManagerProtocol"
            )

    # Constructor drift is checked explicitly because __init__ is a dunder
    # and does not appear in `_public_method_names`.
    all_errors.extend(
        _compare_signature(
            "__init__",
            VllmKvCacheManagerProtocol.__init__,
            KVCacheManager.__init__,
        )
    )

    if all_errors:
        from vllm.version import __version__ as installed_vllm_version

        header = (
            f"VllmKvCacheManagerProtocol drift detected.\n"
            f"  Protocol pinned to vLLM {PINNED_VLLM_VERSION}\n"
            f"  Installed vLLM       : {installed_vllm_version}\n"
            f"Mismatches:\n  - "
        )
        pytest.fail(header + "\n  - ".join(all_errors))


def test_protocol_is_runtime_checkable() -> None:
    """Sanity: the Protocol is marked ``@runtime_checkable`` so Python
    code can do an ``isinstance`` check against concrete impls."""
    from kvbm.v2.vllm.kv_cache_manager_protocol import (
        VllmKvCacheManagerProtocol,
    )

    # Accessing this attribute triggers the runtime_checkable-only behavior.
    assert hasattr(VllmKvCacheManagerProtocol, "__runtime_checkable__") or getattr(
        VllmKvCacheManagerProtocol, "_is_runtime_protocol", False
    )
