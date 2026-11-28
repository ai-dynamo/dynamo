# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot a model's Python-side state, diff two snapshots, apply a diff.

A snapshot describes, as JSON, every instance attribute of every module and of
the module's ``quant_method`` and attention ``impl`` (the "owners"). Registered
parameters, buffers and submodules are compared by name, class and layout, not
carried: GMS carries the tensors.

``apply`` makes another model match a snapshot for the attributes a diff names,
and reports what it could not reproduce. It never guesses: a value it cannot
carry, or a value the target already holds that differs from both sides, is
reported as fatal. Tensors the target still holds on meta are the caller's to
refuse.

Not carried: the state of any other object an owner holds, such as a nested
object, a numpy scalar, bytes or a complex number (only its class is recorded,
so a change inside it does not show), class attributes and module globals, and
CUDA tensors held by owners (only their layout is compared).
"""

from __future__ import annotations

import base64
import enum
import hashlib
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from types import ModuleType
from typing import Iterator

import torch

OWNERS = ("module", "quant_method", "impl")
ABSENT = {"absent": True}
_SIMPLE = (bool, int, float, str, type(None))
_CONTAINERS = (list, tuple, dict, set, frozenset)
# Flat containers of simple values up to this JSON size travel by value;
# larger or nested ones by digest (a change is detected, not applied).
_MAX_INLINE_JSON = 64 * 1024
# CPU tensors up to this many elements travel by value; larger ones by digest.
_MAX_CPU_VALUES = 4096
_SIGNATURE_DEPTH = 6
_DTYPES = {str(v): v for v in vars(torch).values() if isinstance(v, torch.dtype)}
# torch.nn.Module's hook registries. Their keys are process-global handle ids.
_HOOK_TABLES = frozenset(
    key
    for key, value in vars(torch.nn.Module()).items()
    if isinstance(value, dict) and "hooks" in key
)
# Suffixes of the registries that flag hooks of another registry.
_HOOK_FLAGS = ("_with_kwargs", "_always_called")


def _cls(value) -> str:
    return type(value).__module__ + "." + type(value).__qualname__


def _digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _is_simple(value) -> bool:
    return type(value) in _SIMPLE and not (type(value) is float and math.isnan(value))


def _plain(value):
    """``value`` as exact JSON if it is a flat container of simple values."""
    if type(value) is dict:
        if all(type(k) is str and _is_simple(v) for k, v in value.items()):
            return dict(value)
        return None
    if type(value) in (list, tuple, set, frozenset) and all(map(_is_simple, value)):
        if type(value) in (set, frozenset):
            return sorted(value, key=json.dumps)
        return list(value)
    return None


def _describe_tensor(value: torch.Tensor) -> dict:
    desc = {"t": [list(value.shape), str(value.dtype), value.device.type]}
    if value.device.type != "cpu":
        return desc  # in GMS, or rebuilt by the reader: compared by layout
    if (
        value.layout != torch.strided
        or value.is_quantized
        or not (type(value) is torch.Tensor or isinstance(value, torch.nn.Parameter))
    ):
        desc["opaque"] = True
        return desc
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    desc["sha"] = _digest(raw)
    if value.numel() <= _MAX_CPU_VALUES:
        desc["data"] = base64.b64encode(raw).decode()
    return desc


def _token(value) -> str | None:
    if type(value) is float and math.isnan(value):
        return "float:nan"
    if isinstance(value, enum.Enum):
        cls = type(value)
        return f"enum:{cls.__module__}:{cls.__qualname__}:{value.name}"
    if isinstance(value, torch.dtype):
        return f"dtype:{value}"
    if isinstance(value, torch.device):
        return f"device:{value}"
    return None


def _signature(value, depth: int = 0, active: frozenset = frozenset()):
    """Structural signature of a container, so that a change inside it shows.

    Simple values, tokens, tensor layouts and CPU contents, callables by
    qualified name, other objects by class. Cycle- and depth-safe.
    """
    if _is_simple(value):
        return value
    token = _token(value)
    if token is not None:
        return {"token": token}
    if torch.is_tensor(value):
        return _describe_tensor(value)
    if depth >= _SIGNATURE_DEPTH or id(value) in active:
        return {"o": _cls(value), "depth": depth}
    inner = active | {id(value)}
    if isinstance(value, dict):
        items = sorted(
            json.dumps(
                [_signature(k, depth + 1, inner), _signature(v, depth + 1, inner)],
                sort_keys=True,
            )
            for k, v in value.items()
        )
        return {"o": _cls(value), "items": items}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_signature(x, depth + 1, inner) for x in value]
        if isinstance(value, (set, frozenset)):
            items = sorted(json.dumps(x, sort_keys=True) for x in items)
        return {"o": _cls(value), "items": items}
    if callable(value):
        func = getattr(value, "__func__", value)
        name = getattr(func, "__qualname__", None)
        if isinstance(name, str):
            module = getattr(func, "__module__", "")
            return {"o": _cls(value), "fn": f"{module}.{name}"}
    return {"o": _cls(value)}


def describe(value) -> dict:
    """JSON description of one attribute value."""
    if torch.is_tensor(value):
        return _describe_tensor(value)
    if _is_simple(value):
        return {"v": value}
    token = _token(value)
    if token is not None:
        return {"token": token}
    if isinstance(value, _CONTAINERS):
        plain = _plain(value)
        if plain is not None:
            encoded = json.dumps(plain, sort_keys=True)
            if len(encoded) <= _MAX_INLINE_JSON:
                return {"v": plain, "seq": type(value).__name__}
        else:
            encoded = json.dumps(_signature(value), sort_keys=True)
        return {"o": _cls(value), "sig": _digest(encoded.encode())}
    return {"o": _cls(value)}


def _describe_attr(owner, key: str) -> dict:
    """Description of the attribute ``key`` that ``owner`` holds."""
    value = vars(owner)[key]
    if (
        key in _HOOK_TABLES
        and type(value) is OrderedDict
        and value
        and isinstance(owner, torch.nn.Module)
    ):
        value = _by_position(owner, key, value)
    return describe(value)


def _by_position(module: torch.nn.Module, key: str, table: OrderedDict) -> OrderedDict:
    """A hook registry keyed by each hook's position instead of its handle id.

    The registries that flag hooks (``*_with_kwargs``, ``*_always_called``)
    get the positions in the registry they index. A flag whose hook that
    registry no longer has is left out; torch ignores it too.
    """
    for flag in _HOOK_FLAGS:
        key = key.removesuffix(flag)
    position = {handle: i for i, handle in enumerate(vars(module).get(key, ()))}
    return OrderedDict(
        (position[handle], hook) for handle, hook in table.items() if handle in position
    )


def _same(a: dict, b: dict) -> bool:
    """Whether two descriptions are equal. ``==`` takes 1, 1.0 and True for
    one value, and 0.0 for -0.0, so inline values are also compared as text."""
    if a != b:
        return False
    if "v" not in a:
        return True
    left, right = a["v"], b["v"]
    if type(left) in (list, dict):
        return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
    return repr(left) == repr(right)


def _stable(desc: dict) -> dict:
    """The part of a description that two processes building the same model
    agree on: no ``ref``, device index, tensor device or contents, or
    container digest."""
    if "t" in desc:
        return {"t": desc["t"][:2]}
    if "sig" in desc:
        return {"o": desc["o"]}
    kind = _device_type(desc.get("token"))
    if kind is not None:
        return {"token": "device:" + kind}
    return _bare(desc)


def owners(module: torch.nn.Module) -> Iterator[tuple[str, object]]:
    """The module and the owners it holds, as (owner name, object)."""
    yield "module", module
    for name in OWNERS[1:]:
        owner = vars(module).get(name)
        if owner is not None:
            yield name, owner


def snapshot(model: torch.nn.Module) -> dict:
    """``{"<module>|<owner>": {attr: description}}`` for every module.

    An object reachable from two attributes of one module's owners also
    records ``ref``, the other attribute, so that ``apply`` can re-link an
    object that loading wired into a new place, and can tell one object held
    twice from two objects of one class.
    """
    state = {}
    for name, module in model.named_modules():
        found = list(owners(module))
        rows, places = {}, {}
        for owner_name, owner in found:
            rows[owner_name] = {
                key: _describe_attr(owner, key)
                for key in vars(owner)
                if not key.startswith("__")
            }
        # Deepest owner first, so a ref points at the object's home.
        for owner_name, owner in sorted(found, key=lambda o: -OWNERS.index(o[0])):
            for key, value in vars(owner).items():
                if "o" in rows[owner_name].get(key, {}):
                    places.setdefault(id(value), []).append(f"{owner_name}.{key}")
        for owner_name, owner in found:
            for key, desc in rows[owner_name].items():
                if "o" in desc:
                    here = f"{owner_name}.{key}"
                    other = [p for p in places[id(vars(owner)[key])] if p != here]
                    if other:
                        desc["ref"] = other[0]
            state[f"{name}|{owner_name}"] = rows[owner_name]
    return state


def diff(construct: dict, final: dict) -> dict:
    """``{"<path>": {attr: construct description or ABSENT}}`` for every
    attribute that loading added, changed or deleted."""
    changed = {}
    for path in set(construct) | set(final):
        before, after = construct.get(path, {}), final.get(path, {})
        row = {
            key: before.get(key, ABSENT)
            for key in set(before) | set(after)
            if not _same(before.get(key, ABSENT), after.get(key, ABSENT))
        }
        if row:
            changed[path] = row
    return changed


def resolve_owner(model: torch.nn.Module, path: str):
    """The owner object at ``<module>|<owner>``, or None if the model has no
    such module or the module does not hold that owner."""
    name, _, owner_name = path.rpartition("|")
    try:
        module = model.get_submodule(name)
    except AttributeError:
        return None
    return module if owner_name == "module" else vars(module).get(owner_name)


def _tensor(desc: dict) -> torch.Tensor:
    shape, dtype_name, _ = desc["t"]
    dtype = _DTYPES.get(dtype_name)
    if dtype is None:
        raise RuntimeError(f"Load diff has an unknown tensor dtype: {dtype_name}")
    raw = bytearray(base64.b64decode(desc["data"]))
    if _digest(bytes(raw)) != desc["sha"]:
        raise RuntimeError("Load diff CPU tensor data is corrupt")
    if not raw:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(shape)


_INLINE = {
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
}


def _value(desc: dict):
    """The value an inline description carries. A container is a new object,
    so the target does not share it with the snapshot."""
    kind = _INLINE.get(desc.get("seq"))
    return kind(desc["v"]) if kind is not None else desc["v"]


class _Unresolvable(Exception):
    pass


def _token_value(token: str):
    """The value a token names. Enum classes are looked up only in modules the
    process has already imported; nothing is imported on a token's behalf."""
    kind, _, rest = token.partition(":")
    if kind == "float" and rest == "nan":
        return float("nan")
    if kind == "dtype" and rest in _DTYPES:
        return _DTYPES[rest]
    if kind == "enum":
        module_name, qualname, member = rest.split(":")
        cls = sys.modules.get(module_name)
        for part in qualname.split("."):
            cls = vars(cls).get(part) if isinstance(cls, (type, ModuleType)) else None
        if isinstance(cls, type) and issubclass(cls, enum.Enum):
            if member in cls.__members__:
                return cls[member]
    raise _Unresolvable(token)


def _device_type(token) -> str | None:
    if isinstance(token, str) and token.startswith("device:"):
        return token[len("device:") :].partition(":")[0]
    return None


@dataclass
class Report:
    """What ``apply`` did, and what it could not reproduce."""

    applied: list = field(default_factory=list)
    restored: list = field(default_factory=list)
    linked: list = field(default_factory=list)
    deleted: list = field(default_factory=list)
    # Differences in attributes that loading did not change: logged only.
    nondeterministic: list = field(default_factory=list)
    # Fatal: loading produced it and the target does not have it.
    conflicts: list = field(default_factory=list)
    unapplied_tokens: list = field(default_factory=list)
    missing_objects: list = field(default_factory=list)
    missing_tensors: list = field(default_factory=list)
    missing_owners: list = field(default_factory=list)

    @property
    def fatal(self) -> list:
        return (
            self.missing_owners
            + self.missing_objects
            + self.missing_tensors
            + self.conflicts
            + self.unapplied_tokens
        )

    def summary(self) -> str:
        return (
            f"applied {len(self.applied)} load-time values, restored "
            f"{len(self.restored)} CPU tensors, linked {len(self.linked)} objects, "
            f"deleted {len(self.deleted)}; nondeterministic "
            f"{len(self.nondeterministic)}; conflicts {len(self.conflicts)}, "
            f"unapplied tokens {len(self.unapplied_tokens)}, missing objects "
            f"{len(self.missing_objects)}, tensors {len(self.missing_tensors)}, "
            f"owners {len(self.missing_owners)}"
        )


def apply(model: torch.nn.Module, final: dict, changed: dict) -> Report:
    """Make ``model`` match the ``final`` snapshot where loading changed it.

    For each attribute in ``changed``: if the model still has the value it had
    after construction, the final value is applied when the description
    carries it (simple values, flat containers, tokens, small CPU tensors) or
    when the object can be re-linked by ``ref``, and an attribute that loading
    deleted is deleted. A value the model holds that differs from both the
    construction value and the final one is a conflict. Anything else that
    loading produced and the model lacks is fatal, including one object held
    under two attributes where the model holds two. Differences in attributes
    that loading did not change are only reported. A tensor the model holds on
    meta that does not travel by value is left there and not reported.
    """
    report = Report()
    # What the model holds where loading changed something, taken before any
    # tensor is restored in place: whether a value is still what construction
    # made must not depend on a tensor that shares its storage.
    held = {}
    for path, loaded in changed.items():
        owner = resolve_owner(model, path)
        if owner is not None:
            held[path] = owner, _held(owner, loaded)
    # Restored first: an attribute that shares the tensor's storage, or holds
    # the tensor in a container, is compared with the final values.
    storages = _restore_in_place(final, changed, held, report)
    for path, want_row in final.items():
        owner = resolve_owner(model, path)
        loaded = changed.get(path, {})
        if owner is None:
            report.missing_owners.append(path)
            continue
        attrs = vars(owner)
        held_owner, mine = held.get(path, (owner, {}))
        if held_owner is not owner:
            # An earlier row re-linked this owner to another object.
            mine = _held(owner, loaded)
        for key, want in want_row.items():
            have = _describe_attr(owner, key) if key in attrs else ABSENT
            if _same(have, _bare(want)):
                continue
            where = f"{path}.{key}"
            value = attrs.get(key)
            if "t" in want and torch.is_tensor(value) and value.is_meta:
                # A constructor-made CPU tensor that meta construction left on
                # meta. Restored if it travels by value; else left for the
                # caller, which refuses a model with tensors still on meta.
                if "data" in want:
                    setattr(owner, key, _tensor(want))
                    report.restored.append(where)
                continue
            if key not in loaded:
                report.nondeterministic.append((where, have, want))
                continue
            rebuilt = key in mine and not _same(mine[key], _bare(loaded[key]))
            if "v" in want:
                if rebuilt:
                    report.conflicts.append((where, have, want))
                    continue
                setattr(owner, key, _value(want))
                report.applied.append(where)
            elif "token" in want:
                kind = _device_type(want["token"])
                if kind is not None:
                    # Device indices are process-local; the type must match.
                    if _device_type(have.get("token")) != kind:
                        report.conflicts.append((where, have, want))
                    continue
                if rebuilt:
                    report.conflicts.append((where, have, want))
                    continue
                try:
                    setattr(owner, key, _token_value(want["token"]))
                    report.applied.append(where)
                except _Unresolvable:
                    report.unapplied_tokens.append((where, have, want["token"]))
            elif "o" in want:
                if key in mine and _replaced(mine[key], loaded[key]):
                    report.conflicts.append((where, have, want))
                elif "ref" in want and _relink(model, owner, path, key, want):
                    report.linked.append(f"{where} -> {want['ref']}")
                else:
                    report.missing_objects.append((where, have, want))
            elif "t" in want:
                if rebuilt or "data" not in want:
                    report.missing_tensors.append((where, have, want))
                elif torch.is_tensor(value) and have["t"] == want["t"]:
                    # Not restored in place: the tensor was restored for a
                    # holder with another final value. It does not get a
                    # tensor of its own, which would split it from the
                    # holders it still shares with.
                    report.missing_tensors.append((where, have, want))
                else:
                    setattr(owner, key, _tensor(want))
                    report.restored.append(where)
        for key, before in loaded.items():
            if key in want_row or before == ABSENT or key not in mine:
                continue
            if _replaced(mine[key], before):
                report.conflicts.append((f"{path}.{key}", mine[key], ABSENT))
            else:
                delattr(owner, key)
                report.deleted.append(f"{path}.{key}")
    if storages:
        report.conflicts.extend(_other_holders(model, final, storages))
    _check_shared(model, final, changed, report)
    return report


def _held(owner, loaded: dict) -> dict:
    """Descriptions of the attributes in ``loaded`` that ``owner`` holds."""
    attrs = vars(owner)
    return {key: _describe_attr(owner, key) for key in loaded if key in attrs}


def _restore_in_place(final: dict, changed: dict, held: dict, report: Report) -> set:
    """Restore the small CPU tensors that loading changed, where the model
    still holds the tensor construction made, and return their storages.

    In place, so that every holder of the tensor sees the values.
    """
    queued = {}
    for path, (owner, mine) in held.items():
        attrs = vars(owner)
        for key, have in mine.items():
            want = final.get(path, {}).get(key, ABSENT)
            value = attrs[key]
            if "data" not in want or have.get("t") != want["t"] or id(value) in queued:
                continue
            if _same(have, _bare(changed[path][key])):
                queued[id(value)] = (value, _tensor(want))
                report.restored.append(f"{path}.{key}")
    with torch.no_grad():
        for value, source in queued.values():
            value.copy_(source)
    return {_storage(value) for value, _ in queued.values()}


def _replaced(have: dict, before: dict) -> bool:
    """Whether the target's rebuild replaced what construction made, as far
    as two processes can tell: see ``_stable``."""
    return not _same(_stable(have), _stable(before))


def _storage(value) -> int | None:
    """Identity of the storage behind a CPU tensor, else None."""
    if (
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and value.layout == torch.strided
    ):
        return value.untyped_storage()._cdata
    return None


def _other_holders(model: torch.nn.Module, final: dict, storages: set) -> list:
    """Attributes that share storage with a tensor restored in place, as the
    same tensor or as a view, but whose final value is another one: loading
    gave only some of the holders a new tensor."""
    conflicts = []
    for path, want_row in final.items():
        owner = resolve_owner(model, path)
        if owner is None:
            continue
        for key, value in vars(owner).items():
            storage = _storage(value)
            if storage is not None and storage in storages and key in want_row:
                have = describe(value)
                if not _same(have, _bare(want_row[key])):
                    conflicts.append((f"{path}.{key}", have, want_row[key]))
    return conflicts


def _bare(desc: dict) -> dict:
    """``desc`` without ``ref``, which ``describe`` alone cannot produce."""
    return {k: v for k, v in desc.items() if k != "ref"}


def _ref_place(model, path: str, want: dict) -> tuple | None:
    """The owner and attribute that ``want["ref"]`` names, if the model has
    them."""
    ref_owner, _, ref_key = want["ref"].partition(".")
    source = resolve_owner(model, path.rpartition("|")[0] + "|" + ref_owner)
    if source is None or ref_key not in vars(source):
        return None
    return source, ref_key


def _relink(model, owner, path: str, key: str, want: dict) -> bool:
    """Point ``owner.key`` at the object its ``ref`` names, if that matches."""
    place = _ref_place(model, path, want)
    if place is None or not _same(_describe_attr(*place), _bare(want)):
        return False
    setattr(owner, key, vars(place[0])[place[1]])
    return True


def _check_shared(
    model: torch.nn.Module, final: dict, changed: dict, report: Report
) -> None:
    """An object the writer held under two attributes must be one object here.

    Such an object is described by its class, so two objects of that class
    pass for the writer's one. Nothing is re-linked here: the description
    does not say which of the two the writer kept. Where loading made the
    link, two objects are fatal; elsewhere the difference is only reported.
    """
    for path, want_row in final.items():
        owner = resolve_owner(model, path)
        if owner is None:
            continue
        attrs = vars(owner)
        loaded = changed.get(path, {})
        for key, want in want_row.items():
            if "ref" not in want or key not in attrs:
                continue
            have = _describe_attr(owner, key)
            if not _same(have, _bare(want)):
                continue  # reported by apply
            place = _ref_place(model, path, want)
            if place is not None and vars(place[0])[place[1]] is attrs[key]:
                continue
            entry = (f"{path}.{key}", have, want)
            if key in loaded:
                report.missing_objects.append(entry)
            else:
                report.nondeterministic.append(entry)
