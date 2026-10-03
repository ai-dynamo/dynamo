# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical, hashable serialization of policy specs and cells.

Canonical JSON: sorted keys, no whitespace, floats in ``repr`` form, NaN/inf rejected.

Canonical YAML: block style with sorted keys; lists of scalars in flow style; strings as
double-quoted JSON strings; floats in ``repr`` form, except that an exponent form without a
decimal point gets ``.0`` inserted (``1e-05`` -> ``1.0e-05``) so that both YAML 1.1 (PyYAML) and
YAML 1.2 (serde_yaml) read it back as the same float. Integers and booleans stay as they are.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any


def _check_floats(value: Any, where: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where}: non-finite float {value!r} is not canonicalizable")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where}: mapping key {key!r} is not a string")
            _check_floats(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_floats(item, f"{where}[{index}]")


def canonical_json(value: Any) -> str:
    _check_floats(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def float_text(value: float) -> str:
    text = repr(float(value))
    if "e" in text and "." not in text:
        mantissa, exponent = text.split("e", 1)
        text = f"{mantissa}.0e{exponent}"
    return text


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return float_text(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"unsupported scalar {value!r}")


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _flow(value: Any) -> str:
    if _is_scalar(value):
        return _scalar(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_flow(item) for item in value) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = ", ".join(f"{json.dumps(k)}: {_flow(value[k])}" for k in sorted(value))
        return "{" + items + "}"
    raise TypeError(f"unsupported value {value!r}")


def _block(value: Any, indent: int) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key in sorted(value):
            item = value[key]
            label = (
                key
                if key.replace("_", "").replace("-", "").isalnum()
                else json.dumps(key)
            )
            if isinstance(item, dict) and item:
                lines.append(f"{pad}{label}:")
                lines.extend(_block(item, indent + 2))
            elif isinstance(item, (list, tuple)) and any(
                isinstance(x, dict) for x in item
            ):
                lines.append(f"{pad}{label}:")
                lines.extend(_block(item, indent + 2))
            else:
                lines.append(f"{pad}{label}: {_flow(item)}")
        return lines
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict) and item:
                inner = _block(item, indent + 2)
                inner[0] = f"{pad}- " + inner[0][indent + 2 :]
                lines.extend(inner)
            else:
                lines.append(f"{pad}- {_flow(item)}")
        return lines
    return [pad + _flow(value)]


def canonical_yaml(value: dict) -> str:
    _check_floats(value)
    if not isinstance(value, dict):
        raise TypeError("canonical_yaml expects a mapping at the top level")
    return "\n".join(_block(value, 0)) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_once_text(path: Path, text: str) -> Path:
    """Content-addressed write: skip when the file already holds exactly ``text``."""
    path = Path(path)
    if path.exists() and path.read_text() == text:
        return path
    atomic_write_text(path, text)
    return path
