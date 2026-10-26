#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the Python media protocol models from the Rust schemas.

The Rust media types write their schemas to
lib/llm/src/protocols/openai/media_schemas/ (see media_schemas.rs there). The
generate-media-protocols pre-commit hook runs this script when a schema
changes. To run it by hand:

    uvx pre-commit run generate-media-protocols --all-files

With --check, the script writes nothing and fails when a committed model is
stale.

The models carry the constraints of the Rust types: a numeric bound becomes
a field constraint, a scalar is strict, so a string is not read as a number
or a bool, and an enum in a response becomes a ``Literal`` of its values.
An enum in a request becomes a plain ``str``. The frontend checks the value
before it forwards the request, and a rolling upgrade runs the frontend and
a worker at different versions: a worker must accept what a newer frontend
forwards for a field it does not read. A worker builds a response, so its
response model keeps the enum, and the worker fails where it sets a value
the frontend rejects. For a type, a bound, a default, and a response enum,
a generated model accepts and rejects the same JSON as the Rust type; the
shared cases in lib/llm/src/protocols/openai/media_fixtures/ hold both sides
to that, and mark what the frontend alone rejects. A Rust rule the schema
cannot state, such as a validator with no ``#[schema]`` bound or an integer
width, is not mirrored: the frontend checks it before a request reaches a
worker.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCHEMAS = REPO / "lib/llm/src/protocols/openai/media_schemas"
MODELS = REPO / "components/src/dynamo/common/protocols"
MODULES = ("audio", "image", "video")

# The copyright check compares the last year in this range with the year the
# file last changed. Extend the range when a regeneration lands in a new year.
HEADER = """\
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generated from lib/llm/src/protocols/openai/media_schemas/{module}.json by
# scripts/generate_media_protocols.py. Do not edit."""

OPTIONS = (
    "--input-file-type=openapi",
    "--output-model-type=pydantic_v2.BaseModel",
    "--target-python-version=3.10",
    # An enum becomes a Literal of its values, not an Enum class. A request
    # enum is a plain string by then: see loosen_request_enums.
    "--enum-field-as-literal=all",
    "--collapse-root-models",
    # A field with a serde default is not Optional.
    "--strict-nullable",
    # Pydantic reads "4" and 4.0 as the int 4, and "true" as the bool True,
    # where serde rejects them. Strict scalars reject what serde rejects;
    # strict float still reads a JSON integer, as serde's f64 does.
    "--strict-types",
    "int",
    "float",
    "bool",
    "str",
    # A bound is an Annotated[..., Field(...)] constraint, not a conint() or
    # confloat() call: mypy rejects a call as an annotation, and CI runs mypy
    # without the pydantic plugin. The same flag puts a field default after
    # the annotation, where mypy reads it; a positional Field(default) is
    # invisible to it.
    "--use-annotated",
    "--use-field-description",
    "--use-schema-description",
    "--disable-timestamp",
    "--formatters=builtin",
)

REF_PREFIX = "#/components/schemas/"


def _enum_ref(schema: object, enums: set[str]) -> bool:
    return (
        isinstance(schema, dict)
        and isinstance(schema.get("$ref"), str)
        and schema["$ref"].removeprefix(REF_PREFIX) in enums
    )


def _without_enums(schema: object, enums: set[str]) -> object:
    """`schema` with every reference to an enum in it replaced by a string.

    utoipa writes a field of an enum type as a `$ref`, and an `Option` of one
    as a `oneOf` of the `$ref` and `null`. Any other shape around an enum is
    an error here, so a new shape is noticed rather than left as a Literal.
    """
    if isinstance(schema, list):
        return [_without_enums(item, enums) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if _enum_ref(schema, enums):
        return {**{k: v for k, v in schema.items() if k != "$ref"}, "type": "string"}
    if any(_enum_ref(variant, enums) for variant in schema.get("oneOf", [])):
        kinds = []
        for variant in schema["oneOf"]:
            if _enum_ref(variant, enums):
                kinds.append("string")
            elif variant == {"type": "null"}:
                kinds.append("null")
            else:
                raise ValueError(f"an enum next to {variant} is not handled")
        kind = kinds if len(kinds) > 1 else kinds[0]
        return {**{k: v for k, v in schema.items() if k != "oneOf"}, "type": kind}
    return {key: _without_enums(value, enums) for key, value in schema.items()}


def _reachable(schemas: dict, root: str) -> set[str]:
    """The names of the component schemas `root` refers to, and `root`."""
    seen: set[str] = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        text = json.dumps(schemas[name])
        for part in text.split(f'"{REF_PREFIX}')[1:]:
            pending.append(part.split('"', 1)[0])
    return seen


def loosen_request_enums(document: dict) -> None:
    """Make every enum a request schema refers to a plain string, in place.

    The module docstring says why. A response schema keeps its enums, so a
    component both sides refer to is an error: loosening it would loosen
    the response too.
    """
    schemas = document["components"]["schemas"]
    enums = {name for name, schema in schemas.items() if "enum" in schema}
    request = set()
    response = set()
    for name in schemas:
        if name.endswith("Request"):
            request |= _reachable(schemas, name) - enums
        elif name.endswith("Response"):
            response |= _reachable(schemas, name) - enums
    if shared := request & response:
        raise ValueError(f"a request and a response share {sorted(shared)}")
    for name in request:
        schemas[name] = _without_enums(schemas[name], enums)
    # An enum only a request referred to is unreferenced now.
    text = json.dumps(document)
    for name in enums:
        if f'"{REF_PREFIX}{name}"' not in text:
            del schemas[name]


def generate(out_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for module in MODULES:
            document = json.loads((SCHEMAS / f"{module}.json").read_text())
            loosen_request_enums(document)
            schema = Path(tmp) / f"{module}.json"
            schema.write_text(json.dumps(document, indent=2))
            subprocess.run(
                [
                    "datamodel-codegen",
                    f"--input={schema}",
                    f"--output={out_dir / f'{module}_protocol.py'}",
                    f"--custom-file-header={HEADER.format(module=module)}",
                    *OPTIONS,
                ],
                check=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when a committed model is stale, and write nothing",
    )
    if not parser.parse_args().check:
        generate(MODELS)
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        generate(Path(tmp))
        stale = [
            f"{module}_protocol.py"
            for module in MODULES
            if (Path(tmp) / f"{module}_protocol.py").read_text()
            != (MODELS / f"{module}_protocol.py").read_text()
        ]
    if stale:
        print(
            f"stale in {MODELS}: {', '.join(stale)}. Run "
            "`uvx pre-commit run generate-media-protocols --all-files`."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
