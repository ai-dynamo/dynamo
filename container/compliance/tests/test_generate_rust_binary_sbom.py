# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from container.compliance.generate_rust_binary_sbom import (
    _parse_cargo_tree,
    cyclonedx_document,
    runtime_package_ids,
)


def _metadata() -> dict:
    packages = [
        {
            "id": "sidecar 1.0.0 (path+file:///sidecar)",
            "name": "sidecar",
            "version": "1.0.0",
            "license": "Apache-2.0",
            "repository": "https://example.com/sidecar",
        },
        {
            "id": "runtime 1.0.0 (path+file:///runtime)",
            "name": "runtime",
            "version": "1.0.0",
            "license": "Apache-2.0",
        },
        {
            "id": "normal 2.0.0 (registry+https://example.com)",
            "name": "normal",
            "version": "2.0.0",
            "license": "MIT/Apache-2.0",
        },
        {
            "id": "build-only 3.0.0 (registry+https://example.com)",
            "name": "build-only",
            "version": "3.0.0",
            "license": "MIT",
        },
        {
            "id": "unrelated 4.0.0 (path+file:///unrelated)",
            "name": "unrelated",
            "version": "4.0.0",
            "license": "BSD-3-Clause",
        },
    ]
    return {
        "packages": packages,
        "workspace_members": [packages[0]["id"], packages[4]["id"]],
        "resolve": {
            "nodes": [
                {
                    "id": packages[0]["id"],
                    "deps": [
                        {
                            "pkg": packages[1]["id"],
                            "dep_kinds": [{"kind": None, "target": None}],
                        },
                        {
                            "pkg": packages[3]["id"],
                            "dep_kinds": [{"kind": "build", "target": None}],
                        },
                    ],
                },
                {
                    "id": packages[1]["id"],
                    "deps": [
                        {
                            "pkg": packages[2]["id"],
                            "dep_kinds": [{"kind": "normal", "target": None}],
                        }
                    ],
                },
                {"id": packages[2]["id"], "deps": []},
                {"id": packages[3]["id"], "deps": []},
                {"id": packages[4]["id"], "deps": []},
            ]
        },
    }


def test_runtime_package_ids_follow_only_normal_edges() -> None:
    metadata = _metadata()
    names_by_id = {package["id"]: package["name"] for package in metadata["packages"]}

    package_ids = runtime_package_ids(metadata, {"sidecar"})

    assert {names_by_id[package_id] for package_id in package_ids} == {
        "sidecar",
        "runtime",
        "normal",
    }


def test_cyclonedx_document_is_deterministic_and_preserves_licenses() -> None:
    document = cyclonedx_document(_metadata(), {"sidecar"})

    assert document["bomFormat"] == "CycloneDX"
    assert [component["name"] for component in document["components"]] == [
        "normal",
        "runtime",
        "sidecar",
    ]
    normal = document["components"][0]
    assert normal["licenses"] == [{"expression": "MIT OR Apache-2.0"}]
    assert document["components"][-1]["type"] == "application"


def test_missing_root_is_rejected() -> None:
    try:
        runtime_package_ids(_metadata(), {"missing"})
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing root package should fail")


def test_cargo_tree_parser_deduplicates_and_ignores_annotations() -> None:
    output = """\
sidecar v1.0.0 (/src/sidecar)
normal v2.0.0
normal v2.0.0 (*)
"""

    assert _parse_cargo_tree(output) == {
        ("sidecar", "1.0.0"),
        ("normal", "2.0.0"),
    }
