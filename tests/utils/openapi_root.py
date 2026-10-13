# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root-only lookup for known acyclic local OpenAPI schema compositions."""


def root_property(document, node, name):
    """Find a request-root property through these exports' acyclic local refs/allOf.

    Return None for an absent property. Do not descend into arbitrary properties:
    a same-named field inside messages or tools does not describe this request root.
    Broken local refs raise KeyError instead of becoming evidence of absence.
    """
    if name in node.get("properties", {}):
        return node["properties"][name]
    if "$ref" in node:
        reference = node["$ref"]
        assert reference.startswith("#/"), reference
        resolved = document
        for part in reference[2:].split("/"):
            resolved = resolved[part.replace("~1", "/").replace("~0", "~")]
        return root_property(document, resolved, name)
    for part in node.get("allOf", []):
        found = root_property(document, part, name)
        if found is not None:
            return found
    return None
