# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

SKILL_ROOT = Path(__file__).parents[1]
RENDERER_PATH = SKILL_ROOT / "scripts/render_review.py"
TEMPLATE_PATH = SKILL_ROOT / "assets/review-template.html"

module_spec = importlib.util.spec_from_file_location(
    "visual_review_renderer", RENDERER_PATH
)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError(f"Unable to load {RENDERER_PATH}")
renderer = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(renderer)


def base_spec():
    return {
        "title": "Review",
        "correctness_score": {
            "base": 5,
            "value": 5,
            "summary": "No correctness change.",
            "factors": [],
        },
        "risk_score": {
            "base": 10,
            "value": 10,
            "summary": "Safe to merge.",
            "factors": [],
        },
        "findings": [
            {
                "id": "finding-one",
                "severity": "P2",
                "title": "Finding",
                "summary": "A focused finding.",
                "details": [
                    {"label": "Trigger", "items": ["The trigger occurs."]},
                    {"label": "Impact", "items": ["The impact follows."]},
                ],
                "suggested_fix": {
                    "summary": "Fix the behavior.",
                    "steps": ["Apply the focused fix."],
                    "tests": [],
                },
                "agent_prompt": "Fix the focused behavior.",
                "file": "example.py",
                "side": "new",
                "line": 1,
                "involves_api_objects": False,
            }
        ],
        "diagrams": [
            {
                "id": "diagram-information-flow",
                "type": "component",
                "primary": True,
                "finding_ids": [],
                "title": "Information flow",
                "nodes": [
                    {
                        "id": "caller",
                        "label": "Caller",
                        "kind": "changed",
                        "target": "finding-one",
                    },
                    {"id": "callee", "label": "Callee", "kind": "existing"},
                ],
                "edges": [
                    {
                        "from": "caller",
                        "to": "callee",
                        "label": "Call",
                        "target": "finding-one",
                    }
                ],
            }
        ],
    }


def text_diff():
    return renderer.parse_diff(
        "diff --git a/example.py b/example.py\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -0,0 +1 @@\n"
        "+new_call()\n"
    )


def test_parse_diff_handles_metadata_and_binary_payloads():
    parsed = renderer.parse_diff(
        "diff --git a/mode.sh b/mode.sh\n"
        "old mode 100644\n"
        "new mode 100755\n"
        "diff --git a/image.png b/image.png\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "Binary files /dev/null and b/image.png differ\n"
        "diff --git a/data.bin b/data.bin\n"
        "new file mode 100644\n"
        "index 0000000..2222222\n"
        "GIT binary patch\n"
        "literal 3\n"
        "KcmZQzWZ\n"
        "diff --git a/source.py b/copy.py\n"
        "similarity index 100%\n"
        "copy from source.py\n"
        "copy to copy.py\n"
        "diff --git a/example.py b/example.py\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -0,0 +1 @@\n"
        "+new_call()\n"
    )

    assert [item["path"] for item in parsed["files"]] == [
        "mode.sh",
        "image.png",
        "data.bin",
        "copy.py",
        "example.py",
    ]
    assert parsed["additions"] == 1
    assert parsed["deletions"] == 0
    assert ("example.py", "new", 1) in parsed["line_keys"]


def test_validate_spec_rejects_boolean_line_numbers():
    finding_spec = base_spec()
    finding_spec["findings"][0]["line"] = True

    file_spec = base_spec()
    file_spec["file_map"] = [{"path": "example.py", "total_lines": True, "markers": []}]

    marker_spec = base_spec()
    marker_spec["file_map"] = [
        {
            "path": "example.py",
            "total_lines": 1,
            "markers": [
                {
                    "line": True,
                    "target": "finding-one",
                    "kind": "risk",
                }
            ],
        }
    ]

    matrix_spec = base_spec()
    matrix_spec["test_matrix"] = [
        {
            "finding_ids": ["finding-one"],
            "case": "Boolean line",
            "expected": "Rejected",
            "status": "missing",
            "file": "example.py",
            "side": "new",
            "line": True,
        }
    ]

    invalid_specs = [
        (finding_spec, r"findings\[0\]\.line"),
        (file_spec, r"file_map\[0\]\.total_lines"),
        (marker_spec, r"file_map\[0\]\.markers\[0\]\.line"),
        (matrix_spec, r"test_matrix\[0\]\.line"),
    ]
    for spec, error in invalid_specs:
        with pytest.raises(renderer.ReviewError, match=error):
            renderer.validate_spec(copy.deepcopy(spec), text_diff())


def test_component_targets_have_keyboard_links():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'class="component-links"' in template
    assert '<a href="#${esc(link.target)}">' in template
    assert 'aria-describedby="${linksId}"' in template
    assert "componentTargets(diagram)" in template
