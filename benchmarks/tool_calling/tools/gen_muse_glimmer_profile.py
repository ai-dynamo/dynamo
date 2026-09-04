#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit the Muse Glimmer declarative case profile.

Derived from ``gen_qwen3_coder_xml_profile.py`` by REUSE rather than by copy:
this module imports that generator's tool registry, case builders and case list
and re-expresses them for Muse Glimmer. Only what actually differs is restated
here, so a fix to a shared case lands in both profiles.

Muse Glimmer matches the qwen3_coder_xml family on the one structural property
that shapes a profile: **thinking is unconditionally on**. There is no
``enable_thinking`` / ``thinking`` kwarg for this model — its chat template
branches on nothing of the sort — so the thinking-off and drop-history families
have no analogue and every case runs under the template default.

Where it differs is the MARKUP, and that difference is confined to two places:

* Framing is recipient-routed rather than marker-paired. A turn is a sequence of
  ``<|start|>assistant to=RCPT<|message|> ... <|eom|>`` messages, where ``self``
  is reasoning, ``user`` is the visible answer, and any other recipient opens a
  tool channel. There is no reasoning open/close pair to bait, so the
  ``missing_open_think_tag`` case has no analogue; the reasoning-always-on case
  below replaces it.
* Tool calls are ATEM XML inside a tool channel, not ``<parameter=>`` markup.

Both differences are expressed as data: ``FORBIDDEN`` below lists the tokens
that must never survive into visible content or reasoning, and the shared
marker-containment family (F8) then exercises them unchanged, because those
cases bait the model with ordinary prompts and rely on the defaults block to
catch a leak.

Run:  python3 tools/gen_muse_glimmer_profile.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "custom/configs/case_profiles/muse_glimmer.json"

_REF = ROOT / "tools/gen_qwen3_coder_xml_profile.py"
_spec = importlib.util.spec_from_file_location("gen_qwen3_coder_xml_profile", _REF)
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

# Reused verbatim: the tool registry, the case builders, and the token budget.
# These are grammar-independent — a tool schema and an expected argument dict do
# not depend on how the model frames a call on the wire.
TOOLS = _ref.TOOLS
case = _ref.case
tc = _ref.tc

# Muse Glimmer control tokens and ATEM tool markup. None of these may survive
# into visible content or reasoning once parsing is correct. `<|start|>`,
# `<|eom|>` and `<|eot|>` all CUT a body, so `<|message|>` is the one framing
# token that can reach a reader at all, which is why it is listed first.
FORBIDDEN = [
    "<|message|>",
    "<|start|>",
    "<|eom|>",
    "<|eot|>",
    "to=self",
    "<atem:function_calls>",
    "</atem:function_calls>",
    "<atem:invoke",
    "</atem:invoke>",
    "<atem:parameter",
    "</atem:parameter>",
    "<|patch|>",
    "<|video|>",
]

# This template family exposes NO thinking kwargs at all: reasoning is routed by
# recipient and cannot be suppressed per request. Every case runs under the
# template default, exactly as in qwen3_coder_xml.
PRESETS = {"template_default": {}}

# The reference case list, minus the one case whose premise is a reasoning
# open/close PAIR. Muse has no such pair to omit, so the case cannot be
# expressed for this grammar; the replacement below covers the same property
# (reasoning populates without the request asking for it) through the framing
# this model actually uses.
_DROP = {"qx_reasoning_missing_open_think_tag"}

CASES = []
for _c in _ref.CASES:
    if _c["case_id"] in _DROP:
        continue
    _c = json.loads(json.dumps(_c))
    _c["case_id"] = "muse_" + _c["case_id"][len("qx_") :]
    _c["description"] = _c["description"]
    CASES.append(_c)

# ---------------------------------------------------------------------------
# Muse-specific replacement for the dropped case. Reasoning is unconditional
# and recipient-routed, so a plain question with no tools must still populate
# reasoning_content, with none of the routing framing reaching the client.
# ---------------------------------------------------------------------------
_before = len(_ref.CASES)
case(
    "muse_reasoning_recipient_routed_always_on",
    "reasoning is routed by recipient and always on, so it populates with no request control",
    "Think step by step about why 2+2=4, then return only the integer 4.",
    preset="template_default",
    no_tools=True,
    finish=("stop",),
    expected_content="4",
    expect_reasoning=True,
)
CASES.extend(
    {**json.loads(json.dumps(c)), "case_id": "muse_" + c["case_id"][len("qx_") :]}
    if c["case_id"].startswith("qx_")
    else json.loads(json.dumps(c))
    for c in _ref.CASES[_before:]
)


def main() -> None:
    payload = {
        "schema_version": 1,
        "profile": "muse_glimmer",
        "description": (
            "Muse Glimmer recipient-routed reasoning and ATEM tool-calling "
            "qualification matrix."
        ),
        "default_modes": ["nonstream", "stream"],
        "logical_cases": len(CASES),
        "expected_records": len(CASES) * 2,
        "defaults": {"forbidden_output_fragments": FORBIDDEN},
        "request_presets": PRESETS,
        "tools": TOOLS,
        "cases": CASES,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    ids = [c["case_id"] for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    print(f"wrote {OUT} with {len(CASES)} cases")


if __name__ == "__main__":
    main()
