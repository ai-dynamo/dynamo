---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Responses Conflict Resolution
subtitle: Preserve unary response ordering while merging structured tool calls
---

This design resolves the conflict in the unary Chat Completions to Responses API conversion.

## Behavior

Preserve the length-stop detection from `main`. When a response includes a requested reasoning
summary, structured tool calls, and text content, emit output items in this order:

1. Reasoning summary
2. Structured function calls
3. Function calls parsed from text
4. Remaining text message

Keep existing empty-choice and empty-output fallbacks unchanged.

## Implementation

Merge both branches' changes in
`lib/llm/src/protocols/openai/responses/mod.rs`. Retain the structured tool-call conversion and
the PR's length-stop handling without changing request conversion or streaming behavior.

## Validation

Run the focused Rust test for the conversion module. Keep the regression test that asserts a
reasoning item precedes a structured function call and the resulting text message.
