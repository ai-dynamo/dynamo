<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Router load readiness test contract

Tracked issue: https://github.com/ai-dynamo/dynamo/issues/15723
Existing correction: https://github.com/ai-dynamo/dynamo/pull/15753

## Functional Behavior

The advisory selection test must await the registered worker's load record using
its existing bounded helper before asserting projected decode load. Preserve all
load, busy-threshold, and no-booking assertions. Production code stays unchanged.

## Unit Tests

Run advisory_select_reports_worker_load_and_busy_evaluation with
standalone-selection,standalone-slot-tracker enabled, then repeat the exact test
binary 100 times. Confirm the filter executes one test per repetition.

## Integration / Functional Tests

Run the complete dynamo-kv-router library suite with the same features. Run
formatting, targeted Clippy with warnings denied, and changed-file pre-commit.

## Smoke Tests

Fresh upstream CI must exercise the full workspace on the pushed head.

## E2E Tests

Not applicable: this changes only readiness in an existing router test.

## Manual / cURL Tests

Not applicable: no production HTTP path changes. Prior Anthropic live evidence
is retained; this correction must not modify the stop-sequence implementation.
