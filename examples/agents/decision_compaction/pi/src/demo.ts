// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { isAbsolute, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { fixtureSession } from "./fixture-session.js";
import { helperClient } from "./helper.js";

const python = process.env.DYNAMO_COMPACTION_PYTHON;
if (!python || !isAbsolute(python)) throw Error("Set DYNAMO_COMPACTION_PYTHON to an absolute Python executable with the example dependencies installed.");
const repository = fileURLToPath(new URL("../../../../../", import.meta.url));
const invoke = helperClient(python, ["-m", "dynamo.compaction.helper", "--mode", "fixture-sdk"], { PYTHONPATH: resolve(repository, "components/src"), PYTHONNOUSERSITE: "1", PYTHONUNBUFFERED: "1" });
// The helper uses fixture bytes, not a model tokenizer; this is not a token qualification.
const f = await fixtureSession({ coordinator: invoke, maxOutputTokens: 16_384 });
try {
  await f.session.compact();
  assert.equal(f.acknowledgments.length, 1);
  const resumed = await f.reopen();
  await resumed.prompt("Continue the synthetic task", { expandPromptTemplates: false });
  assert.match(JSON.stringify(f.requests.at(-1)), /Constraint 0: do not push/);
  assert.match(JSON.stringify(f.requests.at(-1)), /Checkpoint observations/);
  console.log(JSON.stringify({ mode: "fixture", helper: "official_sdk_mock_transport", pi: "1.1.0", persisted: true, resumed: true, next_actor_payload: true, model_quality_qualified: false }));
} finally { await f.dispose(); }
