/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";

import { DEPLOYMENTS, POC_PEERING_OHIO, POC_PEERING_WEST, selectDeployment } from "../lib/const";
import { loadDeploymentSettings } from "../lib/deployment-settings";
import { VALID_CONFIG } from "./fixtures";

function withSettings(value: unknown, action: (file: string, cwd: string) => void): void {
  const cwd = fs.mkdtempSync(path.join(os.tmpdir(), "cdk-settings-"));
  try {
    fs.writeFileSync(path.join(cwd, "settings.json"), JSON.stringify(value));
    action("settings.json", cwd);
  } finally {
    fs.rmSync(cwd, { recursive: true, force: true });
  }
}

const peering = { ohio: POC_PEERING_OHIO, west: POC_PEERING_WEST };

test("uses checked-in examples unless a local settings file is explicitly selected", () => {
  assert.equal(loadDeploymentSettings(undefined).deployments, DEPLOYMENTS);
});

test("loads operator deployment and peering settings relative to the requested cwd", () => {
  const config = { deployments: { custom: VALID_CONFIG }, peering };
  withSettings(config, (file, cwd) => {
    const loaded = loadDeploymentSettings(file, cwd);
    assert.deepEqual(selectDeployment("custom", loaded.deployments), VALID_CONFIG);
    assert.deepEqual(loaded.peering, peering);
    assert.throws(() => selectDeployment("ohio", loaded.deployments), /Choose one of: custom/);
  });
});

test("rejects missing, invalid or incorrectly shaped local settings", () => {
  assert.throws(() => loadDeploymentSettings(""), /must name a JSON/);
  assert.throws(() => loadDeploymentSettings("missing.json", os.tmpdir()), /Could not read/);
  for (const value of [null, [], {}, { deployments: {}, peering },
    { deployments: { bad: null }, peering }, { deployments: DEPLOYMENTS, peering: {} }]) {
    withSettings(value, (file, cwd) => assert.throws(() => loadDeploymentSettings(file, cwd), /must contain/));
  }
  withSettings({ deployments: DEPLOYMENTS, peering: { ...peering, ohio: { ...peering.ohio, privateRouteTableIds: ["invalid"] } } },
    (file, cwd) => assert.throws(() => loadDeploymentSettings(file, cwd), /Each peering region/));
});
