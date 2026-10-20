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

import { loadGlobalRouterDeployment } from "../lib/global-router-deployment";

test("leaves the Global Router disabled unless image and config are provided", () => {
  assert.equal(loadGlobalRouterDeployment(undefined, undefined), undefined);
});

test("requires both an explicit image and a readable JSON config", () => {
  assert.throws(
    () => loadGlobalRouterDeployment("router:tag", undefined),
    /routerConfigPath/,
  );
  assert.throws(
    () => loadGlobalRouterDeployment(undefined, "router.json"),
    /routerImage/,
  );
  assert.throws(
    () => loadGlobalRouterDeployment("registry.example/router", "router.json"),
    /explicit tag or sha256 digest/,
  );
  assert.throws(
    () => loadGlobalRouterDeployment("router:latest", "router.json"),
    /implicit latest tag/,
  );
});

test("requires the container reachable listen address", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "global-router-"));
  try {
    const file = path.join(directory, "router.json");
    fs.writeFileSync(file, JSON.stringify({ listen: "127.0.0.1:8080" }));
    assert.throws(
      () => loadGlobalRouterDeployment("router:tag", file),
      /listen must be 0.0.0.0:8080/,
    );
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("loads the supplied JSON unchanged for the ConfigMap", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "global-router-"));
  try {
    const config = {
      listen: "0.0.0.0:8080",
      pools: [{
        private_frontend_base_url: "http://frontend.invalid:8000",
        relay_grpc_url: "http://relay.invalid:9000",
        stats_grpc_url: "http://stats.invalid:9001",
      }],
    };
    const file = path.join(directory, "router.json");
    fs.writeFileSync(file, JSON.stringify(config));
    const deployment = loadGlobalRouterDeployment("router:sha256-abc", file);
    assert.ok(deployment);
    assert.equal(deployment.image, "router:sha256-abc");
    assert.deepEqual(deployment.config, config);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
