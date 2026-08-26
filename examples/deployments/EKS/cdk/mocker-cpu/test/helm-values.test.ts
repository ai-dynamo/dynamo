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
import { readFileSync } from "node:fs";
import * as path from "node:path";
import { test } from "node:test";

import {
  createMockerHelmValues,
  DGD_POOL_LABEL_KEY,
  PURPOSE_LABEL_KEY,
  PURPOSE_LABEL_VALUE,
} from "../lib/helm-values";
import { MULTI_POOL_CONFIG, VALID_CONFIG } from "./fixtures";

test("maps a DGD to one or more infrastructure pools", () => {
  const dgd = MULTI_POOL_CONFIG.dgds.find(
    (candidate) => candidate.dgdName === "mocker-flex",
  );
  assert.ok(dgd);
  const values = createMockerHelmValues(
    dgd,
    MULTI_POOL_CONFIG.dynamoPlatformConfig,
  );
  const placement = values.placement as {
    nodeAffinity: {
      requiredDuringSchedulingIgnoredDuringExecution: {
        nodeSelectorTerms: Array<{
          matchExpressions: Array<{
            key: string;
            operator: string;
            values: string[];
          }>;
        }>;
      };
    };
    nodeSelector: Record<string, string>;
  };

  assert.deepEqual(placement.nodeSelector, {
    [PURPOSE_LABEL_KEY]: PURPOSE_LABEL_VALUE,
  });
  assert.deepEqual(
    placement.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution,
    {
      nodeSelectorTerms: [
        {
          matchExpressions: [
            {
              key: DGD_POOL_LABEL_KEY,
              operator: "In",
              values: ["cpu-a", "cpu-b"],
            },
          ],
        },
      ],
    },
  );
});

test("uses each DGD's independent Mocker settings", () => {
  const dgd = VALID_CONFIG.dgds[0];
  const values = createMockerHelmValues(
    dgd,
    VALID_CONFIG.dynamoPlatformConfig,
  );
  const mocker = values.mocker as {
    replicas: number;
    speedup_ratio: number;
  };

  assert.equal(mocker.replicas, 3);
  assert.equal(mocker.speedup_ratio, 1);
  assert.equal(
    values.image,
    "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.4.0",
  );
});

test("applies pool affinity to both DGD components", () => {
  const template = readFileSync(
    path.resolve(
      __dirname,
      "..",
      "..",
      "helm",
      "dynamo-mocker",
      "templates",
      "dgd.yaml",
    ),
    "utf8",
  );
  const decodeStart = template.indexOf("  - name: decode");
  assert.ok(decodeStart > 0);
  const componentSections = [
    template.slice(0, decodeStart),
    template.slice(decodeStart),
  ];

  for (const section of componentSections) {
    assert.match(
      section,
      /nodeAffinity:\n\{\{ toYaml \.Values\.placement\.nodeAffinity \| indent 12 \}\}/,
    );
  }
  assert.match(componentSections[1], /podAntiAffinity:/);
});
