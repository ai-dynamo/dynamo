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
    MULTI_POOL_CONFIG.privateFrontendSourceCidrs,
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
    VALID_CONFIG.privateFrontendSourceCidrs,
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

test("selecting a Frontend KV router leaves worker settings unchanged", () => {
  const dgd = VALID_CONFIG.dgds[0];
  const baseline = createMockerHelmValues(
    dgd,
    VALID_CONFIG.dynamoPlatformConfig,
    VALID_CONFIG.privateFrontendSourceCidrs,
  );
  const selected = createMockerHelmValues(
    { ...dgd, frontendRouterMode: "kv" },
    VALID_CONFIG.dynamoPlatformConfig,
    VALID_CONFIG.privateFrontendSourceCidrs,
  );
  const frontend = selected.frontend as Record<string, unknown>;
  assert.equal(frontend.routerMode, "kv");
  const { routerMode, ...resources } = frontend;
  assert.deepEqual({ ...selected, frontend: resources }, baseline);
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

test("exposes one private endpoint per DGD on the Dynamo Frontend port", () => {
  const chartDir = path.resolve(__dirname, "..", "..", "helm", "dynamo-mocker");
  const dgdTemplate = readFileSync(
    path.join(chartDir, "templates", "dgd.yaml"),
    "utf8",
  );
  const serviceTemplate = readFileSync(
    path.join(chartDir, "templates", "frontend-private-service.yaml"),
    "utf8",
  );

  assert.match(
    dgdTemplate,
    /app\.kubernetes\.io\/name: dynamo-mocker-frontend\n\s+dynamo\.nvidia\.com\/dgd: \{\{ \.Values\.deployment\.name \| quote \}\}/,
  );
  assert.match(
    serviceTemplate,
    /name: \{\{ printf "%s-private" \.Values\.deployment\.name \| quote \}\}/,
  );
  assert.match(serviceTemplate, /aws-load-balancer-type: "external"/);
  assert.match(serviceTemplate, /aws-load-balancer-nlb-target-type: "ip"/);
  assert.match(serviceTemplate, /aws-load-balancer-scheme: "internal"/);
  assert.match(serviceTemplate, /type: LoadBalancer/);
  assert.match(
    serviceTemplate,
    /loadBalancerSourceRanges:\n\{\{ toYaml \.Values\.frontendEndpoint\.sourceCidrs \| indent 4 \}\}/,
  );
  assert.match(
    serviceTemplate,
    /port: 8000\n\s+targetPort: http/,
  );
  assert.match(
    serviceTemplate,
    /app\.kubernetes\.io\/name: dynamo-mocker-frontend\n\s+dynamo\.nvidia\.com\/dgd: \{\{ \.Values\.deployment\.name \| quote \}\}/,
  );
});
