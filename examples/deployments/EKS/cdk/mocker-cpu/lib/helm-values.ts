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

import type { DgdConfig, DynamoPlatformConfig } from "./config";

export const DGD_POOL_LABEL_KEY = "dynamo.nvidia.com/dgd-pool";
export const PURPOSE_LABEL_KEY = "dynamo.nvidia.com/purpose";
export const PURPOSE_LABEL_VALUE = "mocker-cpu";

export function createMockerHelmValues(
  dgd: DgdConfig,
  platform: DynamoPlatformConfig,
): Record<string, unknown> {
  return {
    deployment: {
      name: dgd.dgdName,
    },
    image: `nvcr.io/nvidia/ai-dynamo/dynamo-planner:${platform.dynamoVersion}`,
    model: {
      name: dgd.modelName,
    },
    mocker: {
      replicas: dgd.mockerReplicas,
      speedup_ratio: dgd.mockerSpeedupRatio,
    },
    placement: {
      nodeSelector: {
        [PURPOSE_LABEL_KEY]: PURPOSE_LABEL_VALUE,
      },
      nodeAffinity: {
        requiredDuringSchedulingIgnoredDuringExecution: {
          nodeSelectorTerms: [
            {
              matchExpressions: [
                {
                  key: DGD_POOL_LABEL_KEY,
                  operator: "In",
                  values: [...dgd.poolNames].sort((left, right) =>
                    left < right ? -1 : left > right ? 1 : 0,
                  ),
                },
              ],
            },
          ],
        },
      },
    },
    frontend: {
      resources: {
        requests: {
          cpu: "250m",
          memory: "512Mi",
        },
        limits: {
          cpu: "1",
          memory: "2Gi",
        },
      },
    },
    worker: {
      resources: {
        requests: {
          cpu: "500m",
          memory: "1Gi",
        },
        limits: {
          cpu: "2",
          memory: "4Gi",
        },
      },
    },
  };
}
