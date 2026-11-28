// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ClusterConfig, DgdConfig, DynamoPlatformConfig } from "./config";
import { DGD_POOL_LABEL_KEY, PURPOSE_LABEL_KEY, PURPOSE_LABEL_VALUE } from "./helm-values";

const K8S_DISCOVERY_SA_SUFFIX = "-k8s-service-discovery";

export function createKvDcRelayHelmValues(
  dgd: DgdConfig,
  cluster: ClusterConfig,
  platform: DynamoPlatformConfig,
  image: string,
  sourceCidrs: readonly string[],
): Record<string, unknown> {
  const runtimeNamespace = `${platform.dynamoNamespace}-${dgd.dgdName}`;
  return {
    relay: {
      image,
      dgdName: dgd.dgdName,
      dcId: `${cluster.region}-${dgd.dgdName}`,
      runtimeNamespace,
      watchNamespaces: [dgd.workerRuntimeNamespace ?? runtimeNamespace],
      serviceAccountName: `${dgd.dgdName}${K8S_DISCOVERY_SA_SUFFIX}`,
      wanListen: "0.0.0.0:50051",
      statsListen: "0.0.0.0:50052",
      statsAllowNonLoopback: true,
      sourceCidrs: [...sourceCidrs],
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
      resources: {
        requests: {
          cpu: "100m",
          memory: "128Mi",
        },
        limits: {
          cpu: "1",
          memory: "1Gi",
        },
      },
    },
  };
}

export function hasExplicitImageVersion(image: string): boolean {
  const digestSeparator = image.lastIndexOf("@");
  if (digestSeparator >= 0) {
    return /^sha256:[a-fA-F0-9]{64}$/.test(image.slice(digestSeparator + 1));
  }
  const finalPathSegment = image.slice(image.lastIndexOf("/") + 1);
  const tagSeparator = finalPathSegment.lastIndexOf(":");
  if (tagSeparator <= 0) {
    return false;
  }
  const tag = finalPathSegment.slice(tagSeparator + 1);
  return /^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/.test(tag);
}
