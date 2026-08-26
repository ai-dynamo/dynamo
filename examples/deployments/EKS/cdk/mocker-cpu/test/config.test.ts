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
import { test } from "node:test";

import { App } from "aws-cdk-lib";

import type { DeploymentConfig, DgdPoolConfig } from "../lib/config";
import { DynamoMockerCpuStack } from "../lib/dynamo-mocker-cpu-stack";
import { VALID_CONFIG } from "./fixtures";

function constructStack(config: DeploymentConfig): void {
  const app = new App();
  new DynamoMockerCpuStack(app, "ConfigTestStack", {
    config,
    env: {
      account: "111111111111",
      region: config.clusterConfig.region,
    },
  });
}

test("accepts independent pool and DGD configuration", () => {
  assert.doesNotThrow(() => constructStack(VALID_CONFIG));
});

test("requires unique infrastructure pool names", () => {
  const firstPool = VALID_CONFIG.dgdPools[0];
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        dgdPools: [firstPool, { ...firstPool, subnetSlot: 2 }],
      }),
    /pool names must be unique/,
  );
});

test("requires unique DGD names", () => {
  const firstDgd = VALID_CONFIG.dgds[0];
  assert.throws(
    () => constructStack({ ...VALID_CONFIG, dgds: [firstDgd, firstDgd] }),
    /DGD names must be unique/,
  );
});

test("reserves the shared Dynamo platform Helm release name", () => {
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        dgds: [
          {
            ...VALID_CONFIG.dgds[0],
            dgdName: "dynamo-platform",
          },
        ],
      }),
    /reserved for the shared platform Helm release/,
  );
});

test("requires every pool AZ to belong to the VPC", () => {
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        dgdPools: [
          {
            ...VALID_CONFIG.dgdPools[0],
            availabilityZone: "us-west-2c",
          },
        ],
      }),
    /not in the configured VPC/,
  );
});

test("requires each DGD to reference existing pools exactly once", () => {
  const firstDgd = VALID_CONFIG.dgds[0];
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        dgds: [{ ...firstDgd, poolNames: ["missing-pool"] }],
      }),
    /references unknown pool/,
  );
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        dgds: [{ ...firstDgd, poolNames: ["cpu-a", "cpu-a"] }],
      }),
    /one or more unique pool names/,
  );

  const emptyPoolMapping = {
    ...VALID_CONFIG,
    dgds: [{ ...firstDgd, poolNames: [] }],
  } as unknown as DeploymentConfig;
  assert.throws(
    () => constructStack(emptyPoolMapping),
    /one or more unique pool names/,
  );
});

test("validates pool capacity and DGD Mocker settings separately", () => {
  const firstPool = VALID_CONFIG.dgdPools[0];
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        dgdPools: [
          {
            ...firstPool,
            nodeGroupConfig: {
              ...firstPool.nodeGroupConfig,
              scaling: { minSize: 3, desiredSize: 2, maxSize: 4 },
            },
          },
        ],
      }),
    /node scaling/,
  );
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        dgds: [{ ...VALID_CONFIG.dgds[0], mockerReplicas: 0 }],
      }),
    /positive replica count/,
  );
});

test("requires x86_64 instance types for the configured node image", () => {
  const firstPool = VALID_CONFIG.dgdPools[0];
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        dgdPools: [
          {
            ...firstPool,
            nodeGroupConfig: {
              ...firstPool.nodeGroupConfig,
              instanceTypes: ["m7i.xlarge", "m7g.xlarge"],
            },
          },
        ],
      }),
    /valid x86_64 EC2 instance types/,
  );

  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        clusterConfig: {
          ...VALID_CONFIG.clusterConfig,
          bastionConfig: {
            ...VALID_CONFIG.clusterConfig.bastionConfig,
            instanceType: "t4g.small",
          },
        },
      }),
    /Bastion instanceType must be a valid x86_64 EC2 instance type/,
  );
});

test("requires a unique subnet slot from the supported range", () => {
  const firstPool = VALID_CONFIG.dgdPools[0];
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        dgdPools: [
          firstPool,
          { ...firstPool, name: "cpu-b" },
        ],
      }),
    /unique integer subnetSlot/,
  );

  const dgdPools = Array.from({ length: 7 }, (_, index) => ({
    ...firstPool,
    name: `cpu-${index + 1}`,
    subnetSlot: index + 1,
  })) as unknown as [DgdPoolConfig, ...DgdPoolConfig[]];
  assert.throws(
    () => constructStack({ ...VALID_CONFIG, dgdPools }),
    /subnetSlot from 1 through 6/,
  );
});

test("requires one valid shared Dynamo platform configuration", () => {
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        dynamoPlatformConfig: {
          ...VALID_CONFIG.dynamoPlatformConfig,
          dynamoNamespace: "Not Valid",
        },
      }),
    /Dynamo platform requires/,
  );
});

test("requires a nonempty SSH launch key-pair name for the private cluster", () => {
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        clusterConfig: {
          ...VALID_CONFIG.clusterConfig,
          bastionConfig: {
            ...VALID_CONFIG.clusterConfig.bastionConfig,
            keyPairName: "",
          },
        },
      }),
    /nonempty EC2 key-pair name/,
  );
});

test("requires one Region, VPC, and two standard Availability Zones", () => {
  assert.throws(
    () =>
      constructStack({
        ...VALID_CONFIG,
        clusterConfig: {
          ...VALID_CONFIG.clusterConfig,
          bastionConfig: {
            ...VALID_CONFIG.clusterConfig.bastionConfig,
            region: "us-east-1",
          },
        },
      }),
    /same Region and VPC configuration/,
  );

  const duplicateAzVpcConfig = {
    ...VALID_CONFIG.clusterConfig.vpcConfig,
    availabilityZones: ["us-west-2a", "us-west-2a"],
  } as const;
  const duplicateAzConfig = {
    ...VALID_CONFIG,
    clusterConfig: {
      ...VALID_CONFIG.clusterConfig,
      bastionConfig: {
        ...VALID_CONFIG.clusterConfig.bastionConfig,
        vpcConfig: duplicateAzVpcConfig,
      },
      vpcConfig: duplicateAzVpcConfig,
    },
  } as DeploymentConfig;
  assert.throws(
    () => constructStack(duplicateAzConfig),
    /two distinct standard Availability Zones/,
  );

  const invalidCidrVpcConfig = {
    ...VALID_CONFIG.clusterConfig.vpcConfig,
    cidrBlock: "10.42.1.0/16",
  };
  const invalidCidrConfig: DeploymentConfig = {
    ...VALID_CONFIG,
    clusterConfig: {
      ...VALID_CONFIG.clusterConfig,
      bastionConfig: {
        ...VALID_CONFIG.clusterConfig.bastionConfig,
        vpcConfig: invalidCidrVpcConfig,
      },
      vpcConfig: invalidCidrVpcConfig,
    },
  };
  assert.throws(
    () => constructStack(invalidCidrConfig),
    /aligned private IPv4 CIDR/,
  );
});
