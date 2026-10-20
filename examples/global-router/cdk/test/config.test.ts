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
import { Template } from "aws-cdk-lib/assertions";

import type { DeploymentConfig, DgdPoolConfig } from "../lib/config";
import { selectDeployment } from "../lib/const";
import { DynamoMockerCpuStack } from "../lib/dynamo-mocker-cpu-stack";
import { VALID_CONFIG } from "./fixtures";

test("selects Ohio when explicitly requested and rejects unknown deployments", () => {
  assert.equal(selectDeployment("ohio").clusterConfig.region, "us-east-2");
  assert.equal(selectDeployment("west").clusterConfig.region, "us-west-2");
  assert.throws(() => selectDeployment("unknown"), /Unknown deployment unknown/);
  assert.throws(() => selectDeployment(1), /Unknown deployment 1/);
});

test("synthesizes a deployment-specific Mocker image override", () => {
  const config: DeploymentConfig = {
    ...selectDeployment("west"),
    mockerImage: "example.registry/mocker:poc-42",
  };
  const app = new App();
  const stack = new DynamoMockerCpuStack(app, "ImageOverride", {
    config,
    env: { account: "111111111111", region: "us-west-2" },
  });
  const charts = Template.fromStack(stack).findResources(
    "Custom::AWSCDK-EKS-HelmChart",
  );
  const mockerChart = Object.values(charts).find((resource) => {
    if (resource.Properties.Release !== "mocker-west-1") return false;
    const values = JSON.parse(String(resource.Properties.Values)) as {
      deployment?: { name?: string };
    };
    return values.deployment?.name === "mocker-west-1";
  });
  assert.ok(mockerChart);
  const values = JSON.parse(String(mockerChart.Properties.Values)) as {
    image: string;
    frontendEndpoint: { sourceCidrs: string[] };
  };
  assert.equal(values.image, "example.registry/mocker:poc-42");
  assert.deepEqual(values.frontendEndpoint.sourceCidrs, [
    "10.42.0.0/16",
    "10.43.0.0/16",
  ]);
});

test("requires valid source CIDRs for private Frontend endpoints", () => {
  for (const cidrs of [[], ["10.42.0.0/33"], ["10.42.0.0"], ["999.1.1.1/24"]]) {
    const config = {
      ...VALID_CONFIG,
      privateFrontendSourceCidrs: cidrs,
    } as unknown as DeploymentConfig;
    assert.throws(
      () => constructStack(config),
      /Private Frontend source CIDRs must contain valid IPv4 CIDR ranges/,
    );
  }
  assert.doesNotThrow(() => constructStack(VALID_CONFIG));
});

test("requires valid source CIDRs for optional private Relay endpoints", () => {
  for (const cidrs of [[], ["10.42.0.0/33"], ["10.42.0.0"], ["999.1.1.1/24"]]) {
    const config = {
      ...VALID_CONFIG,
      privateRelaySourceCidrs: cidrs,
    } as unknown as DeploymentConfig;
    assert.throws(
      () => constructStack(config),
      /Private Relay source CIDRs must contain valid IPv4 CIDR ranges/,
    );
  }
  assert.doesNotThrow(() => constructStack(VALID_CONFIG));
});

test("requires an explicitly versioned optional Relay image", () => {
  for (const image of ["", "example.registry/relay", "example.registry/relay:latest"]) {
    assert.throws(
      () => constructStack({ ...VALID_CONFIG, kvDcRelayImage: image }),
      /kvDcRelayImage must provide an explicit image tag or sha256 digest/,
    );
  }
  assert.doesNotThrow(() =>
    constructStack({
      ...VALID_CONFIG,
      kvDcRelayImage: "example.registry/relay:poc-1",
    }),
  );
});

test("synthesizes each named deployment with its own regional cluster", () => {
  for (const [name, region, clusterName] of [
    ["ohio", "us-east-2", "sachal-mocker-cluster"],
    ["west", "us-west-2", "sachal-mocker-west-cluster"],
  ] as const) {
    const config = selectDeployment(name);
    assert.equal(config.clusterConfig.region, region);
    const app = new App();
    const stack = new DynamoMockerCpuStack(app, `Synth-${name}`, {
      config,
      env: { account: "111111111111", region },
    });
    const template = Template.fromStack(stack);
    template.hasResourceProperties("AWS::EKS::Cluster", {
      Name: clusterName,
    });
    const charts = template.findResources("Custom::AWSCDK-EKS-HelmChart");
    const mockerChart = Object.values(charts).find(
      (resource) => resource.Properties.Release === config.dgds[0].dgdName,
    );
    assert.ok(mockerChart);
    const values = JSON.parse(String(mockerChart.Properties.Values)) as {
      frontendEndpoint: { sourceCidrs: string[] };
    };
    assert.deepEqual(values.frontendEndpoint.sourceCidrs, [
    "10.42.0.0/16",
    "10.43.0.0/16",
  ]);
  }
});

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
