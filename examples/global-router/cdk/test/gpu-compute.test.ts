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
import { Template, Match } from "aws-cdk-lib/assertions";
import {
  GpuAssetsStack,
  ReservedGpuComputeStack,
  GpuFoundationStack,
  GpuRegion,
  validateGpuFleet,
} from "../lib/gpu-compute";

const config: GpuRegion = {
  region: "us-west-2",
  availabilityZone: "us-west-2a",
  reservationId: "cr-00000000000000000",
  instanceType: "g6.4xlarge",
  count: 2,
  hourlyRate: 1.4,
  releaseVersion: "1.35.9-20261023",
  clusterName: "example-gpu",
  subnetId: "subnet-00000000000000000",
};
test("compute targets only the supplied reservation and cannot scale down or surge", () => {
  const stack = new ReservedGpuComputeStack(new App(), "Compute", { config });
  const t = Template.fromStack(stack);
  t.resourceCountIs("AWS::EKS::Cluster", 0);
  t.resourceCountIs("AWS::EC2::CapacityReservation", 0);
  t.resourceCountIs("AWS::EC2::VPC", 0);
  t.hasResourceProperties("AWS::EC2::LaunchTemplate", {
    LaunchTemplateData: {
      InstanceType: config.instanceType,
      CapacityReservationSpecification: {
        CapacityReservationTarget: {
          CapacityReservationId: config.reservationId,
        },
      },
      MetadataOptions: { HttpTokens: "required", HttpPutResponseHopLimit: 1 },
      ImageId: Match.absent(),
      UserData: Match.absent(),
    },
  });
  t.hasResourceProperties("AWS::EKS::Nodegroup", {
    ClusterName: config.clusterName,
    Subnets: [config.subnetId],
    AmiType: "AL2023_x86_64_NVIDIA",
    ReleaseVersion: config.releaseVersion,
    ScalingConfig: { MinSize: 2, DesiredSize: 2, MaxSize: 2 },
    UpdateConfig: { MaxUnavailable: 1, UpdateStrategy: "MINIMAL" },
  });
  for (const type of [
    "AWS::EKS::Nodegroup",
    "AWS::EC2::LaunchTemplate",
    "AWS::IAM::Role",
  ]) {
    t.hasResource(type, {
      DeletionPolicy: "Retain",
      UpdateReplacePolicy: "Retain",
    });
  }
  assert.equal(stack.terminationProtection, true);
});
test("foundation supplies private control plane and no GPU lifecycle resources", () => {
  const stack = new GpuFoundationStack(new App(), "Foundation", {
    env: { region: "us-west-2", account: "111111111111" },
    config: {
      ...config,
      subnetId: undefined,
      foundation: {
        cidr: "10.44.0.0/16",
        availabilityZones: ["us-west-2a", "us-west-2b"],
      },
    },
  });
  const t = Template.fromStack(stack);
  t.resourceCountIs("AWS::EKS::Nodegroup", 0);
  t.resourceCountIs("AWS::EC2::CapacityReservation", 0);
  t.hasResourceProperties("AWS::EKS::Cluster", {
    ResourcesVpcConfig: {
      EndpointPrivateAccess: true,
      EndpointPublicAccess: false,
    },
  });
  for (const type of [
    "AWS::EKS::Cluster",
    "AWS::EC2::VPC",
    "AWS::EC2::Subnet",
    "AWS::IAM::Role",
  ]) {
    t.hasResource(type, {
      DeletionPolicy: "RetainExceptOnCreate",
      UpdateReplacePolicy: "Retain",
    });
  }
  assert.equal(stack.terminationProtection, true);
});
test("asset bootstrap has no persistent deployment roles", () => {
  const stack = new GpuAssetsStack(new App(), "Assets", {
    env: { region: "us-west-2", account: "111111111111" },
  });
  const t = Template.fromStack(stack);
  t.resourceCountIs("AWS::IAM::Role", 0);
  t.resourceCountIs("AWS::EKS::Cluster", 0);
  t.hasResourceProperties("AWS::S3::Bucket", {
    PublicAccessBlockConfiguration: {
      BlockPublicAcls: true,
      BlockPublicPolicy: true,
      IgnorePublicAcls: true,
      RestrictPublicBuckets: true,
    },
  });
});
test("fleet rejects duplicate regions, counts, cost violations and incompatible types", () => {
  validateGpuFleet([config]);
  for (const value of [
    { count: 1 },
    { count: 5 },
    { count: 2.5 },
    { hourlyRate: 7.01 },
    { hourlyRate: 0 },
    { instanceType: "p5.48xlarge" },
    { reservationId: "" },
    { availabilityZone: "us-east-2a" },
    { subnetId: "invalid" },
    { releaseVersion: "latest" },
    { subnetId: undefined },
    {
      foundation: {
        cidr: "10.44.0.0/16",
        availabilityZones: ["us-west-2a", "us-west-2a"],
      },
    },
  ]) {
    assert.throws(() => validateGpuFleet([{ ...config, ...value }]));
  }
  assert.throws(() => validateGpuFleet([config, config]), /unique/);
  assert.throws(() => validateGpuFleet([]));
});
