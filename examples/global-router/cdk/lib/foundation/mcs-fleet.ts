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

import {
  CfnOutput,
  CfnResource,
  RemovalPolicy,
  Stack,
  StackProps,
  aws_ec2 as ec2,
} from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { Construct } from "constructs";
import { cidrsOverlap, isValidVpcCidr } from "./cidr";
import { MultiClusterSupport } from "./multi-cluster";

export interface McsMember {
  readonly id: string;
  readonly siteId: string;
  readonly region: string;
  readonly clusterName: string;
  readonly vpcId: string;
  readonly cidr: string;
  readonly routeTableIds: readonly string[];
  readonly clusterSecurityGroupId: string;
  readonly kubectlServiceToken: string;
  /** Existing hub only. Changes API source ranges, never installs or replaces its control plane. */
  readonly hubApiService?: {
    readonly name: string;
    readonly namespace: string;
  };
}
export interface McsLink {
  readonly left: string;
  readonly right: string;
  readonly peeringId?: string;
  /** Another stack owns both this connection and its routes. */
  readonly external?: boolean;
}
export interface McsFleet {
  readonly members: readonly McsMember[];
  readonly links: readonly McsLink[];
}

export function validateMcsFleet(f: McsFleet): void {
  if (
    !Array.isArray(f?.members) ||
    f.members.length < 2 ||
    f.members.length > 4 ||
    !Array.isArray(f.links)
  )
    throw new Error("Configure two to four MCS members and their links");
  const ids = new Set<string>(),
    sites = new Set<string>(),
    regions = new Set<string>();
  for (const m of f.members) {
    if (
      !/^[a-z][a-z0-9]{0,20}$/.test(m.id) ||
      ids.has(m.id) ||
      sites.has(m.siteId) ||
      regions.has(m.region)
    )
      throw new Error("Member IDs, site IDs and regions must be unique");
    ids.add(m.id);
    sites.add(m.siteId);
    regions.add(m.region);
    if (
      !/^[a-z0-9]([-a-z0-9]*[a-z0-9])?$/.test(m.siteId) ||
      m.siteId.length > 48 ||
      !/^[a-z]{2}-[a-z]+-\d$/.test(m.region) ||
      !isValidVpcCidr(m.cidr)
    )
      throw new Error("Invalid member identity, region or private VPC CIDR");
    if (
      !/^vpc-[a-f0-9]{8,17}$/.test(m.vpcId) ||
      !/^sg-[a-f0-9]{8,17}$/.test(m.clusterSecurityGroupId) ||
      !m.clusterName ||
      !m.kubectlServiceToken.startsWith(`arn:aws:lambda:${m.region}:`)
    )
      throw new Error(
        "Specify existing cluster, VPC, security group and regional kubectl provider",
      );
    if (
      !m.routeTableIds.length ||
      new Set(m.routeTableIds).size !== m.routeTableIds.length ||
      m.routeTableIds.some((id: string) => !/^rtb-[a-f0-9]{8,17}$/.test(id))
    )
      throw new Error("Specify unique regional route table IDs");
  }
  if (f.members.filter((m) => m.hubApiService).length > 1)
    throw new Error("Configure at most one existing hub");
  const expected = new Set<string>();
  for (let i = 0; i < f.members.length; i++)
    for (let j = i + 1; j < f.members.length; j++) {
      const a = f.members[i],
        b = f.members[j];
      if (cidrsOverlap(a.cidr, b.cidr) || a.vpcId === b.vpcId)
        throw new Error("Member VPCs must not overlap");
      expected.add([a.id, b.id].sort().join(":"));
    }
  const actual = new Set<string>();
  for (const l of f.links) {
    const key = [l.left, l.right].sort().join(":");
    if (!expected.has(key) || actual.has(key))
      throw new Error("Invalid or duplicate peering link");
    actual.add(key);
    if (l.peeringId && !/^pcx-[a-f0-9]{8,17}$/.test(l.peeringId))
      throw new Error("Invalid peering ID");
    if (l.external && !l.peeringId)
      throw new Error("External links require an existing peering ID");
  }
  if (actual.size !== expected.size)
    throw new Error(
      "MCS direct pod connectivity requires every member pair; VPC peering is not transitive",
    );
}

/** New peerings only; existing connections stay with their original stack. */
export class McsPeeringsStack extends Stack {
  constructor(
    scope: Construct,
    id: string,
    props: StackProps & { fleet: McsFleet; member: McsMember },
  ) {
    super(scope, id, { ...props, terminationProtection: true });
    for (const l of props.fleet.links.filter(
      (l) => l.left === props.member.id && !l.external,
    )) {
      const right = props.fleet.members.find((m) => m.id === l.right)!;
      const connection = new ec2.CfnVPCPeeringConnection(
        this,
        `To${right.id}`,
        {
          vpcId: props.member.vpcId,
          peerVpcId: right.vpcId,
          peerRegion: right.region,
          tags: [{ key: "Project", value: "global-router-mcs" }],
        },
      );
      connection.applyRemovalPolicy(RemovalPolicy.RETAIN);
      new CfnOutput(this, `Peering${props.member.id}To${right.id}`, {
        value: connection.ref,
      });
    }
  }
}

/** Literal peering IDs from the first phase avoid cross-region export/provider dependencies. */
export class McsRoutesStack extends Stack {
  constructor(
    scope: Construct,
    id: string,
    props: StackProps & { fleet: McsFleet; member: McsMember },
  ) {
    super(scope, id, { ...props, terminationProtection: true });
    for (const l of props.fleet.links.filter(
      (l) => !l.external && [l.left, l.right].includes(props.member.id),
    )) {
      if (!l.peeringId)
        throw new Error(
          "Deploy peerings and populate peering IDs before routes",
        );
      const other = props.fleet.members.find(
        (m) => m.id === (l.left === props.member.id ? l.right : l.left),
      )!;
      for (const [index, rt] of props.member.routeTableIds.entries()) {
        const route = new ec2.CfnRoute(this, `To${other.id}Route${index + 1}`, {
          routeTableId: rt,
          destinationCidrBlock: other.cidr,
          vpcPeeringConnectionId: l.peeringId,
        });
        route.applyRemovalPolicy(RemovalPolicy.RETAIN);
      }
    }
  }
}

/** Add MCS to existing clusters without owning their compute or their Karmada control plane. */
export class McsMemberStack extends Stack {
  constructor(
    scope: Construct,
    id: string,
    props: StackProps & { fleet: McsFleet; member: McsMember },
  ) {
    super(scope, id, { ...props, terminationProtection: true });
    const m = props.member;
    const provider = eks.KubectlProvider.fromKubectlProviderAttributes(
      this,
      "Provider",
      { serviceToken: m.kubectlServiceToken },
    );
    const cluster = eks.Cluster.fromClusterAttributes(this, "Cluster", {
      clusterName: m.clusterName,
      clusterSecurityGroupId: m.clusterSecurityGroupId,
      kubectlProvider: provider,
    });
    const peers = props.fleet.members
      .filter((p) => p.id !== m.id)
      .map((p) => p.cidr) as [string, ...string[]];
    new MultiClusterSupport(this, "Mcs", {
      cluster,
      config: { memberName: m.siteId, peerCidrs: peers },
      region: m.region,
      vpcCidr: m.cidr,
      nodeGroups: [],
    });
    if (m.hubApiService) {
      const cidrs = props.fleet.members.map((p) => p.cidr);
      new eks.KubernetesPatch(this, "HubPrivateApiSources", {
        cluster,
        resourceName: `service/${m.hubApiService.name}`,
        resourceNamespace: m.hubApiService.namespace,
        applyPatch: {
          spec: { loadBalancerSourceRanges: cidrs },
          metadata: {
            annotations: {
              "service.beta.kubernetes.io/load-balancer-source-ranges":
                cidrs.join(","),
            },
          },
        },
        restorePatch: {},
      });
    }
    // These resources may predate this stack. Never remove their CRDs, CNI settings or peer access on rollback.
    for (const c of this.node.findAll())
      if (c instanceof CfnResource) c.applyRemovalPolicy(RemovalPolicy.RETAIN);
    new CfnOutput(this, "MemberName", { value: m.siteId });
  }
}
