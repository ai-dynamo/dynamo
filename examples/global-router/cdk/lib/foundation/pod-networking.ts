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

import { aws_ec2 as ec2 } from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { Construct } from "constructs";

import { cidrsOverlap, isValidVpcCidr } from "./cidr";
import type { PeerNetworkingConfig } from "./config";

const EXCLUDE_SNAT_ENV = "AWS_VPC_K8S_CNI_EXCLUDE_SNAT_CIDRS";

export interface PeerNetworkingProps {
  readonly cluster: eks.ICluster;
  readonly config: PeerNetworkingConfig;
  /** This cluster's VPC range. It must not overlap a peer. */
  readonly vpcCidr: string;
}

/** Allows direct pod connectivity over the fleet's VPC peerings. */
export class PeerNetworking extends Construct {
  constructor(scope: Construct, id: string, props: PeerNetworkingProps) {
    super(scope, id);
    const { cluster, config } = props;
    validatePeerNetworkingConfig(config, props.vpcCidr);

    for (const cidr of config.peerCidrs) {
      cluster.clusterSecurityGroup.addIngressRule(
        ec2.Peer.ipv4(cidr),
        ec2.Port.allTcp(),
        `Pod traffic from peer VPC ${cidr}`,
      );
      // Inter-Region peering carries smaller packets than the pods' MTU, and
      // senders learn the limit from this ICMP message (Path MTU Discovery).
      cluster.clusterSecurityGroup.addIngressRule(
        ec2.Peer.ipv4(cidr),
        ec2.Port.icmpTypeAndCode(3, 4),
        `Path MTU Discovery from peer VPC ${cidr}`,
      );
    }

    new eks.KubernetesPatch(this, "ExcludePeerVpcsFromSnat", {
      cluster,
      resourceName: "daemonset/aws-node",
      resourceNamespace: "kube-system",
      applyPatch: awsNodeEnvPatch({
        name: EXCLUDE_SNAT_ENV,
        value: config.peerCidrs.join(","),
      }),
      restorePatch: awsNodeEnvPatch({
        $patch: "delete",
        name: EXCLUDE_SNAT_ENV,
      }),
    });

  }
}

export function validatePeerNetworkingConfig(
  config: PeerNetworkingConfig,
  vpcCidr: string,
): void {
  if (!config.peerCidrs.length || config.peerCidrs.some((cidr) => !isValidVpcCidr(cidr))) {
    throw new Error(
      "Peer CIDRs must be nonempty private, network-aligned VPC ranges from /16 to /24.",
    );
  }
  if (config.peerCidrs.some((cidr) => cidrsOverlap(cidr, vpcCidr))) {
    throw new Error("Peer CIDRs must not overlap the cluster's own VPC.");
  }
}

function awsNodeEnvPatch(env: Record<string, string>): Record<string, unknown> {
  return {
    spec: {
      template: {
        spec: {
          containers: [{ name: "aws-node", env: [env] }],
        },
      },
    },
  };
}
