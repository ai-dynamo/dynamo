// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { Stack, aws_iam as iam } from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { Construct } from "constructs";
import { ClusterTarget } from "../config";

export function importCluster(scope: Construct, target: ClusterTarget): eks.ICluster {
  const stack = Stack.of(scope);
  if (stack.region !== target.region) throw new Error("Application stack and cluster must be in the same region");
  const partition = target.oidcProviderArn?.split(":")[1];
  if (!partition || !target.name || !/^sg-[0-9a-f]+$/.test(target.securityGroupId) ||
      !target.kubectlServiceToken.startsWith(`arn:${partition}:lambda:${target.region}:${stack.account}:`) ||
      !target.oidcProviderArn.startsWith(`arn:${partition}:iam::${stack.account}:oidc-provider/`)) {
    throw new Error("Supply cluster name, security group, same-account kubectl provider and OIDC provider");
  }
  return eks.Cluster.fromClusterAttributes(scope, "Cluster", {
    clusterName: target.name,
    clusterSecurityGroupId: target.securityGroupId,
    kubectlProvider: eks.KubectlProvider.fromKubectlProviderAttributes(scope, "Kubectl", {
      serviceToken: target.kubectlServiceToken,
    }),
    openIdConnectProvider: iam.OpenIdConnectProvider.fromOpenIdConnectProviderArn(scope, "Oidc", target.oidcProviderArn),
  });
}
