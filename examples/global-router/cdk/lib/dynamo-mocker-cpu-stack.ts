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

import * as fs from "node:fs";
import * as path from "node:path";

import { KubectlV35Layer } from "@aws-cdk/lambda-layer-kubectl-v35";
import {
  CfnOutput,
  Duration,
  RemovalPolicy,
  Stack,
  StackProps,
  Tags,
  aws_ec2 as ec2,
  aws_s3_assets as s3Assets,
} from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { Construct } from "constructs";

import { EksAdminBastion } from "./bastion";
import { isValidIpv4Cidr } from "./cidr";
import type { GlobalRouterDeployment } from "./global-router-deployment";
import { createGlobalRouterHelmValues } from "./global-router-helm-values";
import {
  createKvDcRelayHelmValues,
  hasExplicitImageVersion,
} from "./kv-dc-relay-helm-values";
import {
  type DeploymentConfig,
  type DgdConfig,
  type DgdPoolConfig,
  type VpcConfig,
} from "./config";
import {
  createMockerHelmValues,
  DGD_POOL_LABEL_KEY,
  PURPOSE_LABEL_KEY,
  PURPOSE_LABEL_VALUE,
} from "./helm-values";
import {
  KARMADA_API_SERVICE,
  KARMADA_KUBECONFIG_SECRET,
  KARMADA_NAMESPACE,
  MultiClusterSupport,
} from "./multi-cluster";
import { MAX_DGD_POOL_SUBNET_SLOTS, MockerVpc } from "./vpc";

const DYNAMO_PLATFORM_RELEASE = "dynamo-platform";

export interface DynamoMockerCpuStackProps extends StackProps {
  readonly config: DeploymentConfig;
  readonly mockerChartPath?: string;
  readonly globalRouter?: GlobalRouterDeployment;
}

export class DynamoMockerCpuStack extends Stack {
  public readonly vpc: ec2.IVpc;
  public readonly privateRouteTableIds: readonly string[];

  constructor(scope: Construct, id: string, props: DynamoMockerCpuStackProps) {
    super(scope, id, props);

    const { config } = props;
    const { clusterConfig, dgdPools, dgds, dynamoPlatformConfig } = config;
    assertCompatibleDeployment(config);
    if (
      config.privateFrontendSourceCidrs.length === 0 ||
      config.privateFrontendSourceCidrs.some(
        (cidr) => !isValidIpv4Cidr(cidr),
      )
    ) {
      throw new Error(
        "Private Frontend source CIDRs must contain valid IPv4 CIDR ranges.",
      );
    }
    if (
      config.privateRelaySourceCidrs.length === 0 ||
      config.privateRelaySourceCidrs.some(
        (cidr) => !isValidIpv4Cidr(cidr),
      )
    ) {
      throw new Error(
        "Private Relay source CIDRs must contain valid IPv4 CIDR ranges.",
      );
    }
    if (config.kvDcRelayImage !== undefined) {
      const image = config.kvDcRelayImage;
      if (
        image.trim() === "" ||
        !hasExplicitImageVersion(image) ||
        image.endsWith(":latest")
      ) {
        throw new Error(
          "kvDcRelayImage must provide an explicit image tag or sha256 digest, not latest.",
        );
      }
    }
    const orderedPools = orderDgdPools(dgdPools);
    const orderedDgds = orderDgds(dgds);
    const platformConfig = dynamoPlatformConfig;
    const network = new MockerVpc(this, "Vpc", {
      config: clusterConfig.vpcConfig,
      dgdPools: orderedPools,
    });
    this.vpc = network.vpc;
    this.privateRouteTableIds = [...new Set(
      network.vpc.privateSubnets.map((subnet) => subnet.routeTable.routeTableId),
    )];

    const cluster = new eks.Cluster(this, "Cluster", {
      bootstrapClusterCreatorAdminPermissions: false,
      albController: {
        version: eks.AlbControllerVersion.V2_14_1,
      },
      clusterName: clusterConfig.clusterName,
      defaultCapacity: 0,
      defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
      endpointAccess: eks.EndpointAccess.PRIVATE,
      kubectlProviderOptions: {
        kubectlLayer: new KubectlV35Layer(this, "KubectlLayer"),
        // The EKS v2 construct colocates its provider with control-plane ENIs
        // when private endpoint access is enabled. These subnets need NAT
        // egress so Helm can download the Dynamo platform chart from NGC.
        privateSubnets: network.clusterSubnets.subnets,
      },
      version: eks.KubernetesVersion.V1_35,
      vpc: network.vpc,
      vpcSubnets: [{ subnets: network.clusterSubnets.subnets }],
    });

    const bastion = new EksAdminBastion(this, "Bastion", {
      cluster,
      deploymentConfig: config,
      vpc: network.vpc,
    });

    const poolNodeGroups = orderedPools.map((pool) => {
      const poolId = `DgdPool-${pool.name}`;
      const launchTemplate = new ec2.LaunchTemplate(
        this,
        `${poolId}LaunchTemplate`,
        {
          blockDevices: [
            {
              deviceName: "/dev/xvda",
              volume: ec2.BlockDeviceVolume.ebs(50, {
                deleteOnTermination: true,
                encrypted: true,
                volumeType: ec2.EbsDeviceVolumeType.GP3,
              }),
            },
          ],
          requireImdsv2: true,
        },
      );
      const launchTemplateId = launchTemplate.launchTemplateId;
      if (launchTemplateId === undefined) {
        throw new Error(
          `The managed node group launch template for ${pool.name} has no ID.`,
        );
      }

      const { nodeGroupConfig } = pool;
      const nodeGroup = cluster.addNodegroupCapacity(`${poolId}NodeGroup`, {
        amiType: eks.NodegroupAmiType.AL2023_X86_64_STANDARD,
        capacityType: eks.CapacityType.ON_DEMAND,
        desiredSize: nodeGroupConfig.scaling.desiredSize,
        enableNodeAutoRepair: true,
        instanceTypes: nodeGroupConfig.instanceTypes.map(
          (instanceType) => new ec2.InstanceType(instanceType),
        ),
        labels: {
          [DGD_POOL_LABEL_KEY]: pool.name,
          [PURPOSE_LABEL_KEY]: PURPOSE_LABEL_VALUE,
        },
        launchTemplateSpec: {
          id: launchTemplateId,
          version: launchTemplate.latestVersionNumber,
        },
        maxSize: nodeGroupConfig.scaling.maxSize,
        maxUnavailable: 1,
        minSize: nodeGroupConfig.scaling.minSize,
        nodegroupName: `${pool.name}-mocker`,
        subnets: {
          subnets: network.subnetsForPool(pool.name).subnets,
        },
        tags: {
          DgdPool: pool.name,
          Project: "dynamo-mocker-cpu",
        },
      });

      return { nodeGroup, pool, poolId };
    });
    const poolNodeGroupsByName = new Map(
      poolNodeGroups.map(({ nodeGroup, pool }) => [pool.name, nodeGroup]),
    );

    const platformChart = new eks.HelmChart(this, "DynamoPlatform", {
      atomic: true,
      chart:
        "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/" +
        `dynamo-platform-${platformConfig.dynamoVersion}.tgz`,
      cluster,
      createNamespace: true,
      namespace: platformConfig.dynamoNamespace,
      release: DYNAMO_PLATFORM_RELEASE,
      removalPolicy: RemovalPolicy.DESTROY,
      timeout: Duration.minutes(15),
      values: {
        global: {
          etcd: {
            install: false,
          },
          nats: {
            install: false,
          },
        },
      },
      wait: true,
    });
    for (const { nodeGroup } of poolNodeGroups) {
      platformChart.node.addDependency(nodeGroup);
    }

    const mockerChartAsset = new s3Assets.Asset(this, "MockerChartAsset", {
      path: props.mockerChartPath ?? resolveMockerChartPath(),
    });
    for (const dgd of orderedDgds) {
      const dgdId = `Dgd-${dgd.dgdName}`;
      const mockerChart = new eks.HelmChart(this, `${dgdId}DynamoMocker`, {
        atomic: true,
        chartAsset: mockerChartAsset,
        cluster,
        createNamespace: false,
        namespace: platformConfig.dynamoNamespace,
        release: dgd.dgdName,
        removalPolicy: RemovalPolicy.DESTROY,
        timeout: Duration.minutes(10),
        values: createMockerHelmValues(
          dgd,
          platformConfig,
          config.privateFrontendSourceCidrs,
          config.mockerImage,
          config.mockerRuntimeVersionOverride,
        ),
        wait: true,
      });
      if (cluster.albController === undefined) {
        throw new Error("The AWS Load Balancer Controller is not configured.");
      }
      mockerChart.node.addDependency(cluster.albController);
      for (const poolName of orderNames(dgd.poolNames)) {
        const nodeGroup = poolNodeGroupsByName.get(poolName);
        if (nodeGroup === undefined) {
          throw new Error(
            `DGD ${dgd.dgdName} references unknown pool ${poolName}.`,
          );
        }
        mockerChart.node.addDependency(nodeGroup);
      }
      mockerChart.node.addDependency(platformChart);

      if (config.kvDcRelayImage !== undefined) {
        if (dgd.dgdName.length > 40) {
          throw new Error(
            `DGD ${dgd.dgdName} must be at most 40 characters to fit the operator's Kubernetes discovery ServiceAccount name.`,
          );
        }
        const relayChartAsset = new s3Assets.Asset(
          this,
          `${dgdId}RelayChartAsset`,
          { path: resolveKvDcRelayChartPath() },
        );
        const relayChart = new eks.HelmChart(
          this,
          `${dgdId}KvDcRelay`,
          {
            atomic: true,
            chartAsset: relayChartAsset,
            cluster,
            createNamespace: false,
            namespace: platformConfig.dynamoNamespace,
            release: `${dgd.dgdName}-relay`,
            removalPolicy: RemovalPolicy.DESTROY,
            timeout: Duration.minutes(10),
            values: createKvDcRelayHelmValues(
              dgd,
              clusterConfig,
              platformConfig,
              config.kvDcRelayImage,
              config.privateRelaySourceCidrs,
            ),
            wait: true,
          },
        );
        relayChart.node.addDependency(mockerChart);
        if (cluster.albController === undefined) {
          throw new Error("The AWS Load Balancer Controller is not configured.");
        }
        relayChart.node.addDependency(cluster.albController);
        for (const poolName of orderNames(dgd.poolNames)) {
          const nodeGroup = poolNodeGroupsByName.get(poolName);
          if (nodeGroup === undefined) {
            throw new Error(
              `Relay for DGD ${dgd.dgdName} references unknown pool ${poolName}.`,
            );
          }
          relayChart.node.addDependency(nodeGroup);
        }
      }
    }

    if (props.globalRouter !== undefined) {
      const globalRouterChartAsset = new s3Assets.Asset(
        this,
        "GlobalRouterChartAsset",
        { path: resolveGlobalRouterChartPath() },
      );
      const globalRouterChart = new eks.HelmChart(this, "GlobalRouter", {
        atomic: false,
        chartAsset: globalRouterChartAsset,
        cluster,
        createNamespace: false,
        namespace: platformConfig.dynamoNamespace,
        release: "global-router-ohio",
        removalPolicy: RemovalPolicy.DESTROY,
        timeout: Duration.minutes(5),
        values: createGlobalRouterHelmValues(props.globalRouter),
        // Relay sources may be unavailable until their own regional deployment is ready.
        wait: false,
      });
      globalRouterChart.node.addDependency(platformChart);
      for (const { nodeGroup } of poolNodeGroups) {
        globalRouterChart.node.addDependency(nodeGroup);
      }
    }

    if (config.multiCluster !== undefined) {
      new MultiClusterSupport(this, "MultiCluster", {
        cluster,
        config: config.multiCluster,
        region: clusterConfig.region,
        vpcCidr: clusterConfig.vpcConfig.cidrBlock,
        nodeGroups: poolNodeGroups.map(({ nodeGroup }) => nodeGroup),
      });
    }

    Tags.of(this).add("ManagedBy", "AWS-CDK");
    Tags.of(this).add("Project", "dynamo-mocker-cpu");

    new CfnOutput(this, "AwsRegion", {
      description: "AWS Region containing the EKS cluster.",
      value: clusterConfig.region,
    });
    new CfnOutput(this, "ClusterName", {
      description: "EKS cluster name.",
      value: cluster.clusterName,
    });
    new CfnOutput(this, "DynamoNamespace", {
      description: "Kubernetes namespace containing Dynamo and the DGDs.",
      value: platformConfig.dynamoNamespace,
    });
    new CfnOutput(this, "DgdNames", {
      description: "Comma-separated DynamoGraphDeployment names.",
      value: orderedDgds.map((dgd) => dgd.dgdName).join(","),
    });
    new CfnOutput(this, "DgdPoolNames", {
      description: "Comma-separated infrastructure pool names.",
      value: orderedPools.map((pool) => pool.name).join(","),
    });
    new CfnOutput(this, "DgdPoolPlacements", {
      description: "Comma-separated pool-to-Availability-Zone placements.",
      value: orderedPools
        .map((pool) => `${pool.name}=${pool.availabilityZone}`)
        .join(","),
    });
    new CfnOutput(this, "DgdPoolMappings", {
      description: "Semicolon-separated DGD-to-infrastructure-pool mappings.",
      value: orderedDgds
        .map(
          (dgd) => `${dgd.dgdName}=${orderNames(dgd.poolNames).join("+")}`,
        )
        .join(";"),
    });
    new CfnOutput(this, "ModelNames", {
      description: "Comma-separated model identifiers exposed by the DGDs.",
      value: orderedDgds.map((dgd) => dgd.modelName).join(","),
    });
    new CfnOutput(this, "DgdPoolSubnetIds", {
      description: "Comma-separated pool-to-workload-subnet mappings.",
      value: orderedPools
        .map(
          (pool) =>
            `${pool.name}=${network.subnetsForPool(pool.name).subnetIds[0]}`,
        )
        .join(","),
    });
    if (config.kvDcRelayImage !== undefined) {
      new CfnOutput(this, "KvDcRelayEndpointCommands", {
        description:
          "Get each private Relay NLB hostname for Global Router JSON configuration.",
        value: orderedDgds
          .map(
            (dgd) =>
              `${dgd.dgdName}=kubectl get service -n ${platformConfig.dynamoNamespace} ` +
              `${dgd.dgdName}-relay -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'`,
          )
          .join("; "),
      });
    }
    if (config.multiCluster !== undefined) {
      new CfnOutput(this, "KarmadaMemberName", {
        description: "Use with karmadactl register --cluster-name.",
        value: config.multiCluster.memberName,
      });
      if (config.multiCluster.karmadaControlPlane !== undefined) {
        new CfnOutput(this, "KarmadaApiEndpointCommand", {
          description:
            "Run on the bastion: prints the internal Karmada API load balancer hostname (port 5443).",
          value:
            `kubectl get service -n ${KARMADA_NAMESPACE} ${KARMADA_API_SERVICE} ` +
            "-o jsonpath='{.status.loadBalancer.ingress[0].hostname}'",
        });
        new CfnOutput(this, "KarmadaKubeconfigCommand", {
          description:
            "Run on the bastion: writes the Karmada admin kubeconfig. Point its server at the load balancer.",
          value:
            `kubectl get secret -n ${KARMADA_NAMESPACE} ${KARMADA_KUBECONFIG_SECRET} ` +
            "-o jsonpath='{.data.kubeconfig}' | base64 -d > karmada-apiserver.config",
        });
      }
    }
    new CfnOutput(this, "PortForwardFrontends", {
      description: "Semicolon-separated DGD-to-port-forward commands.",
      value: orderedDgds
        .map(
          (dgd) =>
            `${dgd.dgdName}=kubectl port-forward ` +
            `-n ${platformConfig.dynamoNamespace} ` +
            `svc/${dgd.dgdName}-frontend 8000:8000`,
        )
        .join("; "),
    });
    if (props.globalRouter !== undefined) {
      new CfnOutput(this, "PortForwardGlobalRouter", {
        description: "Run on the bastion, then test the Router from another bastion shell.",
        value:
          `kubectl port-forward -n ${platformConfig.dynamoNamespace} ` +
          "svc/global-router 8080:8080",
      });
    }
    new CfnOutput(this, "BastionInstanceId", {
      description: "Instance ID of the public SSH admin bastion.",
      value: bastion.instance.instanceId,
    });
    new CfnOutput(this, "BastionPublicIp", {
      description: "Elastic IPv4 address assigned to the SSH bastion.",
      value: bastion.publicIp,
    });
    new CfnOutput(this, "BastionKeyPairName", {
      description: "EC2 key pair installed for the ec2-user account at launch.",
      value: clusterConfig.bastionConfig.keyPairName,
    });
    new CfnOutput(this, "ConnectToBastion", {
      description: "Use the private key for the configured EC2 key pair.",
      value:
        "ssh -i /path/to/private-key.pem " +
        "-L 8000:127.0.0.1:8000 " +
        `ec2-user@${bastion.publicIp}`,
    });
    new CfnOutput(this, "ConfigureKubectlOnBastion", {
      description: "Run this command from the bastion SSH shell.",
      value:
        `aws eks update-kubeconfig --region ${clusterConfig.region} ` +
        `--name ${cluster.clusterName}`,
    });
  }
}

function orderDgdPools(
  dgdPools: readonly DgdPoolConfig[],
): DgdPoolConfig[] {
  return [...dgdPools].sort((left, right) =>
    compareNames(left.name, right.name),
  );
}

function orderDgds(dgds: readonly DgdConfig[]): DgdConfig[] {
  return [...dgds].sort((left, right) =>
    compareNames(left.dgdName, right.dgdName),
  );
}

function orderNames(names: readonly string[]): string[] {
  return [...names].sort(compareNames);
}

function compareNames(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function assertCompatibleDeployment(config: DeploymentConfig): void {
  const { clusterConfig, dgdPools, dgds, dynamoPlatformConfig } = config;
  const { bastionConfig, vpcConfig } = clusterConfig;
  if (dgdPools.length === 0) {
    throw new Error("DeploymentConfig.dgdPools requires at least one pool.");
  }
  if (dgds.length === 0) {
    throw new Error("DeploymentConfig.dgds requires at least one DGD.");
  }
  if (
    clusterConfig.region !== vpcConfig.region ||
    bastionConfig.region !== vpcConfig.region ||
    !sameVpcConfig(bastionConfig.vpcConfig, vpcConfig)
  ) {
    throw new Error(
      "The cluster, VPC, and bastion must use the same Region and VPC configuration.",
    );
  }
  if (
    new Set(vpcConfig.availabilityZones).size !== 2 ||
    vpcConfig.availabilityZones.some(
      (availabilityZone) =>
        !isStandardAvailabilityZone(availabilityZone, vpcConfig.region),
    )
  ) {
    throw new Error(
      "VpcConfig.availabilityZones requires two distinct standard Availability Zones in its Region.",
    );
  }
  if (!vpcConfig.availabilityZones.includes(bastionConfig.availabilityZone)) {
    throw new Error("The bastion Availability Zone must belong to its VPC.");
  }
  if (!isValidVpcCidr(vpcConfig.cidrBlock)) {
    throw new Error(
      "VpcConfig.cidrBlock must be an aligned private IPv4 CIDR with a /16 through /24 prefix.",
    );
  }
  if (bastionConfig.keyPairName.trim() === "") {
    throw new Error(
      "Bastion keyPairName requires a nonempty EC2 key-pair name.",
    );
  }
  if (!isX86InstanceType(bastionConfig.instanceType)) {
    throw new Error(
      "Bastion instanceType must be a valid x86_64 EC2 instance type.",
    );
  }
  if (
    clusterConfig.clusterName.length > 100 ||
    !/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(clusterConfig.clusterName)
  ) {
    throw new Error("The EKS cluster name is invalid.");
  }

  if (
    !isKubernetesName(dynamoPlatformConfig.dynamoNamespace, 63) ||
    dynamoPlatformConfig.dynamoVersion.trim() === ""
  ) {
    throw new Error(
      "The Dynamo platform requires a version and lowercase DNS-label namespace.",
    );
  }

  const poolNames = new Set<string>();
  const subnetSlots = new Set<number>();

  for (const pool of dgdPools) {
    if (poolNames.has(pool.name)) {
      throw new Error(`DGD pool names must be unique: ${pool.name}.`);
    }
    poolNames.add(pool.name);

    if (!isKubernetesName(pool.name, 56)) {
      throw new Error(
        "Each DGD pool name must be a lowercase DNS label at most 56 characters long.",
      );
    }

    if (
      !Number.isInteger(pool.subnetSlot) ||
      pool.subnetSlot < 1 ||
      pool.subnetSlot > MAX_DGD_POOL_SUBNET_SLOTS ||
      subnetSlots.has(pool.subnetSlot)
    ) {
      throw new Error(
        `Pool ${pool.name} requires a unique integer subnetSlot from 1 through ${MAX_DGD_POOL_SUBNET_SLOTS}.`,
      );
    }
    subnetSlots.add(pool.subnetSlot);

    if (
      !clusterConfig.vpcConfig.availabilityZones.includes(
        pool.availabilityZone,
      )
    ) {
      throw new Error(
        `Pool ${pool.name} selects ${pool.availabilityZone}, which is not in the configured VPC.`,
      );
    }
    if (
      pool.nodeGroupConfig.instanceTypes.length === 0 ||
      pool.nodeGroupConfig.instanceTypes.some(
        (instanceType) => !isX86InstanceType(instanceType),
      )
    ) {
      throw new Error(
        `Pool ${pool.name} requires valid x86_64 EC2 instance types for the AL2023 x86_64 node image.`,
      );
    }
    const { desiredSize, maxSize, minSize } = pool.nodeGroupConfig.scaling;
    if (
      !Number.isInteger(minSize) ||
      !Number.isInteger(desiredSize) ||
      !Number.isInteger(maxSize) ||
      minSize < 0 ||
      minSize > desiredSize ||
      desiredSize > maxSize ||
      desiredSize < 1
    ) {
      throw new Error(
        `Pool ${pool.name} node scaling must use integers with 0 <= minSize <= desiredSize <= maxSize and desiredSize >= 1.`,
      );
    }
  }

  const dgdNames = new Set<string>();
  for (const dgd of dgds) {
    if (dgdNames.has(dgd.dgdName)) {
      throw new Error(`DGD names must be unique: ${dgd.dgdName}.`);
    }
    dgdNames.add(dgd.dgdName);

    if (!isKubernetesName(dgd.dgdName, 53)) {
      throw new Error(
        "Each DGD name must be a lowercase DNS label at most 53 characters long.",
      );
    }
    if (dgd.dgdName === DYNAMO_PLATFORM_RELEASE) {
      throw new Error(
        `DGD name ${DYNAMO_PLATFORM_RELEASE} is reserved for the shared platform Helm release.`,
      );
    }
    if (
      dgd.poolNames.length === 0 ||
      new Set(dgd.poolNames).size !== dgd.poolNames.length
    ) {
      throw new Error(
        `DGD ${dgd.dgdName} requires one or more unique pool names.`,
      );
    }
    for (const poolName of dgd.poolNames) {
      if (!poolNames.has(poolName)) {
        throw new Error(
          `DGD ${dgd.dgdName} references unknown pool ${poolName}.`,
        );
      }
    }
    if (
      !Number.isInteger(dgd.mockerReplicas) ||
      dgd.mockerReplicas < 1 ||
      !Number.isFinite(dgd.mockerSpeedupRatio) ||
      dgd.mockerSpeedupRatio <= 0 ||
      dgd.modelName.trim() === ""
    ) {
      throw new Error(
        `DGD ${dgd.dgdName} requires a model, positive replica count, and positive speedup ratio.`,
      );
    }
  }
}

function sameVpcConfig(left: VpcConfig, right: VpcConfig): boolean {
  return (
    left.region === right.region &&
    left.name === right.name &&
    left.cidrBlock === right.cidrBlock &&
    left.availabilityZones[0] === right.availabilityZones[0] &&
    left.availabilityZones[1] === right.availabilityZones[1]
  );
}

function isStandardAvailabilityZone(
  availabilityZone: string,
  region: string,
): boolean {
  return new RegExp(`^${escapeRegex(region)}[a-z]$`).test(availabilityZone);
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function isValidVpcCidr(value: string): boolean {
  const match =
    /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\/(\d{1,2})$/.exec(
      value,
    );
  if (match === null) {
    return false;
  }
  const octets = match.slice(1, 5).map(Number);
  const prefix = Number(match[5]);
  if (octets.some((octet) => octet > 255) || prefix < 16 || prefix > 24) {
    return false;
  }
  const address =
    octets[0] * 2 ** 24 +
    octets[1] * 2 ** 16 +
    octets[2] * 2 ** 8 +
    octets[3];
  const blockSize = 2 ** (32 - prefix);
  const isPrivate =
    octets[0] === 10 ||
    (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31) ||
    (octets[0] === 192 && octets[1] === 168);
  return isPrivate && address % blockSize === 0;
}

function isKubernetesName(value: string, maximumLength: number): boolean {
  return (
    value.length <= maximumLength &&
    /^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$/.test(value)
  );
}

function isX86InstanceType(value: string): boolean {
  try {
    return (
      new ec2.InstanceType(value).architecture ===
      ec2.InstanceArchitecture.X86_64
    );
  } catch {
    return false;
  }
}

export function resolveGlobalRouterChartPath(): string {
  const candidates = [
    path.resolve(process.cwd(), "helm/global-router"),
    path.resolve(__dirname, "../helm/global-router"),
    path.resolve(__dirname, "../../helm/global-router"),
  ];
  const chartPath = candidates.find((candidate) => {
    try {
      return fs.statSync(path.join(candidate, "Chart.yaml")).isFile();
    } catch {
      return false;
    }
  });
  if (chartPath === undefined) {
    throw new Error(
      `Global Router Helm chart not found. Searched: ${candidates.join(", ")}`,
    );
  }
  return chartPath;
}

function resolveMockerChartPath(): string {
  const candidates = [
    path.resolve(__dirname, "..", "helm", "dynamo-mocker"),
    path.resolve(__dirname, "..", "..", "helm", "dynamo-mocker"),
  ];
  const chartPath = candidates.find((candidate) =>
    fs.existsSync(path.join(candidate, "Chart.yaml")),
  );

  if (chartPath === undefined) {
    throw new Error(
      `Unable to locate the Dynamo Mocker Helm chart; checked ${candidates.join(", ")}.`,
    );
  }
  return chartPath;
}

function resolveKvDcRelayChartPath(): string {
  const candidates = [
    path.resolve(process.cwd(), "helm/kv-dc-relay"),
    path.resolve(__dirname, "../helm/kv-dc-relay"),
    path.resolve(__dirname, "../../helm/kv-dc-relay"),
  ];
  const chartPath = candidates.find((candidate) => {
    try {
      return fs.statSync(path.join(candidate, "Chart.yaml")).isFile();
    } catch {
      return false;
    }
  });
  if (chartPath === undefined) {
    throw new Error(
      `KV DC Relay Helm chart not found. Searched: ${candidates.join(", ")}`,
    );
  }
  return chartPath;
}
