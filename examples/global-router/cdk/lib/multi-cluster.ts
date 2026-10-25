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

import { Duration, RemovalPolicy, aws_ec2 as ec2 } from "aws-cdk-lib";
import * as eks from "aws-cdk-lib/aws-eks-v2";
import { Construct, type IDependable } from "constructs";

import { cidrsOverlap, isValidIpv4Cidr, isValidVpcCidr } from "./cidr";
import type {
  DeploymentConfig,
  KarmadaControlPlaneConfig,
  MultiClusterConfig,
} from "./config";

export const KARMADA_NAMESPACE = "karmada-system";
export const KARMADA_RELEASE = "karmada";
/** The chart names the API server Service and its kubeconfig after the release. */
export const KARMADA_API_SERVICE = `${KARMADA_RELEASE}-apiserver`;
export const KARMADA_KUBECONFIG_SECRET = `${KARMADA_RELEASE}-kubeconfig`;
export const KARMADA_API_PORT = 5443;
/** The release that `crds/` and the setup image pins below come from. */
export const KARMADA_VERSION = "1.19.0";

/** `karmadactl register` rejects longer member names. */
const MAX_MEMBER_NAME_LENGTH = 48;

/**
 * Helm must finish waiting, and roll back if it has to, well inside the
 * kubectl Lambda's 15-minute limit.
 */
const KARMADA_HELM_TIMEOUT = Duration.minutes(10);

// The release chart pins the Karmada component images, but its setup jobs
// default to `latest`. cfssl v1.6.5 was current when Karmada 1.19.0 shipped;
// the kubectl image only publishes `latest`, so it is pinned by digest.
const CFSSL_IMAGE_TAG = "v1.6.5";
const KUBECTL_IMAGE_DIGEST =
  "sha256:9b58231669c7c371ecd976626a61d72ae7e46326f30932e8934a087f5f26fbd8";

// The chart's default certificate names. Setting `certs.auto.hosts` replaces
// the list, so these are kept next to the load balancer's names.
const KARMADA_DEFAULT_CERT_HOSTS = [
  "kubernetes.default.svc",
  "*.etcd.{{ .Release.Namespace }}.svc.{{ .Values.clusterDomain }}",
  "*.{{ .Release.Namespace }}.svc.{{ .Values.clusterDomain }}",
  "*.{{ .Release.Namespace }}.svc",
  "localhost",
  "127.0.0.1",
];

// From Karmada `KARMADA_VERSION`, `charts/karmada/_crds/bases/multicluster`;
// see crds/README.md.
const MCS_CRD_FILES = [
  "multicluster.x-k8s.io_serviceexports.json",
  "multicluster.x-k8s.io_serviceimports.json",
];

const EXCLUDE_SNAT_ENV = "AWS_VPC_K8S_CNI_EXCLUDE_SNAT_CIDRS";

export interface MultiClusterSupportProps {
  readonly cluster: eks.ICluster & { readonly albController?: IDependable };
  readonly config: MultiClusterConfig;
  readonly region: string;
  /** This cluster's VPC range. It must not be one of the peers. */
  readonly vpcCidr: string;
  /** Node groups the Karmada control plane runs on. */
  readonly nodeGroups: readonly IDependable[];
}

/**
 * Prepares one cluster for Karmada and the Multi-Cluster Services (MCS) API:
 *
 * - Pods in peer VPCs may reach this cluster's pods. MCS EndpointSlices list
 *   pod addresses, so callers in other clusters connect to pods directly.
 * - Traffic from pods here to peer VPCs keeps its pod address (no SNAT), so
 *   a peer can tell which pod is calling.
 * - The MCS `ServiceExport` and `ServiceImport` types are installed.
 * - In the hub only: the Karmada control plane, behind an internal load
 *   balancer that every member VPC can reach.
 *
 * Members join Karmada in Pull mode with `karmadactl register` (see the
 * README). The join uses a short-lived token from the hub, so it is a step
 * after deployment rather than a CloudFormation resource.
 */
export class MultiClusterSupport extends Construct {
  constructor(scope: Construct, id: string, props: MultiClusterSupportProps) {
    super(scope, id);
    const { cluster, config } = props;
    validateMultiClusterConfig(config, props.vpcCidr);

    for (const cidr of config.peerCidrs) {
      cluster.clusterSecurityGroup.addIngressRule(
        ec2.Peer.ipv4(cidr),
        ec2.Port.allTcp(),
        `MCS pod traffic from peer VPC ${cidr}`,
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

    // `overwrite` applies instead of creating, so CRDs that already exist
    // in a running cluster are adopted. They are kept on removal and
    // rollback: they may predate this stack, and deleting a CRD deletes every
    // ServiceExport and ServiceImport.
    new eks.KubernetesManifest(this, "McsCrds", {
      cluster,
      manifest: loadMcsCrds(),
      overwrite: true,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const karmada = config.karmadaControlPlane;
    if (karmada !== undefined) {
      if (cluster.albController === undefined) {
        throw new Error(
          "The Karmada API load balancer needs the AWS Load Balancer Controller.",
        );
      }
      const chart = new eks.HelmChart(this, "KarmadaControlPlane", {
        atomic: true,
        chart: karmadaChartUrl(karmada.version),
        cluster,
        createNamespace: true,
        namespace: KARMADA_NAMESPACE,
        release: KARMADA_RELEASE,
        // CDK never uninstalls it: the chart's uninstall hook also deletes the
        // MCS CRDs and its own namespace (see the README).
        removalPolicy: RemovalPolicy.RETAIN,
        timeout: KARMADA_HELM_TIMEOUT,
        values: createKarmadaControlPlaneValues(karmada, props.region),
        // Waits for the control plane pods, not for the load balancer's
        // address; the README's join steps check for that.
        wait: true,
      });
      chart.node.addDependency(cluster.albController);
      for (const nodeGroup of props.nodeGroups) {
        chart.node.addDependency(nodeGroup);
      }
    }
  }
}

/**
 * Returns `config` without the hub's Karmada control plane unless `install`
 * is set (`-c karmada=true`). The running clusters still have a Karmada that
 * was installed by hand, which Helm cannot install over (see the README).
 */
export function withKarmadaControlPlane(
  config: DeploymentConfig,
  install: boolean,
): DeploymentConfig {
  const multiCluster = config.multiCluster;
  if (install) {
    if (multiCluster?.karmadaControlPlane === undefined) {
      throw new Error(
        "-c karmada=true needs a deployment with a Karmada control plane (Ohio).",
      );
    }
    return config;
  }
  if (multiCluster?.karmadaControlPlane === undefined) {
    return config;
  }
  const { karmadaControlPlane: _notInstalled, ...member } = multiCluster;
  return { ...config, multiCluster: member };
}

export function validateMultiClusterConfig(
  config: MultiClusterConfig,
  vpcCidr: string,
): void {
  if (
    config.memberName.length > MAX_MEMBER_NAME_LENGTH ||
    !/^[a-z0-9]([-a-z0-9]*[a-z0-9])?$/.test(config.memberName)
  ) {
    throw new Error(
      `Karmada member name ${config.memberName} must be a lowercase DNS label of at most ${MAX_MEMBER_NAME_LENGTH} characters.`,
    );
  }
  if (config.peerCidrs.some((cidr) => !isValidVpcCidr(cidr))) {
    throw new Error(
      "Multi-cluster peer CIDRs must be private, network-aligned VPC ranges from /16 to /24.",
    );
  }
  if (config.peerCidrs.some((cidr) => cidrsOverlap(cidr, vpcCidr))) {
    throw new Error("Multi-cluster peer CIDRs must not overlap the cluster's own VPC.");
  }
  const karmada = config.karmadaControlPlane;
  if (karmada === undefined) {
    return;
  }
  if (karmada.version !== KARMADA_VERSION) {
    throw new Error(
      `Karmada ${karmada.version} is not supported: crds/ and the pinned setup images come from ${KARMADA_VERSION}. crds/README.md says how to move them together.`,
    );
  }
  if (karmada.apiSourceCidrs.some((cidr) => !isValidIpv4Cidr(cidr))) {
    throw new Error("Karmada API source CIDRs must be valid IPv4 CIDR ranges.");
  }
}

/** The chart that each Karmada release publishes. */
export function karmadaChartUrl(version: string): string {
  return (
    `https://github.com/karmada-io/karmada/releases/download/v${version}/` +
    `karmada-chart-v${version}.tgz`
  );
}

export function createKarmadaControlPlaneValues(
  config: KarmadaControlPlaneConfig,
  region: string,
): Record<string, unknown> {
  return {
    installMode: "host",
    cfssl: { image: { tag: CFSSL_IMAGE_TAG } },
    kubectl: { image: { digest: KUBECTL_IMAGE_DIGEST } },
    // Helm 4 waits until every resource is ready at the same time and counts
    // a deleted Job as never ready, so the setup Job must outlive the wait.
    staticResourceJob: {
      ttlSecondsAfterFinished: KARMADA_HELM_TIMEOUT.toSeconds() + 300,
    },
    certs: {
      mode: "auto",
      auto: {
        // The internal load balancer gets a name in this pattern.
        hosts: [...KARMADA_DEFAULT_CERT_HOSTS, `*.elb.${region}.amazonaws.com`],
      },
    },
    apiServer: {
      serviceType: "LoadBalancer",
      serviceAnnotations: {
        "service.beta.kubernetes.io/aws-load-balancer-type": "external",
        "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type": "ip",
        "service.beta.kubernetes.io/aws-load-balancer-scheme": "internal",
        "service.beta.kubernetes.io/load-balancer-source-ranges":
          config.apiSourceCidrs.join(","),
      },
    },
    // A propagated ServiceImport brings its derived Service and EndpointSlices
    // only with this gate on. The chart turns it off; Karmada's own default,
    // which `karmadactl init` keeps, is on.
    controllerManager: { featureGates: { PropagateDeps: true } },
  };
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

function loadMcsCrds(): Record<string, unknown>[] {
  const directory = resolveCrdDirectory();
  return MCS_CRD_FILES.map((file) => {
    const crd: unknown = JSON.parse(
      fs.readFileSync(path.join(directory, file), "utf8"),
    );
    if (
      crd === null ||
      typeof crd !== "object" ||
      (crd as { kind?: unknown }).kind !== "CustomResourceDefinition"
    ) {
      throw new Error(`crds/${file} is not a CustomResourceDefinition; see crds/README.md.`);
    }
    return crd as Record<string, unknown>;
  });
}

function resolveCrdDirectory(): string {
  const candidates = [
    path.resolve(__dirname, "..", "crds"),
    path.resolve(__dirname, "..", "..", "crds"),
  ];
  const directory = candidates.find((candidate) =>
    fs.existsSync(path.join(candidate, MCS_CRD_FILES[0])),
  );
  if (directory === undefined) {
    throw new Error(
      `Unable to locate the MCS CRDs; checked ${candidates.join(", ")}.`,
    );
  }
  return directory;
}
