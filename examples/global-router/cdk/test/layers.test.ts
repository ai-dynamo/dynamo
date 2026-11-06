// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { App } from "aws-cdk-lib";
import { Template } from "aws-cdk-lib/assertions";
import { ClusterTarget, GlobalLayerConfig, GlobalStoresConfig, PoolConfig } from "../lib/config";
import { GlobalStoresStack } from "../lib/global/stores";
import { GlobalLayerStack } from "../lib/global/stack";
import { globalManifests } from "../lib/global/manifests";
import { PoolStack } from "../lib/pools/stack";
import { poolManifests } from "../lib/pools/manifests";
import { RegionalFoundationStack } from "../lib/foundation/stack";
import { ClusterServicesStack } from "../lib/foundation/services";
import { mockerDgdSpec } from "../testing/workloads/mocker";
import { TestEnvironment, testEnvironmentStack } from "../testing/environment";

const account = "111111111111", region = "us-east-2";
const cluster: ClusterTarget = { name: "test-east", region, securityGroupId: "sg-00000000000000000",
  kubectlServiceToken: `arn:aws:lambda:${region}:${account}:function:kubectl`,
  oidcProviderArn: `arn:aws:iam::${account}:oidc-provider/oidc.eks.${region}.amazonaws.com/id/EXAMPLE` };
const stores: GlobalStoresConfig = { homeRegion: region, replicaRegions: ["us-west-2"],
  affinityTableName: "test-affinity", discoveryTableName: "test-discovery" };
const global: GlobalLayerConfig = { name: "gr", namespace: "global", cluster, stores, affinityScope: "test",
  frontendImage: "example/frontend:v1", discoveryImage: "example/discovery:v1",
  frontendReplicas: 2, discoveryReplicas: 2, discoveryMode: "registrations",
  ingress: { sourceCidrs: ["10.0.0.0/8"] } };
const pool: PoolConfig = { name: "blue", namespace: "pools", cluster, clusterId: "test-east",
  discoveryUrl: "http://gr-discovery.global.svc:8080", relayImage: "example/runtime:v1", relayReplicas: 2,
  discoveryMode: "registrations", frontendServiceName: "blue-frontend",
  frontendUrls: ["http://blue.example.invalid:8000"],
  dgdSpec: mockerDgdSpec({ image: "example/runtime:v1", model: "test/model", workers: 2, speedup: 1 }) };
const app = () => new App({ context: { "aws-cdk-lib:enablePartitionLiterals": true } });
const resources = (template: Template): any[] => Object.values(template.toJSON().Resources);

test("stores have distinct runtime schemas, protected replicas, backups and TTL", () => {
  const stack = new GlobalStoresStack(app(), "Stores", { env: { account, region }, config: stores });
  assert.equal(stack.terminationProtection, true);
  const rs = resources(Template.fromStack(stack));
  const tables = rs.filter(r => r.Type === "AWS::DynamoDB::GlobalTable");
  assert.equal(tables.length, 2);
  for (const table of tables) {
    assert.equal(table.DeletionPolicy, "Retain");
    assert.equal(table.UpdateReplacePolicy, "Retain");
    assert.equal(table.Properties.BillingMode, "PAY_PER_REQUEST");
    assert.deepEqual(table.Properties.Replicas.map((r: any) => r.Region).sort(), [region, "us-west-2"].sort());
    for (const replica of table.Properties.Replicas) {
      assert.equal(replica.DeletionProtectionEnabled, true);
      assert.equal(replica.PointInTimeRecoverySpecification.PointInTimeRecoveryEnabled, true);
    }
  }
  const affinity = tables.find(t => t.Properties.TableName === "test-affinity").Properties;
  assert.deepEqual(affinity.KeySchema, [{ AttributeName: "itemHashKey", KeyType: "HASH" }, { AttributeName: "sortKey", KeyType: "RANGE" }]);
  assert.deepEqual(affinity.TimeToLiveSpecification, { AttributeName: "expiresAt", Enabled: true });
  const discovery = tables.find(t => t.Properties.TableName === "test-discovery").Properties;
  assert.deepEqual(discovery.KeySchema, [{ AttributeName: "pk", KeyType: "HASH" }, { AttributeName: "sk", KeyType: "RANGE" }]);
  assert.deepEqual(discovery.TimeToLiveSpecification, { AttributeName: "expires_at", Enabled: true });
  assert.ok(!rs.some(r => r.Type.startsWith("AWS::EKS") || r.Type.startsWith("AWS::EC2")));
});

test("global stack has local-table-only IAM and no pools, cluster, compute or tables", () => {
  const template = Template.fromStack(new GlobalLayerStack(app(), "Global", { env: { account, region }, config: global }));
  const rs = resources(template);
  assert.ok(!rs.some(r => /AWS::(EKS::Cluster|EKS::Nodegroup|EC2::|DynamoDB::)/.test(r.Type)));
  const statements = rs.filter(r => r.Type === "AWS::IAM::Policy").flatMap(r => r.Properties.PolicyDocument.Statement);
  const ddb = statements.filter(s => JSON.stringify(s.Action).includes("dynamodb:"));
  assert.equal(ddb.length, 2);
  assert.deepEqual(ddb.find(s => s.Action.includes("dynamodb:GetItem")).Action.sort(), ["dynamodb:GetItem", "dynamodb:PutItem"]);
  assert.deepEqual(ddb.find(s => s.Action.includes("dynamodb:Query")).Action.sort(), ["dynamodb:DescribeTable", "dynamodb:Query", "dynamodb:UpdateItem"]);
  assert.ok(JSON.stringify(ddb.find(s => s.Action.includes("dynamodb:GetItem")).Resource).includes(`dynamodb:${region}:${account}:table/test-affinity`));
  assert.ok(JSON.stringify(ddb.find(s => s.Action.includes("dynamodb:Query")).Resource).includes(`dynamodb:${region}:${account}:table/test-discovery`));
  assert.ok(!JSON.stringify(ddb).includes("*"));
  const manifests = globalManifests(global);
  assert.ok(!manifests.some(m => ["DynamoGraphDeployment", "DynamoPoolRelay", "Namespace"].includes(m.kind)));
  const frontend = manifests.find(m => m.kind === "Deployment" && m.metadata.name === "gr-router");
  const container = frontend!.spec.template.spec.containers[0];
  assert.deepEqual(container.command, ["global-frontend"]);
  assert.ok(container.args.includes("--affinity-ddb-table") && container.args.includes("test-affinity"));
  assert.ok(container.args.includes("--relay-view-port"));
  assert.equal(container.args[container.args.indexOf("--router-policy") + 1], "affinity");
  assert.equal(container.args[container.args.indexOf("--affinity-store") + 1], "dynamodb");
  assert.equal(container.readinessProbe.httpGet.path, "/live");
  const discovery = manifests.find(m => m.kind === "Deployment" && m.metadata.name === "gr-discovery")!;
  assert.ok(discovery.spec.template.spec.containers[0].args.includes("test-discovery"));
  const discoveryArgs = discovery.spec.template.spec.containers[0].args;
  assert.ok(!discoveryArgs.includes("--dynamodb-ensure-table"));
  assert.equal(discoveryArgs[discoveryArgs.indexOf("--pool-discovery") + 1], "registrations");
  assert.equal(discoveryArgs[discoveryArgs.indexOf("--registration-storage") + 1], "dynamodb");
  assert.ok(!discoveryArgs.includes("--router-service"));
  assert.ok(!manifests.some(m => ["ServiceExport", "ServiceImport", "ClusterRole", "ClusterRoleBinding"].includes(m.kind)));
  assert.equal(discoveryArgs[discoveryArgs.indexOf("--router-discovery") + 1], "registrations");
  assert.ok(!discoveryArgs.includes("--router"));
  assert.equal(container.args[container.args.indexOf("--router-replica-id") + 1], "test-east/global/$(POD_UID)");
  assert.equal(container.args[container.args.indexOf("--router-advertise-url") + 1], "http://$(POD_IP):50051");
  assert.deepEqual(container.env, [
    { name: "POD_UID", valueFrom: { fieldRef: { fieldPath: "metadata.uid" } } },
    { name: "POD_IP", valueFrom: { fieldRef: { fieldPath: "status.podIP" } } },
  ]);
});

test("each pool owns just its DGD and relay; recipe is preserved", () => {
  const rs = resources(Template.fromStack(new PoolStack(app(), "Pool", { env: { account, region }, config: pool })));
  assert.ok(rs.every(r => r.Type === "Custom::AWSCDK-EKS-KubernetesResource"));
  assert.equal(rs.length, 2);
  const manifests = poolManifests(pool);
  assert.deepEqual(manifests.dgd.spec, pool.dgdSpec);
  assert.equal(manifests.dgd.apiVersion, "nvidia.com/v1beta1");
  const spec = manifests.dgd.spec;
  assert.ok((spec.components as any[]).every((c: any) => c.podTemplate.spec.containers[0].env.some((v: any) => v.name === "DYN_SERVING_LOAD_PUBLISH" && v.value === "1")));
  assert.equal(manifests.relay.kind, "DynamoPoolRelay");
  assert.equal(manifests.relay.metadata.name, manifests.dgd.metadata.name);
  const env = manifests.relay.spec.podTemplate.spec.containers[0].env;
  assert.equal(env.find((v: any) => v.name === "DYN_RELAY_ROUTING_POOL").value, "test-east/pools/blue");
  assert.equal(env.find((v: any) => v.name === "DYN_RELAY_ROUTING_POOL_FRONTENDS").value, "http://blue.example.invalid:8000");
  const gpu = { ...pool, dgdSpec: { components: [{ name: "Frontend", type: "frontend", replicas: 1, podTemplate: { spec: { containers: [{ name: "main", image: "gpu:v1" }] } } }] } };
  assert.deepEqual(poolManifests(gpu).dgd.spec, gpu.dgdSpec);
});

test("foundation has no workload charts; shared services retain namespaces", () => {
  const foundation = Template.fromStack(new RegionalFoundationStack(app(), "Foundation", { env: { account, region }, config: {
    name: "test-east", region, vpcCidr: "10.44.0.0/16", availabilityZones: ["us-east-2a", "us-east-2b"],
    systemInstanceType: "m5.large", systemNodeCount: 2,
  } }));
  assert.ok(!JSON.stringify(foundation.toJSON()).includes("DynamoGraphDeployment"));
  assert.ok(!JSON.stringify(foundation.toJSON()).includes("global-frontend"));
  const rs = resources(Template.fromStack(new ClusterServicesStack(app(), "Services", { env: { account, region }, config: {
    cluster, operatorNamespace: "dynamo-system", dynamoVersion: "1.6.0", workloadNamespaces: ["global", "pools"],
  } })));
  const ns = rs.find(r => r.Type === "Custom::AWSCDK-EKS-KubernetesResource");
  assert.equal(ns.DeletionPolicy, "Retain");
  assert.equal(ns.Properties.PruneLabel, undefined);
  assert.ok(!rs.some(r => r.Type === "AWS::EKS::Cluster"));
});

test("existing fleet services create only new namespaces without owning the operator", () => {
  const rs = resources(Template.fromStack(new ClusterServicesStack(app(), "ReuseServices", { env: { account, region }, config: {
    cluster, operatorNamespace: "existing-operator", manageOperator: false, workloadNamespaces: ["new-global", "new-pools"],
  } })));
  assert.ok(!rs.some(r => r.Type.includes("Helm")));
  const ns = rs.find(r => r.Type === "Custom::AWSCDK-EKS-KubernetesResource");
  assert.equal(ns.DeletionPolicy, "Retain");
  assert.ok(JSON.stringify(ns).includes("new-global"));
  assert.ok(JSON.stringify(ns).includes("new-pools"));
  assert.ok(!JSON.stringify(ns).includes("existing-operator"));
});

test("managed pool endpoint is private and relay registration waits for its hostname", () => {
  const config = { ...pool, frontendUrls: undefined, frontendEndpoint: { sourceCidrs: ["10.44.0.0/16", "10.45.0.0/16"] } };
  const endpoint = poolManifests(config).endpoint!;
  assert.equal(endpoint.metadata.annotations["service.beta.kubernetes.io/aws-load-balancer-scheme"], "internal");
  assert.deepEqual(endpoint.spec.loadBalancerSourceRanges, config.frontendEndpoint.sourceCidrs);
  assert.deepEqual(endpoint.spec.selector, { "nvidia.com/dynamo-graph-deployment-name": "blue", "nvidia.com/dynamo-component-type": "frontend" });
  const template = Template.fromStack(new PoolStack(app(), "ManagedEndpoint", { env: { account, region }, config })).toJSON();
  const entries = Object.entries(template.Resources) as [string, any][];
  const [addressId, address] = entries.find(([, r]) => r.Type === "Custom::AWSCDK-EKS-KubernetesObjectValue")!;
  assert.equal(address.Properties.JsonPath, ".status.loadBalancer.ingress[0].hostname");
  const relay = entries.find(([id]) => id.startsWith("Relay"))![1];
  assert.ok(JSON.stringify(relay.Properties.Manifest).includes(addressId));
  assert.ok(JSON.stringify(relay.Properties.Manifest).includes("DYN_RELAY_ROUTING_POOL_FRONTENDS"));
  assert.throws(() => poolManifests({ ...config, frontendUrls: pool.frontendUrls }), /Choose/);
  assert.throws(() => poolManifests({ ...config, frontendEndpoint: { sourceCidrs: [] } }), /source CIDRs/);
});

test("digest-pinned mocker components carry the explicit runtime version for admission", () => {
  const spec = mockerDgdSpec({ image: `example/runtime@sha256:${"a".repeat(64)}`,
    model: "test/model", workers: 2, speedup: 1, runtimeVersionOverride: "1.7.0" });
  for (const component of spec.components as any[]) assert.equal(component.runtimeVersionOverride, "1.7.0");
});

test("reject incomplete regional storage and discovery configurations before deployment", () => {
  assert.throws(() => globalManifests({ ...global, cluster: { ...cluster, region: "ap-northeast-1" } }), /local replica/);
  assert.throws(() => globalManifests({ ...global, stores: { ...stores, discoveryTableName: stores.affinityTableName } }), /separate tables/);
  assert.throws(() => globalManifests({ ...global, stores: { ...stores, replicaRegions: [region] } }), /unique/);
  assert.throws(() => poolManifests({ ...pool, frontendUrls: [] }), /frontendUrls/);
  assert.throws(() => poolManifests({ ...pool, relayImage: "runtime:latest" }), /explicit version/);
  assert.throws(() => new GlobalStoresStack(app(), "Stores", { env: { account, region: "us-west-2" }, config: stores }), /homeRegion/);
});

test("test composition selects independent lifecycles using the same application stacks", () => {
  const config: TestEnvironment = { account, name: "isolated-test", stores, regions: [{ id: "east", global,
    pools: [{ ...pool, workload: { image: "example/runtime:v1", model: "test/model", workers: 2, speedup: 1 } }],
  }] };
  for (const layer of ["stores", "global", "pool"]) {
    const a = app();
    const stack = testEnvironmentStack(a, config, layer, "east", "blue");
    assert.equal(a.node.children.filter(c => c instanceof GlobalStoresStack || c instanceof GlobalLayerStack || c instanceof PoolStack).length, 1);
    const rs = resources(Template.fromStack(stack));
    assert.ok(!rs.some(r => /AWS::(EKS::|EC2::)/.test(r.Type)));
    assert.equal(rs.filter(r => r.Type === "AWS::DynamoDB::GlobalTable").length, layer === "stores" ? 2 : 0);
    if (layer === "pool") assert.ok(rs.every(r => r.Type === "Custom::AWSCDK-EKS-KubernetesResource"));
  }
  assert.throws(() => testEnvironmentStack(app(), config, "pool", "east", "missing"), /configured -c pool/);
  assert.throws(() => testEnvironmentStack(app(), config, "global", "missing"), /configured -c member/);
});

test("global and pool layers reject non-DynamoDB discovery and static router membership", () => {
  for (const discoveryMode of ["mcs", "karmada", "memory"]) {
    assert.throws(() => globalManifests({ ...global, discoveryMode } as GlobalLayerConfig), /backed by DynamoDB/);
    assert.throws(() => poolManifests({ ...pool, discoveryMode } as PoolConfig), /backed by DynamoDB/);
  }
  assert.throws(() => globalManifests({ ...global, routers: [
    { id: "east/a", url: "http://router.example.invalid:50051" },
  ] } as unknown as GlobalLayerConfig), /Remove routers/);
});

test("router self-registration formats the primary IPv6 pod address as an authority", () => {
  const manifests = globalManifests({ ...global, podAddressFamily: "ipv6" });
  const router = manifests.find(m => m.kind === "Deployment" && m.metadata.name === "gr-router")!;
  const args = router.spec.template.spec.containers[0].args;
  assert.equal(args[args.indexOf("--router-advertise-url") + 1], "http://[$(POD_IP)]:50051");
});
