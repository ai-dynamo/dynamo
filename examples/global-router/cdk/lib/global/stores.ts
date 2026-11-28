// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { CfnOutput, RemovalPolicy, Stack, StackProps, aws_dynamodb as dynamodb } from "aws-cdk-lib";
import { Construct } from "constructs";
import { GlobalStoresConfig, validateStores } from "../config";

/** A single owner for both MREC Global Tables, independent of regional workloads. */
export class GlobalStoresStack extends Stack {
  constructor(scope: Construct, id: string, props: StackProps & { config: GlobalStoresConfig }) {
    super(scope, id, { ...props, terminationProtection: true });
    const config = props.config;
    validateStores(config);
    if (this.region !== config.homeRegion) throw new Error("Deploy stores once, in homeRegion");
    for (const [id, name, partition, sort, ttl] of [
      ["Affinity", config.affinityTableName, "itemHashKey", "sortKey", "expiresAt"],
      ["Discovery", config.discoveryTableName, "pk", "sk", "expires_at"],
    ]) {
      const table = new dynamodb.TableV2(this, id, {
        tableName: name,
        partitionKey: { name: partition, type: dynamodb.AttributeType.STRING },
        sortKey: { name: sort, type: dynamodb.AttributeType.STRING },
        timeToLiveAttribute: ttl,
        billing: dynamodb.Billing.onDemand(),
        multiRegionConsistency: dynamodb.MultiRegionConsistency.EVENTUAL,
        deletionProtection: true,
        pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
        removalPolicy: RemovalPolicy.RETAIN,
        replicas: config.replicaRegions.map(region => ({
          region,
          deletionProtection: true,
          pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
        })),
      });
      new CfnOutput(this, `${id}TableName`, { value: table.tableName });
    }
    new CfnOutput(this, "TableRegions", { value: [config.homeRegion, ...config.replicaRegions].join(",") });
  }
}
