---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Global Discovery Service Registration Storage
subtitle: Keep pool registrations in memory or in a shared DynamoDB table
---

**Experimental.** With `--pool-discovery registrations`, the Global Discovery Service lists the pools
that [Pool Relays](pool-relay.md) register with `POST /pools`. The registration store decides where
those registrations live. This page explains when to use each store, how to set up DynamoDB, and how
to run and verify the service. For every flag and endpoint, see
[Global Discovery Service Configuration](../../reference/components/global-discovery-service-configuration.md).

## Choose a store

| | `memory` (default) | `dynamodb` |
| --- | --- | --- |
| Replicas | One. Each replica keeps only the registrations it receives. | Any number. Every replica reads the same table. |
| Restart | State is lost. | State survives. |
| `complete` in `GET /pools` | `false` until one lease has passed after start. | `true` at once. |
| Regions | One. | One, or several through a DynamoDB Global Table. |
| Extra setup | None. | A table, IAM permissions, and AWS credentials for the pods. |

Use `memory` for a single-replica development or lab deployment. Use `dynamodb` when the service must
survive restarts without a gap in the pool list, run more than one replica, or span AWS regions.

## Create the table

The table has a string partition key `pk` and a string sort key `sk`, on-demand billing, and a TTL
attribute named `expires_at`. Create it once per account and region:

```bash
aws dynamodb create-table \
  --table-name global-discovery-pools \
  --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
  --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

aws dynamodb wait table-exists --table-name global-discovery-pools --region us-east-1

aws dynamodb update-time-to-live \
  --table-name global-discovery-pools \
  --time-to-live-specification Enabled=true,AttributeName=expires_at \
  --region us-east-1

aws dynamodb update-continuous-backups \
  --table-name global-discovery-pools \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true \
  --region us-east-1
```

TTL removes registrations that nobody has renewed for seven days. It is garbage collection only; it
never decides whether a pool is live. Point-in-time recovery is optional but recommended.

### Add regions with a Global Table

To serve several regions, add a replica of the table in each one. Run this after the table is active:

```bash
aws dynamodb update-table \
  --table-name global-discovery-pools \
  --replica-updates '[{"Create": {"RegionName": "eu-west-1"}}]' \
  --region us-east-1
```

Repeat with another `RegionName` for each region. Enable point-in-time recovery in each region
separately. Replicas in each region point at their regional endpoint with `--dynamodb-region`. A Global
Table resolves concurrent writes to one item by last writer wins on the whole item, which matches how
the service treats registrations: whichever copy wins is a complete registration at most one heartbeat
old.

## Grant access

The service needs three actions on the table ARN, and the same in every replica region:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["dynamodb:Query", "dynamodb:UpdateItem", "dynamodb:DescribeTable"],
      "Resource": "arn:aws:dynamodb:us-east-1:111122223333:table/global-discovery-pools"
    }
  ]
}
```

Add `dynamodb:CreateTable` and `dynamodb:UpdateTimeToLive` only for a development deployment that uses
`--dynamodb-ensure-table`.

The service reads credentials from the AWS SDK default chain. On EKS, attach the policy to an IAM
role and give the Pod's ServiceAccount access to it.

With IAM Roles for Service Accounts (IRSA), annotate the ServiceAccount with the role ARN:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: global-discovery-service
  namespace: global-discovery
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::111122223333:role/global-discovery-service
```

With EKS Pod Identity, create an association instead and leave the ServiceAccount unannotated:

```bash
aws eks create-pod-identity-association \
  --cluster-name my-cluster \
  --namespace global-discovery \
  --service-account global-discovery-service \
  --role-arn arn:aws:iam::111122223333:role/global-discovery-service
```

## Run the service

Add the DynamoDB flags to the container arguments. Two replicas share one table:

```yaml
spec:
  replicas: 2
  template:
    spec:
      serviceAccountName: global-discovery-service
      containers:
        - name: global-discovery
          image: IMAGE
          args:
            - --listen=0.0.0.0:8080
            - --pool-discovery=registrations
            - --registration-storage=dynamodb
            - --dynamodb-table=global-discovery-pools
            - --dynamodb-region=us-east-1
            - --router=aws-ohio/global-router-0=http://10.42.0.5:50051
```

The complete manifest, with the Service and ServiceAccount, is
[`container/global-discovery-service-dynamodb.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/container/global-discovery-service-dynamodb.yaml).

The service checks the table at startup with `DescribeTable` and exits with an error that names the
table if it is missing. `--dynamodb-poll-interval-secs` (default 2) must be less than
`--heartbeat-interval-secs` (default 5), or the service refuses to start.

### Local development

For development, run [DynamoDB Local](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DynamoDBLocal.html)
and let the service create the table:

```yaml
          args:
            - --pool-discovery=registrations
            - --registration-storage=dynamodb
            - --dynamodb-table=global-discovery-pools
            - --dynamodb-region=us-east-1
            - --dynamodb-endpoint=http://dynamodb-local:8000
            - --dynamodb-ensure-table
          env:
            - name: AWS_ACCESS_KEY_ID
              value: local
            - name: AWS_SECRET_ACCESS_KEY
              value: local
```

`--dynamodb-ensure-table` creates the table and enables TTL, without point-in-time recovery or
replicas. Do not use it in production.

## How leases and expiry work

A registration is stored with the time it was renewed and the lease the accepting replica granted.
Replicas decide whether a pool is live when they read it, not when it is written: a pool is live while
`renewed_at + lease + skew` has not passed on the reading replica's clock. Nothing in the table is
removed when a lease lapses, so a pool that resumes heartbeating reappears without any repair.

`--lease-skew-secs` (default 2) adds slack for clock differences between replicas. Raise it if your
nodes' clocks drift further than that. Without it, a replica with a fast clock hides a pool slightly
early. The effect is only a pool appearing or disappearing a moment sooner or later on one replica.

Each item stores the lease its writer granted, so replicas configured with different
`--lease-duration-secs` values still judge each pool by the lease it was given.

## Verify

Port-forward to a replica and list the pools:

```bash
kubectl -n global-discovery port-forward deploy/global-discovery-service 8080:8080 &
curl -s localhost:8080/pools
```

With a registered Pool Relay, the response lists the pool, and `complete` is `true` even right after a
restart:

```json
{"pools": [{"key": {"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
            "location": {"region": "us-west-2", "availability_zone": null, "datacenter": null},
            "model": "Qwen/Qwen3-0.6B", "capabilities": ["aggregated"],
            "frontends": ["http://10.43.1.2:8000/"], "capacity": [], "relay_identities": []}],
 "complete": true}
```

Query another replica, or delete a Pod, and the same pools appear. `relay_identities` is empty because
registrations do not carry relay identities.

`GET /healthz` returns 200 while the table answers. It returns 503 when the table is unreachable, when
the Pod lacks permission, or when the table does not exist. The readiness probe in the example manifest
uses it, so the Pod leaves the Service until the table is back.

## Operate

- **Replicas.** Put any number of replicas behind one Kubernetes Service. Any replica accepts
  `POST /pools` and any replica answers `GET /pools`.
- **Rolling restarts.** Nothing special is needed. A new replica lists every live pool at once, and Pool
  Relays keep heartbeating through the Service.
- **Throttling and outages.** A failed DynamoDB call returns 503 from the endpoint that made it. A
  Pool Relay registers again on its next heartbeat, and pools stay listed while their leases last. On-demand
  capacity absorbs the expected write rate, roughly `pools x relay replicas / heartbeat interval`, for
  example 80 writes per second for 200 pools with two relay replicas at a 5 second heartbeat.
- **Corrupt items.** An item that cannot be decoded is logged and skipped. The rest of the list is served.
- **Scale limit.** All items share one partition key. That is sized for tens to hundreds of pools.
