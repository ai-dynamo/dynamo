<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Registry IAM

Two roles touch the registry's DynamoDB table: the registry role that
`mcs-registry` pods assume, and an admin role for `mcs-admin` and Global
Table changes. Neither needs anything outside DynamoDB (and KMS, when the
table uses a customer managed key). Replace `<account>`, the regions and the
table name (`global-discovery-mcs` by default) below.

## Registry role

`mcs-registry` reads and writes the replica of its own region. One role is
usually shared by every region, so the policy lists the table ARN in each
replica region; a per-region role can list only its own.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "McsRegistryTableReadWrite",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:BatchWriteItem"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-east-1:<account>:table/global-discovery-mcs",
        "arn:aws:dynamodb:us-west-2:<account>:table/global-discovery-mcs"
      ]
    }
  ]
}
```

Why each action:

| Action | Used by |
| :---- | :---- |
| `GetItem` | `GetCluster`, the read before a registration, `Ping` (readiness) |
| `Query` | `LoadClusterSet`, `ListClusterSets`, `DeleteCluster` |
| `PutItem` | Cluster registration, export and endpoint set puts, the clusterset index and endpoint partition marker |
| `UpdateItem` | Cluster heartbeats |
| `DeleteItem` | `DeleteEndpointSet` |
| `BatchWriteItem` | `DeleteExport`, `DeleteCluster` |

Not granted on purpose: `Scan`, `DescribeTable`, any table or backup
management, the `by_updated_at` index (`.../table/global-discovery-mcs/index/*`; add
it to `Query` when an incremental refresh uses it) and DynamoDB Streams (add
`dynamodb:DescribeStream`, `GetRecords`, `GetShardIterator` and
`ListStreams` on `.../table/global-discovery-mcs/stream/*` if a change-hint reader is
added).

With a customer managed KMS key, also allow `kms:Decrypt`, `kms:Encrypt`,
`kms:GenerateDataKey*` and `kms:DescribeKey` on the key of each region,
with the condition `"StringEquals": {"kms:ViaService":
"dynamodb.<region>.amazonaws.com"}`. The default AWS owned key needs
nothing.

## Admin role

Used by `mcs-admin create-table` and by whoever adds or removes Global
Table replicas. Grant it to operators or a deployment pipeline, not to
pods. `mcs-admin describe` and `delete-cluster` need only the registry
role's actions.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "McsRegistryCreateTable",
      "Effect": "Allow",
      "Action": [
        "dynamodb:CreateTable",
        "dynamodb:DescribeTable",
        "dynamodb:DescribeTimeToLive",
        "dynamodb:UpdateTimeToLive",
        "dynamodb:DescribeContinuousBackups",
        "dynamodb:UpdateContinuousBackups",
        "dynamodb:TagResource",
        "dynamodb:ListTagsOfResource"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-east-1:<account>:table/global-discovery-mcs",
        "arn:aws:dynamodb:us-west-2:<account>:table/global-discovery-mcs"
      ]
    },
    {
      "Sid": "McsRegistryGlobalTableReplicas",
      "Effect": "Allow",
      "Action": [
        "dynamodb:UpdateTable",
        "dynamodb:CreateTableReplica",
        "dynamodb:DeleteTableReplica",
        "dynamodb:Scan",
        "dynamodb:Query",
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:BatchWriteItem"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-east-1:<account>:table/global-discovery-mcs",
        "arn:aws:dynamodb:us-west-2:<account>:table/global-discovery-mcs"
      ]
    },
    {
      "Sid": "McsRegistryReplicationServiceLinkedRole",
      "Effect": "Allow",
      "Action": "iam:CreateServiceLinkedRole",
      "Resource": "arn:aws:iam::<account>:role/aws-service-role/replication.dynamodb.amazonaws.com/AWSServiceRoleForDynamoDBReplication",
      "Condition": {
        "StringEquals": {"iam:AWSServiceName": "replication.dynamodb.amazonaws.com"}
      }
    }
  ]
}
```

- `CreateTable` and the TTL and backup actions are what
  `mcs-admin create-table` calls. `TagResource` is needed only when tags
  are passed at creation.
- Adding a replica (`update-table --replica-updates`) requires
  `UpdateTable` and `CreateTableReplica` on the table in the source region
  and the new region, plus the data actions DynamoDB uses to copy the
  existing items, and the replication service-linked role, created once
  per account.
- Deleting the table is deliberately absent; deletion protection is on by
  default. For a teardown, grant `dynamodb:DeleteTable` temporarily and
  turn protection off first with
  `aws dynamodb update-table --no-deletion-protection-enabled`.
- With a customer managed KMS key, the admin role also needs
  `kms:CreateGrant`, `kms:DescribeKey`, `kms:Decrypt` and
  `kms:GenerateDataKey*` on the key of each region.

## IRSA

`mcs-registry` runs in an EKS cluster in each registry region. With IAM
roles for service accounts, annotate its ServiceAccount with the role:

```yaml
metadata:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::<account>:role/mcs-registry
```

The EKS pod identity webhook injects `AWS_ROLE_ARN`,
`AWS_WEB_IDENTITY_TOKEN_FILE`, `AWS_REGION` and `AWS_DEFAULT_REGION`; the
AWS SDK default credential chain used by the store picks them up.

Trust policy skeleton of the registry role. Each EKS cluster has its own
OIDC provider, so there is one statement per cluster (one per region here).
Pin both the audience and the exact ServiceAccount:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "McsRegistryUsEast1",
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<account>:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/<EAST_OIDC_ID>"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.us-east-1.amazonaws.com/id/<EAST_OIDC_ID>:aud": "sts.amazonaws.com",
          "oidc.eks.us-east-1.amazonaws.com/id/<EAST_OIDC_ID>:sub": "system:serviceaccount:<namespace>:<serviceaccount>"
        }
      }
    },
    {
      "Sid": "McsRegistryUsWest2",
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<account>:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/<WEST_OIDC_ID>"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.us-west-2.amazonaws.com/id/<WEST_OIDC_ID>:aud": "sts.amazonaws.com",
          "oidc.eks.us-west-2.amazonaws.com/id/<WEST_OIDC_ID>:sub": "system:serviceaccount:<namespace>:<serviceaccount>"
        }
      }
    }
  ]
}
```

- `<namespace>` and `<serviceaccount>` are the namespace and name of the
  registry's ServiceAccount.
- Use regional STS endpoints (the default of current SDKs), so token
  exchange does not depend on another region.
- EKS Pod Identity works as well: the trust principal is
  `pods.eks.amazonaws.com` with `sts:AssumeRole` and `sts:TagSession`, and a
  pod identity association replaces the annotation. The permission policy is
  the same.
- A Job that runs `mcs-admin create-table` needs its own ServiceAccount
  bound to the admin role; keep it separate from the registry
  ServiceAccount.

## Local development

dynamodb-local accepts any credentials: export `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` and `AWS_REGION` with dummy values and pass
`--dynamodb-endpoint` (`mcs-registry`, and `mcs-admin` before the command)
or `--endpoint` (`mcs-admin create-table`).
