<!-- SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Audit Subsystem

Captures per-request `AuditRecord`s (full request + response bodies) and
fans them out to one or more sinks selected via `DYN_AUDIT_SINKS`.

## S3 Sink

Selected by adding `s3` to `DYN_AUDIT_SINKS`. Uploads rotated,
gzip-compressed NDJSON segments to an S3-compatible endpoint. Each record
contains the complete request and response for one inference call.

### IAM Setup (EKS / IRSA)

The frontend pod's ServiceAccount must carry an IRSA annotation
pointing at an IAM role that grants S3 write access. Minimum viable
setup:

**1. Trust policy** on the IAM role (substitute your cluster's OIDC
issuer, account ID, namespace, and SA name):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/oidc.eks.<REGION>.amazonaws.com/id/<OIDC_ID>"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringLike": {
        "oidc.eks.<REGION>.amazonaws.com/id/<OIDC_ID>:sub":
          "system:serviceaccount:<NAMESPACE>:*-k8s-service-discovery"
      },
      "StringEquals": {
        "oidc.eks.<REGION>.amazonaws.com/id/<OIDC_ID>:aud": "sts.amazonaws.com"
      }
    }
  }]
}
```

**2. Permission policy** (inline on the role):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
    "Resource": "arn:aws:s3:::<BUCKET>/<PREFIX>/*"
  }]
}
```

If using SSE-KMS, add:

```json
{ "Effect": "Allow", "Action": "kms:GenerateDataKey", "Resource": "<KMS_KEY_ARN>" }
```

**3. Annotate the ServiceAccount:**

```bash
kubectl annotate sa <DGD_NAME>-k8s-service-discovery \
  -n <NAMESPACE> \
  eks.amazonaws.com/role-arn=arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_NAME>
```

**4. Restart the frontend** so the EKS pod-identity-webhook injects the
OIDC token:

```bash
kubectl rollout restart deployment <DGD_NAME>-frontend -n <NAMESPACE>
```

### Recommended Pod Settings

**`terminationGracePeriodSeconds`**: The S3 sink flushes and uploads the
in-flight segment during shutdown. `PutObject` can take 5–30 s under
throttling or large-segment conditions. The default K8s value of 30 s
may not be enough if the final segment is large AND S3 is throttling.

Recommended: set `terminationGracePeriodSeconds: 60` (or higher) on
frontend pods that have audit enabled. This gives the sink time to drain
the bus, flush the gzip buffer, and complete the final upload before
SIGKILL.

If the pod is killed before the upload completes, the in-flight segment
(up to `DYN_AUDIT_S3_ROLL_INTERVAL_MS` worth of records) is lost. For
test-fixture and tuning-corpus use cases this is acceptable; for strict
compliance it may warrant the `spool-to-disk-on-failure` extension
described in the design docs.

### Environment Variables

See `lib/runtime/src/config/environment_names.rs` for the full list.
Key ones:

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DYN_AUDIT_SINKS` | yes | (empty) | Comma-separated sink names. Include `s3`. |
| `DYN_AUDIT_S3_BUCKET` | yes (when s3) | — | S3 bucket name |
| `DYN_AUDIT_S3_REGION` | recommended | SDK chain | Bucket region |
| `DYN_AUDIT_S3_PREFIX` | no | `dynamo-audit` | Key prefix |
| `DYN_AUDIT_S3_ROLL_INTERVAL_MS` | no | `60000` | Time-based rotation (ms) |
| `DYN_AUDIT_S3_SSE` | no | `AES256` | `AES256`, `aws:kms`, or `none` |
| `DYN_AUDIT_FORCE_LOGGING` | no | `false` | Capture all requests regardless of `store` flag |
| `DYN_AUDIT_SAMPLE_RATE` | no | `1.0` | Head-based sampling [0.0, 1.0] |
| `DYN_AUDIT_DEPLOYMENT` | no | auto from `DYN_PARENT_DGD_K8S_NAME` | Deployment label in record + key partition |

### Object Key Format

```
<prefix>/[<deployment>/]YYYY/MM/DD/HH/<pod-name>-<startup-uuid8>-<seq>.jsonl.gz
```

Partition-friendly for Athena / Glue. Each `<pod-name>-<startup-uuid8>`
pair is unique per process lifetime.
