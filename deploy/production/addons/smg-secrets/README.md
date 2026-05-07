<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SMG Secrets - Production Addon

SMG Secrets mirrors the Hugging Face token into the `smg` namespace so the SMG router can load the gated DeepSeek tokenizer. It depends on the External Secrets Operator addon and a cluster-level secret store.

## Deployment

| Field | Value |
|---|---|
| Argo CD app | [`gitops/apps/66-smg-secrets.yaml`](../../gitops/apps/66-smg-secrets.yaml) |
| Source | This repository, [`hf-token-mirror.yaml`](hf-token-mirror.yaml) |
| Namespace | `smg` |
| Kustomization | [`kustomization.yaml`](kustomization.yaml) |
| Sync wave | `6` |
| Baseline | Yes, as part of the SMG path |

## What This Addon Owns

| Resource | Purpose |
|---|---|
| `ExternalSecret/smg-hf-token` | Reads `huggingface/token` from the configured secret store. |
| `Secret/smg-hf-token` | Target Kubernetes Secret with key `HF_TOKEN`. |

## What This Addon Does Not Own

- It does not install External Secrets Operator.
- It does not create the `ClusterSecretStore`.
- It does not create the upstream cloud secret.
- It does not mirror the Postgres password; `smg-postgres` owns that local secret until replaced by a production external secret.

## Required External Contract

The manifest expects:

- `ClusterSecretStore/aws-secrets`
- remote secret key `huggingface/token`
- remote property `token`

If the cluster uses another provider or secret path, update `hf-token-mirror.yaml` and the matching infrastructure code together.

## Verify

```bash
kubectl -n smg get externalsecret smg-hf-token
kubectl -n smg get secret smg-hf-token
kubectl -n smg describe externalsecret smg-hf-token
```

## Upgrade Notes

ESO API changes can affect this manifest. Keep this file aligned with the installed External Secrets Operator CRD version.

Upstream reference: [External Secrets Operator documentation](https://external-secrets.io/).
