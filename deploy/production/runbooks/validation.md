<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Validation

Run the baseline production checks after the add-ons are healthy:

```bash
deploy/pre-deployment/pre-deployment-check.sh --profile production --output json
```

Run the Dynamo-specific checks after `dynamo-platform` has synced:

```bash
deploy/pre-deployment/pre-deployment-check.sh --require dynamo-crds,dynamo-webhooks,kai-queue --output json
```

Check optional integrations only when they are installed:

```bash
deploy/pre-deployment/pre-deployment-check.sh --require keda,opentelemetry,parca,lws-volcano --output json
```
