<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Backup and Restore

Confirm the Velero server and Dynamo backup schedule:

```bash
deploy/pre-deployment/pre-deployment-check.sh --require velero,velero-schedule --output json
velero schedule get
```

Create a manual backup before risky changes:

```bash
velero backup create dynamo-manual-$(date +%Y%m%d%H%M%S) --include-namespaces dynamo-system
```

Restore into a recovery cluster after confirming storage credentials and snapshot locations:

```bash
velero restore create --from-backup <backup-name>
```
