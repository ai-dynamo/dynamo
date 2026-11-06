<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Global Router infrastructure

**Experimental.** AWS CDK infrastructure for NVIDIA Dynamo global routing, DynamoDB-backed discovery and
session affinity, serving pools, and optional test environments and observability.

- [Architecture, configuration and deployment](cdk/README.md)
- [Build the three service containers](cdk/images/README.md)
- [Reusable test environments and smoke checks](cdk/testing/README.md)
- [CloudWatch dashboards and metric semantics](cdk/lib/observability/README.md)
- [Prepare a source export](SHARING.md)

The registration-based deployment requires images incorporating
[router self-registration PR #18198](https://github.com/ai-dynamo/dynamo/pull/18198),
which is not yet merged. Unmodified `main` images do not support these deployment flags.

Deployment requires your own AWS account, networking, compatible Dynamo runtime
images and operator. Example configurations use placeholder resource identifiers.

Licensed under [Apache-2.0](LICENSE). Vendored CRD attribution is preserved in
[cdk/crds/README.md](cdk/crds/README.md).
