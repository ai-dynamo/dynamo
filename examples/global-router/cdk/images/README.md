<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Build service containers

Run from `cdk/`. Requirements: Git, Python 3, curl, Docker with Buildx, and a Dynamo
checkout containing the runtime changes described in the [deployment guide](../README.md#dynamodb-discovery-and-endpoint-requirements).
The script rejects tracked modifications in that checkout. Keep credentials and
unrelated files out of the Dynamo Docker build context.

```bash
bash images/build.sh /path/to/dynamo global-router-dev tested-revision
```

The multi-stage [Dockerfile](Dockerfile) builds three Linux/amd64 images:

| Image | Contents |
|---|---|
| `global-discovery-service` | Rust discovery service binary |
| `global-frontend` | Rust frontend including the global router |
| `pool-relay` | Python PoolRelay component and rebuilt Dynamo Python bindings |

The script checks the two binaries with `--help` and checks PoolRelay imports.
It downloads and checksum-verifies a pinned Swagger UI archive into ignored
`images/assets/`. Images are loaded into the selected Docker daemon; they are not
pushed automatically. The default prefix is `global-router-dev` and the default
tag is the first 12 characters of the Dynamo checkout's Git revision. Images also
carry the full source revision as a label. Build context files are governed by
the Dynamo checkout's Docker ignore rules, not this repository's Git ignore rules.

For a remote builder, configure `DOCKER_CONTEXT` and `BUILDX_BUILDER` before
invoking the script. Build caches and images live on the selected builder/daemon.
The release build disables LTO and limits compiler parallelism to reduce memory
use; account for this profile when comparing performance.

## Push to your registry

For ECR, create the three repositories first and use your own account and region:

```bash
export AWS_REGION=us-east-2
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
export IMAGE_PREFIX="$REGISTRY/global-router"
export IMAGE_TAG=tested-revision

aws ecr get-login-password --region "$AWS_REGION" |
  docker login --username AWS --password-stdin "$REGISTRY"
bash images/build.sh /path/to/dynamo "$IMAGE_PREFIX" "$IMAGE_TAG"
for component in global-discovery-service global-frontend pool-relay; do
  docker push "$IMAGE_PREFIX/$component:$IMAGE_TAG"
done
```

Set `discoveryImage` and `frontendImage` in your global configuration and
`relayImage` in your pool configuration to the pushed references, preferably
immutable digests. See [global](../config/examples/global.json) and
[pool](../config/examples/pool.json) examples. CDK consumes these references;
deployment does not build or push containers.

Pool frontend, mocker and GPU backend images are separate runtime images, such as
compatible Dynamo CI images. This build script does not build GPU serving images
or the Kubernetes operator.
