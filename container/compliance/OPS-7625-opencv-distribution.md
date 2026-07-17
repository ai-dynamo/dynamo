<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OPS-7625 — opencv-python-headless is still distributed after the whiteout

## TL;DR

PR #11607 removes `opencv-python-headless` with `rm -rf` (a **whiteout**) in the
`pre_runtime` stage, which is `FROM ${RUNTIME_IMAGE}` (upstream
`nvcr.io/nvidia/tensorrt-llm/release`, which ships opencv in its base layers).

A whiteout only *hides* files from the merged filesystem view. **The bytes remain
in the upstream base layer and are still pushed/distributed.** Harrison and Tyler
are correct. The runtime build uses `docker buildx build --push` with **no
`--squash`** anywhere, so nothing flattens those layers away.

The media-codec scan gate (PR #11632, `scan_codecs.py --root /`) walks the
*filesystem view*, so it reports opencv as absent and **passes green** even though
the bytes still ship — a false green.

## Proof (reproducible, ~10s)

Reproduces #11607's exact `runtime_full → pre_runtime (whiteout + squash COPY) →
runtime` pattern with a synthetic base that ships opencv's vendored libavcodec:

```bash
mkdir -p /tmp/opencv-repro && cd /tmp/opencv-repro

cat > Dockerfile.base <<'EOF'
FROM busybox:latest
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/opencv_python_headless.libs && \
    echo "OPENCV_LIBAVCODEC_SECRET_BYTES_v1" > /usr/local/lib/python3.12/dist-packages/opencv_python_headless.libs/libavcodec.so.62
EOF
docker build -f Dockerfile.base -t repro-upstream:latest .

cat > Dockerfile.dynamo <<'EOF'
FROM repro-upstream:latest AS runtime_full
RUN rm -rf /usr/local/lib/python3.12/dist-packages/opencv_python_headless.libs
FROM repro-upstream:latest AS pre_runtime
RUN rm -rf /usr/local/lib/python3.12/dist-packages/opencv_python_headless.libs
COPY --from=runtime_full / /
FROM pre_runtime AS runtime
EOF
docker build -f Dockerfile.dynamo -t repro-runtime:latest .

# (A) filesystem view — what the container and scan_codecs.py --root / see:
docker run --rm repro-runtime:latest ls /usr/local/.../opencv_python_headless.libs/
#   -> No such file or directory  (gate PASSES, container clean)

# (B) distributed bytes — what `docker push`/`docker save` actually ship:
docker save repro-runtime:latest -o r.tar && mkdir u && tar -xf r.tar -C u
grep -rl OPENCV_LIBAVCODEC_SECRET_BYTES_v1 u/
#   -> u/blobs/sha256/<...>   THE OPENCV BYTES ARE STILL IN A SHIPPED LAYER
```

Result: **(A) absent from the fs view, (B) present in the pushed layers.** That is
the exact gap: the running container is clean, the gate is green, the distributed
artifact still contains opencv.

## Why there is no Dockerfile-only fix

To drop the bytes the final image must not include the upstream base layers. The
only ways to achieve that:

- **`FROM scratch` + `COPY --from=<stage> / /`** (buildkit-native flatten). Copies
  the *merged* view, so whiteout-hidden files never make it in. **Impractical
  here:** `scratch` inherits none of upstream's ~50 env vars (PATH, LD_LIBRARY_PATH,
  CUDA_*, NIXL_*, DALI_*, …). `pre_runtime` even relies on inherited `${PATH}`.
  Re-declaring all of it by hand is brittle and breaks on every upstream tag bump.
- **Build-time squash.** Not supported by `docker buildx` (buildkit) or the classic
  builder in this toolchain.

## Options (ranked)

1. **Confirm the requirement with OSRB/legal first.** The opencv in question is
   shipped by NVIDIA's *own* `tensorrt-llm/release` base image. Is byte-level
   removal from the derived image required, or is runtime-inaccessibility (the
   current whiteout) sufficient given the base already carries it under its own
   clearance? OPS-7625 reads as "remove it," so assume byte-removal until told
   otherwise — but this question could make the rest moot.
2. **Strip opencv in the upstream base** (`tensorrt-llm/release`) — cleanest, no
   build hack, but cross-team and slow; unlikely to land for an imminent RC.
3. **Post-build squash in CI** (`docker-squash`). Validated below: it removes the
   whiteout-hidden bytes *and* preserves config/filesystem. Cost: the runtime build
   is `buildx --push` multi-arch, so this means pull-per-arch → squash → re-push →
   rebuild the manifest list. Real rework of release-critical CI; must be validated
   on a real build.

### docker-squash mechanism is validated

```bash
docker-squash -t repro-runtime:squashed repro-runtime:latest
docker save repro-runtime:squashed -o s.tar && tar -xf s.tar -C u2
grep -rl OPENCV_LIBAVCODEC_SECRET_BYTES_v1 u2/    # -> (nothing) bytes GONE
docker run --rm repro-runtime:squashed cat /usr/local/lib/keepme.txt  # -> config/fs intact
```

## Note on the scan gate (#11632)

Because it scans the filesystem view, the gate cannot see residual layer bytes. If
byte-removal is the requirement, the gate should additionally scan the layer blobs
(`docker save` tarballs) so it fails on whiteout-only "removals" instead of passing
them green.
