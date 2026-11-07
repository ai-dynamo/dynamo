<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bootstrap the contributor compiler cache

The canonical runner policy names `ai-dynamo-sccache`. Contributor builds use
`dynamo-ci-210086341041-us-west-2-pr-sccache`; they never read the canonical store.
The new Velonix main-only seeder role can list/read the source and write only
the contributor destination. It cannot write the source, delete cache entries,
or publish images/models. Confirm the live bucket owner, region, encryption,
and current `SCCACHE_S3_BUCKET` configuration before enabling the operation.
An SSE-KMS source additionally needs a reviewed decrypt grant on its specific
key; this draft grants no blanket KMS access.

Run `seed-contributor-sccache.yml` from current main with `execute=false` first.
The default preview considers at most 100,000 objects / 50 GiB written in the
last 30 days. It writes no objects. Review its JSON output, then run with
`execute=true` and the same bounds. `resume_after` supplies the `after_key`
input for subsequent bounded batches. Use age zero to consider older entries.

Copies stay inside S3, preserve object keys and sccache prefixes, check the
source ETag and use destination `If-None-Match: *`. Existing destination keys
are never replaced. New keys retain the destination's AES256 encryption and
normal PR-cache expiry. The 50-GiB bound counts attempted source bytes, so it
also bounds work when destinations already exist. The job fails on access,
encryption or other errors; it does not interpret those failures as misses.

Seeded entries are reusable only when the compiler, inputs, flags and cache-key
environment match. Copying the bucket does not guarantee hits. Read-only
preview and live cache-hit statistics are rollout checks; this local draft
has not copied data or changed AWS permissions.
