# SGLang runtime hotfix patches

Applied in sorted order by `container/templates/sglang_runtime.Dockerfile`
(`git apply` against `/sgl-workspace/sglang`, the upstream SGLang repo-root
layout carried by the `lmsysorg/sglang:<runtime_image_tag>` base image pinned
in `container/context.yaml`). Only files ending in `.patch` are applied; this
README is ignored by the build step. Patches must not touch test paths.

Target base: `lmsysorg/sglang:v0.5.18-cu130-runtime` (SGLang v0.5.18).

| File | Scope | Upstream status |
| --- | --- | --- |
| `01-pd-dsa-state-index-exact-match.patch` | PD disaggregation: DSA indexer state silently truncated on prefix-cache hit (prefill payload sliced to `extend_range` while decode registers `origin_input_len`); widen the prefill payload, add `StateType.DSA` to the exact-length set, log every mismatch. | Not yet upstream. |
| `02-pd-eagle-sampled-verification.patch` | TODO (placeholder, not present yet): EAGLE speculative decoding sampled-verification mismatch under PD. Root-cause analysis pending; add the patch file here once ready. | Pending RCA. |
