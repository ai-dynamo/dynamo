# Multimodal Profile System

This directory holds the per-backend lists of multimodal serving tests
(one file per backend — currently `vllm.py`). Each file declares a
`<BACKEND>_MULTIMODAL_PROFILES` list that
`tests/serve/test_<backend>.py` flattens into pytest configs via
`make_multimodal_configs(...)`.

## Why

Before the profile system, every multimodal test was a hand-written
`VLLMConfig(...)` block in `tests/serve/test_vllm.py`. Adding a new
model, a new topology, or a new payload variant required copy-pasting
~40 lines and keeping the marks / timeouts / `script_args` /
`request_payloads` aligned by hand. The profile system declares the
*shape* (model, topology, payload) and lets one generator build all
the configs.

## Three-layer hierarchy

```
MultimodalModelProfile           ← one HF model
  └─ topologies: dict[str, TopologyConfig]
       └─ TopologyConfig         ← one deployment shape (agg / e_pd / epd / p_d / ...)
            └─ tests: list[MmCase]
                 └─ MmCase       ← one concrete test case (payload + per-case overrides)
```

Each layer adds something the layer above can't:

| Layer | Owns | Why this layer? |
|-------|------|-----------------|
| `MultimodalModelProfile` | `name`, `short_name`, `gpu_marker`, `extra_vllm_args`, `gated`, `marks` | Per-model facts that are constant across topologies (HF id, default GPU class, model-wide pytest marks like `gated`). |
| `TopologyConfig` | `marks`, `timeout_s`, `delayed_start`, `profiled_vram_gib`, `requested_vllm_kv_cache_bytes`, `single_gpu` / `two_gpu`, `gpu_marker` override, `env` | Per-topology launch shape (how the deployment is wired — e.g. `agg` is one process, `epd` is encode + prefill + decode on three GPUs). The shape determines GPU count, VRAM bounds, and timeout. |
| `MmCase` | `payload`, `suffix`, `extra_script_args`, per-case `marks` / `timeout_s` / `env` overrides | Per-test variation within a topology — same launch shape, different payload (HTTP-URL vs b64-inline vs video) or different launch flag (e.g. `--frontend-decoding`). |

The topology key (`"agg"`, `"e_pd"`, etc.) is what `make_multimodal_configs`
looks up in `<BACKEND>_TOPOLOGY_SCRIPTS` to find the launch script.
**Multiple topology keys can map to the same script** — see the
`agg`/`agg_video` and `epd`/`epd_video` pairs in `vllm.py`. The split
exists because video tests need different VRAM bounds / timeouts /
`delayed_start` than image tests, even though both run
`agg_multimodal.sh`.

## What gets generated

For each `(profile, topology, case)` triple, `make_multimodal_configs`
emits one `EngineConfig` keyed:

```
mm_{topology}_{profile.short_name}[_{case.suffix}]
```

So `qwen3-vl-2b` running on the `agg` topology with two MmCases
(`""` and `"b64"`) produces:

```
mm_agg_qwen3-vl-2b
mm_agg_qwen3-vl-2b_b64
```

Marks attached to each EngineConfig are layered, in this order
(later wins on conflicts that pytest cares about — pytest just
collects all marks):

1. GPU marker (`gpu_1` / `gpu_2` / `gpu_4`) — from `topology.gpu_marker`
   or falling back to `profile.gpu_marker`
2. `pytest.mark.timeout(...)` — from `case.timeout_s` if set, else
   `topology.timeout_s`
3. Case marks (`case.marks`) if non-empty, else topology marks
   (`topology.marks`) — case overrides topology entirely
4. `profiled_vram_gib(...)` and `requested_vllm_kv_cache_bytes(...)`
   if set on the topology
5. `skipif(DYN_HF_GATED_MODELS_ENABLED unset)` if `profile.gated`
6. Profile-wide marks (`profile.marks`)

## End-to-end example

```python
# tests/serve/multimodal_profiles/vllm.py

VLLM_TOPOLOGY_SCRIPTS = {
    "agg": "agg_multimodal.sh",
    "agg_video": "agg_multimodal.sh",   # same script, different config row
    "e_pd": "disagg_multimodal_e_pd.sh",
    "epd": "disagg_multimodal_epd.sh",
    "epd_video": "disagg_multimodal_epd.sh",
    "p_d": "disagg_multimodal_p_d.sh",
}

VLLM_MULTIMODAL_PROFILES = [
    MultimodalModelProfile(
        name="Qwen/Qwen3-VL-2B-Instruct",
        short_name="qwen3-vl-2b",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=220,
                tests=[
                    MmCase(payload=make_image_payload(["green"])),
                    MmCase(suffix="b64", payload=make_image_payload_b64(["green"])),
                    MmCase(
                        suffix="frontend_decoding",
                        payload=make_image_payload(["green"]),
                        extra_script_args=["--frontend-decoding"],
                    ),
                ],
            ),
            "agg_video": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=600,
                delayed_start=60,
                profiled_vram_gib=8.2,
                requested_vllm_kv_cache_bytes=1_719_075_000,
                tests=[MmCase(payload=make_video_payload(["red", "static", "still"]))],
            ),
            ...
        },
    ),
    ...
]
```

Expands to:

```
mm_agg_qwen3-vl-2b                           # post_merge, gpu_1, t=220, image
mm_agg_qwen3-vl-2b_b64                       # post_merge, gpu_1, t=220, b64-image
mm_agg_qwen3-vl-2b_frontend_decoding         # post_merge, gpu_1, t=220, image, --frontend-decoding
mm_agg_video_qwen3-vl-2b                     # pre_merge,  gpu_1, t=600, video, kv-bounded
...
```

Each becomes one `VLLMConfig(...)` instance with the right
`script_name`, `script_args`, `marks`, `request_payloads`, and `env`,
and is wired into `vllm_configs` in `tests/serve/test_vllm.py`.

## Adding a new variant

| Want to add... | Where it goes |
|----------------|---------------|
| A new model | New `MultimodalModelProfile(...)` entry in `VLLM_MULTIMODAL_PROFILES` |
| A new topology shape (e.g. tensor-parallel) | New entry in `VLLM_TOPOLOGY_SCRIPTS` + a corresponding launch script in `examples/backends/vllm/launch/` |
| A new test case for an existing topology (e.g. test the same model with a different payload) | New `MmCase(...)` in that topology's `tests` list |
| A new topology bound (e.g. video needs longer timeout) | New `TopologyConfig` keyed by a distinct topology key (`agg_video`) — multiple keys can map to the same launch script |
| A new launch flag for one case only | `MmCase(extra_script_args=[...], ...)` |

## Layered helpers in `tests/utils/multimodal.py`

`tests/utils/multimodal.py` holds the dataclasses (`MmCase`,
`TopologyConfig`, `MultimodalModelProfile`), the generator
(`make_multimodal_configs`), and the payload factories shared across
profiles:

- `make_image_payload(expected_colors, *, max_attempts=1)` — HTTP-URL
  color-identification payload using the standard test image
- `make_image_payload_b64(expected_colors)` — same prompt, but inlines
  the LFS-tracked PNG as a `data:image/png;base64,...` URL.
  Materialization is deferred to first `.body` read so pytest
  collection doesn't fail when LFS hasn't pulled the image yet
- `make_video_payload(expected_phrases)` — local test video
- `make_audio_payload(expected_phrases)` — remote test WAV

Backend-specific files (`vllm.py`, future `trtllm.py`, etc.) compose
these factories with `MmCase`s; they don't define their own.
