---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: MiniMax-H3
subtitle: Generate video with a synchronized audio track (T2VA) using vLLM-Omni
---

MiniMax-H3 generates video with a synchronized audio track (text-to-video-and-audio, T2VA). It runs on vLLM-Omni and needs 4×B200 GPUs. Because the standard Dynamo image encodes VP9 video only, MiniMax-H3 also needs a separate image layer that muxes H.264 video and AAC audio.

For text-to-video without audio, see [Text-to-Video](text-to-video.md).

## Before You Begin

Set up vLLM-Omni as described in [Text-to-Video › Set up the backend](text-to-video.md#set-up-the-backend).

## Run MiniMax-H3

The initial MiniMax-H3 qualification serves text-to-video-and-audio (T2VA) with
one aggregated diffusion worker. The launcher passes `--task-type fl2va`, which
loads only H3's FL2VA checkpoint partition. Dynamo's standard image stays on
its VP9-only media stack, so build the opt-in video-audio overlay to mux H.264
video and AAC audio:

```bash
export DYN_H3_DYNAMO_REVISION="$(git rev-parse HEAD)"
export DYN_H3_BASE_IMAGE="dynamo:${DYN_H3_DYNAMO_REVISION}-vllm-runtime"
export DYN_H3_IMAGE="dynamo:minimax-h3-${DYN_H3_DYNAMO_REVISION}"

python container/render.py --framework vllm --target runtime --output-short-filename
docker build -t "$DYN_H3_BASE_IMAGE" -f container/rendered.Dockerfile .
docker build \
  --build-arg BASE_IMAGE="$DYN_H3_BASE_IMAGE" \
  -f examples/backends/vllm/omni/video_audio.Dockerfile \
  -t "$DYN_H3_IMAGE" .
export DYN_H3_IMAGE_REF="$(docker image inspect --format '{{.Id}}' "$DYN_H3_IMAGE")"
```

Start a shell with exactly four B200 GPUs visible. The commands pass the tested
source revision and image ID into the qualification record:

```bash
container/run.sh \
  --image "$DYN_H3_IMAGE" \
  --gpus device=0,1,2,3 \
  --mount-workspace \
  -e DYN_H3_DYNAMO_REVISION \
  -e DYN_H3_IMAGE_REF \
  -e DYN_H3_MODEL \
  -e DYN_H3_MODEL_PATH \
  -e DYN_H3_MODEL_REVISION \
  -e DYN_H3_FASTH3_LORA_PATH \
  -e DYN_H3_FASTH3_VARIANT \
  -e DYN_H3_FASTVIDEO_VSA_TOPK \
  -e HF_TOKEN \
  -it
```

Launch the profile inside the container. It uses four-way Ulysses and text-encoder tensor
parallelism while keeping VAE patch parallelism at one, so small supported
resolutions such as 448x256 still have at least one tile per participating
rank. Larger resolutions can opt into four-way VAE patch parallelism with
`DYN_H3_VAE_PATCH_PARALLEL_SIZE=4`:

```bash
bash examples/backends/vllm/launch/agg_omni_minimax_h3.sh \
  > /tmp/dynamo-minimax-h3.log 2>&1 &
```

When an offline cache contains only H3's `FL2VA/` partition, set
`DYN_H3_MODEL_PATH` to that partition directory. The launcher keeps
`DYN_H3_MODEL` as the public API model name and passes the local partition as
the engine path.

To use the four-step Dense FastH3 adapter, download and pin the startup adapter:

```bash
export DYN_H3_FASTH3_LORA_PATH="$(hf download \
  FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors \
  --revision bcf40ca6f457ed66f8badf13514943e390205fca)"
```

The launcher fuses this adapter into the checkpoint at load time. It is not a
request-time LoRA and cannot be combined with another request-time adapter.
The qualification script verifies the adapter SHA-256, requests four inference
steps, omits request scheduler-shift overrides, and records request duration and
adapter provenance. FastH3 v1 is T2VA-only and does not support CPU or layerwise
offload.

To use FastH3 with Variable Sparse Attention (VSA), download a VSA adapter and
identify the variant to the launcher and qualification script:

```bash
export DYN_H3_FASTH3_VARIANT=vsa-datafree
export DYN_H3_FASTH3_LORA_PATH="$(hf download \
  FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  vsa-datafree/adapter_model.safetensors \
  --revision bcf40ca6f457ed66f8badf13514943e390205fca)"
```

This path requires vLLM-Omni `v0.29.0rc1` or later. The video-audio overlay
pins `fastvideo-kernel==0.3.5`. For a `vsa-*` adapter, the launcher selects
`FASTVIDEO_VSA` and uses the portable Triton route, retaining 64 key/value
blocks per query block. Override the top-k value with
`DYN_H3_FASTVIDEO_VSA_TOPK`. FastH3 VSA supports pure Ulysses sequence
parallelism; ring and all-gather degrees must remain 1. Check the first-forward
worker logs for `FASTVIDEO_VSA H3 routing` and verify that it does not fall back
to dense SDPA.

Run the supplied end-to-end qualification against the same worker. It requests
a 10-second clip, then verifies a non-empty H.264 stream at 24 FPS and a
non-silent 32-kHz stereo AAC stream:

```bash
export DYN_H3_QUAL_DIR=/tmp/dynamo_minimax_h3_qualification
bash examples/backends/vllm/launch/validate_omni_minimax_h3.sh
```

```json
{
  "model": "MiniMaxAI/MiniMax-H3",
  "prompt": "A cat playing Canon in D on a grand piano.",
  "size": "448x256",
  "response_format": "url",
  "output_format": "mp4",
  "nvext": {
    "fps": 24,
    "num_inference_steps": 50,
    "seed": 42
  },
  "task": "t2va",
  "duration": 10.0,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "aspect_ratio": "16:9"
}
```

The launcher pins model revision `42ed227ee7df40d41602854ae760620d6eb651fe`.
Set `DYN_H3_MODEL_REVISION` to test another revision. Dynamo preserves
unrecognized top-level video fields and the vLLM-Omni adapter merges them into
the pipeline's `extra_args`, where the upstream model validates them. Client
libraries that expose an `extra_body` option can use it to add these top-level
fields; do not send a literal nested `extra_body` object. Generated responses
report `fps` and `audio_sample_rate` for each MP4. The qualification output
records the exact Dynamo revision, container image ID, model revision, package
versions, visible GPU models, request, media checksum, and stream probe data.

First/last-frame FL2VA and reference-driven Ref2VA are not part of this initial
qualification. They additionally require typed image, video, and audio input
references.
