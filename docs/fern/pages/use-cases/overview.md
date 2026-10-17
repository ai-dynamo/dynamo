---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Use Cases
subtitle: Find the guide for your workload
sidebar-title: Overview
---

Each section in this tab is organized around a workload. Start with the first
page of the section that matches what you are building; it covers backend
setup and a first working request.

## Workloads

| You want to… | Start here |
| --- | --- |
| Serve rollouts for reinforcement learning training | [Reinforcement Learning](reinforcement-learning/overview.md) |
| Serve coding agents and multi-step agent sessions | [Agents](agents/overview.md) |
| Answer questions about images | [Image Understanding](image-understanding/overview.md) |
| Answer questions about video | [Video Understanding](video-understanding/overview.md) |
| Transcribe speech, synthesize speech, or build a voice agent | [Speech](speech/voice-pipelines.md) |
| Generate images or video from a prompt | [Diffusion](diffusion/overview.md) |
| Speed up image and video workloads | [VLM Performance Optimization](image-understanding/vlm-performance-optimization.md) |
| Pack many small models onto shared GPUs | [GPU Sharing](gpu-sharing/overview.mdx) |

Tool calling and reasoning output parsing apply to every chat deployment and
are documented with the frontend in the Developer Guide:
[Tool Calling and Reasoning](tool-calling-and-reasoning/overview.mdx).

## Media Input and Output

| Modality | Understand (input) | Generate (output) |
| --- | --- | --- |
| Image | `image_url` in `/v1/chat/completions` on vLLM, SGLang, and TensorRT-LLM. See [Image Understanding](image-understanding/overview.md). | `/v1/images/generations` on vLLM-Omni, SGLang, and TensorRT-LLM (experimental). See [Text to Image](diffusion/text-to-image.md). |
| Video | `video_url` in `/v1/chat/completions` on vLLM, SGLang, and TensorRT-LLM. See [Video Understanding](video-understanding/overview.md). | `/v1/videos` on vLLM-Omni, SGLang, TensorRT-LLM (experimental), and FastVideo. See [Text to Video (T2V)](diffusion/text-to-video.md) and [Image to Video (I2V)](diffusion/image-to-video.md). |
| Speech and audio | `/v1/realtime` transcription on vLLM and Nemotron Speech NIMs. See [Speech to Text](speech/speech-to-text.md). | `/v1/audio/speech` and `/v1/realtime` on vLLM-Omni. See [Text-to-Speech](speech/text-to-speech.md) and [Speech-to-Speech](speech/speech-to-speech.md). |
