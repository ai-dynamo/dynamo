/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export const EXAMPLE_BACKENDS: Record<string, string> = {
  vllm: "vLLM",
  sglang: "SGLang",
  trtllm: "TensorRT-LLM",
  fastvideo: "FastVideo",
  tritonserver: "Triton Server",
  mocker: "Mocker",
  sample: "Sample",
  custom: "Custom",
};

export const EXAMPLE_PLATFORMS: Record<string, string> = {
  kubernetes: "Kubernetes",
  local: "Local / CLI",
};

export type ExampleVariant = {
  id: string;
  target: string;
  label: string;
  description: string;
};
