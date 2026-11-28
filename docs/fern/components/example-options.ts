/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export const EXAMPLE_BACKENDS: Record<string, string> = {
  vllm: "vLLM",
  sglang: "SGLang",
  trtllm: "TensorRT-LLM",
  fastvideo: "FastVideo",
  tokenspeed: "TokenSpeed",
  mocker: "Mocker",
  sample: "Sample",
  custom: "Custom",
};

export const EXAMPLE_PLATFORMS: Record<string, string> = {
  kubernetes: "Kubernetes",
  local: "Local / CLI",
};

// Compact filter-chip labels for the catalog topics. Cards, search, and the
// navigation keep the full data-topic-label; unlisted topics fall back to it.
export const EXAMPLE_TOPIC_SHORT_LABELS: Record<string, string> = {
  "basic-serving": "Serving",
  routing: "Routing",
  "kv-cache-offloading": "KV Cache",
  "multimodal-serving": "Multimodal",
  workloads: "Workloads",
  diffusion: "Diffusion",
  "autoscaling-and-budgets": "Autoscaling",
  "observability-and-recovery": "Observability",
  "cloud-and-integrations": "Cloud",
  "development-examples": "Development",
};

export type ExampleVariant = {
  id: string;
  target: string;
  label: string;
  description: string;
};
