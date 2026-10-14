/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Choices offered by DeploymentCustomizer. Image tags and published container
 * repositories derive from releases.data.ts, so a release bump updates them.
 * The model list holds models that the examples, recipes, and guides already
 * reference; it is a starting point, and readers can type any model ID.
 */

import { ARTIFACTS, CURRENT_TAG, RELEASES } from "./releases.data";

export const DEFAULT_REGISTRY = "nvcr.io/nvidia/ai-dynamo";
export const DEFAULT_NAMESPACE = "dynamo-system";
export const DEFAULT_MOUNT_PATH = "/home/dynamo/.cache/huggingface";
export const LATEST_TAG = CURRENT_TAG;

export const PUBLISHED_REPOS: string[] = ARTIFACTS.filter((artifact) => artifact.category === "container").map(
  (artifact) => artifact.name,
);

export interface ImageTagOption {
  tag: string;
  date?: string;
  latest: boolean;
}

// The examples use the v1beta1 DynamoGraphDeployment API, served since Dynamo 1.2.0.
const MIN_RELEASE = [1, 2, 0];

function versionParts(version: string): number[] | null {
  const match = version.match(/^v?(\d+)\.(\d+)\.(\d+)$/);
  return match ? match.slice(1).map(Number) : null;
}

function atLeast(parts: number[], minimum: number[]): boolean {
  for (let i = 0; i < minimum.length; i++) {
    if (parts[i] !== minimum[i]) return parts[i] > minimum[i];
  }
  return true;
}

export const IMAGE_TAGS: ImageTagOption[] = RELEASES.filter(
  (release) => release.kind === "stable" || release.kind === "patch",
)
  .map((release) => ({ release, parts: versionParts(release.version) }))
  .filter(({ parts }) => parts !== null && atLeast(parts, MIN_RELEASE))
  .map(({ release }) => {
    const tag = release.version.replace(/^v/, "");
    return { tag, date: release.date, latest: tag === CURRENT_TAG };
  });

export type ModelCategory = "text" | "vision" | "image" | "video" | "audio";

export const MODEL_CATEGORIES: Record<ModelCategory, string> = {
  text: "Text generation",
  vision: "Vision language",
  image: "Image generation",
  video: "Video generation",
  audio: "Speech",
};

export interface ModelOption {
  id: string;
  size: string;
  category: ModelCategory;
  gated?: boolean;
}

export const MODEL_OPTIONS: ModelOption[] = [
  { id: "Qwen/Qwen3-0.6B", size: "0.6B", category: "text" },
  { id: "Qwen/Qwen3-8B", size: "8B", category: "text" },
  { id: "Qwen/Qwen3-14B", size: "14B", category: "text" },
  { id: "Qwen/Qwen3-32B", size: "32B", category: "text" },
  { id: "Qwen/Qwen3-32B-FP8", size: "32B · FP8", category: "text" },
  { id: "Qwen/Qwen3-30B-A3B", size: "30B MoE", category: "text" },
  { id: "Qwen/Qwen3-235B-A22B-FP8", size: "235B MoE · FP8", category: "text" },
  { id: "meta-llama/Llama-3.1-8B-Instruct", size: "8B", category: "text", gated: true },
  { id: "nvidia/Llama-3.1-8B-Instruct-FP8", size: "8B · FP8", category: "text" },
  { id: "meta-llama/Llama-3.3-70B-Instruct", size: "70B", category: "text", gated: true },
  { id: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", size: "8B", category: "text" },
  { id: "deepseek-ai/DeepSeek-R1", size: "671B MoE", category: "text" },
  { id: "openai/gpt-oss-20b", size: "21B MoE", category: "text" },
  { id: "openai/gpt-oss-120b", size: "117B MoE", category: "text" },
  { id: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8", size: "120B MoE · FP8", category: "text" },
  { id: "google/gemma-4-31B-it", size: "31B", category: "text" },
  { id: "Qwen/Qwen3-VL-2B-Instruct", size: "2B", category: "vision" },
  { id: "Qwen/Qwen2.5-VL-7B-Instruct", size: "7B", category: "vision" },
  { id: "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8", size: "30B MoE · FP8", category: "vision" },
  { id: "black-forest-labs/FLUX.1-schnell", size: "12B", category: "image" },
  { id: "black-forest-labs/FLUX.2-klein-4B", size: "4B", category: "image" },
  { id: "Qwen/Qwen-Image", size: "20B", category: "image" },
  { id: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", size: "1.3B", category: "video" },
  { id: "Wan-AI/Wan2.2-TI2V-5B-Diffusers", size: "5B", category: "video" },
  { id: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", size: "1.7B", category: "audio" },
];

export function modelCategory(id: string | null | undefined): ModelCategory | null {
  return MODEL_OPTIONS.find((option) => option.id === id)?.category ?? null;
}
