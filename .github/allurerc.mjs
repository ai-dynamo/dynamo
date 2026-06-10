// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from "allure";

// Labels use "dynamo_" prefix to avoid collision with allure-pytest's
// built-in "framework" label (always set to "pytest").
const byLabel = (name, value) => ({ labels }) =>
  labels.find((l) => l.name === name && l.value === value);

// Also match on pytest marker tags (e.g. @pytest.mark.vllm adds tag "vllm")
// as a fallback for tests where conftest.py hook doesn't fire.
const byTagOrLabel = (tagValue, labelName, labelValue) => ({ labels }) =>
  labels.find((l) => l.name === labelName && l.value === labelValue) ||
  labels.find((l) => l.name === "tag" && l.value === tagValue);

// Environment matcher: matches on framework tag + platform label
const envMatcher = (framework, platform) => ({ labels }) => {
  const hasFramework =
    labels.find((l) => l.name === "dynamo_framework" && l.value === framework) ||
    labels.find((l) => l.name === "tag" && l.value === framework);
  const hasPlatform = platform
    ? labels.find((l) => l.name === "dynamo_platform" && l.value === platform)
    : true;
  return hasFramework && hasPlatform;
};

export default defineConfig({
  name: "Dynamo Test Health",
  output: "./allure-report",
  historyPath: "./history.jsonl",
  environments: {
    "vllm-amd64": { name: "vLLM (amd64)", matcher: envMatcher("vllm", "amd64") },
    "vllm-arm64": { name: "vLLM (arm64)", matcher: envMatcher("vllm", "arm64") },
    "sglang-amd64": { name: "SGLang (amd64)", matcher: envMatcher("sglang", "amd64") },
    "sglang-arm64": { name: "SGLang (arm64)", matcher: envMatcher("sglang", "arm64") },
    "trtllm-amd64": { name: "TRT-LLM (amd64)", matcher: envMatcher("trtllm", "amd64") },
    "trtllm-arm64": { name: "TRT-LLM (arm64)", matcher: envMatcher("trtllm", "arm64") },
  },
  plugins: {
    pr: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "PR",
        singleFile: false,
        filter: byLabel("dynamo_workflow", "PR"),
        publish: true,
      },
    },
    postMerge: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "Post-Merge",
        singleFile: false,
        filter: byLabel("dynamo_workflow", "Post-Merge CI Pipeline"),
        publish: true,
      },
    },
    nightly: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "Nightly",
        singleFile: false,
        filter: byLabel("dynamo_workflow", "Nightly CI Pipeline"),
        publish: true,
      },
    },
    release: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "Release",
        singleFile: false,
        filter: byLabel("dynamo_workflow", "Release Pipeline"),
        publish: true,
      },
    },
    vllm: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "vLLM",
        singleFile: false,
        filter: byTagOrLabel("vllm", "dynamo_framework", "vllm"),
        publish: true,
      },
    },
    sglang: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "SGLang",
        singleFile: false,
        filter: byTagOrLabel("sglang", "dynamo_framework", "sglang"),
        publish: true,
      },
    },
    trtllm: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "TRT-LLM",
        singleFile: false,
        filter: byTagOrLabel("trtllm", "dynamo_framework", "trtllm"),
        publish: true,
      },
    },
  },
});
