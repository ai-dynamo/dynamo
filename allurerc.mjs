// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from "allure";

const byLabel = (name, value) => ({ labels }) =>
  labels.find((l) => l.name === name && l.value === value);

export default defineConfig({
  name: "Dynamo Test Health",
  output: "./allure-report",
  historyPath: "./history.jsonl",
  plugins: {
    dashboard: {
      options: {
        reportName: "Dashboard",
        singleFile: false,
        reportLanguage: "en",
        publish: true,
      },
    },
    pr: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "PR",
        singleFile: false,
        filter: byLabel("workflow", "PR"),
        publish: true,
      },
    },
    postMerge: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "Post-Merge",
        singleFile: false,
        filter: byLabel("workflow", "Post-Merge CI Pipeline"),
        publish: true,
      },
    },
    nightly: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "Nightly",
        singleFile: false,
        filter: byLabel("workflow", "Nightly CI Pipeline"),
        publish: true,
      },
    },
    release: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "Release",
        singleFile: false,
        filter: byLabel("workflow", "Release Pipeline"),
        publish: true,
      },
    },
    vllm: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "vLLM",
        singleFile: false,
        filter: byLabel("framework", "vllm"),
        publish: true,
      },
    },
    sglang: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "SGLang",
        singleFile: false,
        filter: byLabel("framework", "sglang"),
        publish: true,
      },
    },
    trtllm: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "TRT-LLM",
        singleFile: false,
        filter: byLabel("framework", "trtllm"),
        publish: true,
      },
    },
  },
});
