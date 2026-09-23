/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * GENERATED FILE - DO NOT EDIT.
 * Written by docs/fern/scripts/gen_nightly_selector.py and gitignored;
 * the Fern docs workflow rebuilds it on every publish.
 */

export interface NightlyBackendBuild {
  backend: "sglang" | "trtllm" | "vllm";
  backendVersion: string;
  /** Newest nightly wheel that shipped this backend version, null when unpublished. */
  dynamo: string | null;
  date: string;
  /** Immutable NGC nightly tag, YYYYMMDD-<short sha>. */
  tag: string;
  /** Tip of main: the rolling *-runtime-nightly:latest tag points here. */
  latest?: boolean;
}

export const NIGHTLY_BACKEND_BUILDS: NightlyBackendBuild[] = [
  { backend: "sglang", backendVersion: "0.5.19", dynamo: "1.6.0.dev20260922", date: "Sep 22, 2026", tag: "20260922-8612fc1", latest: true },
  { backend: "sglang", backendVersion: "0.5.18", dynamo: "1.5.0.dev20260908", date: "Sep 8, 2026", tag: "20260908-946acce" },
  { backend: "sglang", backendVersion: "0.5.17", dynamo: "1.5.0.dev20260826", date: "Aug 26, 2026", tag: "20260826-27f09d5" },
  { backend: "trtllm", backendVersion: "1.3.0rc26", dynamo: "1.6.0.dev20260922", date: "Sep 22, 2026", tag: "20260922-8612fc1", latest: true },
  { backend: "trtllm", backendVersion: "1.3.0rc25", dynamo: "1.5.0.dev20260914", date: "Sep 14, 2026", tag: "20260914-11b85b9" },
  { backend: "trtllm", backendVersion: "1.3.0rc24", dynamo: "1.5.0.dev20260830", date: "Aug 30, 2026", tag: "20260830-4c7e981" },
  { backend: "vllm", backendVersion: "0.29.0", dynamo: "1.6.0.dev20260922", date: "Sep 22, 2026", tag: "20260922-8612fc1", latest: true },
  { backend: "vllm", backendVersion: "0.28.0", dynamo: "1.5.0.dev20260914", date: "Sep 14, 2026", tag: "20260914-11b85b9" },
  { backend: "vllm", backendVersion: "0.27.1", dynamo: "1.5.0.dev20260830", date: "Aug 30, 2026", tag: "20260830-4c7e981" },
];

export default NIGHTLY_BACKEND_BUILDS;
