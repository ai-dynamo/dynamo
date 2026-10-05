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
  { backend: "sglang", backendVersion: "0.5.19", dynamo: "1.6.0.dev20261004", date: "Oct 4, 2026", tag: "20261004-1cbc578", latest: true },
  { backend: "sglang", backendVersion: "0.5.18", dynamo: "1.5.0.dev20260908", date: "Sep 8, 2026", tag: "20260908-946acce" },
  { backend: "sglang", backendVersion: "0.5.17", dynamo: "1.5.0.dev20260826", date: "Aug 26, 2026", tag: "20260826-27f09d5" },
  { backend: "trtllm", backendVersion: "1.3.0rc29", dynamo: "1.6.0.dev20261004", date: "Oct 4, 2026", tag: "20261004-1cbc578", latest: true },
  { backend: "trtllm", backendVersion: "1.3.0rc28", dynamo: "1.6.0.dev20261002", date: "Oct 2, 2026", tag: "20261002-e07d871" },
  { backend: "trtllm", backendVersion: "1.3.0rc27", dynamo: "1.6.0.dev20260929", date: "Sep 29, 2026", tag: "20260929-51b83df" },
  { backend: "vllm", backendVersion: "0.30.0", dynamo: "1.6.0.dev20261004", date: "Oct 4, 2026", tag: "20261004-1cbc578", latest: true },
  { backend: "vllm", backendVersion: "0.29.0", dynamo: "1.6.0.dev20260928", date: "Sep 28, 2026", tag: "20260928-b75173c" },
  { backend: "vllm", backendVersion: "0.28.0", dynamo: "1.5.0.dev20260914", date: "Sep 14, 2026", tag: "20260914-11b85b9" },
];

export default NIGHTLY_BACKEND_BUILDS;
