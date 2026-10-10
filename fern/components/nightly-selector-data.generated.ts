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

export interface NightlyBuild {
  version: string;
  date: string;
  packages: string[];
  note?: string;
}

export const NIGHTLY_BACKEND_BUILDS: NightlyBackendBuild[] = [
  { backend: "sglang", backendVersion: "0.5.21", dynamo: "1.6.0.dev20261008", date: "Oct 8, 2026", tag: "20261008-049eece", latest: true },
  { backend: "sglang", backendVersion: "0.5.19", dynamo: "1.6.0.dev20261004", date: "Oct 4, 2026", tag: "20261004-1cbc578" },
  { backend: "sglang", backendVersion: "0.5.18", dynamo: "1.5.0.dev20260908", date: "Sep 8, 2026", tag: "20260908-946acce" },
  { backend: "trtllm", backendVersion: "1.3.0rc29", dynamo: "1.6.0.dev20261008", date: "Oct 8, 2026", tag: "20261008-049eece", latest: true },
  { backend: "trtllm", backendVersion: "1.3.0rc28", dynamo: "1.6.0.dev20261002", date: "Oct 2, 2026", tag: "20261002-e07d871" },
  { backend: "trtllm", backendVersion: "1.3.0rc27", dynamo: "1.6.0.dev20260929", date: "Sep 29, 2026", tag: "20260929-51b83df" },
  { backend: "vllm", backendVersion: "0.31.0", dynamo: "1.6.0.dev20261008", date: "Oct 8, 2026", tag: "20261008-049eece", latest: true },
  { backend: "vllm", backendVersion: "0.30.0", dynamo: "1.6.0.dev20261007", date: "Oct 7, 2026", tag: "20261007-16820ec" },
  { backend: "vllm", backendVersion: "0.29.0", dynamo: "1.6.0.dev20260928", date: "Sep 28, 2026", tag: "20260928-b75173c" },
];

export const NIGHTLY_BUILDS: NightlyBuild[] = [
  { version: "1.6.0.dev20261010", date: "Oct 10, 2026", packages: ["ai-dynamo", "ai-dynamo-runtime", "kvbm"] },
  { version: "1.6.0.dev20261009", date: "Oct 9, 2026", packages: ["ai-dynamo", "ai-dynamo-runtime"] },
  { version: "1.6.0.dev20261008", date: "Oct 8, 2026", packages: ["ai-dynamo", "ai-dynamo-runtime", "kvbm"] },
];

export default NIGHTLY_BACKEND_BUILDS;
