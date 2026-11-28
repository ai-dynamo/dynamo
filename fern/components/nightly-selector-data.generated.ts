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
  { backend: "sglang", backendVersion: "0.5.23", dynamo: "1.8.0.dev20261128", date: "Nov 27, 2026", tag: "20261127-cb68225", latest: true },
  { backend: "sglang", backendVersion: "0.5.22", dynamo: "1.7.0.dev20261103", date: "Nov 3, 2026", tag: "20261103-e37f0b9" },
  { backend: "sglang", backendVersion: "0.5.21", dynamo: "1.6.0.dev20261018", date: "Oct 18, 2026", tag: "20261018-3763cee" },
  { backend: "trtllm", backendVersion: "1.4.0rc4", dynamo: "1.8.0.dev20261128", date: "Nov 27, 2026", tag: "20261127-cb68225", latest: true },
  { backend: "trtllm", backendVersion: "1.4.0rc3", dynamo: "1.8.0.dev20261123", date: "Nov 23, 2026", tag: "20261123-d8a5fba" },
  { backend: "trtllm", backendVersion: "1.4.0rc2", dynamo: "1.7.0.dev20261116", date: "Nov 16, 2026", tag: "20261116-b20cd50" },
  { backend: "vllm", backendVersion: "0.34.0", dynamo: "1.8.0.dev20261128", date: "Nov 27, 2026", tag: "20261127-cb68225", latest: true },
  { backend: "vllm", backendVersion: "0.33.0", dynamo: "1.8.0.dev20261123", date: "Nov 23, 2026", tag: "20261123-d8a5fba" },
  { backend: "vllm", backendVersion: "0.32.0", dynamo: "1.7.0.dev20261105", date: "Nov 5, 2026", tag: "20261105-2fb3b13" },
];

export const NIGHTLY_BUILDS: NightlyBuild[] = [
  { version: "1.8.0.dev20261128", date: "Nov 28, 2026", packages: ["ai-dynamo", "ai-dynamo-runtime"] },
  { version: "1.8.0.dev20261127", date: "Nov 27, 2026", packages: ["ai-dynamo", "ai-dynamo-runtime"] },
  { version: "1.8.0.dev20261126", date: "Nov 26, 2026", packages: ["ai-dynamo", "ai-dynamo-runtime"] },
];

export default NIGHTLY_BACKEND_BUILDS;
