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
  { backend: "sglang", backendVersion: "0.5.22", dynamo: "1.7.0.dev20261027", date: "Oct 27, 2026", tag: "20261027-53258de", latest: true },
  { backend: "sglang", backendVersion: "0.5.21", dynamo: "1.6.0.dev20261018", date: "Oct 18, 2026", tag: "20261018-3763cee" },
  { backend: "sglang", backendVersion: "0.5.19", dynamo: "1.6.0.dev20260929", date: "Sep 29, 2026", tag: "20260929-51b83df" },
  { backend: "trtllm", backendVersion: "1.4.0rc0", dynamo: "1.7.0.dev20261027", date: "Oct 27, 2026", tag: "20261027-53258de", latest: true },
  { backend: "trtllm", backendVersion: "1.3.0rc29", dynamo: "1.6.0.dev20261021", date: "Oct 21, 2026", tag: "20261021-bc2b624" },
  { backend: "trtllm", backendVersion: "1.3.0rc28", dynamo: null, date: "Oct 2, 2026", tag: "20261002-e07d871" },
  { backend: "vllm", backendVersion: "0.32.0", dynamo: "1.7.0.dev20261027", date: "Oct 27, 2026", tag: "20261027-53258de", latest: true },
  { backend: "vllm", backendVersion: "0.31.0", dynamo: "1.6.0.dev20261023", date: "Oct 23, 2026", tag: "20261023-75d0686" },
  { backend: "vllm", backendVersion: "0.30.0", dynamo: "1.6.0.dev20260929", date: "Sep 29, 2026", tag: "20260929-51b83df" },
];

export const NIGHTLY_BUILDS: NightlyBuild[] = [
  { version: "1.7.0.dev20261029", date: "Oct 29, 2026", packages: ["ai-dynamo", "ai-dynamo-runtime"] },
  { version: "1.7.0.dev20261028", date: "Oct 28, 2026", packages: ["ai-dynamo", "ai-dynamo-runtime"] },
  { version: "1.7.0.dev20261027", date: "Oct 27, 2026", packages: ["ai-dynamo", "ai-dynamo-runtime"] },
];

export default NIGHTLY_BACKEND_BUILDS;
