/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package lpx implements LPU build normalization, workload projection,
// runtime configuration, and node-local LPX materialization.
//
// Builds target XT (XT8888) or HX (HX16x8x2x3). The scheduler API's V2 and
// V3 workload modes denote these hardware families, respectively. Manifest v2
// and deployment v1 are independent compiler schema revisions, both of which
// can describe either hardware family.
package lpx
