/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package lpx implements LPU build normalization, workload projection,
// runtime configuration, and node-local LPX materialization.
//
// Builds target one of two LPU families: XT (XT8888, xtFamily) and HX
// (HX16x8x2x3, hxFamily). The generated scheduler API still names them by
// generation, so its V2 and V3 workload modes are XT and HX respectively.
// Separately, "v2" in manifest.v2.capnp.bin and the manifest package is the
// compiler manifest schema revision, which describes builds of either family.
package lpx
