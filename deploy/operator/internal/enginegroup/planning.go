/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

// ScalePlanResolution distinguishes executable work, a definitive rejection, and no required operation.
// On a method error, none of these fields is authoritative.
type ScalePlanResolution struct {
	Plan      *ResolvedPlan
	Rejection *Failure
}
