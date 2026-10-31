/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting

// DGDRV1beta1Options configures the defaults of v1beta1 Requests.
type DGDRV1beta1Options struct {
	// DefaultImage is the profiler image put into spec.image when a Request is created
	// without one; empty derives dynamo-planner:<operator version>.
	DefaultImage string
}
