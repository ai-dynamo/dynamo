/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting

import (
	"fmt"

	semver "github.com/Masterminds/semver/v3"
)

// defaultImage is the default profiler image used when spec.image is not set.
// Starting with Dynamo 1.1.0, the profiler's runtime dependencies ship only in
// the dedicated dynamo-planner image.
const defaultImage = "nvcr.io/nvidia/ai-dynamo/dynamo-planner"

func (d *DGDRDefaulter) defaultImageFor() string {
	if d.DefaultImage != "" {
		return d.DefaultImage
	}
	version, err := semver.NewVersion(d.OperatorVersion)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%s:%s", defaultImage, version.String())
}
