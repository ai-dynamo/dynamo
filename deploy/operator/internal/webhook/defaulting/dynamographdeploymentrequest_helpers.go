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

// PlannerImageFor returns explicitImage when set, else the planner image with a canonical
// operator-version tag, or an empty string when the version cannot be parsed. The same resolved
// image defaults v1beta1 Requests and executes v1beta2 Runs.
func PlannerImageFor(operatorVersion, explicitImage string) string {
	if explicitImage != "" {
		return explicitImage
	}
	version, err := semver.NewVersion(operatorVersion)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%s:%s", defaultImage, version.String())
}
