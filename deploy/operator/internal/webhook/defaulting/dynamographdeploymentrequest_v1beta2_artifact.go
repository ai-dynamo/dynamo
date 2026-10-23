/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package defaulting

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/aisimulate"
)

// resolveAISimulateDefault returns the AISimulate artifact admission persists as one pair: the
// configured planner image, or the official one derived from the operator version when none is
// configured, and the AISimulate version it carries. The explicit override wins, the derived image
// included; otherwise the version is the one bundled in the Dynamo release named by the image tag
// (aisimulate.ResolveVersion). Both are empty when there is nothing to persist: no image is
// configured and the operator version names no release. An override then has no image to pair with.
func resolveAISimulateDefault(operatorVersion, image, versionOverride string) (resolvedImage, version string, err error) {
	resolvedImage = defaultPlannerImageFor(operatorVersion, image)
	if resolvedImage == "" {
		if versionOverride != "" {
			return "", "", fmt.Errorf("the AISimulate default version override needs an image: operator version %q "+
				"names no release to derive the planner image from; set the AISimulate default image", operatorVersion)
		}
		return "", "", nil
	}
	version, err = aisimulate.ResolveVersion(resolvedImage, versionOverride)
	if err != nil {
		return "", "", fmt.Errorf("the AISimulate default image %q cannot be used as it is: %w; "+
			"set the AISimulate default version override to the AISimulate version it bundles", resolvedImage, err)
	}
	return resolvedImage, version, nil
}

// ValidateAISimulateDefault rejects an operator configuration whose default AISimulate artifact
// admission could not persist: an image, configured or derived from the operator version, whose
// Dynamo release has no bundled AISimulate version in the release contract needs the override that
// names it, and an override without any image to pair with is rejected too. An override with an
// empty image configuration is valid: it pairs with the official planner image.
func ValidateAISimulateDefault(operatorVersion, image, versionOverride string) error {
	_, _, err := resolveAISimulateDefault(operatorVersion, image, versionOverride)
	return err
}
