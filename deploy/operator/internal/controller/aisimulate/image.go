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

package aisimulate

import (
	"errors"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/versionoverride"
)

// aisimulateVersionByDynamoMinor is the release contract between a Dynamo release and the AISimulate
// package its planner image bundles, keyed by the Dynamo major.minor. The image tag is the Dynamo
// version, not the version of the AISimulate package inside the image, and the AISimulate version
// selects the version-specific input schema, so the two are never inferred from one tag. Every
// Dynamo minor bump adds its entry; release_contract_test.go fails until it does.
var aisimulateVersionByDynamoMinor = map[string]string{
	"1.6": "0.13.0",
}

// ResolveVersion returns the AISimulate version a Request selects: the explicit override when set,
// otherwise the AISimulate version bundled in the Dynamo release named by the image tag. The
// override is the escape hatch for images the contract does not cover (a nightly or custom planner
// image) and names the AISimulate version itself. An image whose tag is not a Dynamo release with a
// mapping entry, or a missing image, is an error; the caller reports it on the field that fixes it.
func ResolveVersion(image, versionOverride string) (string, error) {
	if versionOverride != "" {
		version, err := versionoverride.Parse(versionOverride)
		if err != nil {
			return "", fmt.Errorf("AISimulate version override %q: %w", versionOverride, err)
		}
		return version.String(), nil
	}
	if image == "" {
		return "", errors.New("no AISimulate image")
	}
	release, err := versionoverride.ParseImageVersion(image)
	if err != nil {
		return "", fmt.Errorf("resolve AISimulate version: %w", err)
	}
	minor := fmt.Sprintf("%d.%d", release.Major, release.Minor)
	version, ok := aisimulateVersionByDynamoMinor[minor]
	if !ok {
		return "", fmt.Errorf("image %q is Dynamo %s, which has no bundled AISimulate version in the release contract", image, minor)
	}
	return version, nil
}
