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

// ResolveVersion returns the AISimulate compatibility version of a Request: the explicit
// override when set, otherwise the version in the image tag. An image whose tag is not a
// version, or a missing image, is an error; the caller reports it on the field that fixes it.
// The parsing is shared with the runtime version of the other Request kinds.
func ResolveVersion(image, versionOverride string) (string, error) {
	if image == "" && versionOverride == "" {
		return "", errors.New("no AISimulate image")
	}
	version, err := versionoverride.Resolve(image, versionOverride)
	if err != nil {
		return "", fmt.Errorf("resolve AISimulate version: %w", err)
	}
	return version.String(), nil
}
