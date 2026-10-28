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
	_ "embed"
	"strings"
)

// hardwareSKUs mirrors the system specifications packaged in the pinned AISimulate wheel.
// It is a compatibility hint for warnings and discovery, not an API enum: explicit
// identifiers outside this set remain valid and are passed through unchanged.
//
//go:embed hardware_skus.txt
var hardwareSKUsText string

var hardwareSKUs = func() map[string]struct{} {
	result := map[string]struct{}{}
	for _, line := range strings.Split(hardwareSKUsText, "\n") {
		value := strings.TrimSpace(line)
		if value == "" || strings.HasPrefix(value, "#") {
			continue
		}
		result[value] = struct{}{}
	}
	return result
}()

// IsKnownHardwareSKU reports whether sku is packaged by the AISimulate release pinned by
// this Dynamo checkout. False does not make a value invalid: an external system catalog may
// support additional identifiers.
func IsKnownHardwareSKU(sku string) bool {
	_, found := hardwareSKUs[sku]
	return found
}
