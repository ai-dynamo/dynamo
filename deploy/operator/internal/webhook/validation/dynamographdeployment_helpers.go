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

package validation

import (
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

const (
	// maxCombinedResourceNameLength is kept as a local alias for readability.
	maxCombinedResourceNameLength = consts.MaxCombinedGroveResourceNameLength

	unsetValue = "<unset>"

	vllmDistributedExecutorBackendMP  = "mp"
	vllmDistributedExecutorBackendRay = "ray"
)

type clusterTopologyInfo struct {
	domainIndex map[string]int
	domains     []string
}

func getUnique[T comparable](slice []T) []T {
	seen := make(map[T]struct{}, len(slice))
	uniqueSlice := make([]T, 0, len(slice))
	for _, element := range slice {
		if _, exists := seen[element]; !exists {
			seen[element] = struct{}{}
			uniqueSlice = append(uniqueSlice, element)
		}
	}
	return uniqueSlice
}

// difference returns elements in set a that are not in set b (a - b).
func difference(a, b map[string]struct{}) []string {
	var result []string
	for name := range a {
		if _, exists := b[name]; !exists {
			result = append(result, name)
		}
	}
	return result
}

func validateVLLMDistributedExecutorBackendAnnotation(fieldPath string, annotations map[string]string) error {
	if annotations == nil {
		return nil
	}

	value, exists := annotations[consts.KubeAnnotationVLLMDistributedExecutorBackend]
	if !exists {
		return nil
	}

	switch strings.ToLower(value) {
	case vllmDistributedExecutorBackendMP, vllmDistributedExecutorBackendRay:
		return nil
	default:
		if fieldPath == "" {
			return fmt.Errorf("annotation %s has invalid value %q: must be \"mp\" or \"ray\"",
				consts.KubeAnnotationVLLMDistributedExecutorBackend, value)
		}
		return fmt.Errorf("%s[%s] has invalid value %q: must be \"mp\" or \"ray\"",
			fieldPath, consts.KubeAnnotationVLLMDistributedExecutorBackend, value)
	}
}
