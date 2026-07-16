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

package dynamo

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
)

// ValidateRenderedPodSpecContainerNames rejects names reused across any container kind.
func ValidateRenderedPodSpecContainerNames(podSpec *corev1.PodSpec) error {
	if podSpec == nil {
		return nil
	}

	seen := make(map[string]string, len(podSpec.Containers)+len(podSpec.InitContainers)+len(podSpec.EphemeralContainers))
	validateName := func(name, path string) error {
		if previousPath, ok := seen[name]; ok {
			return fmt.Errorf("container name %q is used by both %s and %s", name, previousPath, path)
		}
		seen[name] = path
		return nil
	}

	for i := range podSpec.Containers {
		if err := validateName(podSpec.Containers[i].Name, fmt.Sprintf("spec.containers[%d]", i)); err != nil {
			return err
		}
	}
	for i := range podSpec.InitContainers {
		if err := validateName(podSpec.InitContainers[i].Name, fmt.Sprintf("spec.initContainers[%d]", i)); err != nil {
			return err
		}
	}
	for i := range podSpec.EphemeralContainers {
		if err := validateName(podSpec.EphemeralContainers[i].Name, fmt.Sprintf("spec.ephemeralContainers[%d]", i)); err != nil {
			return err
		}
	}

	return nil
}
