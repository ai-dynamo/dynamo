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

package v1alpha1

import (
	"bytes"
	"encoding/json"
)

type dynamoGraphDeploymentForMarshal DynamoGraphDeployment

// MarshalJSON serializes a DGD and removes native-container zero-value fields
// introduced by the typed v1beta1-to-v1alpha1 conversion path.
// Normalization stays at the root so ExtraPodSpec's historical JSON, which
// participates in the legacy v1alpha1 worker hash, remains unchanged.
func (d DynamoGraphDeployment) MarshalJSON() ([]byte, error) {
	raw, err := json.Marshal(dynamoGraphDeploymentForMarshal(d))
	if err != nil {
		return nil, err
	}

	return normalizeDynamoGraphDeploymentJSON(raw)
}

type dynamoComponentDeploymentForMarshal DynamoComponentDeployment

// MarshalJSON serializes a DCD and removes native-container zero-value fields
// introduced by the typed v1beta1-to-v1alpha1 conversion path.
// See DynamoGraphDeployment.MarshalJSON for why normalization is root-scoped.
func (d DynamoComponentDeployment) MarshalJSON() ([]byte, error) {
	raw, err := json.Marshal(dynamoComponentDeploymentForMarshal(d))
	if err != nil {
		return nil, err
	}

	return normalizeDynamoComponentDeploymentJSON(raw)
}

func normalizeDynamoGraphDeploymentJSON(raw []byte) ([]byte, error) {
	// Decode the DGD into a mutable root before normalizing its service containers.
	root, err := decodeV1alpha1JSONObject(raw)
	if err != nil {
		return nil, err
	}

	// Limit normalization to service specs so unrelated DGD fields retain their encoding.
	if spec, ok := root["spec"].(map[string]any); ok {
		if services, ok := spec["services"].(map[string]any); ok {
			for _, service := range services {
				if serviceSpec, ok := service.(map[string]any); ok {
					normalizeV1alpha1ExtraPodSpecJSON(serviceSpec)
				}
			}
		}
	}

	return json.Marshal(root)
}

func normalizeDynamoComponentDeploymentJSON(raw []byte) ([]byte, error) {
	// Decode the DCD into a mutable root before normalizing its component container.
	root, err := decodeV1alpha1JSONObject(raw)
	if err != nil {
		return nil, err
	}

	// Limit normalization to the component spec so unrelated DCD fields retain their encoding.
	if spec, ok := root["spec"].(map[string]any); ok {
		normalizeV1alpha1ExtraPodSpecJSON(spec)
	}

	return json.Marshal(root)
}

func decodeV1alpha1JSONObject(raw []byte) (map[string]any, error) {
	// Preserve JSON numbers while configuring the generic object decoder.
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()

	// Decode the root object for targeted container normalization.
	var root map[string]any
	if err := decoder.Decode(&root); err != nil {
		return nil, err
	}
	return root, nil
}

func normalizeV1alpha1ExtraPodSpecJSON(component map[string]any) {
	extraPodSpec, ok := component["extraPodSpec"].(map[string]any)
	if !ok {
		return
	}

	// MainContainer has no required name and conversion clears its synthetic one.
	if mainContainer, ok := extraPodSpec["mainContainer"].(map[string]any); ok {
		if name, ok := mainContainer["name"].(string); ok && name == "" {
			delete(mainContainer, "name")
		}
		removeEmptyContainerResourcesJSON(mainContainer)
	}

	// Native container lists also materialize zero ResourceRequirements as {}.
	for _, field := range []string{"containers", "initContainers", "ephemeralContainers"} {
		containers, ok := extraPodSpec[field].([]any)
		if !ok {
			continue
		}
		for _, container := range containers {
			if container, ok := container.(map[string]any); ok {
				removeEmptyContainerResourcesJSON(container)
			}
		}
	}
}

func removeEmptyContainerResourcesJSON(container map[string]any) {
	resources, ok := container["resources"].(map[string]any)
	if ok && len(resources) == 0 {
		delete(container, "resources")
	}
}
