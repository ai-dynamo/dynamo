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
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestV1alpha1WireShapeSurvivesV1beta1StorageMigration(t *testing.T) {
	tests := []struct {
		name       string
		rawAlpha   string
		alpha      conversion.Convertible
		hub        conversion.Hub
		storedHub  conversion.Hub
		afterAlpha conversion.Convertible
	}{
		{
			name:       "DynamoGraphDeployment",
			rawAlpha:   `{"apiVersion":"nvidia.com/v1alpha1","kind":"DynamoGraphDeployment","metadata":{"name":"preupgrade-spec-probe","namespace":"dynamo-cloud"},"spec":{"services":{"Frontend":{"replicas":0,"componentType":"frontend","extraPodSpec":{"mainContainer":{"image":"busybox:1.36"}}}}}}`,
			alpha:      &DynamoGraphDeployment{},
			hub:        &v1beta1.DynamoGraphDeployment{},
			storedHub:  &v1beta1.DynamoGraphDeployment{},
			afterAlpha: &DynamoGraphDeployment{},
		},
		{
			name:       "DynamoComponentDeployment",
			rawAlpha:   `{"apiVersion":"nvidia.com/v1alpha1","kind":"DynamoComponentDeployment","metadata":{"name":"preupgrade-component","namespace":"dynamo-cloud"},"spec":{"componentType":"frontend","extraPodSpec":{"mainContainer":{"image":"busybox:1.36"}},"replicas":0}}`,
			alpha:      &DynamoComponentDeployment{},
			hub:        &v1beta1.DynamoComponentDeployment{},
			storedHub:  &v1beta1.DynamoComponentDeployment{},
			afterAlpha: &DynamoComponentDeployment{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Decode the sparse v1alpha1 object and retain its original wire-level spec")
			var before map[string]any
			if err := json.Unmarshal([]byte(tt.rawAlpha), &before); err != nil {
				t.Fatalf("unmarshal raw v1alpha1 object: %v", err)
			}
			if err := json.Unmarshal([]byte(tt.rawAlpha), tt.alpha); err != nil {
				t.Fatalf("unmarshal typed v1alpha1 object: %v", err)
			}

			t.Log("Convert the object to v1beta1 and round-trip it through storage JSON")
			if err := tt.alpha.ConvertTo(tt.hub); err != nil {
				t.Fatalf("ConvertTo() error = %v", err)
			}
			hubJSON, err := json.Marshal(tt.hub)
			if err != nil {
				t.Fatalf("marshal v1beta1 object: %v", err)
			}
			if err := json.Unmarshal(hubJSON, tt.storedHub); err != nil {
				t.Fatalf("unmarshal stored v1beta1 object: %v", err)
			}

			t.Log("Convert the stored object back to the v1alpha1 API wire representation")
			if err := tt.afterAlpha.ConvertFrom(tt.storedHub); err != nil {
				t.Fatalf("ConvertFrom() error = %v", err)
			}
			afterJSON, err := json.Marshal(tt.afterAlpha)
			if err != nil {
				t.Fatalf("marshal converted v1alpha1 object: %v", err)
			}
			var after map[string]any
			if err := json.Unmarshal(afterJSON, &after); err != nil {
				t.Fatalf("unmarshal converted v1alpha1 object: %v", err)
			}

			t.Log("Verify storage migration preserves the original sparse spec shape")
			if diff := cmp.Diff(before["spec"], after["spec"]); diff != "" {
				t.Fatalf("v1alpha1 spec changed across v1beta1 storage migration (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDynamoGraphDeploymentMarshalNormalizesExtraPodSpecContainers(t *testing.T) {
	t.Log("Build a DGD containing empty and non-empty native container resources")
	dgd := DynamoGraphDeployment{
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ExtraPodSpec: &ExtraPodSpec{
						PodSpec: &corev1.PodSpec{
							Containers: []corev1.Container{
								{Name: "empty-sidecar"},
								{Name: "resource-sidecar", Resources: corev1.ResourceRequirements{Claims: []corev1.ResourceClaim{{Name: "sidecar-gpu"}}}},
							},
							InitContainers: []corev1.Container{{Name: "init"}},
							EphemeralContainers: []corev1.EphemeralContainer{{
								EphemeralContainerCommon: corev1.EphemeralContainerCommon{Name: "debug"},
							}},
						},
						MainContainer: &corev1.Container{Image: "busybox"},
					},
				},
			},
		},
	}

	t.Log("Marshal through the public v1alpha1 root object")
	raw, err := json.Marshal(dgd)
	if err != nil {
		t.Fatalf("marshal DGD: %v", err)
	}
	var root map[string]any
	if err := json.Unmarshal(raw, &root); err != nil {
		t.Fatalf("unmarshal DGD JSON: %v", err)
	}
	extraPodSpec := root["spec"].(map[string]any)["services"].(map[string]any)["frontend"].(map[string]any)["extraPodSpec"].(map[string]any)

	t.Log("Verify the main container omits its empty synthetic fields")
	mainContainer := extraPodSpec["mainContainer"].(map[string]any)
	if _, ok := mainContainer["name"]; ok {
		t.Fatalf("mainContainer.name was not omitted: %v", mainContainer)
	}
	if _, ok := mainContainer["resources"]; ok {
		t.Fatalf("mainContainer.resources was not omitted: %v", mainContainer)
	}

	t.Log("Verify every PodSpec container list omits empty resources")
	containers := extraPodSpec["containers"].([]any)
	if _, ok := containers[0].(map[string]any)["resources"]; ok {
		t.Fatalf("containers[0].resources was not omitted: %v", containers[0])
	}
	initContainers := extraPodSpec["initContainers"].([]any)
	if _, ok := initContainers[0].(map[string]any)["resources"]; ok {
		t.Fatalf("initContainers[0].resources was not omitted: %v", initContainers[0])
	}
	ephemeralContainers := extraPodSpec["ephemeralContainers"].([]any)
	if _, ok := ephemeralContainers[0].(map[string]any)["resources"]; ok {
		t.Fatalf("ephemeralContainers[0].resources was not omitted: %v", ephemeralContainers[0])
	}

	t.Log("Verify non-empty resources remain present")
	resources, ok := containers[1].(map[string]any)["resources"].(map[string]any)
	if !ok || len(resources) == 0 {
		t.Fatalf("containers[1].resources = %v, want non-empty", containers[1])
	}
}
