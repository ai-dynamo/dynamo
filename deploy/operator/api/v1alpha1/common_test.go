/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestExtraPodSpec_MarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		spec     ExtraPodSpec
		wantJSON string
	}{
		{
			name: "nil PodSpec with mainContainer",
			spec: ExtraPodSpec{
				PodSpec:       nil,
				MainContainer: &corev1.Container{Name: "main"},
			},
			wantJSON: `{"mainContainer":{"name":"main","resources":{}}}`,
		},
		{
			name: "nil Containers omits containers key entirely",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					NodeSelector: map[string]string{"gpu": "true"},
				},
			},
			wantJSON: `{"nodeSelector":{"gpu":"true"}}`,
		},
		{
			name: "empty Containers omits containers key",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{},
				},
			},
			wantJSON: `{}`,
		},
		{
			name: "populated Containers are serialized",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Name: "sidecar"}},
				},
			},
			wantJSON: `{"containers":[{"name":"sidecar","resources":{}}]}`,
		},
		{
			name: "tolerations preserved without containers",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					Tolerations: []corev1.Toleration{{Key: "nvidia.com/gpu"}},
				},
			},
			wantJSON: `{"tolerations":[{"key":"nvidia.com/gpu"}]}`,
		},
		{
			name: "mainContainer alongside PodSpec fields",
			spec: ExtraPodSpec{
				PodSpec: &corev1.PodSpec{
					NodeSelector: map[string]string{"zone": "us-east"},
				},
				MainContainer: &corev1.Container{Name: "main"},
			},
			wantJSON: `{"nodeSelector":{"zone":"us-east"},"mainContainer":{"name":"main","resources":{}}}`,
		},
		{
			name:     "nil PodSpec and nil mainContainer",
			spec:     ExtraPodSpec{},
			wantJSON: `{}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := json.Marshal(tt.spec)
			if err != nil {
				t.Fatalf("MarshalJSON() error = %v", err)
			}

			if string(got) != tt.wantJSON {
				t.Errorf("MarshalJSON() mismatch\n got: %s\nwant: %s", string(got), tt.wantJSON)
			}
		})
	}
}

func TestExtraPodSpec_MarshalJSON_RoundTrip(t *testing.T) {
	original := ExtraPodSpec{
		PodSpec: &corev1.PodSpec{
			Containers:   []corev1.Container{{Name: "main", Image: "nginx"}},
			NodeSelector: map[string]string{"gpu": "true"},
			Tolerations:  []corev1.Toleration{{Key: "nvidia.com/gpu", Operator: corev1.TolerationOpExists}},
		},
		MainContainer: &corev1.Container{Name: "override", Image: "custom"},
	}

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("MarshalJSON() error = %v", err)
	}

	var restored ExtraPodSpec
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatalf("UnmarshalJSON() error = %v", err)
	}

	if !reflect.DeepEqual(original.PodSpec, restored.PodSpec) {
		t.Errorf("round-trip: PodSpec mismatch\n got: %+v\nwant: %+v", restored.PodSpec, original.PodSpec)
	}

	if !reflect.DeepEqual(original.MainContainer, restored.MainContainer) {
		t.Errorf("round-trip: MainContainer mismatch\n got: %+v\nwant: %+v", restored.MainContainer, original.MainContainer)
	}
}
