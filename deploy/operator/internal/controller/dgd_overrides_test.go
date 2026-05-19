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

package controller

import (
	"reflect"
	"testing"

	dgdv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestApplyDGDOverrides_MergesServiceImageWithoutDroppingGeneratedFields(t *testing.T) {
	replicas := int32(2)
	generated := &dgdv1alpha1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: dgdv1alpha1.GroupVersion.String(),
			Kind:       dgdOverrideKind,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:   "generated-dgd",
			Labels: map[string]string{"generated": "keep"},
		},
		Spec: dgdv1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*dgdv1alpha1.DynamoComponentDeploymentSharedSpec{
				"TRTLLMWorker": {
					Replicas: &replicas,
					ExtraPodSpec: &dgdv1alpha1.ExtraPodSpec{
						MainContainer: &corev1.Container{
							Image: "nvcr.io/nvstaging/ai-dynamo/tensorrtllm-runtime:1.2.0rc0",
							Args:  []string{"--generated-arg"},
						},
					},
				},
				"Frontend": {
					Replicas: &replicas,
				},
			},
		},
	}

	override := &runtime.RawExtension{Raw: []byte(`{
		"apiVersion": "nvidia.com/v1alpha1",
		"kind": "DynamoGraphDeployment",
		"metadata": {
			"labels": {
				"override": "applied"
			}
		},
		"spec": {
			"services": {
				"TRTLLMWorker": {
					"extraPodSpec": {
						"mainContainer": {
							"image": "nvcr.io/nvstaging/ai-dynamo/tensorrtllm-runtime:1.2.0rc0-cuda13"
						}
					}
				}
			}
		}
	}`)}

	if err := applyDGDOverrides(generated, override); err != nil {
		t.Fatalf("applyDGDOverrides() error = %v", err)
	}

	if got := generated.Labels["generated"]; got != "keep" {
		t.Fatalf("generated label = %q, want keep", got)
	}
	if got := generated.Labels["override"]; got != "applied" {
		t.Fatalf("override label = %q, want applied", got)
	}

	worker := generated.Spec.Services["TRTLLMWorker"]
	if worker == nil {
		t.Fatal("TRTLLMWorker service missing after override")
	}
	if worker.Replicas == nil || *worker.Replicas != 2 {
		t.Fatalf("TRTLLMWorker replicas = %v, want 2", worker.Replicas)
	}
	if worker.ExtraPodSpec == nil || worker.ExtraPodSpec.MainContainer == nil {
		t.Fatalf("TRTLLMWorker main container missing after override: %#v", worker)
	}
	if got, want := worker.ExtraPodSpec.MainContainer.Image, "nvcr.io/nvstaging/ai-dynamo/tensorrtllm-runtime:1.2.0rc0-cuda13"; got != want {
		t.Fatalf("TRTLLMWorker image = %q, want %q", got, want)
	}
	if got, want := worker.ExtraPodSpec.MainContainer.Args, []string{"--generated-arg"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("TRTLLMWorker args = %#v, want %#v", got, want)
	}
	if generated.Spec.Services["Frontend"] == nil {
		t.Fatal("Frontend service missing after override")
	}
}

func TestApplyDGDOverrides_RejectsWrongKind(t *testing.T) {
	generated := &dgdv1alpha1.DynamoGraphDeployment{}
	override := &runtime.RawExtension{Raw: []byte(`{
		"apiVersion": "nvidia.com/v1alpha1",
		"kind": "ConfigMap"
	}`)}

	if err := applyDGDOverrides(generated, override); err == nil {
		t.Fatal("applyDGDOverrides() error = nil, want invalid kind error")
	}
}

func TestApplyDGDOverrides_RejectsUnknownFields(t *testing.T) {
	generated := &dgdv1alpha1.DynamoGraphDeployment{}
	override := &runtime.RawExtension{Raw: []byte(`{
		"apiVersion": "nvidia.com/v1alpha1",
		"kind": "DynamoGraphDeployment",
		"spec": {
			"services": {
				"TRTLLMWorker": {
					"extraPodSpec": {
						"mainContainer": {
							"imag": "nvcr.io/nvstaging/ai-dynamo/tensorrtllm-runtime:1.2.0rc0-cuda13"
						}
					}
				}
			}
		}
	}`)}

	if err := applyDGDOverrides(generated, override); err == nil {
		t.Fatal("applyDGDOverrides() error = nil, want unknown field error")
	}
}
