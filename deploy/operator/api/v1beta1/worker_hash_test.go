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

package v1beta1

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestComputeDGDWorkersSpecHashNil(t *testing.T) {
	if _, err := ComputeDGDWorkersSpecHash(nil); err == nil {
		t.Fatal("ComputeDGDWorkersSpecHash(nil) error = nil, want error")
	}
}

func TestComputeDGDWorkersSpecHashDeterministic(t *testing.T) {
	const goldenDGDWorkerHash = "2bc34b39"

	dgd := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test"},
		Spec: DynamoGraphDeploymentSpec{
			Components: []DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					ComponentType: ComponentTypeWorker,
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name:  "main",
								Image: "worker:latest",
								Env: []corev1.EnvVar{{
									Name:  "FOO",
									Value: "bar",
								}},
							}},
						},
					},
				},
			},
		},
	}

	got1, err := ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash: %v", err)
	}
	got2, err := ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		t.Fatalf("second ComputeDGDWorkersSpecHash: %v", err)
	}
	if got1 != got2 {
		t.Fatalf("ComputeDGDWorkersSpecHash is not deterministic: %q != %q", got1, got2)
	}
	if got1 != goldenDGDWorkerHash {
		t.Fatalf("ComputeDGDWorkersSpecHash() = %q, want golden %q", got1, goldenDGDWorkerHash)
	}
}

func TestComputeDGDWorkersSpecHashTracksDGDSpecMetadata(t *testing.T) {
	dgd := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test"},
		Spec: DynamoGraphDeploymentSpec{
			Components: []DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					ComponentType: ComponentTypeWorker,
				},
			},
		},
	}
	baseHash, err := ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash(base): %v", err)
	}

	withAnnotation := dgd.DeepCopy()
	withAnnotation.Spec.Annotations = map[string]string{"rollout": "annotation"}
	annotationHash, err := ComputeDGDWorkersSpecHash(withAnnotation)
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash(withAnnotation): %v", err)
	}
	if annotationHash == baseHash {
		t.Fatalf("DGD spec annotations did not affect worker hash: %q", annotationHash)
	}

	withLabel := dgd.DeepCopy()
	withLabel.Spec.Labels = map[string]string{"rollout": "label"}
	labelHash, err := ComputeDGDWorkersSpecHash(withLabel)
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash(withLabel): %v", err)
	}
	if labelHash == baseHash {
		t.Fatalf("DGD spec labels did not affect worker hash: %q", labelHash)
	}
}
