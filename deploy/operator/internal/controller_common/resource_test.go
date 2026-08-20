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

package controller_common

import (
	"context"
	"errors"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/bsm/gomega"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

func TestGetSpecChangeResult(t *testing.T) {
	tests := []struct {
		name          string
		current       client.Object
		desired       client.Object
		expectedHash  bool
		expectedError bool
	}{
		{
			name: "no change in hash with deployment spec and env variables",
			current: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name":      "nim-deployment",
						"namespace": "default",
					},
					"spec": map[string]interface{}{
						"replicas": int64(2),
						"selector": map[string]interface{}{
							"matchLabels": map[string]interface{}{
								"app": "nim",
							},
						},
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"labels": map[string]interface{}{
									"app": "nim",
								},
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nim",
										"image": "nim:v0.1.0",
										"ports": []interface{}{
											map[string]interface{}{
												"containerPort": int64(80),
											},
										},
										"env": []interface{}{
											map[string]interface{}{"name": "ENV_VAR1", "value": "value1"},
											map[string]interface{}{"name": "ENV_VAR2", "value": "value2"},
										},
									},
								},
							},
						},
					},
				},
			},
			desired: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name":      "nim-deployment",
						"namespace": "default",
					},
					"spec": map[string]interface{}{
						"replicas": int64(2),
						"selector": map[string]interface{}{
							"matchLabels": map[string]interface{}{
								"app": "nim",
							},
						},
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"labels": map[string]interface{}{
									"app": "nim",
								},
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nim",
										"image": "nim:v0.1.0",
										"ports": []interface{}{
											map[string]interface{}{
												"containerPort": int64(80),
											},
										},
										"env": []interface{}{
											map[string]interface{}{"name": "ENV_VAR1", "value": "value1"},
											map[string]interface{}{"name": "ENV_VAR2", "value": "value2"},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedHash:  false,
			expectedError: false,
		},
		{
			name: "no change in hash with deployment spec and env variables, change in order",
			current: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name":      "nim-deployment",
						"namespace": "default",
					},
					"spec": map[string]interface{}{
						"replicas": int64(2),
						"selector": map[string]interface{}{
							"matchLabels": map[string]interface{}{
								"app": "nim",
							},
						},
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"labels": map[string]interface{}{
									"app": "nim",
								},
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nim",
										"image": "nim:v0.1.0",
										"ports": []interface{}{
											map[string]interface{}{
												"containerPort": int64(80),
											},
										},
										"env": []interface{}{
											map[string]interface{}{"name": "ENV_VAR1", "value": "value1"},
											map[string]interface{}{"name": "ENV_VAR2", "value": "value2"},
										},
									},
								},
							},
						},
					},
				},
			},
			desired: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name":      "nim-deployment",
						"namespace": "default",
					},
					"spec": map[string]interface{}{
						"replicas": int64(2),
						"selector": map[string]interface{}{
							"matchLabels": map[string]interface{}{
								"app": "nim",
							},
						},
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"labels": map[string]interface{}{
									"app": "nim",
								},
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nim",
										"image": "nim:v0.1.0",
										"ports": []interface{}{
											map[string]interface{}{
												"containerPort": int64(80),
											},
										},
										"env": []interface{}{
											map[string]interface{}{"name": "ENV_VAR2", "value": "value2"},
											map[string]interface{}{"name": "ENV_VAR1", "value": "value1"},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedHash:  false,
			expectedError: false,
		},
		{
			name: "no change in hash with change in metadata and status",
			current: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name":      "nim-deployment",
						"namespace": "default",
					},
					"spec": map[string]interface{}{
						"replicas": int64(2),
						"selector": map[string]interface{}{
							"matchLabels": map[string]interface{}{
								"app": "nim",
							},
						},
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"labels": map[string]interface{}{
									"app": "nim",
								},
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nim",
										"image": "nim:v0.1.0",
										"ports": []interface{}{
											map[string]interface{}{
												"containerPort": int64(80),
											},
										}, // switch order of env
										"env": []interface{}{
											map[string]interface{}{"name": "ENV_VAR1", "value": "value1"},
											map[string]interface{}{"name": "ENV_VAR2", "value": "value2"},
										},
									},
								},
							},
						},
					},
				},
			},
			desired: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name":      "nim-deployment",
						"namespace": "default",
						"blah":      "blah",
					},
					"spec": map[string]interface{}{
						"replicas": int64(2),
						"selector": map[string]interface{}{
							"matchLabels": map[string]interface{}{
								"app": "nim",
							},
						},
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"labels": map[string]interface{}{
									"app": "nim",
								},
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nim",
										"image": "nim:v0.1.0",
										"ports": []interface{}{
											map[string]interface{}{
												"containerPort": int64(80),
											},
										},
										"env": []interface{}{
											map[string]interface{}{"name": "ENV_VAR1", "value": "value1"},
											map[string]interface{}{"name": "ENV_VAR2", "value": "value2"},
										},
									},
								},
							},
						},
					},
					"status": map[string]interface{}{
						"ready": true,
					},
				},
			},
			expectedHash:  false,
			expectedError: false,
		},
		{
			name: "change in hash with change in value of elements",
			current: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name":      "nim-deployment",
						"namespace": "default",
					},
					"spec": map[string]interface{}{
						"replicas": int64(2),
						"selector": map[string]interface{}{
							"matchLabels": map[string]interface{}{
								"app": "nim",
							},
						},
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"labels": map[string]interface{}{
									"app": "nim",
								},
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nim",
										"image": "nim:v0.1.0",
										"ports": []interface{}{
											map[string]interface{}{
												"containerPort": int64(80),
											},
										},
										"env": []interface{}{
											map[string]interface{}{"name": "ENV_VAR1", "value": "value2"},
											map[string]interface{}{"name": "ENV_VAR2", "value": "value1"},
										},
									},
								},
							},
						},
					},
				},
			},
			desired: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name":      "nim-deployment",
						"namespace": "default",
					},
					"spec": map[string]interface{}{
						"replicas": int64(3),
						"selector": map[string]interface{}{
							"matchLabels": map[string]interface{}{
								"app": "nim",
							},
						},
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"labels": map[string]interface{}{
									"app": "nim",
								},
							},
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":  "nim",
										"image": "nim:v0.1.0",
										"ports": []interface{}{
											map[string]interface{}{
												"containerPort": int64(80),
											},
										},
										"env": []interface{}{
											map[string]interface{}{"name": "ENV_VAR1", "value": "asdf"},
											map[string]interface{}{"name": "ENV_VAR2", "value": "jljl"},
										},
									},
								},
							},
						},
					},
				},
			},
			expectedHash:  true,
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hash, err := GetSpecHash(tt.current)
			if err != nil {
				t.Errorf("failed to get spec hash in test for resource %s: %s", tt.current.GetName(), err)
			}
			// Set both hash and generation annotations (generation=1 simulates initial state)
			updateAnnotations(tt.current, hash, 1)
			result, err := GetSpecChangeResult(tt.current, tt.desired)
			if err != nil {
				t.Errorf("failed to check if spec has changed in test for resource %s: %s", tt.current.GetName(), err)
			}
			if tt.expectedHash && !result.NeedsUpdate {
				t.Errorf("GetSpecChangeResult() NeedsUpdate = %v, want %v", result.NeedsUpdate, tt.expectedHash)
			}
			if !tt.expectedHash && result.NeedsUpdate {
				t.Errorf("GetSpecChangeResult() NeedsUpdate = %v, want %v", result.NeedsUpdate, tt.expectedHash)
			}
		})
	}
}

func TestGetSpecChangeResult_AnnotationOnlyForEquivalentSpec(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	current := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      "worker",
				"namespace": "default",
			},
			"spec": map[string]interface{}{
				"replicas": int64(1),
			},
		},
	}
	desired := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      "worker",
				"namespace": "default",
			},
			"spec": map[string]interface{}{
				"replicas": int64(1),
			},
		},
	}
	current.SetGeneration(7)

	result, err := GetSpecChangeResult(current, desired)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(result.NeedsUpdate).To(gomega.BeTrue())
	g.Expect(result.SpecNeedsUpdate).To(gomega.BeFalse())
	g.Expect(result.NewGeneration).To(gomega.Equal(int64(7)))
	g.Expect(result.NewHash).ToNot(gomega.BeNil())
}

func TestGetSpecChangeResult_OrderOnlyManualDriftNeedsSpecUpdate(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	desired := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      "worker",
				"namespace": "default",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"initContainers": []interface{}{
							map[string]interface{}{"name": "setup", "image": "busybox"},
							map[string]interface{}{"name": "migrate", "image": "busybox"},
						},
						"containers": []interface{}{
							map[string]interface{}{"name": "main", "image": "worker:v1"},
						},
					},
				},
			},
		},
	}
	current := desired.DeepCopy()
	current.SetGeneration(6)
	current.Object["spec"].(map[string]interface{})["template"].(map[string]interface{})["spec"].(map[string]interface{})["initContainers"] = []interface{}{
		map[string]interface{}{"name": "migrate", "image": "busybox"},
		map[string]interface{}{"name": "setup", "image": "busybox"},
	}
	desiredHash, err := GetSpecHash(desired)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	currentHash, err := GetSpecHash(current)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(currentHash).To(gomega.Equal(desiredHash), "canonical hash must reproduce the historical blind spot")
	current.SetAnnotations(map[string]string{
		NvidiaAnnotationHashKey:       desiredHash,
		NvidiaAnnotationGenerationKey: "5",
	})

	result, err := GetSpecChangeResult(current, desired)
	g.Expect(err).ToNot(gomega.HaveOccurred())
	g.Expect(result.NeedsUpdate).To(gomega.BeTrue())
	g.Expect(result.SpecNeedsUpdate).To(gomega.BeTrue())
	g.Expect(result.ManualChangeDetected).To(gomega.BeTrue())
	g.Expect(result.NewGeneration).To(gomega.Equal(int64(7)))
}

func TestGetSpecChangeResult_GenerationTracking(t *testing.T) {
	tests := []struct {
		name                       string
		currentGeneration          int64
		lastAppliedGeneration      string // empty string means annotation not set
		lastAppliedHash            string // empty string means annotation not set, "match" means compute from current
		desiredReplicas            int64  // different from current (2) means hash will differ
		expectNeedsUpdate          bool
		expectSpecNeedsUpdate      bool
		expectManualChangeDetected bool
		expectNewGeneration        int64 // 0 means don't check
	}{
		{
			name:                  "no change - generations and hash match",
			currentGeneration:     5,
			lastAppliedGeneration: "5",
			lastAppliedHash:       "match",
			desiredReplicas:       2, // same as current
			expectNeedsUpdate:     false,
		},
		{
			name:                       "generation increased but spec is equivalent - annotations only",
			currentGeneration:          7,
			lastAppliedGeneration:      "5",
			lastAppliedHash:            "match",
			desiredReplicas:            2,
			expectNeedsUpdate:          true,
			expectManualChangeDetected: false,
			expectNewGeneration:        7,
		},
		{
			// Upgrade scenario: hash matches but no generation annotation yet.
			name:                  "missing generation annotation - annotations only when spec is equivalent",
			currentGeneration:     5,
			lastAppliedGeneration: "", // missing
			lastAppliedHash:       "match",
			desiredReplicas:       2,
			expectNeedsUpdate:     true,
			expectNewGeneration:   5,
		},
		{
			name:                  "missing hash annotation - annotations only when spec is equivalent",
			currentGeneration:     5,
			lastAppliedGeneration: "5",
			lastAppliedHash:       "", // missing
			desiredReplicas:       2,
			expectNeedsUpdate:     true,
			expectNewGeneration:   5,
		},
		{
			name:                  "hash changed - needs full update",
			currentGeneration:     5,
			lastAppliedGeneration: "5",
			lastAppliedHash:       "match",
			desiredReplicas:       3, // different from current (2)
			expectNeedsUpdate:     true,
			expectSpecNeedsUpdate: true,
			expectNewGeneration:   6, // current(5) + 1
		},
		{
			name:                  "corrupted generation annotation - annotations only when spec is equivalent",
			currentGeneration:     5,
			lastAppliedGeneration: "invalid",
			lastAppliedHash:       "match",
			desiredReplicas:       2,
			expectNeedsUpdate:     true,
			expectNewGeneration:   5,
		},
		{
			name:                  "both annotations missing - annotations only when spec is equivalent",
			currentGeneration:     5,
			lastAppliedGeneration: "",
			lastAppliedHash:       "",
			desiredReplicas:       2,
			expectNeedsUpdate:     true,
			expectNewGeneration:   5,
		},
		{
			name:                       "manual change with hash also changed",
			currentGeneration:          7,
			lastAppliedGeneration:      "5",
			lastAppliedHash:            "match",
			desiredReplicas:            3, // different
			expectNeedsUpdate:          true,
			expectSpecNeedsUpdate:      true,
			expectManualChangeDetected: false, // hash change takes precedence
			expectNewGeneration:        8,
		},
		{
			// Generation=0 can occur with CRDs that don't have generation tracking enabled,
			// or as a safety net for edge cases. When gen=0, we skip generation-based
			// manual change detection and rely solely on hash comparison.
			name:                  "generation zero - skip generation check",
			currentGeneration:     0,
			lastAppliedGeneration: "0",
			lastAppliedHash:       "match",
			desiredReplicas:       2,
			expectNeedsUpdate:     false, // gen check skipped when gen=0, hash matches
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			// Create current resource
			current := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name":        "test-deployment",
						"namespace":   "default",
						"generation":  tt.currentGeneration,
						"annotations": map[string]interface{}{},
					},
					"spec": map[string]interface{}{
						"replicas": int64(2),
					},
				},
			}

			// Create desired resource
			desired := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
					"metadata": map[string]interface{}{
						"name":      "test-deployment",
						"namespace": "default",
					},
					"spec": map[string]interface{}{
						"replicas": tt.desiredReplicas,
					},
				},
			}

			// Set annotations based on test case
			// "match" means the lastAppliedHash should match the CURRENT spec's hash
			// (simulating that operator last applied what's currently in the cluster)
			annotations := make(map[string]string)
			if tt.lastAppliedHash == "match" {
				hash, err := GetSpecHash(current)
				g.Expect(err).To(gomega.BeNil())
				annotations[NvidiaAnnotationHashKey] = hash
			} else if tt.lastAppliedHash != "" {
				annotations[NvidiaAnnotationHashKey] = tt.lastAppliedHash
			}
			if tt.lastAppliedGeneration != "" {
				annotations[NvidiaAnnotationGenerationKey] = tt.lastAppliedGeneration
			}
			if len(annotations) > 0 {
				current.SetAnnotations(annotations)
			}

			result, err := GetSpecChangeResult(current, desired)
			g.Expect(err).To(gomega.BeNil())
			g.Expect(result.NeedsUpdate).To(gomega.Equal(tt.expectNeedsUpdate), "NeedsUpdate mismatch")
			g.Expect(result.SpecNeedsUpdate).To(gomega.Equal(tt.expectSpecNeedsUpdate), "SpecNeedsUpdate mismatch")
			g.Expect(result.ManualChangeDetected).To(gomega.Equal(tt.expectManualChangeDetected), "ManualChangeDetected mismatch")
			if tt.expectNewGeneration != 0 {
				g.Expect(result.NewGeneration).To(gomega.Equal(tt.expectNewGeneration), "NewGeneration mismatch")
			}
		})
	}
}

func TestCopySpec(t *testing.T) {
	src := appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "nim-deployment",
			Namespace: "default",
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &[]int32{2}[0],
		},
	}

	dst := appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "nim-deployment",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "nim-deployment",
					UID:        "1234567890",
				},
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &[]int32{1}[0],
		},
	}

	err := CopySpec(&src, &dst)
	if err != nil {
		t.Errorf("failed to copy spec in test for resource %s: %s", src.GetName(), err)
	}

	expected := appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "nim-deployment",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "nim-deployment",
					UID:        "1234567890",
				},
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &[]int32{2}[0],
		},
	}

	g := gomega.NewGomegaWithT(t)
	g.Expect(dst).To(gomega.Equal(expected))
}

func TestAppendUniqueImagePullSecrets(t *testing.T) {
	tests := []struct {
		name       string
		existing   []corev1.LocalObjectReference
		additional []corev1.LocalObjectReference
		expected   []corev1.LocalObjectReference
	}{
		{
			name:       "empty existing, empty additional",
			existing:   []corev1.LocalObjectReference{},
			additional: []corev1.LocalObjectReference{},
			expected:   []corev1.LocalObjectReference{},
		},
		{
			name:       "empty existing, some additional",
			existing:   []corev1.LocalObjectReference{},
			additional: []corev1.LocalObjectReference{{Name: "secret-a"}, {Name: "secret-b"}},
			expected:   []corev1.LocalObjectReference{{Name: "secret-a"}, {Name: "secret-b"}},
		},
		{
			name:       "some existing, empty additional",
			existing:   []corev1.LocalObjectReference{{Name: "secret-a"}},
			additional: []corev1.LocalObjectReference{},
			expected:   []corev1.LocalObjectReference{{Name: "secret-a"}},
		},
		{
			name:       "no duplicates",
			existing:   []corev1.LocalObjectReference{{Name: "secret-a"}},
			additional: []corev1.LocalObjectReference{{Name: "secret-b"}, {Name: "secret-c"}},
			expected:   []corev1.LocalObjectReference{{Name: "secret-a"}, {Name: "secret-b"}, {Name: "secret-c"}},
		},
		{
			name:       "all duplicates",
			existing:   []corev1.LocalObjectReference{{Name: "secret-a"}, {Name: "secret-b"}},
			additional: []corev1.LocalObjectReference{{Name: "secret-a"}, {Name: "secret-b"}},
			expected:   []corev1.LocalObjectReference{{Name: "secret-a"}, {Name: "secret-b"}},
		},
		{
			name:       "some duplicates",
			existing:   []corev1.LocalObjectReference{{Name: "secret-a"}, {Name: "secret-b"}},
			additional: []corev1.LocalObjectReference{{Name: "secret-b"}, {Name: "secret-c"}},
			expected:   []corev1.LocalObjectReference{{Name: "secret-a"}, {Name: "secret-b"}, {Name: "secret-c"}},
		},
		{
			name:       "duplicates within additional",
			existing:   []corev1.LocalObjectReference{{Name: "secret-a"}},
			additional: []corev1.LocalObjectReference{{Name: "secret-b"}, {Name: "secret-b"}, {Name: "secret-c"}},
			expected:   []corev1.LocalObjectReference{{Name: "secret-a"}, {Name: "secret-b"}, {Name: "secret-c"}},
		},
		{
			name:       "nil existing",
			existing:   nil,
			additional: []corev1.LocalObjectReference{{Name: "secret-a"}},
			expected:   []corev1.LocalObjectReference{{Name: "secret-a"}},
		},
		{
			name:       "nil additional",
			existing:   []corev1.LocalObjectReference{{Name: "secret-a"}},
			additional: nil,
			expected:   []corev1.LocalObjectReference{{Name: "secret-a"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			result := AppendUniqueImagePullSecrets(tt.existing, tt.additional)
			g.Expect(result).To(gomega.Equal(tt.expected))
		})
	}
}

func TestGetSpecChangeResult_ConfigMap(t *testing.T) {
	baseHash := func(t *testing.T, obj client.Object) string {
		t.Helper()
		h, err := GetSpecHash(obj)
		if err != nil {
			t.Fatalf("GetSpecHash: %v", err)
		}
		return h
	}

	baseCM := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "test-cm", Namespace: "ns"},
		Data:       map[string]string{"script.py": "print('v1')"},
	}

	tests := []struct {
		name        string
		current     client.Object
		desired     client.Object
		needsUpdate bool
	}{
		{
			name: "same ConfigMap data does not need update",
			current: func() client.Object {
				cm := baseCM.DeepCopy()
				cm.Annotations = map[string]string{
					NvidiaAnnotationHashKey:       baseHash(t, baseCM),
					NvidiaAnnotationGenerationKey: "1",
				}
				cm.Generation = 1
				return cm
			}(),
			desired: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "test-cm", Namespace: "ns"},
				Data:       map[string]string{"script.py": "print('v1')"},
			},
			needsUpdate: false,
		},
		{
			name: "changed ConfigMap data needs update",
			current: func() client.Object {
				cm := baseCM.DeepCopy()
				cm.Annotations = map[string]string{
					NvidiaAnnotationHashKey:       baseHash(t, baseCM),
					NvidiaAnnotationGenerationKey: "1",
				}
				cm.Generation = 1
				return cm
			}(),
			desired: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "test-cm", Namespace: "ns"},
				Data:       map[string]string{"script.py": "print('v2')"},
			},
			needsUpdate: true,
		},
		{
			name: "metadata-only change does not need update",
			current: func() client.Object {
				cm := baseCM.DeepCopy()
				cm.Annotations = map[string]string{
					NvidiaAnnotationHashKey:       baseHash(t, baseCM),
					NvidiaAnnotationGenerationKey: "1",
				}
				cm.Generation = 1
				return cm
			}(),
			desired: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "different-name", Namespace: "ns", Labels: map[string]string{"foo": "bar"}},
				Data:       map[string]string{"script.py": "print('v1')"},
			},
			needsUpdate: false,
		},
		{
			name: "added key needs update",
			current: func() client.Object {
				cm := baseCM.DeepCopy()
				cm.Annotations = map[string]string{
					NvidiaAnnotationHashKey:       baseHash(t, baseCM),
					NvidiaAnnotationGenerationKey: "1",
				}
				cm.Generation = 1
				return cm
			}(),
			desired: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "test-cm", Namespace: "ns"},
				Data:       map[string]string{"script.py": "print('v1')", "extra.py": "pass"},
			},
			needsUpdate: true,
		},
		{
			name: "no hash annotation needs update (pre-upgrade resource)",
			current: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "test-cm", Namespace: "ns"},
				Data:       map[string]string{"script.py": "print('v1')"},
			},
			desired: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "test-cm", Namespace: "ns"},
				Data:       map[string]string{"script.py": "print('v1')"},
			},
			needsUpdate: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			result, err := GetSpecChangeResult(tt.current, tt.desired)
			g.Expect(err).ToNot(gomega.HaveOccurred())
			g.Expect(result.NeedsUpdate).To(gomega.Equal(tt.needsUpdate))
		})
	}
}

const (
	ownershipTestNamespace   = "default"
	ownershipTestServiceName = "shared-service"
	ownershipTestParentName  = "test-dgd"
	ownershipTestParentUID   = types.UID("parent-dgd-uid")
	ownershipTestForeignUID  = types.UID("foreign-dgd-uid")
)

type ownershipTestReconciler struct {
	client.Client
	recorder events.EventRecorder
}

func (r *ownershipTestReconciler) GetRecorder() events.EventRecorder {
	return r.recorder
}

func newOwnershipTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(s))
	require.NoError(t, appsv1.AddToScheme(s))
	require.NoError(t, v1beta1.AddToScheme(s))
	return s
}

func newOwnershipTestParent() *v1beta1.DynamoGraphDeployment {
	return &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ownershipTestParentName,
			Namespace: ownershipTestNamespace,
			UID:       ownershipTestParentUID,
		},
	}
}

// foreignControllerRef differs from the parent only by UID.
func foreignControllerRef() metav1.OwnerReference {
	return metav1.OwnerReference{
		APIVersion:         v1beta1.GroupVersion.String(),
		Kind:               "DynamoGraphDeployment",
		Name:               ownershipTestParentName,
		UID:                ownershipTestForeignUID,
		Controller:         ptr.To(true),
		BlockOwnerDeletion: ptr.To(true),
	}
}

func parentControllerRef() metav1.OwnerReference {
	ref := foreignControllerRef()
	ref.UID = ownershipTestParentUID
	return ref
}

func ownershipTestService(port int32) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ownershipTestServiceName,
			Namespace: ownershipTestNamespace,
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: corev1.ClusterIPNone,
			Selector:  map[string]string{"app": "dynamo"},
			Ports: []corev1.ServicePort{
				{Name: "http", Port: port, Protocol: corev1.ProtocolTCP},
			},
		},
	}
}

// seedService marks svc as the last applied content.
func seedService(t *testing.T, svc *corev1.Service) *corev1.Service {
	t.Helper()
	existing := svc.DeepCopy()
	hash, err := GetSpecHash(svc)
	require.NoError(t, err)
	existing.Annotations = map[string]string{
		NvidiaAnnotationHashKey:       hash,
		NvidiaAnnotationGenerationKey: "1",
	}
	existing.Generation = 1
	return existing
}

func newOwnershipTestReconciler(t *testing.T, s *runtime.Scheme, objects ...client.Object) (*ownershipTestReconciler, client.Client) {
	t.Helper()
	c := fake.NewClientBuilder().WithScheme(s).WithObjects(objects...).Build()
	return &ownershipTestReconciler{Client: c, recorder: events.NewFakeRecorder(100)}, c
}

func storedService(t *testing.T, c client.Client) *corev1.Service {
	t.Helper()
	stored := &corev1.Service{}
	require.NoError(t, c.Get(context.Background(), types.NamespacedName{
		Name:      ownershipTestServiceName,
		Namespace: ownershipTestNamespace,
	}, stored))
	return stored
}

func serviceGenerator(svc *corev1.Service, toDelete bool) ResourceGenerator[*corev1.Service] {
	return func(_ context.Context) (*corev1.Service, bool, error) {
		return svc, toDelete, nil
	}
}

func requireOwnershipConflict(t *testing.T, err error) {
	t.Helper()
	require.Error(t, err)
	var alreadyOwned *controllerutil.AlreadyOwnedError
	require.True(t, errors.As(err, &alreadyOwned), "expected an *controllerutil.AlreadyOwnedError, got %v", err)
	assert.Equal(t, ownershipTestForeignUID, alreadyOwned.Owner.UID)
}

func TestSyncResource_ForeignControllerBlocksMutation(t *testing.T) {
	tests := []struct {
		name         string
		existingPort int32
		desiredPort  int32
		toDelete     bool
	}{
		{name: "differing content is not overwritten", existingPort: 9090, desiredPort: 8080},
		{name: "existing object is not deleted", existingPort: 9090, desiredPort: 9090, toDelete: true},
		{name: "matching content does not rewrite owner references", existingPort: 8080, desiredPort: 8080},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("seed an object held by a foreign controller that shares the parent's kind and name")
			s := newOwnershipTestScheme(t)
			existing := seedService(t, ownershipTestService(tt.existingPort))
			existing.OwnerReferences = []metav1.OwnerReference{foreignControllerRef()}
			r, c := newOwnershipTestReconciler(t, s, existing)

			t.Log("sync on behalf of a different parent")
			modified, _, err := SyncResource(context.Background(), r, newOwnershipTestParent(),
				serviceGenerator(ownershipTestService(tt.desiredPort), tt.toDelete))

			t.Log("the sync is refused and the stored object is left exactly as it was")
			requireOwnershipConflict(t, err)
			assert.False(t, modified)
			stored := storedService(t, c)
			assert.Equal(t, tt.existingPort, stored.Spec.Ports[0].Port, "spec of a foreign-owned object must not be overwritten")
			assert.Equal(t, []metav1.OwnerReference{foreignControllerRef()}, stored.OwnerReferences)
			assert.Equal(t, existing.Annotations, stored.Annotations, "bookkeeping annotations must not be written either")
		})
	}
}

func TestSyncResource_SyncProceeds(t *testing.T) {
	tests := []struct {
		name          string
		existingOwner []metav1.OwnerReference
		existingPort  int32
		desiredPort   int32
		parent        client.Object
		opts          []SyncOption
		wantModified  bool
		wantOwner     []metav1.OwnerReference
		wantPort      int32
		reason        string
	}{
		{
			name:         "ownerless object with matching content is adopted",
			existingPort: 8080, desiredPort: 8080,
			parent: newOwnershipTestParent(), wantModified: true,
			wantOwner: []metav1.OwnerReference{parentControllerRef()}, wantPort: 8080,
			reason: "adoption must attach the owner reference without changing the content",
		},
		{
			name:         "ownerless object with differing content is adopted and updated",
			existingPort: 9090, desiredPort: 8080,
			parent: newOwnershipTestParent(), wantModified: true,
			wantOwner: []metav1.OwnerReference{parentControllerRef()}, wantPort: 8080,
			reason: "adoption rides along with the content update in one write",
		},
		{
			name:          "the controller that already owns the object reconciles normally",
			existingOwner: []metav1.OwnerReference{parentControllerRef()},
			existingPort:  9090, desiredPort: 8080,
			parent: newOwnershipTestParent(), wantModified: true,
			wantOwner: []metav1.OwnerReference{parentControllerRef()}, wantPort: 8080,
			reason: "the guard must not disturb the ordinary same-owner path",
		},
		{
			name:          "a nil parent asks for an independent lifecycle",
			existingOwner: []metav1.OwnerReference{foreignControllerRef()},
			existingPort:  9090, desiredPort: 8080,
			parent: nil, wantModified: true,
			wantOwner: []metav1.OwnerReference{foreignControllerRef()}, wantPort: 8080,
			reason: "a nil parent means no ownership opinion, so behavior is unchanged from before the guard",
		},
		{
			name:          "the shared-ownership opt-out tolerates a foreign controller",
			existingOwner: []metav1.OwnerReference{foreignControllerRef()},
			existingPort:  8080, desiredPort: 8080,
			parent: newOwnershipTestParent(), opts: []SyncOption{WithSharedOwnership()},
			wantModified: false,
			wantOwner:    []metav1.OwnerReference{foreignControllerRef()}, wantPort: 8080,
			reason: "an object legitimately shared between owners must neither fail nor be re-owned",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("seed the existing object with the owner references under test")
			ctx := context.Background()
			s := newOwnershipTestScheme(t)
			existing := seedService(t, ownershipTestService(tt.existingPort))
			existing.OwnerReferences = tt.existingOwner
			r, c := newOwnershipTestReconciler(t, s, existing)

			t.Log("sync once")
			modified, res, err := SyncResource(ctx, r, tt.parent,
				serviceGenerator(ownershipTestService(tt.desiredPort), false), tt.opts...)

			t.Log("the sync is allowed and leaves the expected content and ownership")
			require.NoError(t, err, tt.reason)
			assert.Equal(t, tt.wantModified, modified, tt.reason)
			require.NotNil(t, res)
			stored := storedService(t, c)
			assert.Equal(t, tt.wantPort, stored.Spec.Ports[0].Port)
			assert.Equal(t, tt.wantOwner, stored.OwnerReferences)

			t.Log("a second sync over the settled object writes nothing further")
			modified, _, err = SyncResource(ctx, r, tt.parent,
				serviceGenerator(ownershipTestService(tt.desiredPort), false), tt.opts...)
			require.NoError(t, err)
			assert.False(t, modified, "reconciliation must converge rather than write on every pass")
		})
	}
}
