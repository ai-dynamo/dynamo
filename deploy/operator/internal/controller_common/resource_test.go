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
	stderrors "errors"
	"reflect"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"

	"github.com/bsm/gomega"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
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

func TestCopySpecPreservesUnstructuredFields(t *testing.T) {
	t.Log("Build live and desired unstructured specs with an opaque future field")
	src := &unstructured.Unstructured{Object: map[string]interface{}{
		"spec": map[string]interface{}{
			"known":  "value",
			"future": map[string]interface{}{"enabled": true},
		},
	}}
	dst := &unstructured.Unstructured{Object: map[string]interface{}{
		"metadata": map[string]interface{}{"name": "resource"},
		"spec":     map[string]interface{}{"known": "old"},
	}}

	t.Log("Copy the desired spec without decoding it through registered Go types")
	if err := CopySpec(src, dst); err != nil {
		t.Fatalf("CopySpec() error = %v", err)
	}

	t.Log("Verify the opaque field and its value were preserved")
	want := map[string]interface{}{
		"known":  "value",
		"future": map[string]interface{}{"enabled": true},
	}
	got, found, err := unstructured.NestedMap(dst.Object, "spec")
	if err != nil {
		t.Fatalf("read copied spec: %v", err)
	}
	if !found {
		t.Fatal("copied spec was not found")
	}
	gomega.NewWithT(t).Expect(got).To(gomega.Equal(want))
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

type observedResourceTestReconciler struct {
	client.Client
	recorder events.EventRecorder
}

func (r observedResourceTestReconciler) GetRecorder() events.EventRecorder {
	return r.recorder
}

func TestSyncObservedResourceUsesProvidedObservation(t *testing.T) {
	t.Log("Build an observed ConfigMap and record client reads")
	ctx := context.Background()
	g := gomega.NewGomegaWithT(t)
	scheme := runtime.NewScheme()
	g.Expect(corev1.AddToScheme(scheme)).To(gomega.Succeed())

	existing := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "config", Namespace: "default"},
		Data:       map[string]string{"value": "before"},
	}
	getCalls := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(existing).
		WithInterceptorFuncs(interceptor.Funcs{
			Get: func(
				ctx context.Context,
				reader client.WithWatch,
				key client.ObjectKey,
				object client.Object,
				options ...client.GetOption,
			) error {
				getCalls++
				return reader.Get(ctx, key, object, options...)
			},
		}).
		Build()
	reconciler := observedResourceTestReconciler{
		Client:   kubeClient,
		recorder: events.NewFakeRecorder(10),
	}

	observed := &corev1.ConfigMap{}
	g.Expect(kubeClient.Get(ctx, client.ObjectKeyFromObject(existing), observed)).To(gomega.Succeed())
	desired := observed.DeepCopy()
	desired.Data["value"] = "after"

	t.Log("Sync the desired ConfigMap from the exact caller observation")
	modified, synced, err := SyncObservedResource(ctx, reconciler, nil, observed, desired)

	t.Log("Verify sync did not reread or mutate the supplied observation")
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(modified).To(gomega.BeTrue())
	g.Expect(getCalls).To(gomega.Equal(1), "sync must use the caller's exact observation")
	g.Expect(observed.Data["value"]).To(gomega.Equal("before"), "sync must not mutate the caller's observation")
	g.Expect(synced.Data["value"]).To(gomega.Equal("after"))
}

type syncResourceTestReconciler struct {
	client.Client
	recorder events.EventRecorder
}

func (r syncResourceTestReconciler) GetRecorder() events.EventRecorder {
	return r.recorder
}

func newSyncResourceTestReconciler(t *testing.T, objects ...client.Object) syncResourceTestReconciler {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add corev1 to scheme: %v", err)
	}
	return syncResourceTestReconciler{
		Client:   fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build(),
		recorder: events.NewFakeRecorder(16),
	}
}

func syncResourceTestParent(uid string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "parent",
			Namespace: "default",
			UID:       types.UID(uid),
		},
	}
}

func syncResourceTestControllerRef(parent *corev1.ConfigMap) metav1.OwnerReference {
	controller := true
	return metav1.OwnerReference{
		APIVersion: "v1",
		Kind:       "ConfigMap",
		Name:       parent.Name,
		UID:        parent.UID,
		Controller: &controller,
	}
}

func syncResourceTestForeignRef() metav1.OwnerReference {
	controller := true
	return metav1.OwnerReference{
		APIVersion: "apps/v1",
		Kind:       "Deployment",
		Name:       "foreign",
		UID:        "foreign-uid",
		Controller: &controller,
	}
}

func syncResourceTestConfigMap(value string, refs ...metav1.OwnerReference) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "managed",
			Namespace:       "default",
			OwnerReferences: refs,
		},
		Data: map[string]string{"value": value},
	}
}

func TestSyncResourceOwnership(t *testing.T) {
	parent := syncResourceTestParent("current-uid")
	staleParent := syncResourceTestParent("stale-uid")
	currentRef := syncResourceTestControllerRef(parent)
	foreignRef := syncResourceTestForeignRef()
	staleRef := syncResourceTestControllerRef(staleParent)

	tests := []struct {
		name          string
		parent        client.Object
		existing      *corev1.ConfigMap
		desiredValue  string
		toDelete      bool
		opts          []SyncOption
		wantErr       bool
		wantModified  bool
		wantExists    bool
		wantValue     string
		wantOwnerRefs []metav1.OwnerReference
	}{
		{
			name:          "matching controller owner updates",
			parent:        parent,
			existing:      syncResourceTestConfigMap("before", currentRef),
			desiredValue:  "after",
			wantModified:  true,
			wantExists:    true,
			wantValue:     "after",
			wantOwnerRefs: []metav1.OwnerReference{currentRef},
		},
		{
			name:          "nil parent updates foreign-owned resource",
			existing:      syncResourceTestConfigMap("before", foreignRef),
			desiredValue:  "after",
			wantModified:  true,
			wantExists:    true,
			wantValue:     "after",
			wantOwnerRefs: []metav1.OwnerReference{foreignRef},
		},
		{
			name:          "shared ownership updates foreign-owned resource",
			parent:        parent,
			existing:      syncResourceTestConfigMap("before", foreignRef),
			desiredValue:  "after",
			opts:          []SyncOption{WithSharedOwnership()},
			wantModified:  true,
			wantExists:    true,
			wantValue:     "after",
			wantOwnerRefs: []metav1.OwnerReference{foreignRef},
		},
		{
			name:          "foreign controller update is rejected",
			parent:        parent,
			existing:      syncResourceTestConfigMap("foreign", foreignRef),
			desiredValue:  "desired",
			wantErr:       true,
			wantExists:    true,
			wantValue:     "foreign",
			wantOwnerRefs: []metav1.OwnerReference{foreignRef},
		},
		{
			name:          "foreign controller delete is rejected",
			parent:        parent,
			existing:      syncResourceTestConfigMap("foreign", foreignRef),
			desiredValue:  "foreign",
			toDelete:      true,
			wantErr:       true,
			wantExists:    true,
			wantValue:     "foreign",
			wantOwnerRefs: []metav1.OwnerReference{foreignRef},
		},
		{
			name:         "ownerless existing resource is rejected before noop",
			parent:       parent,
			existing:     syncResourceTestConfigMap("desired"),
			desiredValue: "desired",
			wantErr:      true,
			wantExists:   true,
			wantValue:    "desired",
		},
		{
			name:          "same parent name with stale UID delete is rejected",
			parent:        parent,
			existing:      syncResourceTestConfigMap("foreign", staleRef),
			desiredValue:  "foreign",
			toDelete:      true,
			wantErr:       true,
			wantExists:    true,
			wantValue:     "foreign",
			wantOwnerRefs: []metav1.OwnerReference{staleRef},
		},
		{
			name:         "not found delete is a noop",
			parent:       parent,
			desiredValue: "unused",
			toDelete:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objects []client.Object
			if tt.existing != nil {
				objects = append(objects, tt.existing)
			}
			r := newSyncResourceTestReconciler(t, objects...)
			modified, _, err := SyncResource(
				context.Background(),
				r,
				tt.parent,
				func(context.Context) (*corev1.ConfigMap, bool, error) {
					return syncResourceTestConfigMap(tt.desiredValue), tt.toDelete, nil
				},
				tt.opts...,
			)
			if tt.wantErr && err == nil {
				t.Fatal("SyncResource() error = nil, want ownership conflict")
			}
			if tt.wantErr {
				var ownershipConflict *OwnershipConflictError
				if !stderrors.As(err, &ownershipConflict) {
					t.Fatalf("SyncResource() error = %T %v, want OwnershipConflictError", err, err)
				}
				if recorder := r.recorder.(*events.FakeRecorder); len(recorder.Events) != 0 {
					t.Fatalf("SyncResource() recorded %d helper-level ownership event(s), want none", len(recorder.Events))
				}
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("SyncResource() error = %v", err)
			}
			if modified != tt.wantModified {
				t.Fatalf("SyncResource() modified = %t, want %t", modified, tt.wantModified)
			}

			got := &corev1.ConfigMap{}
			err = r.Get(context.Background(), client.ObjectKey{Name: "managed", Namespace: "default"}, got)
			if !tt.wantExists {
				if !apierrors.IsNotFound(err) {
					t.Fatalf("get synced ConfigMap error = %v, want NotFound", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("get synced ConfigMap: %v", err)
			}
			if got.Data["value"] != tt.wantValue {
				t.Fatalf("stored value = %q, want %q", got.Data["value"], tt.wantValue)
			}
			if !reflect.DeepEqual(got.OwnerReferences, tt.wantOwnerRefs) {
				t.Fatalf("owner references = %#v, want %#v", got.OwnerReferences, tt.wantOwnerRefs)
			}
		})
	}
}

func TestSyncResourceDeleteUsesObservedPreconditions(t *testing.T) {
	parent := syncResourceTestParent("parent-uid")
	existing := syncResourceTestConfigMap("before", syncResourceTestControllerRef(parent))
	existing.UID = "resource-uid"

	var deleteUID *types.UID
	var deleteResourceVersion *string
	var observedResourceVersion string
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add corev1 to scheme: %v", err)
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(existing).
		WithInterceptorFuncs(interceptor.Funcs{
			Delete: func(ctx context.Context, delegate client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
				options := (&client.DeleteOptions{}).ApplyOptions(opts)
				observedResourceVersion = object.GetResourceVersion()
				if options.Preconditions == nil || options.Preconditions.UID == nil || options.Preconditions.ResourceVersion == nil {
					t.Fatal("SyncResource() delete does not have both UID and ResourceVersion preconditions")
				}
				deleteUID = options.Preconditions.UID
				deleteResourceVersion = options.Preconditions.ResourceVersion
				return delegate.Delete(ctx, object, opts...)
			},
		}).
		Build()
	r := syncResourceTestReconciler{
		Client:   kubeClient,
		recorder: events.NewFakeRecorder(16),
	}

	modified, _, err := SyncResource(
		context.Background(),
		r,
		parent,
		func(context.Context) (*corev1.ConfigMap, bool, error) {
			return syncResourceTestConfigMap("unused"), true, nil
		},
	)
	if err != nil {
		t.Fatalf("SyncResource() error = %v", err)
	}
	if !modified {
		t.Fatal("SyncResource() modified = false, want true")
	}
	if deleteUID == nil || *deleteUID != existing.UID {
		t.Fatalf("delete UID precondition = %v, want %q", deleteUID, existing.UID)
	}
	if deleteResourceVersion == nil || *deleteResourceVersion != observedResourceVersion {
		t.Fatalf("delete ResourceVersion precondition = %v, want %q", deleteResourceVersion, observedResourceVersion)
	}
}
