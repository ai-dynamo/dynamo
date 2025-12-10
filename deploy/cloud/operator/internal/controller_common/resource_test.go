/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/bsm/gomega"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestIsSpecChanged2(t *testing.T) {
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
			result, err := IsSpecChanged(tt.current, tt.desired)
			if err != nil {
				t.Errorf("failed to check if spec has changed in test for resource %s: %s", tt.current.GetName(), err)
			}
			if tt.expectedHash && !result.NeedsUpdate {
				t.Errorf("IsSpecChanged() NeedsUpdate = %v, want %v", result.NeedsUpdate, tt.expectedHash)
			}
			if !tt.expectedHash && result.NeedsUpdate {
				t.Errorf("IsSpecChanged() NeedsUpdate = %v, want %v", result.NeedsUpdate, tt.expectedHash)
			}
		})
	}
}

func TestIsSpecChanged_GenerationTracking(t *testing.T) {
	tests := []struct {
		name                       string
		currentGeneration          int64
		lastAppliedGeneration      string // empty string means annotation not set
		lastAppliedHash            string // empty string means annotation not set, "match" means compute from desired
		desiredReplicas            int64  // different from current (2) means hash will differ
		expectNeedsUpdate          bool
		expectManualChangeDetected bool
		expectAnnotationOnly       bool
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
			name:                       "manual change detected - generation increased",
			currentGeneration:          7,
			lastAppliedGeneration:      "5",
			lastAppliedHash:            "match",
			desiredReplicas:            2,
			expectNeedsUpdate:          true,
			expectManualChangeDetected: true,
			expectNewGeneration:        8, // current(7) + 1
		},
		{
			name:                  "missing generation annotation - annotation only update",
			currentGeneration:     5,
			lastAppliedGeneration: "", // missing
			lastAppliedHash:       "match",
			desiredReplicas:       2,
			expectNeedsUpdate:     true,
			expectAnnotationOnly:  true,
			expectNewGeneration:   5, // current, not +1 (no spec change)
		},
		{
			name:                  "missing hash annotation - needs full update",
			currentGeneration:     5,
			lastAppliedGeneration: "5",
			lastAppliedHash:       "", // missing
			desiredReplicas:       2,
			expectNeedsUpdate:     true,
			expectNewGeneration:   6, // current(5) + 1
		},
		{
			name:                  "hash changed - needs full update",
			currentGeneration:     5,
			lastAppliedGeneration: "5",
			lastAppliedHash:       "match",
			desiredReplicas:       3, // different from current (2)
			expectNeedsUpdate:     true,
			expectNewGeneration:   6, // current(5) + 1
		},
		{
			name:                  "corrupted generation annotation - needs full update",
			currentGeneration:     5,
			lastAppliedGeneration: "invalid",
			lastAppliedHash:       "match",
			desiredReplicas:       2,
			expectNeedsUpdate:     true,
			expectNewGeneration:   6, // current(5) + 1
		},
		{
			name:                  "both annotations missing - needs full update",
			currentGeneration:     5,
			lastAppliedGeneration: "",
			lastAppliedHash:       "",
			desiredReplicas:       2,
			expectNeedsUpdate:     true,
			expectNewGeneration:   6, // current(5) + 1
		},
		{
			name:                       "manual change with hash also changed",
			currentGeneration:          7,
			lastAppliedGeneration:      "5",
			lastAppliedHash:            "match",
			desiredReplicas:            3, // different
			expectNeedsUpdate:          true,
			expectManualChangeDetected: false, // hash change takes precedence
			expectNewGeneration:        8,
		},
		{
			name:                  "generation zero - skip generation check",
			currentGeneration:     0,
			lastAppliedGeneration: "0",
			lastAppliedHash:       "match",
			desiredReplicas:       2,
			expectNeedsUpdate:     false, // gen check skipped when gen=0
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

			result, err := IsSpecChanged(current, desired)
			g.Expect(err).To(gomega.BeNil())
			g.Expect(result.NeedsUpdate).To(gomega.Equal(tt.expectNeedsUpdate), "NeedsUpdate mismatch")
			g.Expect(result.ManualChangeDetected).To(gomega.Equal(tt.expectManualChangeDetected), "ManualChangeDetected mismatch")
			g.Expect(result.AnnotationOnly).To(gomega.Equal(tt.expectAnnotationOnly), "AnnotationOnly mismatch")
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

func TestGetResourcesConfig(t *testing.T) {
	tests := []struct {
		name             string
		resources        *v1alpha1.Resources
		expectedGPULimit corev1.ResourceName
		expectedGPUValue string
		expectError      bool
	}{
		{
			name: "limits.gpu defined with no gpuType",
			resources: &v1alpha1.Resources{
				Limits: &v1alpha1.ResourceItem{
					GPU: "4",
				},
			},
			expectedGPULimit: corev1.ResourceName(consts.KubeResourceGPUNvidia),
			expectedGPUValue: "4",
			expectError:      false,
		},
		{
			name: "limits.gpu defined with custom gpuType",
			resources: &v1alpha1.Resources{
				Limits: &v1alpha1.ResourceItem{
					GPU:     "8",
					GPUType: "gpu.custom-type.com/test",
				},
			},
			expectedGPULimit: corev1.ResourceName("gpu.custom-type.com/test"),
			expectedGPUValue: "8",
			expectError:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			result, err := GetResourcesConfig(tt.resources)

			if tt.expectError {
				g.Expect(err).To(gomega.HaveOccurred())
				return
			}

			g.Expect(err).To(gomega.BeNil())
			g.Expect(result).ToNot(gomega.BeNil())
			g.Expect(result.Limits).ToNot(gomega.BeNil())

			gpuQuantity, exists := result.Limits[tt.expectedGPULimit]
			g.Expect(exists).To(gomega.BeTrue(), "GPU resource %s should exist in limits", tt.expectedGPULimit)
			g.Expect(gpuQuantity.String()).To(gomega.Equal(tt.expectedGPUValue))
		})
	}
}
