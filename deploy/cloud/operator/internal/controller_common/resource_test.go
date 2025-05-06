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

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

type MyResource struct {
	unstructured.Unstructured
}

func (r *MyResource) GetSpec() any {
	return r.Object["spec"]
}

func (r *MyResource) SetSpec(spec any) {
	r.Object["spec"] = spec
}

func TestIsSpecChanged(t *testing.T) {
	tests := []struct {
		name     string
		current  Resource
		desired  Resource
		expected bool
	}{
		{
			name: "no change in hash with deployment spec and env variables",
			current: &MyResource{
				Unstructured: unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "apps/v1",
						"kind":       "Deployment",
						"metadata": map[string]interface{}{
							"name":      "nim-deployment",
							"namespace": "default",
						},
						"spec": map[string]interface{}{
							"replicas": 2,
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
													"containerPort": 80,
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
			},
			desired: &MyResource{
				Unstructured: unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "apps/v1",
						"kind":       "Deployment",
						"metadata": map[string]interface{}{
							"name":      "nim-deployment",
							"namespace": "default",
						},
						"spec": map[string]interface{}{
							"replicas": 2,
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
													"containerPort": 80,
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
			},
			expected: false,
		},
		{
			name: "no change in hash with change in order of elements",
			current: &MyResource{
				Unstructured: unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "apps/v1",
						"kind":       "Deployment",
						"metadata": map[string]interface{}{
							"name":      "nim-deployment",
							"namespace": "default",
						},
						"spec": map[string]interface{}{
							"replicas": 2,
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
													"containerPort": 80,
												},
											}, // switch order of env
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
			},
			desired: &MyResource{
				Unstructured: unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "apps/v1",
						"kind":       "Deployment",
						"metadata": map[string]interface{}{
							"name":      "nim-deployment",
							"namespace": "default",
						},
						"spec": map[string]interface{}{
							"replicas": 2,
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
													"containerPort": 80,
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
			},
			expected: false,
		},
		{
			name: "change in hash with change in value of elements",
			current: &MyResource{
				Unstructured: unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "apps/v1",
						"kind":       "Deployment",
						"metadata": map[string]interface{}{
							"name":      "nim-deployment",
							"namespace": "default",
						},
						"spec": map[string]interface{}{
							"replicas": 2,
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
													"containerPort": 80,
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
			},
			desired: &MyResource{
				Unstructured: unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "apps/v1",
						"kind":       "Deployment",
						"metadata": map[string]interface{}{
							"name":      "nim-deployment",
							"namespace": "default",
						},
						"spec": map[string]interface{}{
							"replicas": 3,
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
													"containerPort": 80,
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
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.current.SetAnnotations(map[string]string{
				NvidiaAnnotationHashKey: GetResourceHash(tt.current.GetSpec()),
			})
			if got := IsSpecChanged(tt.current, tt.desired); got != tt.expected {
				t.Errorf("IsSpecChanged() = %v, want %v", got, tt.expected)
			}
		})
	}
}
