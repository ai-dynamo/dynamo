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

package modelendpoint

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

const (
	testPodWorker0 = "worker-0"
	testPodWorker1 = "worker-1"
	testPodWorker2 = "worker-2"
)

func TestExtractCandidates(t *testing.T) {
	trueVal := true
	falseVal := false

	tests := []struct {
		name                 string
		endpointSlices       *discoveryv1.EndpointSliceList
		expectedCandidates   int
		expectedServiceNames map[string]bool
		validateCandidates   func(t *testing.T, candidates []Candidate)
	}{
		{
			name: "empty endpoint slice list",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{},
			},
			expectedCandidates:   0,
			expectedServiceNames: map[string]bool{},
		},
		{
			name: "endpoint with pod target ref - included",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{
					{
						Labels: map[string]string{
							discoveryv1.LabelServiceName: "my-service",
						},
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses: []string{"10.0.1.5"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker0,
								},
							},
						},
					},
				},
			},
			expectedCandidates: 1,
			expectedServiceNames: map[string]bool{
				"my-service": true,
			},
			validateCandidates: func(t *testing.T, candidates []Candidate) {
				if len(candidates) != 1 {
					t.Fatalf("expected 1 candidate, got %d", len(candidates))
				}
				if candidates[0].Address != "http://10.0.1.5:9090" {
					t.Errorf("expected address http://10.0.1.5:9090, got %s", candidates[0].Address)
				}
				if candidates[0].PodName != testPodWorker0 {
					t.Errorf("expected podName %s, got %s", testPodWorker0, candidates[0].PodName)
				}
			},
		},
		{
			name: "endpoint with nil target ref - skipped",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{
					{
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses: []string{"10.0.1.5"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: nil,
							},
						},
					},
				},
			},
			expectedCandidates:   0,
			expectedServiceNames: map[string]bool{},
		},
		{
			name: "endpoint with non-pod target ref - skipped",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{
					{
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses: []string{"10.0.1.5"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Node",
									Name: "node-1",
								},
							},
						},
					},
				},
			},
			expectedCandidates:   0,
			expectedServiceNames: map[string]bool{},
		},
		{
			name: "endpoint not ready - included",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{
					{
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses: []string{"10.0.1.5"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &falseVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker0,
								},
							},
						},
					},
				},
			},
			expectedCandidates:   1,
			expectedServiceNames: map[string]bool{},
		},
		{
			name: "endpoint with nil ready condition - included",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{
					{
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses: []string{"10.0.1.5"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: nil,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker0,
								},
							},
						},
					},
				},
			},
			expectedCandidates:   1,
			expectedServiceNames: map[string]bool{},
		},
		{
			name: "endpoint with no addresses - skipped",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{
					{
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses: []string{},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker0,
								},
							},
						},
					},
				},
			},
			expectedCandidates:   0,
			expectedServiceNames: map[string]bool{},
		},
		{
			name: "multiple valid endpoints",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{
					{
						Labels: map[string]string{
							discoveryv1.LabelServiceName: "service-1",
						},
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses: []string{"10.0.1.5"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker0,
								},
							},
							{
								Addresses: []string{"10.0.1.6"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker1,
								},
							},
						},
					},
				},
			},
			expectedCandidates: 2,
			expectedServiceNames: map[string]bool{
				"service-1": true,
			},
			validateCandidates: func(t *testing.T, candidates []Candidate) {
				if len(candidates) != 2 {
					t.Fatalf("expected 2 candidates, got %d", len(candidates))
				}
			},
		},
		{
			name: "mixed valid and invalid endpoints",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{
					{
						Labels: map[string]string{
							discoveryv1.LabelServiceName: "my-service",
						},
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses: []string{"10.0.1.5"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker0,
								},
							},
							{
								Addresses: []string{"10.0.1.6"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &falseVal, // Not ready - now included (readiness determined by probing)
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker1,
								},
							},
							{
								Addresses: []string{"10.0.1.7"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Node", // Not a Pod - should be skipped
									Name: "node-1",
								},
							},
							{
								Addresses: []string{"10.0.1.8"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: nil, // Nil TargetRef - should be skipped
							},
							{
								Addresses: []string{"10.0.1.9"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker2,
								},
							},
						},
					},
				},
			},
			expectedCandidates: 3, // testPodWorker0, testPodWorker1 (unready), and testPodWorker2
			expectedServiceNames: map[string]bool{
				"my-service": true,
			},
			validateCandidates: func(t *testing.T, candidates []Candidate) {
				if len(candidates) != 3 {
					t.Fatalf("expected 3 candidates, got %d", len(candidates))
				}
				// Verify only valid pods are included (all 3 pod-backed endpoints)
				validPods := map[string]bool{testPodWorker0: false, testPodWorker1: false, testPodWorker2: false}
				for _, c := range candidates {
					if _, exists := validPods[c.PodName]; exists {
						validPods[c.PodName] = true
					} else {
						t.Errorf("unexpected pod in candidates: %s", c.PodName)
					}
				}
				for pod, found := range validPods {
					if !found {
						t.Errorf("expected pod %s not found in candidates", pod)
					}
				}
			},
		},
		{
			name: "endpoint with multiple addresses - physical pod included once",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{
					{
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses: []string{"10.0.1.5", "10.0.2.5"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker0,
								},
							},
						},
					},
				},
			},
			expectedCandidates:   1,
			expectedServiceNames: map[string]bool{},
			validateCandidates: func(t *testing.T, candidates []Candidate) {
				if len(candidates) != 1 {
					t.Fatalf("expected 1 candidate, got %d", len(candidates))
				}
				if candidates[0].PodName != testPodWorker0 {
					t.Errorf("expected candidate to have podName %s", testPodWorker0)
				}
			},
		},
		{
			name: "duplicate pod endpoint across services - included once",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{
					{
						Labels: map[string]string{
							discoveryv1.LabelServiceName: "component-service",
						},
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(12345))}},
						Endpoints: []discoveryv1.Endpoint{{
							Addresses:  []string{"10.0.1.5"},
							Conditions: discoveryv1.EndpointConditions{Ready: &falseVal},
							TargetRef:  &corev1.ObjectReference{Kind: "Pod", Name: testPodWorker0, UID: "worker-uid"},
						}},
					},
					{
						Labels: map[string]string{
							discoveryv1.LabelServiceName: "base-model-service",
						},
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(12345))}},
						Endpoints: []discoveryv1.Endpoint{{
							Addresses:  []string{"10.0.1.5"},
							Conditions: discoveryv1.EndpointConditions{Ready: &trueVal},
							TargetRef:  &corev1.ObjectReference{Kind: "Pod", Name: testPodWorker0, UID: "worker-uid"},
						}},
					},
				},
			},
			expectedCandidates: 1,
			expectedServiceNames: map[string]bool{
				"component-service":  true,
				"base-model-service": true,
			},
			validateCandidates: func(t *testing.T, candidates []Candidate) {
				if len(candidates) != 1 {
					t.Fatalf("expected one candidate, got %d", len(candidates))
				}
				candidate := candidates[0]
				require.Equal(t, "http://10.0.1.5:12345", candidate.Address)
				if !candidate.KubernetesReady {
					t.Fatal("expected Kubernetes readiness to survive deduplication")
				}
			},
		},
		{
			name: "multiple services",
			endpointSlices: &discoveryv1.EndpointSliceList{
				Items: []discoveryv1.EndpointSlice{
					{
						Labels: map[string]string{
							discoveryv1.LabelServiceName: "service-1",
						},
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses: []string{"10.0.1.5"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker0,
								},
							},
						},
					},
					{
						Labels: map[string]string{
							discoveryv1.LabelServiceName: "service-2",
						},
						Ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
						Endpoints: []discoveryv1.Endpoint{
							{
								Addresses: []string{"10.0.1.6"},
								Conditions: discoveryv1.EndpointConditions{
									Ready: &trueVal,
								},
								TargetRef: &corev1.ObjectReference{
									Kind: "Pod",
									Name: testPodWorker1,
								},
							},
						},
					},
				},
			},
			expectedCandidates: 2,
			expectedServiceNames: map[string]bool{
				"service-1": true,
				"service-2": true,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			candidates, serviceNames := ExtractCandidates(tt.endpointSlices)

			// Check candidate count
			if len(candidates) != tt.expectedCandidates {
				t.Errorf("expected %d candidates, got %d", tt.expectedCandidates, len(candidates))
			}

			// Check service names
			if len(serviceNames) != len(tt.expectedServiceNames) {
				t.Errorf("expected %d service names, got %d", len(tt.expectedServiceNames), len(serviceNames))
			}
			for name := range tt.expectedServiceNames {
				if !serviceNames[name] {
					t.Errorf("expected service name %s not found", name)
				}
			}

			// Run additional validation if provided
			if tt.validateCandidates != nil {
				tt.validateCandidates(t, candidates)
			}
		})
	}
}

func TestExtractCandidatesUsesNamedSystemPort(t *testing.T) {
	tests := []struct {
		name        string
		ports       []discoveryv1.EndpointPort
		address     string
		wantAddress string
	}{
		{
			name:        "custom system port",
			ports:       []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(12345))}},
			address:     "10.0.1.5",
			wantAddress: "http://10.0.1.5:12345",
		},
		{
			name:        "IPv6 custom system port",
			ports:       []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(12345)), Protocol: new(corev1.ProtocolTCP)}},
			address:     "2001:db8::1",
			wantAddress: "http://[2001:db8::1]:12345",
		},
		{
			name:        "explicit default port remains supported",
			ports:       []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(9090))}},
			address:     "10.0.1.5",
			wantAddress: "http://10.0.1.5:9090",
		},
		{
			name: "unrelated port does not override system",
			ports: []discoveryv1.EndpointPort{
				{Name: new("http"), Port: new(int32(8000))},
				{Name: new("system"), Port: new(int32(12345))},
			},
			address:     "10.0.1.5",
			wantAddress: "http://10.0.1.5:12345",
		},
		{name: "missing ports", address: "10.0.1.5"},
		{name: "missing name", ports: []discoveryv1.EndpointPort{{Port: new(int32(12345))}}, address: "10.0.1.5"},
		{name: "unrelated name", ports: []discoveryv1.EndpointPort{{Name: new("http"), Port: new(int32(12345))}}, address: "10.0.1.5"},
		{name: "missing port", ports: []discoveryv1.EndpointPort{{Name: new("system")}}, address: "10.0.1.5"},
		{name: "zero port", ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(0))}}, address: "10.0.1.5"},
		{name: "negative port", ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(-1))}}, address: "10.0.1.5"},
		{name: "out of range port", ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(65536))}}, address: "10.0.1.5"},
		{name: "UDP port", ports: []discoveryv1.EndpointPort{{Name: new("system"), Port: new(int32(12345)), Protocol: new(corev1.ProtocolUDP)}}, address: "10.0.1.5"},
		{
			name: "failover engine ports do not invent a pod-level model endpoint",
			ports: []discoveryv1.EndpointPort{
				{Name: new("system-0"), Port: new(int32(12345))},
				{Name: new("system-1"), Port: new(int32(12346))},
			},
			address: "10.0.1.5",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Discover one Pod through the ports actually published by its Service")
			slices := &discoveryv1.EndpointSliceList{Items: []discoveryv1.EndpointSlice{{
				Labels: map[string]string{discoveryv1.LabelServiceName: "model-service"},
				Ports:  tt.ports,
				Endpoints: []discoveryv1.Endpoint{{
					Addresses:  []string{tt.address},
					TargetRef:  &corev1.ObjectReference{Kind: "Pod", Name: testPodWorker0, UID: "worker-uid"},
					Conditions: discoveryv1.EndpointConditions{Ready: new(true)},
				}},
			}}}
			candidates, serviceNames := ExtractCandidates(slices)

			t.Log("Use the resolved system port or leave the endpoint undiscovered without falling back")
			require.Equal(t, map[string]bool{"model-service": true}, serviceNames)
			if tt.wantAddress == "" {
				require.Empty(t, candidates)
				return
			}
			require.Equal(t, []Candidate{{Address: tt.wantAddress, PodName: testPodWorker0, KubernetesReady: true}}, candidates)
		})
	}
}

func TestFindModelsForBaseModel(t *testing.T) {
	tests := []struct {
		name           string
		namespace      string
		baseModelName  string
		indexField     string
		existingModels []v1alpha1.DynamoModel
		expectedCount  int
		expectedNames  []string
		expectError    bool
	}{
		{
			name:          "finds multiple models for base model",
			namespace:     "default",
			baseModelName: "llama-2-7b",
			indexField:    ".spec.baseModelName",
			existingModels: []v1alpha1.DynamoModel{
				{
					Name:      "lora-1",
					Namespace: "default",
					Spec: v1alpha1.DynamoModelSpec{
						ModelName:     "lora-1",
						BaseModelName: "llama-2-7b",
						ModelType:     "lora",
					},
				},
				{
					Name:      "lora-2",
					Namespace: "default",
					Spec: v1alpha1.DynamoModelSpec{
						ModelName:     "lora-2",
						BaseModelName: "llama-2-7b",
						ModelType:     "lora",
					},
				},
				{
					Name:      "different-base",
					Namespace: "default",
					Spec: v1alpha1.DynamoModelSpec{
						ModelName:     "different-base",
						BaseModelName: "gpt-3",
						ModelType:     "lora",
					},
				},
			},
			expectedCount: 2,
			expectedNames: []string{"lora-1", "lora-2"},
			expectError:   false,
		},
		{
			name:          "finds no models for base model",
			namespace:     "default",
			baseModelName: "non-existent-base",
			indexField:    ".spec.baseModelName",
			existingModels: []v1alpha1.DynamoModel{
				{
					Name:      "lora-1",
					Namespace: "default",
					Spec: v1alpha1.DynamoModelSpec{
						ModelName:     "lora-1",
						BaseModelName: "llama-2-7b",
						ModelType:     "lora",
					},
				},
			},
			expectedCount: 0,
			expectedNames: []string{},
			expectError:   false,
		},
		{
			name:          "filters by namespace",
			namespace:     "ns1",
			baseModelName: "llama-2-7b",
			indexField:    ".spec.baseModelName",
			existingModels: []v1alpha1.DynamoModel{
				{
					Name:      "lora-ns1",
					Namespace: "ns1",
					Spec: v1alpha1.DynamoModelSpec{
						ModelName:     "lora-ns1",
						BaseModelName: "llama-2-7b",
						ModelType:     "lora",
					},
				},
				{
					Name:      "lora-ns2",
					Namespace: "ns2",
					Spec: v1alpha1.DynamoModelSpec{
						ModelName:     "lora-ns2",
						BaseModelName: "llama-2-7b",
						ModelType:     "lora",
					},
				},
			},
			expectedCount: 1,
			expectedNames: []string{"lora-ns1"},
			expectError:   false,
		},
		{
			name:           "handles empty model list",
			namespace:      "default",
			baseModelName:  "any-base",
			indexField:     ".spec.baseModelName",
			existingModels: []v1alpha1.DynamoModel{},
			expectedCount:  0,
			expectedNames:  []string{},
			expectError:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a fake client with existing models
			scheme := runtime.NewScheme()
			_ = v1alpha1.AddToScheme(scheme)
			_ = corev1.AddToScheme(scheme)

			objs := make([]client.Object, len(tt.existingModels))
			for i := range tt.existingModels {
				objs[i] = &tt.existingModels[i]
			}

			// Create fake client with indexer support
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(objs...).
				WithIndex(&v1alpha1.DynamoModel{}, tt.indexField, func(obj client.Object) []string {
					model := obj.(*v1alpha1.DynamoModel)
					return []string{model.Spec.BaseModelName}
				}).
				Build()

			ctx := context.Background()

			// Call the function
			requests, err := FindModelsForBaseModel(ctx, fakeClient, tt.namespace, tt.baseModelName, tt.indexField)

			// Verify error
			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Verify count
			if len(requests) != tt.expectedCount {
				t.Errorf("expected %d requests, got %d", tt.expectedCount, len(requests))
			}

			// Verify names
			foundNames := make(map[string]bool)
			for _, req := range requests {
				foundNames[req.Name] = true
			}

			for _, expectedName := range tt.expectedNames {
				if !foundNames[expectedName] {
					t.Errorf("expected to find model %s, but it was not in the results", expectedName)
				}
			}

			// Verify all returned names were expected
			if len(foundNames) != len(tt.expectedNames) {
				t.Errorf("found unexpected models in results")
			}
		})
	}
}
