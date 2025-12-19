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

package controller

import (
	"context"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDynamoGraphDeploymentScalingAdapterReconciler_Reconcile(t *testing.T) {
	// Register custom types with the scheme
	if err := v1alpha1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1alpha1 to scheme: %v", err)
	}

	tests := []struct {
		name                   string
		adapter                *v1alpha1.DynamoGraphDeploymentScalingAdapter
		dgd                    *v1alpha1.DynamoGraphDeployment
		expectedDGDReplicas    int32
		expectedStatusReplicas int32
		expectError            bool
		expectRequeue          bool
	}{
		{
			name: "updates DGD replicas when DGDSA spec differs",
			adapter: &v1alpha1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd-frontend",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 5,
					DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
						Name:        "test-dgd",
						ServiceName: "Frontend",
					},
				},
			},
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(2)),
						},
					},
				},
			},
			expectedDGDReplicas:    5,
			expectedStatusReplicas: 5,
			expectError:            false,
		},
		{
			name: "no update when replicas already match",
			adapter: &v1alpha1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd-frontend",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 3,
					DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
						Name:        "test-dgd",
						ServiceName: "Frontend",
					},
				},
			},
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(3)),
						},
					},
				},
			},
			expectedDGDReplicas:    3,
			expectedStatusReplicas: 3,
			expectError:            false,
		},
		{
			name: "uses default replicas (1) when DGD service has no replicas set",
			adapter: &v1alpha1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd-worker",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 4,
					DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
						Name:        "test-dgd",
						ServiceName: "worker",
					},
				},
			},
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {}, // no replicas set
					},
				},
			},
			expectedDGDReplicas:    4,
			expectedStatusReplicas: 4,
			expectError:            false,
		},
		{
			name: "error when service not found in DGD",
			adapter: &v1alpha1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd-missing",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 2,
					DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
						Name:        "test-dgd",
						ServiceName: "nonexistent",
					},
				},
			},
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(1)),
						},
					},
				},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build initial objects
			var initObjs []client.Object
			initObjs = append(initObjs, tt.adapter, tt.dgd)

			// Create fake client with status subresource support
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme.Scheme).
				WithObjects(initObjs...).
				WithStatusSubresource(&v1alpha1.DynamoGraphDeploymentScalingAdapter{}).
				Build()

			// Create reconciler
			r := &DynamoGraphDeploymentScalingAdapterReconciler{
				Client:   fakeClient,
				Scheme:   scheme.Scheme,
				Recorder: record.NewFakeRecorder(10),
			}

			// Run Reconcile
			ctx := context.Background()
			req := ctrl.Request{
				NamespacedName: types.NamespacedName{
					Name:      tt.adapter.Name,
					Namespace: tt.adapter.Namespace,
				},
			}

			result, err := r.Reconcile(ctx, req)

			// Check error expectation
			if tt.expectError && err == nil {
				t.Errorf("Expected error, but got none")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			// Skip further checks if error was expected
			if tt.expectError {
				return
			}

			// Check requeue
			if tt.expectRequeue && result.RequeueAfter == 0 {
				t.Errorf("Expected requeue, but got none")
			}

			// Verify DGD replicas were updated
			updatedDGD := &v1alpha1.DynamoGraphDeployment{}
			if err := fakeClient.Get(ctx, types.NamespacedName{Name: tt.dgd.Name, Namespace: tt.dgd.Namespace}, updatedDGD); err != nil {
				t.Fatalf("Failed to get updated DGD: %v", err)
			}

			service, exists := updatedDGD.Spec.Services[tt.adapter.Spec.DGDRef.ServiceName]
			if !exists {
				t.Fatalf("Service %s not found in updated DGD", tt.adapter.Spec.DGDRef.ServiceName)
			}

			actualReplicas := int32(1)
			if service.Replicas != nil {
				actualReplicas = *service.Replicas
			}

			if actualReplicas != tt.expectedDGDReplicas {
				t.Errorf("DGD service replicas = %d, expected %d", actualReplicas, tt.expectedDGDReplicas)
			}

			// Verify adapter status was updated
			updatedAdapter := &v1alpha1.DynamoGraphDeploymentScalingAdapter{}
			if err := fakeClient.Get(ctx, types.NamespacedName{Name: tt.adapter.Name, Namespace: tt.adapter.Namespace}, updatedAdapter); err != nil {
				t.Fatalf("Failed to get updated adapter: %v", err)
			}

			if updatedAdapter.Status.Replicas != tt.expectedStatusReplicas {
				t.Errorf("Adapter status.replicas = %d, expected %d", updatedAdapter.Status.Replicas, tt.expectedStatusReplicas)
			}

			// Verify selector is set
			if updatedAdapter.Status.Selector == "" {
				t.Errorf("Adapter status.selector is empty, expected non-empty")
			}
		})
	}
}

func TestDynamoGraphDeploymentScalingAdapterReconciler_Reconcile_NotFound(t *testing.T) {
	// Register custom types with the scheme
	if err := v1alpha1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1alpha1 to scheme: %v", err)
	}

	// Create fake client with no objects
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		Build()

	r := &DynamoGraphDeploymentScalingAdapterReconciler{
		Client:   fakeClient,
		Scheme:   scheme.Scheme,
		Recorder: record.NewFakeRecorder(10),
	}

	ctx := context.Background()
	req := ctrl.Request{
		NamespacedName: types.NamespacedName{
			Name:      "nonexistent",
			Namespace: "default",
		},
	}

	// Should return no error when adapter not found (client.IgnoreNotFound)
	result, err := r.Reconcile(ctx, req)
	if err != nil {
		t.Errorf("Expected no error for not found adapter, got: %v", err)
	}
	if result.RequeueAfter != 0 {
		t.Errorf("Expected no requeueAfter for not found adapter, got: %v", result.RequeueAfter)
	}
}

func TestDynamoGraphDeploymentScalingAdapterReconciler_Reconcile_DGDNotFound(t *testing.T) {
	// Register custom types with the scheme
	if err := v1alpha1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1alpha1 to scheme: %v", err)
	}

	adapter := &v1alpha1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-frontend",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: 5,
			DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
				Name:        "nonexistent-dgd",
				ServiceName: "Frontend",
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithObjects(adapter).
		Build()

	r := &DynamoGraphDeploymentScalingAdapterReconciler{
		Client:   fakeClient,
		Scheme:   scheme.Scheme,
		Recorder: record.NewFakeRecorder(10),
	}

	ctx := context.Background()
	req := ctrl.Request{
		NamespacedName: types.NamespacedName{
			Name:      adapter.Name,
			Namespace: adapter.Namespace,
		},
	}

	// Should return error when DGD not found
	_, err := r.Reconcile(ctx, req)
	if err == nil {
		t.Errorf("Expected error when DGD not found, got none")
	}
}

func TestDynamoGraphDeploymentScalingAdapterReconciler_Reconcile_BeingDeleted(t *testing.T) {
	// Register custom types with the scheme
	if err := v1alpha1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1alpha1 to scheme: %v", err)
	}

	now := metav1.Now()
	adapter := &v1alpha1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test-dgd-frontend",
			Namespace:         "default",
			DeletionTimestamp: &now,
			Finalizers:        []string{"test-finalizer"}, // Required for deletion timestamp to be set
		},
		Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: 5,
			DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
				Name:        "test-dgd",
				ServiceName: "Frontend",
			},
		},
	}

	dgd := &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"Frontend": {
					Replicas: ptr.To(int32(2)),
				},
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithObjects(adapter, dgd).
		Build()

	r := &DynamoGraphDeploymentScalingAdapterReconciler{
		Client:   fakeClient,
		Scheme:   scheme.Scheme,
		Recorder: record.NewFakeRecorder(10),
	}

	ctx := context.Background()
	req := ctrl.Request{
		NamespacedName: types.NamespacedName{
			Name:      adapter.Name,
			Namespace: adapter.Namespace,
		},
	}

	// Should return no error and skip reconciliation
	result, err := r.Reconcile(ctx, req)
	if err != nil {
		t.Errorf("Expected no error for deleting adapter, got: %v", err)
	}
	if result.RequeueAfter != 0 {
		t.Errorf("Expected no requeueAfter for deleting adapter, got: %v", result.RequeueAfter)
	}

	// DGD replicas should NOT be updated (still 2)
	updatedDGD := &v1alpha1.DynamoGraphDeployment{}
	if err := fakeClient.Get(ctx, types.NamespacedName{Name: dgd.Name, Namespace: dgd.Namespace}, updatedDGD); err != nil {
		t.Fatalf("Failed to get DGD: %v", err)
	}

	if *updatedDGD.Spec.Services["Frontend"].Replicas != 2 {
		t.Errorf("DGD replicas should remain unchanged, got %d", *updatedDGD.Spec.Services["Frontend"].Replicas)
	}
}

func TestDynamoGraphDeploymentScalingAdapterReconciler_findAdaptersForDGD(t *testing.T) {
	// Register custom types with the scheme
	if err := v1alpha1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1alpha1 to scheme: %v", err)
	}

	dgd := &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
	}

	// Adapters belonging to test-dgd
	adapter1 := &v1alpha1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-frontend",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
		},
		Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
			DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
				Name:        "test-dgd",
				ServiceName: "Frontend",
			},
		},
	}

	adapter2 := &v1alpha1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-decode",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
		},
		Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
			DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
				Name:        "test-dgd",
				ServiceName: "decode",
			},
		},
	}

	// Adapter belonging to different DGD
	adapterOther := &v1alpha1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "other-dgd-frontend",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "other-dgd",
			},
		},
		Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
			DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
				Name:        "other-dgd",
				ServiceName: "Frontend",
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithObjects(adapter1, adapter2, adapterOther).
		Build()

	r := &DynamoGraphDeploymentScalingAdapterReconciler{
		Client: fakeClient,
	}

	ctx := context.Background()
	requests := r.findAdaptersForDGD(ctx, dgd)

	// Should return 2 requests (for test-dgd adapters only)
	if len(requests) != 2 {
		t.Errorf("findAdaptersForDGD() returned %d requests, expected 2", len(requests))
	}

	// Verify correct adapters are returned
	expectedNames := map[string]bool{
		"test-dgd-frontend": true,
		"test-dgd-decode":   true,
	}

	for _, req := range requests {
		if !expectedNames[req.Name] {
			t.Errorf("Unexpected adapter in results: %s", req.Name)
		}
	}
}
