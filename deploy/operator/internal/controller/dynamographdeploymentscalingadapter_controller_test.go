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

package controller

import (
	"context"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDynamoGraphDeploymentScalingAdapterReconciler_Reconcile(t *testing.T) {
	// Register custom types with the scheme
	if err := v1beta1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1beta1 to scheme: %v", err)
	}

	tests := []struct {
		name                   string
		adapter                *v1beta1.DynamoGraphDeploymentScalingAdapter
		dgd                    *v1beta1.DynamoGraphDeployment
		expectedDGDReplicas    int32
		expectedStatusReplicas int32
		expectError            bool
		expectMissingComponent bool
		expectSkipped          bool
		expectRequeue          bool
	}{
		{
			name: "updates DGD replicas when DGDSA spec differs",
			adapter: &v1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd-frontend",
					Namespace: "default",
				},
				Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 5,
					DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
						Name:          "test-dgd",
						ComponentName: "Frontend",
					},
				},
			},
			dgd: &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName:  "Frontend",
							Replicas:       ptr.To(int32(2)),
							ScalingAdapter: &v1beta1.ScalingAdapter{},
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
			adapter: &v1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd-frontend",
					Namespace: "default",
				},
				Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 3,
					DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
						Name:          "test-dgd",
						ComponentName: "Frontend",
					},
				},
			},
			dgd: &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName:  "Frontend",
							Replicas:       ptr.To(int32(3)),
							ScalingAdapter: &v1beta1.ScalingAdapter{},
						},
					},
				},
			},
			expectedDGDReplicas:    3,
			expectedStatusReplicas: 3,
			expectError:            false,
		},
		{
			name: "uses default replicas (1) when DGD component has no replicas set",
			adapter: &v1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd-worker",
					Namespace: "default",
				},
				Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 4,
					DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
						Name:          "test-dgd",
						ComponentName: "worker",
					},
				},
			},
			dgd: &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName:  "worker",
							ScalingAdapter: &v1beta1.ScalingAdapter{},
						}, // no replicas set
					},
				},
			},
			expectedDGDReplicas:    4,
			expectedStatusReplicas: 4,
			expectError:            false,
		},
		{
			name: "returns without retry when component not found in DGD",
			adapter: &v1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd-missing",
					Namespace: "default",
				},
				Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 2,
					DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
						Name:          "test-dgd",
						ComponentName: "nonexistent",
					},
				},
			},
			dgd: &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName: "Frontend",
							Replicas:      ptr.To(int32(1)),
						},
					},
				},
			},
			expectError:            false,
			expectMissingComponent: true,
		},
		{
			name: "does not propagate replicas after component opts out",
			adapter: &v1beta1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd-frontend",
					Namespace: "default",
				},
				Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: 5,
					DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
						Name:          "test-dgd",
						ComponentName: "Frontend",
					},
				},
			},
			dgd: &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName: "Frontend",
							Replicas:      ptr.To(int32(2)),
						},
					},
				},
			},
			expectedDGDReplicas: 2,
			expectSkipped:       true,
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
				WithStatusSubresource(&v1beta1.DynamoGraphDeploymentScalingAdapter{}).
				Build()

			// Create reconciler
			r := &DynamoGraphDeploymentScalingAdapterReconciler{
				Client:   fakeClient,
				Scheme:   scheme.Scheme,
				Recorder: events.NewFakeRecorder(10),
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

			if tt.expectMissingComponent {
				updatedAdapter := &v1beta1.DynamoGraphDeploymentScalingAdapter{}
				if err := fakeClient.Get(ctx, types.NamespacedName{Name: tt.adapter.Name, Namespace: tt.adapter.Namespace}, updatedAdapter); err != nil {
					t.Fatalf("Failed to get adapter: %v", err)
				}
				if updatedAdapter.Status.Selector != "" {
					t.Errorf("Adapter status.selector = %q, expected empty selector when component is missing", updatedAdapter.Status.Selector)
				}
				return
			}

			// Check requeue
			if tt.expectRequeue && result.RequeueAfter == 0 {
				t.Errorf("Expected requeue, but got none")
			}

			// Verify DGD replicas were updated
			updatedDGD := &v1beta1.DynamoGraphDeployment{}
			if err := fakeClient.Get(ctx, types.NamespacedName{Name: tt.dgd.Name, Namespace: tt.dgd.Namespace}, updatedDGD); err != nil {
				t.Fatalf("Failed to get updated DGD: %v", err)
			}

			component := updatedDGD.GetComponentByName(tt.adapter.Spec.DGDRef.ComponentName)
			if component == nil {
				t.Fatalf("Component %s not found in updated DGD", tt.adapter.Spec.DGDRef.ComponentName)
			}

			actualReplicas := int32(1)
			if component.Replicas != nil {
				actualReplicas = *component.Replicas
			}

			if actualReplicas != tt.expectedDGDReplicas {
				t.Errorf("DGD component replicas = %d, expected %d", actualReplicas, tt.expectedDGDReplicas)
			}

			// Verify adapter status was updated
			updatedAdapter := &v1beta1.DynamoGraphDeploymentScalingAdapter{}
			if err := fakeClient.Get(ctx, types.NamespacedName{Name: tt.adapter.Name, Namespace: tt.adapter.Namespace}, updatedAdapter); err != nil {
				t.Fatalf("Failed to get updated adapter: %v", err)
			}

			if tt.expectSkipped {
				t.Log("Verify adapter status remains unchanged after component opt-out")
				if updatedAdapter.Status.Selector != "" || updatedAdapter.Status.Replicas != 0 {
					t.Errorf("Adapter status was updated after opt-out: %+v", updatedAdapter.Status)
				}
				return
			}

			if updatedAdapter.Status.Replicas != tt.expectedStatusReplicas {
				t.Errorf("Adapter status.replicas = %d, expected %d", updatedAdapter.Status.Replicas, tt.expectedStatusReplicas)
			}

			expectedSelector := labels.SelectorFromSet(labels.Set{
				consts.KubeLabelDynamoGraphDeploymentName: tt.dgd.Name,
				consts.KubeLabelDynamoComponent:           tt.adapter.Spec.DGDRef.ComponentName,
			}).String()
			if updatedAdapter.Status.Selector != expectedSelector {
				t.Errorf("Adapter status.selector = %q, expected %q", updatedAdapter.Status.Selector, expectedSelector)
			}
		})
	}
}

func TestDGDSARequestOnly(t *testing.T) {
	t.Log("Register the storage API and create distinct requested, committed, and actual targets")
	if err := v1beta1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("add v1beta1 to scheme: %v", err)
	}
	requested := int32(5)
	committed := int32(3)
	actual := int32(2)
	adapter := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd-worker", Namespace: "default"},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: requested,
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name:          "test-dgd",
				ComponentName: "worker",
			},
		},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			Annotations: map[string]string{
				v1beta1.DynamoGraphPowerControlModeAnnotation: v1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName:  "worker",
				ComponentType:  v1beta1.ComponentTypeWorker,
				Replicas:       ptr.To(committed),
				ScalingAdapter: nil,
			}},
		},
		Status: v1beta1.DynamoGraphDeploymentStatus{
			Components: map[string]v1beta1.ComponentReplicaStatus{
				"worker": {Replicas: actual},
			},
		},
	}
	dgpb := &v1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{Name: dgd.Name, Namespace: dgd.Namespace},
		Status: v1beta1.DynamoGraphPowerBudgetStatus{
			CommittedReplicaTargets: map[string]int32{"worker": committed},
		},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithObjects(adapter, dgd, dgpb).
		WithStatusSubresource(&v1beta1.DynamoGraphDeploymentScalingAdapter{}).
		Build()
	reconciler := &DynamoGraphDeploymentScalingAdapterReconciler{
		Client:   fakeClient,
		Scheme:   scheme.Scheme,
		Recorder: events.NewFakeRecorder(10),
	}

	t.Log("Defer single-adapter status until the complete-vector reconciler publishes a bounded reason")
	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{
		Name: adapter.Name, Namespace: adapter.Namespace,
	}})
	if err != nil {
		t.Fatalf("reconcile transactional adapter: %v", err)
	}
	deferredAdapter := &v1beta1.DynamoGraphDeploymentScalingAdapter{}
	if err := fakeClient.Get(context.Background(), client.ObjectKeyFromObject(adapter), deferredAdapter); err != nil {
		t.Fatalf("read deferred adapter: %v", err)
	}
	if deferredAdapter.Status != (v1beta1.DynamoGraphDeploymentScalingAdapterStatus{}) {
		t.Fatalf("single-adapter reconcile published an empty pending row: %+v", deferredAdapter.Status)
	}
	deferredAdapter.Status.RequestedReplicas = requested
	deferredAdapter.Status.CommittedReplicas = committed
	deferredAdapter.Status.ActualReplicas = actual
	deferredAdapter.Status.Replicas = actual
	deferredAdapter.Status.PendingReason = v1beta1.DynamoGraphPowerBudgetPendingReasonBudgetExceeded
	if err := fakeClient.Status().Update(context.Background(), deferredAdapter); err != nil {
		t.Fatalf("publish complete-vector pending reason: %v", err)
	}

	t.Log("Reconcile actual capacity without mirroring the pending request into the DGD")
	_, err = reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{
		Name: adapter.Name, Namespace: adapter.Namespace,
	}})
	if err != nil {
		t.Fatalf("reconcile reasoned transactional adapter: %v", err)
	}

	t.Log("Verify requested, committed, and actual targets remain distinct and status reports actual")
	updatedAdapter := &v1beta1.DynamoGraphDeploymentScalingAdapter{}
	if err := fakeClient.Get(context.Background(), client.ObjectKeyFromObject(adapter), updatedAdapter); err != nil {
		t.Fatalf("read adapter: %v", err)
	}
	updatedDGD := &v1beta1.DynamoGraphDeployment{}
	if err := fakeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), updatedDGD); err != nil {
		t.Fatalf("read DGD: %v", err)
	}
	updatedDGPB := &v1beta1.DynamoGraphPowerBudget{}
	if err := fakeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), updatedDGPB); err != nil {
		t.Fatalf("read DGPB: %v", err)
	}
	if updatedAdapter.Spec.Replicas != requested {
		t.Fatalf("requested replicas = %d, want %d", updatedAdapter.Spec.Replicas, requested)
	}
	if got := updatedDGPB.Status.CommittedReplicaTargets["worker"]; got != committed {
		t.Fatalf("committed replicas = %d, want %d", got, committed)
	}
	if got := ptr.Deref(updatedDGD.GetComponentByName("worker").Replicas, int32(1)); got != committed {
		t.Fatalf("committed DGD target = %d, want %d", got, committed)
	}
	if updatedAdapter.Status.Replicas != actual {
		t.Fatalf("adapter status replicas = %d, want actual %d", updatedAdapter.Status.Replicas, actual)
	}
	if updatedAdapter.Status.RequestedReplicas != requested ||
		updatedAdapter.Status.CommittedReplicas != committed ||
		updatedAdapter.Status.ActualReplicas != actual {
		t.Fatalf("adapter transactional status = %+v, want requested=%d committed=%d actual=%d",
			updatedAdapter.Status, requested, committed, actual)
	}
	if updatedAdapter.Status.PendingReason != v1beta1.DynamoGraphPowerBudgetPendingReasonBudgetExceeded {
		t.Fatalf("adapter pending reason = %q, want BudgetExceeded", updatedAdapter.Status.PendingReason)
	}
	if updatedAdapter.Status.LastScaleTime != nil {
		t.Fatalf("request-only reconcile unexpectedly set lastScaleTime: %v", updatedAdapter.Status.LastScaleTime)
	}

	t.Log("Keep the prior decision tuple when a newer request arrives")
	updatedAdapter.Spec.Replicas = 1
	if err := fakeClient.Update(context.Background(), updatedAdapter); err != nil {
		t.Fatalf("publish replacement request: %v", err)
	}
	_, err = reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{
		Name: adapter.Name, Namespace: adapter.Namespace,
	}})
	if err != nil {
		t.Fatalf("reconcile replacement request: %v", err)
	}
	if err := fakeClient.Get(context.Background(), client.ObjectKeyFromObject(adapter), updatedAdapter); err != nil {
		t.Fatalf("read adapter after replacement request: %v", err)
	}
	if updatedAdapter.Status.RequestedReplicas != requested ||
		updatedAdapter.Status.CommittedReplicas != committed ||
		updatedAdapter.Status.PendingReason != v1beta1.DynamoGraphPowerBudgetPendingReasonBudgetExceeded {
		t.Fatalf("replacement request rebound stale decision: %+v", updatedAdapter.Status)
	}
}

func TestDGDSAPreservesZeroSeedBelowMinimum(t *testing.T) {
	t.Log("Create the vector-authoritative all-zero bootstrap rejection")
	if err := v1beta1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("add v1beta1 to scheme: %v", err)
	}
	adapter := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd-worker", Namespace: "default"},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: 0,
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name: "test-dgd", ComponentName: "worker",
			},
		},
		Status: v1beta1.DynamoGraphDeploymentScalingAdapterStatus{
			RequestedReplicas: 0,
			CommittedReplicas: 0,
			PendingReason:     v1beta1.DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline,
		},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			Annotations: map[string]string{
				v1beta1.DynamoGraphPowerControlModeAnnotation: v1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker, Replicas: ptr.To(int32(0)),
			}},
		},
		Status: v1beta1.DynamoGraphDeploymentStatus{
			Components: map[string]v1beta1.ComponentReplicaStatus{"worker": {Replicas: 0}},
		},
	}
	dgpb := &v1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{Name: dgd.Name, Namespace: dgd.Namespace},
		Status: v1beta1.DynamoGraphPowerBudgetStatus{
			Phase:                   v1beta1.DynamoGraphPowerBudgetPhaseInitializing,
			CommittedReplicaTargets: map[string]int32{"worker": 0},
		},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithObjects(adapter, dgd, dgpb).
		WithStatusSubresource(
			&v1beta1.DynamoGraphDeploymentScalingAdapter{},
			&v1beta1.DynamoGraphPowerBudget{},
		).
		Build()
	reconciler := &DynamoGraphDeploymentScalingAdapterReconciler{
		Client: fakeClient, Scheme: scheme.Scheme, Recorder: events.NewFakeRecorder(10),
	}

	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(adapter)})
	if err != nil {
		t.Fatalf("reconcile zero bootstrap status: %v", err)
	}
	storedAdapter := &v1beta1.DynamoGraphDeploymentScalingAdapter{}
	if err := fakeClient.Get(context.Background(), client.ObjectKeyFromObject(adapter), storedAdapter); err != nil {
		t.Fatalf("read zero bootstrap adapter: %v", err)
	}
	if storedAdapter.Status.PendingReason != v1beta1.DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline {
		t.Fatalf("zero durability-boundary reason = %q, want UnenforcedBaseline", storedAdapter.Status.PendingReason)
	}

	t.Log("Preserve the exact BelowMinimum reason after full-vector evaluation")
	storedAdapter.Status.PendingReason = v1beta1.DynamoGraphPowerBudgetPendingReasonBelowMinimum
	if err := fakeClient.Status().Update(context.Background(), storedAdapter); err != nil {
		t.Fatalf("publish zero bootstrap admission reason: %v", err)
	}
	_, err = reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(adapter)})
	if err != nil {
		t.Fatalf("reconcile evaluated zero bootstrap status: %v", err)
	}
	if err := fakeClient.Get(context.Background(), client.ObjectKeyFromObject(adapter), storedAdapter); err != nil {
		t.Fatalf("read evaluated zero bootstrap adapter: %v", err)
	}
	if storedAdapter.Status.PendingReason != v1beta1.DynamoGraphPowerBudgetPendingReasonBelowMinimum {
		t.Fatalf("evaluated zero bootstrap reason = %q, want BelowMinimum", storedAdapter.Status.PendingReason)
	}

	t.Log("Clear the old reason after a normal request becomes the accepted commitment")
	storedAdapter.Spec.Replicas = 2
	if err := fakeClient.Update(context.Background(), storedAdapter); err != nil {
		t.Fatalf("publish accepted request: %v", err)
	}
	storedDGPB := &v1beta1.DynamoGraphPowerBudget{}
	if err := fakeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), storedDGPB); err != nil {
		t.Fatalf("read power budget before acceptance: %v", err)
	}
	storedDGPB.Status.Phase = v1beta1.DynamoGraphPowerBudgetPhaseApplying
	storedDGPB.Status.CommittedReplicaTargets["worker"] = 2
	if err := fakeClient.Status().Update(context.Background(), storedDGPB); err != nil {
		t.Fatalf("publish accepted commitment: %v", err)
	}
	_, err = reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(adapter)})
	if err != nil {
		t.Fatalf("reconcile accepted request status: %v", err)
	}
	if err := fakeClient.Get(context.Background(), client.ObjectKeyFromObject(adapter), storedAdapter); err != nil {
		t.Fatalf("read accepted adapter: %v", err)
	}
	if storedAdapter.Status.PendingReason != "" || storedAdapter.Status.RequestedReplicas != 2 ||
		storedAdapter.Status.CommittedReplicas != 2 {
		t.Fatalf("accepted request retained stale reason: %+v", storedAdapter.Status)
	}
}

func TestStaticDGDSAUnchanged(t *testing.T) {
	t.Log("Create a static DGD whose adapter requests a larger target")
	if err := v1beta1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("add v1beta1 to scheme: %v", err)
	}
	adapter := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd-worker", Namespace: "default"},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: 5,
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name:          "test-dgd",
				ComponentName: "worker",
			},
		},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName:  "worker",
				ComponentType:  v1beta1.ComponentTypeWorker,
				Replicas:       ptr.To(int32(2)),
				ScalingAdapter: &v1beta1.ScalingAdapter{},
			}},
		},
	}
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithObjects(adapter, dgd).
		WithStatusSubresource(&v1beta1.DynamoGraphDeploymentScalingAdapter{}).
		Build()
	reconciler := &DynamoGraphDeploymentScalingAdapterReconciler{
		Client:   fakeClient,
		Scheme:   scheme.Scheme,
		Recorder: events.NewFakeRecorder(10),
	}

	t.Log("Reconcile through the Phase 1 direct adapter propagation path")
	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(adapter)})
	if err != nil {
		t.Fatalf("reconcile static adapter: %v", err)
	}

	t.Log("Verify the static DGD target and Scale status both follow the request")
	storedDGD := &v1beta1.DynamoGraphDeployment{}
	if err := fakeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD); err != nil {
		t.Fatalf("read static DGD: %v", err)
	}
	if got := ptr.Deref(storedDGD.GetComponentByName("worker").Replicas, int32(1)); got != 5 {
		t.Fatalf("static DGD replicas = %d, want 5", got)
	}
	storedAdapter := &v1beta1.DynamoGraphDeploymentScalingAdapter{}
	if err := fakeClient.Get(context.Background(), client.ObjectKeyFromObject(adapter), storedAdapter); err != nil {
		t.Fatalf("read static adapter: %v", err)
	}
	if storedAdapter.Status.Replicas != 5 || storedAdapter.Status.LastScaleTime == nil {
		t.Fatalf("static adapter status = %+v, want replicas=5 and lastScaleTime", storedAdapter.Status)
	}
	if storedAdapter.Status.RequestedReplicas != 5 || storedAdapter.Status.CommittedReplicas != 5 ||
		storedAdapter.Status.ActualReplicas != 5 || storedAdapter.Status.PendingReason != "" {
		t.Fatalf("static adapter compatibility status = %+v, want requested=committed=actual=5", storedAdapter.Status)
	}
}

func TestDynamoGraphDeploymentScalingAdapterReconciler_Reconcile_NotFound(t *testing.T) {
	// Register custom types with the scheme
	if err := v1beta1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1beta1 to scheme: %v", err)
	}

	// Create fake client with no objects
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		Build()

	r := &DynamoGraphDeploymentScalingAdapterReconciler{
		Client:   fakeClient,
		Scheme:   scheme.Scheme,
		Recorder: events.NewFakeRecorder(10),
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
	if err := v1beta1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1beta1 to scheme: %v", err)
	}

	adapter := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-frontend",
			Namespace: "default",
		},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: 5,
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name:          "nonexistent-dgd",
				ComponentName: "Frontend",
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
		Recorder: events.NewFakeRecorder(10),
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
	if err := v1beta1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1beta1 to scheme: %v", err)
	}

	now := metav1.Now()
	adapter := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test-dgd-frontend",
			Namespace:         "default",
			DeletionTimestamp: &now,
			Finalizers:        []string{"test-finalizer"}, // Required for deletion timestamp to be set
		},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: 5,
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name:          "test-dgd",
				ComponentName: "Frontend",
			},
		},
	}

	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "Frontend",
					Replicas:      ptr.To(int32(2)),
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
		Recorder: events.NewFakeRecorder(10),
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
	updatedDGD := &v1beta1.DynamoGraphDeployment{}
	if err := fakeClient.Get(ctx, types.NamespacedName{Name: dgd.Name, Namespace: dgd.Namespace}, updatedDGD); err != nil {
		t.Fatalf("Failed to get DGD: %v", err)
	}

	frontend := updatedDGD.GetComponentByName("Frontend")
	if frontend == nil {
		t.Fatalf("Frontend component not found in updated DGD")
	}
	if *frontend.Replicas != 2 {
		t.Errorf("DGD replicas should remain unchanged, got %d", *frontend.Replicas)
	}
}

func TestDynamoGraphDeploymentScalingAdapterReconciler_findAdaptersForDGD(t *testing.T) {
	// Register custom types with the scheme
	if err := v1beta1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1beta1 to scheme: %v", err)
	}

	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
	}

	// Adapters belonging to test-dgd
	adapter1 := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-frontend",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name:          "test-dgd",
				ComponentName: "Frontend",
			},
		},
	}

	adapter2 := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-decode",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name:          "test-dgd",
				ComponentName: "decode",
			},
		},
	}

	// Adapter belonging to test-dgd without the optional lookup label
	adapterUnlabeled := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker",
			Namespace: "default",
		},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name:          "test-dgd",
				ComponentName: "worker",
			},
		},
	}

	// Adapter belonging to different DGD
	adapterOther := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "other-dgd-frontend",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "other-dgd",
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name:          "other-dgd",
				ComponentName: "Frontend",
			},
		},
	}

	// Adapter with a stale/mismatched label should follow spec.dgdRef.name.
	adapterMislabeled := &v1beta1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mislabeled-dgd-frontend",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentScalingAdapterSpec{
			DGDRef: v1beta1.DynamoGraphDeploymentComponentRef{
				Name:          "other-dgd",
				ComponentName: "Frontend",
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithObjects(adapter1, adapter2, adapterUnlabeled, adapterOther, adapterMislabeled).
		Build()

	r := &DynamoGraphDeploymentScalingAdapterReconciler{
		Client: fakeClient,
	}

	ctx := context.Background()
	requests := r.findAdaptersForDGD(ctx, dgd)

	// Should return 3 requests (for test-dgd adapters only)
	if len(requests) != 3 {
		t.Errorf("findAdaptersForDGD() returned %d requests, expected 3", len(requests))
	}

	// Verify correct adapters are returned
	expectedNames := map[string]bool{
		"test-dgd-frontend": true,
		"test-dgd-decode":   true,
		"test-dgd-worker":   true,
	}

	for _, req := range requests {
		if !expectedNames[req.Name] {
			t.Errorf("Unexpected adapter in results: %s", req.Name)
		}
	}
}
