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

	grovev1alpha1 "github.com/NVIDIA/grove/operator/api/core/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/onsi/gomega"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/scale"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDynamoGraphDeploymentReconciler_reconcileScalingAdapters(t *testing.T) {
	// Register custom types with the scheme
	if err := v1alpha1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1alpha1 to scheme: %v", err)
	}

	tests := []struct {
		name                 string
		dgd                  *v1alpha1.DynamoGraphDeployment
		existingAdapters     []v1alpha1.DynamoGraphDeploymentScalingAdapter
		expectedAdapterCount int
		expectedAdapters     map[string]int32 // map of adapter name to expected replicas
		expectDeleted        []string         // adapter names that should be deleted
	}{
		{
			name: "creates adapters for all services",
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
						"decode": {
							Replicas: ptr.To(int32(3)),
						},
					},
				},
			},
			expectedAdapterCount: 2,
			expectedAdapters: map[string]int32{
				"test-dgd-frontend": 2,
				"test-dgd-decode":   3,
			},
		},
		{
			name: "uses default replicas when not specified",
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {},
					},
				},
			},
			expectedAdapterCount: 1,
			expectedAdapters: map[string]int32{
				"test-dgd-worker": 1, // default replicas
			},
		},
		{
			name: "skips adapter creation when disabled",
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
						"decode": {
							Replicas: ptr.To(int32(3)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Disable: true,
							},
						},
					},
				},
			},
			expectedAdapterCount: 1,
			expectedAdapters: map[string]int32{
				"test-dgd-frontend": 2,
			},
		},
		{
			name: "deletes adapter when service is removed",
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					UID:       "test-uid",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(2)),
						},
					},
				},
			},
			existingAdapters: []v1alpha1.DynamoGraphDeploymentScalingAdapter{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
						Labels: map[string]string{
							consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
						},
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "nvidia.com/v1alpha1",
								Kind:       "DynamoGraphDeployment",
								Name:       "test-dgd",
								UID:        "test-uid",
							},
						},
					},
					Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
						Replicas: 2,
						DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
							Name:        "test-dgd",
							ServiceName: "Frontend",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-removed",
						Namespace: "default",
						Labels: map[string]string{
							consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
						},
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "nvidia.com/v1alpha1",
								Kind:       "DynamoGraphDeployment",
								Name:       "test-dgd",
								UID:        "test-uid",
							},
						},
					},
					Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
						Replicas: 1,
						DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
							Name:        "test-dgd",
							ServiceName: "removed",
						},
					},
				},
			},
			expectedAdapterCount: 1,
			expectedAdapters: map[string]int32{
				"test-dgd-frontend": 2,
			},
			expectDeleted: []string{"test-dgd-removed"},
		},
		{
			name: "deletes adapter when scalingAdapter.disable is set to true",
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					UID:       "test-uid",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(2)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Disable: true,
							},
						},
					},
				},
			},
			existingAdapters: []v1alpha1.DynamoGraphDeploymentScalingAdapter{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
						Labels: map[string]string{
							consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
						},
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "nvidia.com/v1alpha1",
								Kind:       "DynamoGraphDeployment",
								Name:       "test-dgd",
								UID:        "test-uid",
							},
						},
					},
					Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
						Replicas: 2,
						DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
							Name:        "test-dgd",
							ServiceName: "Frontend",
						},
					},
				},
			},
			expectedAdapterCount: 0,
			expectedAdapters:     map[string]int32{},
			expectDeleted:        []string{"test-dgd-frontend"},
		},
		{
			name: "adapter name uses lowercase service name",
			dgd: &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "my-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"MyService": {
							Replicas: ptr.To(int32(1)),
						},
					},
				},
			},
			expectedAdapterCount: 1,
			expectedAdapters: map[string]int32{
				"my-dgd-myservice": 1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build initial objects
			var initObjs []client.Object
			initObjs = append(initObjs, tt.dgd)
			for i := range tt.existingAdapters {
				initObjs = append(initObjs, &tt.existingAdapters[i])
			}

			// Create fake client
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme.Scheme).
				WithObjects(initObjs...).
				Build()

			// Create reconciler
			r := &DynamoGraphDeploymentReconciler{
				Client:   fakeClient,
				Recorder: record.NewFakeRecorder(10),
			}

			// Run reconcileScalingAdapters
			ctx := context.Background()
			err := r.reconcileScalingAdapters(ctx, tt.dgd)
			if err != nil {
				t.Fatalf("reconcileScalingAdapters() error = %v", err)
			}

			// Verify adapters
			adapterList := &v1alpha1.DynamoGraphDeploymentScalingAdapterList{}
			if err := fakeClient.List(ctx, adapterList, client.InNamespace("default")); err != nil {
				t.Fatalf("Failed to list adapters: %v", err)
			}

			if len(adapterList.Items) != tt.expectedAdapterCount {
				t.Errorf("Expected %d adapters, got %d", tt.expectedAdapterCount, len(adapterList.Items))
			}

			// Check expected adapters exist with correct replicas
			for name, expectedReplicas := range tt.expectedAdapters {
				adapter := &v1alpha1.DynamoGraphDeploymentScalingAdapter{}
				err := fakeClient.Get(ctx, types.NamespacedName{Name: name, Namespace: "default"}, adapter)
				if err != nil {
					t.Errorf("Expected adapter %s to exist, but got error: %v", name, err)
					continue
				}
				if adapter.Spec.Replicas != expectedReplicas {
					t.Errorf("Adapter %s has replicas=%d, expected %d", name, adapter.Spec.Replicas, expectedReplicas)
				}
			}

			// Check that deleted adapters don't exist
			for _, name := range tt.expectDeleted {
				adapter := &v1alpha1.DynamoGraphDeploymentScalingAdapter{}
				err := fakeClient.Get(ctx, types.NamespacedName{Name: name, Namespace: "default"}, adapter)
				if err == nil {
					t.Errorf("Expected adapter %s to be deleted, but it still exists", name)
				}
			}
		})
	}
}

// mockScaleClient implements scale.ScalesGetter for testing
type mockScaleClient struct{}

func (m *mockScaleClient) Scales(namespace string) scale.ScaleInterface {
	return &mockScaleInterface{}
}

// mockScaleInterface implements scale.ScaleInterface for testing
type mockScaleInterface struct{}

func (m *mockScaleInterface) Get(ctx context.Context, resource schema.GroupResource, name string, opts metav1.GetOptions) (*autoscalingv1.Scale, error) {
	// Return a dummy scale object - we don't actually need scaling in the test
	return &autoscalingv1.Scale{}, nil
}

func (m *mockScaleInterface) Update(ctx context.Context, resource schema.GroupResource, scale *autoscalingv1.Scale, opts metav1.UpdateOptions) (*autoscalingv1.Scale, error) {
	// Return success without actually doing anything
	return scale, nil
}

func (m *mockScaleInterface) Patch(ctx context.Context, gvr schema.GroupVersionResource, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions) (*autoscalingv1.Scale, error) {
	// Return a dummy scale object
	return &autoscalingv1.Scale{}, nil
}

func Test_reconcileGroveResources(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                   string
		dgdSpec                v1alpha1.DynamoGraphDeploymentSpec
		existingGroveResources []client.Object
		wantReconcileResult    ReconcileResult
	}{
		{
			name: "singular frontend service with 2 replicas - creates a PodClique with 2 replicas - ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ComponentType: string(commonconsts.ComponentTypeFrontend),
						Replicas:      ptr.To(int32(2)),
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-frontend",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:        2,
						UpdatedReplicas: 2,
						ReadyReplicas:   2,
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   ReadyState,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ServiceStatus: map[string]v1alpha1.ServiceReplicaStatus{
					"frontend": {
						ComponentKind:   v1alpha1.ComponentKindPodClique,
						ComponentName:   "test-dgd-0-frontend",
						Replicas:        2,
						UpdatedReplicas: 2,
						ReadyReplicas:   ptr.To(int32(2)),
					},
				},
			},
		},
		{
			name: "frontend service with 1 replica, decode service with 2 replicas - 2 PodCliques - one unready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ComponentType: string(commonconsts.ComponentTypeFrontend),
						Replicas:      ptr.To(int32(1)),
					},
					"decode": {
						ComponentType: string(commonconsts.ComponentTypeDecode),
						Replicas:      ptr.To(int32(2)),
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-frontend",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 1,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:        1,
						UpdatedReplicas: 1,
						ReadyReplicas:   1,
					},
				},
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-decode",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:        2,
						UpdatedReplicas: 1,
						ReadyReplicas:   1, // Only 1 ready, but 2 desired
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   PendingState,
				Reason:  "some_resources_are_not_ready",
				Message: Message("Resources not ready: test-dgd: podclique/test-dgd-0-decode: desired=2, ready=1"),
				ServiceStatus: map[string]v1alpha1.ServiceReplicaStatus{
					"frontend": {
						ComponentKind:   v1alpha1.ComponentKindPodClique,
						ComponentName:   "test-dgd-0-frontend",
						Replicas:        1,
						UpdatedReplicas: 1,
						ReadyReplicas:   ptr.To(int32(1)),
					},
					"decode": {
						ComponentKind:   v1alpha1.ComponentKindPodClique,
						ComponentName:   "test-dgd-0-decode",
						Replicas:        2,
						UpdatedReplicas: 1,
						ReadyReplicas:   ptr.To(int32(1)),
					},
				},
			},
		},
		{
			name: "decode worker multinode (PCSG), prefill worker multinode (PCSG) - both ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"decode": {
						ComponentType: string(commonconsts.ComponentTypeDecode),
						Replicas:      ptr.To(int32(1)),
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 2,
						},
					},
					"prefill": {
						ComponentType: string(commonconsts.ComponentTypeWorker),
						Replicas:      ptr.To(int32(1)),
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 4,
						},
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodCliqueScalingGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-decode",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
						Replicas: 1,
					},
					Status: grovev1alpha1.PodCliqueScalingGroupStatus{
						Replicas:          1,
						UpdatedReplicas:   1,
						AvailableReplicas: 1,
					},
				},
				&grovev1alpha1.PodCliqueScalingGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-prefill",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
						Replicas: 1,
					},
					Status: grovev1alpha1.PodCliqueScalingGroupStatus{
						Replicas:          1,
						UpdatedReplicas:   1,
						AvailableReplicas: 1,
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   ReadyState,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ServiceStatus: map[string]v1alpha1.ServiceReplicaStatus{
					"decode": {
						ComponentKind:     v1alpha1.ComponentKindPodCliqueScalingGroup,
						ComponentName:     "test-dgd-0-decode",
						Replicas:          1,
						UpdatedReplicas:   1,
						AvailableReplicas: ptr.To(int32(1)),
					},
					"prefill": {
						ComponentKind:     v1alpha1.ComponentKindPodCliqueScalingGroup,
						ComponentName:     "test-dgd-0-prefill",
						Replicas:          1,
						UpdatedReplicas:   1,
						AvailableReplicas: ptr.To(int32(1)),
					},
				},
			},
		},
		{
			name: "frontend worker (PodClique), aggregated worker multinode (PCSG) - PCSG unready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ComponentType: string(commonconsts.ComponentTypeFrontend),
						Replicas:      ptr.To(int32(1)),
					},
					"aggregated": {
						ComponentType: string(commonconsts.ComponentTypeWorker),
						Replicas:      ptr.To(int32(2)),
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 8,
						},
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-frontend",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 1,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:        1,
						UpdatedReplicas: 1,
						ReadyReplicas:   1,
					},
				},
				&grovev1alpha1.PodCliqueScalingGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-aggregated",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueScalingGroupStatus{
						Replicas:          2,
						UpdatedReplicas:   2,
						AvailableReplicas: 1, // Only 1 available, but 2 desired
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   PendingState,
				Reason:  "some_resources_are_not_ready",
				Message: Message("Resources not ready: test-dgd: pcsg/test-dgd-0-aggregated: desired=2, available=1"),
				ServiceStatus: map[string]v1alpha1.ServiceReplicaStatus{
					"frontend": {
						ComponentKind:   v1alpha1.ComponentKindPodClique,
						ComponentName:   "test-dgd-0-frontend",
						Replicas:        1,
						UpdatedReplicas: 1,
						ReadyReplicas:   ptr.To(int32(1)),
					},
					"aggregated": {
						ComponentKind:     v1alpha1.ComponentKindPodCliqueScalingGroup,
						ComponentName:     "test-dgd-0-aggregated",
						Replicas:          2,
						UpdatedReplicas:   2,
						AvailableReplicas: ptr.To(int32(1)),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			s := scheme.Scheme
			err := v1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			err = grovev1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			dgd := &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: tt.dgdSpec,
			}

			var objects []client.Object
			objects = append(objects, dgd)
			objects = append(objects, tt.existingGroveResources...)

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			recorder := record.NewFakeRecorder(100)
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:      fakeKubeClient,
				Recorder:    recorder,
				Config:      controller_common.Config{},
				ScaleClient: &mockScaleClient{},
				DockerSecretRetriever: &mockDockerSecretRetriever{
					GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
						return []string{}, nil
					},
				},
			}

			result, err := reconciler.reconcileGroveResources(ctx, dgd)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			g.Expect(result).To(gomega.Equal(tt.wantReconcileResult))
		})
	}
}

func Test_reconcileDynamoComponentsDeployments(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                string
		dgdSpec             v1alpha1.DynamoGraphDeploymentSpec
		existingDCDs        []client.Object
		wantReconcileResult ReconcileResult
	}{
		{
			name: "single service - DCD ready (Available condition = True)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(2)),
					},
				},
			},
			existingDCDs: []client.Object{
				&v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "frontend",
							Replicas:    ptr.To(int32(2)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentName:     "test-dgd-frontend-deployment",
							Replicas:          2,
							UpdatedReplicas:   2,
							ReadyReplicas:     ptr.To(int32(2)),
							AvailableReplicas: ptr.To(int32(2)),
						},
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   ReadyState,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ServiceStatus: map[string]v1alpha1.ServiceReplicaStatus{
					"frontend": {
						ComponentKind:     v1alpha1.ComponentKindDeployment,
						ComponentName:     "test-dgd-frontend-deployment",
						Replicas:          2,
						UpdatedReplicas:   2,
						ReadyReplicas:     ptr.To(int32(2)),
						AvailableReplicas: ptr.To(int32(2)),
					},
				},
			},
		},
		{
			name: "single service - DCD not ready (Available condition = False)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(2)),
					},
				},
			},
			existingDCDs: []client.Object{
				&v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "frontend",
							Replicas:    ptr.To(int32(2)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
						Service: v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentName:     "test-dgd-frontend-deployment",
							Replicas:          2,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   PendingState,
				Reason:  "some_resources_are_not_ready",
				Message: "Resources not ready: test-dgd-frontend: Component deployment not ready - Available condition not true",
				ServiceStatus: map[string]v1alpha1.ServiceReplicaStatus{
					"frontend": {
						ComponentKind:     v1alpha1.ComponentKindDeployment,
						ComponentName:     "test-dgd-frontend-deployment",
						Replicas:          2,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(0)),
					},
				},
			},
		},
		{
			name: "multiple services - all DCDs ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(1)),
					},
					"decode": {
						ServiceName:     "decode",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
						Replicas:        ptr.To(int32(2)),
					},
					"prefill": {
						ServiceName:     "prefill",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypePrefill),
						Replicas:        ptr.To(int32(3)),
					},
				},
			},
			existingDCDs: []client.Object{
				&v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "frontend",
							Replicas:    ptr.To(int32(1)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentName:     "test-dgd-frontend-deployment",
							Replicas:          1,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(1)),
						},
					},
				},
				&v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-decode",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "decode",
							Replicas:    ptr.To(int32(2)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentName:     "test-dgd-decode-deployment",
							Replicas:          2,
							UpdatedReplicas:   2,
							ReadyReplicas:     ptr.To(int32(2)),
							AvailableReplicas: ptr.To(int32(2)),
						},
					},
				},
				&v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-prefill",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "prefill",
							Replicas:    ptr.To(int32(3)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentName:     "test-dgd-prefill-deployment",
							Replicas:          3,
							UpdatedReplicas:   3,
							ReadyReplicas:     ptr.To(int32(3)),
							AvailableReplicas: ptr.To(int32(3)),
						},
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   ReadyState,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ServiceStatus: map[string]v1alpha1.ServiceReplicaStatus{
					"frontend": {
						ComponentKind:     v1alpha1.ComponentKindDeployment,
						ComponentName:     "test-dgd-frontend-deployment",
						Replicas:          1,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(1)),
					},
					"decode": {
						ComponentKind:     v1alpha1.ComponentKindDeployment,
						ComponentName:     "test-dgd-decode-deployment",
						Replicas:          2,
						UpdatedReplicas:   2,
						ReadyReplicas:     ptr.To(int32(2)),
						AvailableReplicas: ptr.To(int32(2)),
					},
					"prefill": {
						ComponentKind:     v1alpha1.ComponentKindDeployment,
						ComponentName:     "test-dgd-prefill-deployment",
						Replicas:          3,
						UpdatedReplicas:   3,
						ReadyReplicas:     ptr.To(int32(3)),
						AvailableReplicas: ptr.To(int32(3)),
					},
				},
			},
		},
		{
			name: "multiple services - some DCDs ready, some not ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(1)),
					},
					"decode": {
						ServiceName:     "decode",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
						Replicas:        ptr.To(int32(2)),
					},
					"prefill": {
						ServiceName:     "prefill",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypePrefill),
						Replicas:        ptr.To(int32(3)),
					},
				},
			},
			existingDCDs: []client.Object{
				&v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "frontend",
							Replicas:    ptr.To(int32(1)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentName:     "test-dgd-frontend-deployment",
							Replicas:          1,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(1)),
						},
					},
				},
				&v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-decode",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "decode",
							Replicas:    ptr.To(int32(2)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
						Service: v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentName:     "test-dgd-decode-deployment",
							Replicas:          2,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				},
				&v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-prefill",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "prefill",
							Replicas:    ptr.To(int32(3)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentName:     "test-dgd-prefill-deployment",
							Replicas:          3,
							UpdatedReplicas:   3,
							ReadyReplicas:     ptr.To(int32(3)),
							AvailableReplicas: ptr.To(int32(3)),
						},
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   PendingState,
				Reason:  "some_resources_are_not_ready",
				Message: "Resources not ready: test-dgd-decode: Component deployment not ready - Available condition not true",
				ServiceStatus: map[string]v1alpha1.ServiceReplicaStatus{
					"frontend": {
						ComponentKind:     v1alpha1.ComponentKindDeployment,
						ComponentName:     "test-dgd-frontend-deployment",
						Replicas:          1,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(1)),
					},
					"decode": {
						ComponentKind:     v1alpha1.ComponentKindDeployment,
						ComponentName:     "test-dgd-decode-deployment",
						Replicas:          2,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(0)),
					},
					"prefill": {
						ComponentKind:     v1alpha1.ComponentKindDeployment,
						ComponentName:     "test-dgd-prefill-deployment",
						Replicas:          3,
						UpdatedReplicas:   3,
						ReadyReplicas:     ptr.To(int32(3)),
						AvailableReplicas: ptr.To(int32(3)),
					},
				},
			},
		},
		{
			name: "multiple services - all DCDs not ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(1)),
					},
					"decode": {
						ServiceName:     "decode",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
						Replicas:        ptr.To(int32(2)),
					},
				},
			},
			existingDCDs: []client.Object{
				&v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "frontend",
							Replicas:    ptr.To(int32(1)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
						Service: v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentName:     "test-dgd-frontend-deployment",
							Replicas:          1,
							UpdatedReplicas:   0,
							ReadyReplicas:     ptr.To(int32(0)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				},
				&v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-decode",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "decode",
							Replicas:    ptr.To(int32(2)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
						Service: v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentName:     "test-dgd-decode-deployment",
							Replicas:          2,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   PendingState,
				Reason:  "some_resources_are_not_ready",
				Message: "Resources not ready: test-dgd-decode: Component deployment not ready - Available condition not true; test-dgd-frontend: Component deployment not ready - Available condition not true",
				ServiceStatus: map[string]v1alpha1.ServiceReplicaStatus{
					"frontend": {
						ComponentKind:     v1alpha1.ComponentKindDeployment,
						ComponentName:     "test-dgd-frontend-deployment",
						Replicas:          1,
						UpdatedReplicas:   0,
						ReadyReplicas:     ptr.To(int32(0)),
						AvailableReplicas: ptr.To(int32(0)),
					},
					"decode": {
						ComponentKind:     v1alpha1.ComponentKindDeployment,
						ComponentName:     "test-dgd-decode-deployment",
						Replicas:          2,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(0)),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			s := scheme.Scheme
			err := v1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			dgd := &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: tt.dgdSpec,
			}

			var objects []client.Object
			objects = append(objects, dgd)
			objects = append(objects, tt.existingDCDs...)

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			recorder := record.NewFakeRecorder(100)
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:   fakeKubeClient,
				Recorder: recorder,
				Config:   controller_common.Config{},
			}

			result, err := reconciler.reconcileDynamoComponentsDeployments(ctx, dgd)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			g.Expect(result).To(gomega.Equal(tt.wantReconcileResult))
		})
	}
}
