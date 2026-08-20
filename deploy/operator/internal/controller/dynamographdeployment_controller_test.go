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
	"fmt"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

func newDynamoGraphDeploymentControllerTestScheme(t testing.TB) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	for _, addToScheme := range []func(*runtime.Scheme) error{
		corev1.AddToScheme,
		autoscalingv1.AddToScheme,
		networkingv1.AddToScheme,
		resourcev1.AddToScheme,
		v1alpha1.AddToScheme,
		v1beta1.AddToScheme,
		grovev1alpha1.AddToScheme,
	} {
		if err := addToScheme(s); err != nil {
			t.Fatalf("failed to add type to scheme: %v", err)
		}
	}
	return s
}

func newTestDGDResourceSyncer(reconciler *DynamoGraphDeploymentReconciler) dgdResourceSyncer {
	return newDGDResourceSyncer(reconciler.Client, reconciler.Recorder)
}

func TestDynamoGraphDeploymentReconcileLocksProviderBeforeRejectingStoredCheckpointIncompatibility(t *testing.T) {
	t.Log("Store a DGD with an incompatible checkpoint configuration")
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "test-dgd",
			Namespace:  "default",
			Generation: 7,
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "prefill",
					Experimental: &v1beta1.ExperimentalSpec{
						Checkpoint:       &v1beta1.ComponentCheckpointConfig{Enabled: true},
						GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{Mode: v1beta1.GMSModeInterPod},
						Failover:         &v1beta1.FailoverSpec{},
					},
				},
				{
					ComponentName: "decode",
					Experimental: &v1beta1.ExperimentalSpec{
						Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true},
						Failover:   &v1beta1.FailoverSpec{},
					},
				},
			},
		},
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd).
		WithStatusSubresource(&v1beta1.DynamoGraphDeployment{}).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        kubeClient,
		Recorder:      events.NewFakeRecorder(10),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{},
	}

	t.Log("Reconcile once to persist the provider before reporting incompatibility")
	result, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKeyFromObject(dgd),
	})
	require.NoError(t, err)
	require.Equal(t, ctrl.Result{}, result)

	var stored v1beta1.DynamoGraphDeployment
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), &stored))
	require.False(t, controller_common.ContainsFinalizer(&stored))
	require.Equal(t, commonconsts.WorkloadProviderComponent, stored.Annotations[commonconsts.KubeAnnotationWorkloadProvider])
	require.Equal(t, v1beta1.DGDStateFailed, stored.Status.State)
	ready := meta.FindStatusCondition(stored.Status.Conditions, "Ready")
	require.NotNil(t, ready)
	require.Equal(t, metav1.ConditionFalse, ready.Status)
	require.Equal(t, dgd.Generation, ready.ObservedGeneration)
	require.Equal(t, string(reasonFailedToReconcileResources), ready.Reason)
	require.Equal(t,
		"component \"prefill\": Snapshot with gpuMemoryService.mode=InterPod is unsupported\n"+
			"component \"prefill\": Snapshot with active/passive failover is temporarily unsupported\n"+
			"component \"decode\": Snapshot with active/passive failover is temporarily unsupported",
		ready.Message,
	)
	require.Zero(t, stored.Status.ObservedGeneration)
}

func TestDynamoGraphDeploymentReconcileFinalizesDeletingStoredCheckpointIncompatibility(t *testing.T) {
	now := metav1.Now()
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test-dgd",
			Namespace:         "default",
			DeletionTimestamp: &now,
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker",
				Experimental: &v1beta1.ExperimentalSpec{
					Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true},
					Failover:   &v1beta1.FailoverSpec{},
				},
			}},
		},
	}
	controller_common.AddFinalizer(dgd)
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd).
		WithStatusSubresource(&v1beta1.DynamoGraphDeployment{}).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        kubeClient,
		Recorder:      events.NewFakeRecorder(10),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{},
	}

	result, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKeyFromObject(dgd),
	})
	require.NoError(t, err)
	require.Equal(t, ctrl.Result{}, result)

	var stored v1beta1.DynamoGraphDeployment
	err = kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), &stored)
	if !apierrors.IsNotFound(err) {
		require.NoError(t, err)
		require.False(t, controller_common.ContainsFinalizer(&stored))
	}
}

func TestDGDScalingAdaptersReconciler_Reconcile(t *testing.T) {
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)

	tests := []struct {
		name                 string
		dgd                  *v1beta1.DynamoGraphDeployment
		existingAdapters     []v1alpha1.DynamoGraphDeploymentScalingAdapter
		expectedAdapterCount int
		expectedAdapters     map[string]int32 // map of adapter name to expected replicas
		expectDeleted        []string         // adapter names that should be deleted
		assertNoReplicaPatch bool
	}{
		{
			name: "creates adapters for services with scalingAdapter.enabled=true",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(2)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
						"decode": {
							Replicas: ptr.To(int32(3)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
					},
				},
			}),
			expectedAdapterCount: 2,
			expectedAdapters: map[string]int32{
				"test-dgd-frontend": 2,
				"test-dgd-decode":   3,
			},
		},
		{
			name: "uses default replicas when not specified",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
					},
				},
			}),
			expectedAdapterCount: 1,
			expectedAdapters: map[string]int32{
				"test-dgd-worker": 1, // default replicas
			},
		},
		{
			name: "preserves existing adapter replicas across components",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
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
								Enabled: true,
							},
						},
						"decode": {
							Replicas: ptr.To(int32(3)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
					},
				},
			}),
			existingAdapters: []v1alpha1.DynamoGraphDeploymentScalingAdapter{
				{
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
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-decode",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
						Replicas: 0,
						DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
							Name:        "test-dgd",
							ServiceName: "decode",
						},
					},
				},
			},
			expectedAdapterCount: 2,
			expectedAdapters: map[string]int32{
				"test-dgd-frontend": 5,
				"test-dgd-decode":   0,
			},
			assertNoReplicaPatch: true,
		},
		{
			name: "skips adapter creation when not enabled",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(2)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
						"decode": {
							Replicas: ptr.To(int32(3)),
							// No ScalingAdapter or Enabled=false means no adapter created
						},
					},
				},
			}),
			expectedAdapterCount: 1,
			expectedAdapters: map[string]int32{
				"test-dgd-frontend": 2,
			},
		},
		{
			name: "deletes adapter when service is removed",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
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
								Enabled: true,
							},
						},
					},
				},
			}),
			existingAdapters: []v1alpha1.DynamoGraphDeploymentScalingAdapter{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
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
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
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
			name: "deletes adapter when scalingAdapter.enabled is not set",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					UID:       "test-uid",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(2)),
							// No ScalingAdapter means adapter should be deleted
						},
					},
				},
			}),
			existingAdapters: []v1alpha1.DynamoGraphDeploymentScalingAdapter{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
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
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "my-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"MyService": {
							Replicas: ptr.To(int32(1)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
					},
				},
			}),
			expectedAdapterCount: 1,
			expectedAdapters: map[string]int32{
				"my-dgd-myservice": 1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build the DGD and any pre-existing scaling adapters")
			var initObjs []client.Object
			initObjs = append(initObjs, tt.dgd)
			for i := range tt.existingAdapters {
				initObjs = append(initObjs, &tt.existingAdapters[i])
			}

			t.Log("Build the fake client and scaling-adapters reconciler")
			clientBuilder := fake.NewClientBuilder().
				WithScheme(testScheme).
				WithObjects(initObjs...)
			if tt.assertNoReplicaPatch {
				t.Log("Intercept adapter patches and verify they exclude replicas")
				clientBuilder = clientBuilder.WithInterceptorFuncs(interceptor.Funcs{
					Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
						data, err := patch.Data(obj)
						require.NoError(t, err)
						assert.NotContains(t, string(data), `"replicas"`,
							"existing adapter patches must never include spec.replicas")
						return c.Patch(ctx, obj, patch, opts...)
					},
				})
			}
			fakeClient := clientBuilder.Build()

			r := &DynamoGraphDeploymentReconciler{
				Client:   fakeClient,
				Recorder: events.NewFakeRecorder(10),
			}

			t.Log("Reconcile scaling adapters")
			ctx := context.Background()
			err := newDGDScalingAdaptersReconciler(r.Client, r.Recorder).Reconcile(ctx, tt.dgd)
			if err != nil {
				t.Fatalf("dgdScalingAdaptersReconciler.Reconcile() error = %v", err)
			}

			t.Log("Verify the resulting adapter set")
			adapterList := &v1alpha1.DynamoGraphDeploymentScalingAdapterList{}
			if err := fakeClient.List(ctx, adapterList, client.InNamespace("default")); err != nil {
				t.Fatalf("Failed to list adapters: %v", err)
			}

			if len(adapterList.Items) != tt.expectedAdapterCount {
				t.Errorf("Expected %d adapters, got %d", tt.expectedAdapterCount, len(adapterList.Items))
			}

			t.Log("Verify expected adapters and replicas")
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

			t.Log("Verify stale adapters were deleted")
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

func TestDGDScalingAdaptersReconciler_EmitsDeleteEventOnlyAfterSuccessfulDelete(t *testing.T) {
	notFound := apierrors.NewNotFound(
		schema.GroupResource{
			Group:    v1alpha1.GroupVersion.Group,
			Resource: "dynamographdeploymentscalingadapters",
		},
		"test-dgd-removed",
	)
	tests := []struct {
		name      string
		deleteErr error
		wantEvent bool
	}{
		{
			name:      "successful delete emits event",
			wantEvent: true,
		},
		{
			name:      "already absent adapter emits no event",
			deleteErr: notFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
			}
			adapter := &v1alpha1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd-removed",
					Namespace: "default",
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
					},
				},
				Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
					DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
						Name:        dgd.Name,
						ServiceName: "removed",
					},
				},
			}
			kubeClient := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithObjects(dgd, adapter).
				WithInterceptorFuncs(interceptor.Funcs{
					Delete: func(
						ctx context.Context,
						writer client.WithWatch,
						obj client.Object,
						opts ...client.DeleteOption,
					) error {
						if tt.deleteErr != nil {
							return tt.deleteErr
						}
						return writer.Delete(ctx, obj, opts...)
					},
				}).
				Build()
			recorder := events.NewFakeRecorder(10)
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:   kubeClient,
				Recorder: recorder,
			}

			require.NoError(t, newDGDScalingAdaptersReconciler(reconciler.Client, reconciler.Recorder).Reconcile(context.Background(), dgd))
			if tt.wantEvent {
				assert.Len(t, recorder.Events, 1)
				return
			}
			assert.Empty(t, recorder.Events)
		})
	}
}

func TestDynamoGraphDeploymentReconciler_mapAutoCheckpointToDGDRequestsAllowsRetainedWithoutOwnerReference(t *testing.T) {
	reconciler := &DynamoGraphDeploymentReconciler{}
	ckpt := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "retained",
			Namespace: "default",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
			Annotations: map[string]string{
				commonconsts.CheckpointAutoAnnotation:           commonconsts.KubeLabelValueTrue,
				commonconsts.CheckpointDeletionPolicyAnnotation: string(v1alpha1.CheckpointDeletionPolicyRetain),
			},
		},
	}

	got := reconciler.mapAutoCheckpointToDGDRequests(context.Background(), ckpt)
	require.Len(t, got, 1)
	assert.Equal(t, types.NamespacedName{Namespace: "default", Name: "test-dgd"}, got[0].NamespacedName)
}

func TestGroveWorkloadsReconciler_Reconcile(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                   string
		dgdSpec                v1alpha1.DynamoGraphDeploymentSpec
		existingGroveResources []client.Object
		draEnabled             bool
		wantReconcileResult    ReconcileResult
		wantErrSubstring       string
		interceptorFuncs       interceptor.Funcs
	}{
		{
			// Covers the error-propagation fix: a non-NotFound Grove read error
			// must surface as a reconcile error (so it retries and does not
			// advance ObservedGeneration), not be folded into a not-ready result.
			name: "transient PodClique API error is propagated",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ComponentType: string(commonconsts.ComponentTypeFrontend),
						Replicas:      ptr.To(int32(1)),
					},
				},
			},
			interceptorFuncs: interceptor.Funcs{
				Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
					if _, ok := obj.(*grovev1alpha1.PodClique); ok {
						return fmt.Errorf("transient API error")
					}
					return c.Get(ctx, key, obj, opts...)
				},
			},
			wantErrSubstring: "transient API error",
		},
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
						Replicas:           2,
						UpdatedReplicas:    2,
						ReadyReplicas:      2,
						ScheduledReplicas:  2,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStateSuccessful,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindPodClique,
						ComponentNames:    []string{"test-dgd-0-frontend"},
						Replicas:          2,
						UpdatedReplicas:   2,
						ReadyReplicas:     ptr.To(int32(2)),
						ScheduledReplicas: ptr.To(int32(2)),
						RuntimeNamespace:  "default-test-dgd",
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
						Replicas:           1,
						UpdatedReplicas:    1,
						ReadyReplicas:      1,
						ScheduledReplicas:  1,
						ObservedGeneration: ptr.To(int64(1)),
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
						Replicas:           2,
						UpdatedReplicas:    1,
						ReadyReplicas:      1, // Only 1 ready, but 2 desired
						ScheduledReplicas:  2, // both scheduled; rollout in progress
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStatePending,
				Reason:  "updating",
				Message: Message("Resources not ready: test-dgd: decode: desired=2, updated=1"),
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindPodClique,
						ComponentNames:    []string{"test-dgd-0-frontend"},
						Replicas:          1,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						ScheduledReplicas: ptr.To(int32(1)),
						RuntimeNamespace:  "default-test-dgd",
					},
					"decode": {
						ComponentKind:     v1beta1.ComponentKindPodClique,
						ComponentNames:    []string{"test-dgd-0-decode"},
						Replicas:          2,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						RuntimeNamespace:  "default-test-dgd",
						ScheduledReplicas: ptr.To(int32(2)),
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
						Replicas:           1,
						UpdatedReplicas:    1,
						AvailableReplicas:  1,
						ScheduledReplicas:  1,
						ObservedGeneration: ptr.To(int64(1)),
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
						Replicas:           1,
						UpdatedReplicas:    1,
						AvailableReplicas:  1,
						ScheduledReplicas:  1,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStateSuccessful,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"decode": {
						ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
						ComponentNames:    []string{"test-dgd-0-decode"},
						Replicas:          1,
						UpdatedReplicas:   1,
						AvailableReplicas: ptr.To(int32(1)),
						RuntimeNamespace:  "default-test-dgd",
						ScheduledReplicas: ptr.To(int32(1)),
					},
					"prefill": {
						ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
						ComponentNames:    []string{"test-dgd-0-prefill"},
						Replicas:          1,
						UpdatedReplicas:   1,
						AvailableReplicas: ptr.To(int32(1)),
						RuntimeNamespace:  "default-test-dgd",
						ScheduledReplicas: ptr.To(int32(1)),
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
						Replicas:           1,
						UpdatedReplicas:    1,
						ReadyReplicas:      1,
						ScheduledReplicas:  1,
						ObservedGeneration: ptr.To(int64(1)),
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
						Replicas:           2,
						UpdatedReplicas:    2,
						AvailableReplicas:  1, // Only 1 available, but 2 desired
						ScheduledReplicas:  2, // both scheduled; availability (not scheduling) is the shortfall
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStatePending,
				Reason:  "pods_not_ready",
				Message: Message("Resources not ready: test-dgd: aggregated: scheduled but available=1/2"),
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindPodClique,
						ComponentNames:    []string{"test-dgd-0-frontend"},
						Replicas:          1,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						ScheduledReplicas: ptr.To(int32(1)),
						RuntimeNamespace:  "default-test-dgd",
					},
					"aggregated": {
						ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
						ComponentNames:    []string{"test-dgd-0-aggregated"},
						Replicas:          2,
						UpdatedReplicas:   2,
						AvailableReplicas: ptr.To(int32(1)),
						RuntimeNamespace:  "default-test-dgd",
						ScheduledReplicas: ptr.To(int32(2)),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			s := newDynamoGraphDeploymentControllerTestScheme(t)

			dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: tt.dgdSpec,
			})

			var objects []client.Object
			objects = append(objects, dgd)
			objects = append(objects, tt.existingGroveResources...)

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithRESTMapper(groveScaleRESTMapper()).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				WithInterceptorFuncs(groveScaleInterceptor(tt.interceptorFuncs, nil)).
				Build()

			recorder := events.NewFakeRecorder(100)
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:        fakeKubeClient,
				Recorder:      recorder,
				Config:        &configv1alpha1.OperatorConfiguration{},
				RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{DRA: tt.draEnabled}},
				DockerSecretRetriever: &mockDockerSecretRetriever{
					GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
						return []string{}, nil
					},
				},
			}

			result, err := reconciler.newGroveProgram().workloads.Reconcile(
				ctx,
				dgd,
				nil,
				nil,
			)
			if tt.wantErrSubstring != "" {
				g.Expect(err).To(gomega.HaveOccurred())
				g.Expect(err.Error()).To(gomega.ContainSubstring(tt.wantErrSubstring))
				return
			}
			g.Expect(err).NotTo(gomega.HaveOccurred())

			t.Log("Expect workers to withhold their runtime namespace until a PCS revision is accepted")
			want := tt.wantReconcileResult
			want.ComponentStatus = make(map[string]v1beta1.ComponentReplicaStatus, len(tt.wantReconcileResult.ComponentStatus))
			for componentName, componentStatus := range tt.wantReconcileResult.ComponentStatus {
				want.ComponentStatus[componentName] = componentStatus
			}
			for i := range dgd.Spec.Components {
				component := &dgd.Spec.Components[i]
				if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
					continue
				}
				componentStatus := want.ComponentStatus[component.ComponentName]
				componentStatus.RuntimeNamespace = ""
				want.ComponentStatus[component.ComponentName] = componentStatus
			}

			g.Expect(result).To(gomega.Equal(want))
		})
	}
}

func TestGroveWorkloadsReconciler_UsesPreservedAlphaServiceIngress(t *testing.T) {
	ctx := context.Background()
	g := gomega.NewGomegaWithT(t)

	className := "custom-nginx"
	alpha := &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Labels:           map[string]string{"graph-label": "kept"},
			Annotations:      map[string]string{"graph-annotation": "kept"},
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ComponentType: commonconsts.ComponentTypeFrontend,
					Replicas:      ptr.To(int32(1)),
					Labels:        map[string]string{"legacy-label": "kept"},
					Annotations:   map[string]string{"legacy-annotation": "kept"},
					Ingress: &v1alpha1.IngressSpec{
						Enabled:                    true,
						Host:                       "legacy-frontend",
						IngressControllerClassName: &className,
					},
				},
			},
		},
	}
	dgd := &v1beta1.DynamoGraphDeployment{}
	g.Expect(alpha.ConvertTo(dgd)).NotTo(gomega.HaveOccurred())

	s := scheme.Scheme
	g.Expect(v1alpha1.AddToScheme(s)).NotTo(gomega.HaveOccurred())
	g.Expect(v1beta1.AddToScheme(s)).NotTo(gomega.HaveOccurred())
	g.Expect(corev1.AddToScheme(s)).NotTo(gomega.HaveOccurred())
	g.Expect(networkingv1.AddToScheme(s)).NotTo(gomega.HaveOccurred())
	g.Expect(grovev1alpha1.AddToScheme(s)).NotTo(gomega.HaveOccurred())

	fakeKubeClient := fake.NewClientBuilder().
		WithScheme(s).
		WithRESTMapper(groveScaleRESTMapper()).
		WithObjects(dgd).
		Build()

	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fakeKubeClient,
		Recorder:      events.NewFakeRecorder(100),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{},
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
				return []string{}, nil
			},
		},
	}

	_, err := reconciler.newGroveProgram().workloads.Reconcile(
		ctx,
		dgd,
		nil,
		nil,
	)
	g.Expect(err).NotTo(gomega.HaveOccurred())

	ingress := &networkingv1.Ingress{}
	g.Expect(fakeKubeClient.Get(ctx, types.NamespacedName{Name: "test-dgd-frontend", Namespace: "default"}, ingress)).NotTo(gomega.HaveOccurred())
	g.Expect(ingress.Spec.IngressClassName).NotTo(gomega.BeNil())
	g.Expect(*ingress.Spec.IngressClassName).To(gomega.Equal(className))
	g.Expect(ingress.Spec.Rules).To(gomega.HaveLen(1))
	g.Expect(ingress.Spec.Rules[0].Host).To(gomega.Equal("legacy-frontend.local"))

	service := &corev1.Service{}
	g.Expect(fakeKubeClient.Get(ctx, types.NamespacedName{Name: "test-dgd-frontend", Namespace: "default"}, service)).NotTo(gomega.HaveOccurred())
	g.Expect(service.Labels["graph-label"]).To(gomega.Equal("kept"))
	g.Expect(service.Labels["legacy-label"]).To(gomega.Equal("kept"))
	g.Expect(service.Annotations["graph-annotation"]).To(gomega.Equal("kept"))
	g.Expect(service.Annotations["legacy-annotation"]).To(gomega.Equal("kept"))
}

func TestGroveWorkloadRendererRenderPreservesLegacyWorkerSelectors(t *testing.T) {
	ctx := context.Background()
	g := gomega.NewGomegaWithT(t)

	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-disagg-planner",
			Namespace: "jsm",
			Annotations: map[string]string{
				commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.1.0",
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Frontend", ComponentType: v1beta1.ComponentTypeFrontend, Replicas: ptr.To(int32(1))},
				{ComponentName: "Planner", ComponentType: v1beta1.ComponentTypePlanner, Replicas: ptr.To(int32(1))},
				{ComponentName: "VllmDecodeWorker", ComponentType: v1beta1.ComponentTypeDecode, Replicas: ptr.To(int32(1))},
				{ComponentName: "VllmPrefillWorker", ComponentType: v1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(1))},
			},
		},
	}
	existingPCS := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-disagg-planner",
			Namespace: "jsm",
		},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{
						Name: "vllmprefillworker",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:        "VllmPrefillWorker",
							commonconsts.KubeLabelDynamoComponentType:    commonconsts.ComponentTypeWorker,
							commonconsts.KubeLabelDynamoSubComponentType: commonconsts.ComponentTypePrefill,
						},
						Annotations: map[string]string{
							commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.1.0",
						},
					},
					{Name: "frontend"},
					{
						Name: "vllmdecodeworker",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:        "VllmDecodeWorker",
							commonconsts.KubeLabelDynamoComponentType:    commonconsts.ComponentTypeWorker,
							commonconsts.KubeLabelDynamoSubComponentType: commonconsts.ComponentTypeDecode,
						},
					},
				},
			},
		},
	}

	fakeKubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd, existingPCS).
		Build()
	renderer := newGroveWorkloadRenderer(
		fakeKubeClient,
		&configv1alpha1.OperatorConfiguration{},
		&controller_common.RuntimeConfig{},
		nil,
	)

	renderedPCS, err := renderer.Render(ctx, dgd, nil, nil, false)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	generatedPCS := renderedPCS.desired
	renderDGD := renderedPCS.renderDeployment
	g.Expect(dgd.GetComponentByName("VllmDecodeWorker").ComponentType).To(gomega.Equal(v1beta1.ComponentTypeDecode))

	prefill := renderDGD.GetComponentByName("VllmPrefillWorker")
	if prefill == nil {
		t.Fatal("expected rendered prefill component")
	}
	g.Expect(prefill.ComponentType).To(gomega.Equal(v1beta1.ComponentTypeWorker))
	g.Expect(prefill.PodTemplate.Labels[commonconsts.KubeLabelDynamoSubComponentType]).To(gomega.Equal(commonconsts.ComponentTypePrefill))

	decode := renderDGD.GetComponentByName("VllmDecodeWorker")
	if decode == nil {
		t.Fatal("expected rendered decode component")
	}
	g.Expect(decode.ComponentType).To(gomega.Equal(v1beta1.ComponentTypeWorker))
	g.Expect(decode.PodTemplate.Labels[commonconsts.KubeLabelDynamoSubComponentType]).To(gomega.Equal(commonconsts.ComponentTypeDecode))

	g.Expect(generatedPCS.Spec.Template.Cliques[0].Name).To(gomega.Equal("vllmprefillworker"))

	var prefillClique *grovev1alpha1.PodCliqueTemplateSpec
	for _, clique := range generatedPCS.Spec.Template.Cliques {
		if clique.Name == "vllmprefillworker" {
			prefillClique = clique
			break
		}
	}
	if prefillClique == nil {
		t.Fatal("expected rendered prefill clique")
	}
	g.Expect(prefillClique.Labels[commonconsts.KubeLabelDynamoComponentType]).To(gomega.Equal(commonconsts.ComponentTypeWorker))
	g.Expect(prefillClique.Labels[commonconsts.KubeLabelDynamoSubComponentType]).To(gomega.Equal(commonconsts.ComponentTypePrefill))
	g.Expect(prefillClique.Annotations[commonconsts.KubeAnnotationDynamoOperatorOriginVersion]).To(gomega.Equal("1.1.0"))

	decodeService, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
		ServiceName:     dynamo.GetDCDResourceName(renderDGD, "VllmDecodeWorker", ""),
		Namespace:       renderDGD.Namespace,
		ComponentType:   string(decode.ComponentType),
		DynamoNamespace: renderDGD.GetDynamoNamespaceForComponent(decode),
		ComponentName:   "VllmDecodeWorker",
		Labels:          dynamo.GetDGDComponentResourceLabels(renderDGD, "VllmDecodeWorker", decode),
		Annotations:     dynamo.GetDGDComponentResourceAnnotations(renderDGD, "VllmDecodeWorker", decode),
		IsK8sDiscovery:  true,
	})
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(decodeService.Spec.Selector[commonconsts.KubeLabelDynamoComponentType]).To(gomega.Equal(commonconsts.ComponentTypeWorker))
}

func TestPrepareGroveTopologyConstraintUpgrade(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	modernConstraint := func(topologyName string, domain grovev1alpha1.TopologyDomain) *grovev1alpha1.TopologyConstraint {
		return &grovev1alpha1.TopologyConstraint{
			TopologyName: topologyName,
			Pack: &grovev1alpha1.TopologyPackConstraint{
				RequiredDomain: domain,
			},
		}
	}
	legacyConstraint := func(domain grovev1alpha1.TopologyDomain) *grovev1alpha1.TopologyConstraint {
		return &grovev1alpha1.TopologyConstraint{PackDomain: domain}
	}

	modern := &grovev1alpha1.PodCliqueSet{
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				TopologyConstraint: modernConstraint("grove-topology", "zone"),
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{Name: "explicit", TopologyConstraint: modernConstraint("grove-topology", "rack")},
					{Name: "inherited", TopologyConstraint: modernConstraint("", "rack")},
				},
				PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
					{Name: "workers", TopologyConstraint: modernConstraint("grove-topology", "block")},
				},
			},
		},
	}
	existing := &grovev1alpha1.PodCliqueSet{
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				TopologyConstraint: legacyConstraint("zone"),
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{Name: "inherited", TopologyConstraint: legacyConstraint("rack")},
					{Name: "explicit", TopologyConstraint: legacyConstraint("rack")},
				},
				PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
					{Name: "workers", TopologyConstraint: legacyConstraint("block")},
				},
			},
		},
	}

	firstStep := modern.DeepCopy()
	prepareGroveTopologyConstraintUpgrade(firstStep, existing)

	g.Expect(firstStep.Spec.Template.TopologyConstraint).To(gomega.Equal(&grovev1alpha1.TopologyConstraint{
		TopologyName: "grove-topology",
		PackDomain:   "zone",
	}))
	g.Expect(firstStep.Spec.Template.Cliques[0].TopologyConstraint).To(gomega.Equal(&grovev1alpha1.TopologyConstraint{
		TopologyName: "grove-topology",
		PackDomain:   "rack",
	}))
	// This constraint can inherit the topology name repaired on its parent, so
	// Grove can migrate its packing field in the same update.
	g.Expect(firstStep.Spec.Template.Cliques[1].TopologyConstraint).To(gomega.Equal(modern.Spec.Template.Cliques[1].TopologyConstraint))
	g.Expect(firstStep.Spec.Template.PodCliqueScalingGroupConfigs[0].TopologyConstraint).To(gomega.Equal(&grovev1alpha1.TopologyConstraint{
		TopologyName: "grove-topology",
		PackDomain:   "block",
	}))

	secondStep := modern.DeepCopy()
	prepareGroveTopologyConstraintUpgrade(secondStep, firstStep)
	g.Expect(secondStep).To(gomega.Equal(modern), "a repaired constraint should proceed to pack.required on the next reconciliation")
}

func TestPreserveGrovePodCliqueSetReplicas(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	desired := &grovev1alpha1.PodCliqueSet{
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{Name: "frontend", Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1}},
					{Name: "prefill", Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1}},
					{Name: "new-worker", Spec: grovev1alpha1.PodCliqueSpec{Replicas: 5}},
				},
				PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
					{Name: "decode-group", CliqueNames: []string{"decode"}, Replicas: ptr.To(int32(1))},
					{Name: "prefill-group", CliqueNames: []string{"prefill"}, Replicas: ptr.To(int32(1))},
					{Name: "new-group", Replicas: ptr.To(int32(7))},
				},
			},
		},
	}
	existing := &grovev1alpha1.PodCliqueSet{
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{Name: "frontend", Spec: grovev1alpha1.PodCliqueSpec{Replicas: 2}},
					{Name: "prefill", Spec: grovev1alpha1.PodCliqueSpec{Replicas: 4}},
				},
				PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
					{Name: "decode-group", CliqueNames: []string{"decode"}},
					{Name: "prefill-group", CliqueNames: []string{"prefill"}, Replicas: ptr.To(int32(6))},
				},
			},
		},
	}

	preserveGrovePodCliqueSetReplicas(desired, existing)

	replicasByClique := map[string]int32{}
	for _, clique := range desired.Spec.Template.Cliques {
		replicasByClique[clique.Name] = clique.Spec.Replicas
	}
	g.Expect(replicasByClique).To(gomega.Equal(map[string]int32{
		"frontend":   2,
		"prefill":    1,
		"new-worker": 5,
	}))
	g.Expect(desired.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas).To(gomega.BeNil())
	g.Expect(desired.Spec.Template.PodCliqueScalingGroupConfigs[1].Replicas).NotTo(gomega.BeNil())
	g.Expect(*desired.Spec.Template.PodCliqueScalingGroupConfigs[1].Replicas).To(gomega.Equal(int32(6)))
	g.Expect(*desired.Spec.Template.PodCliqueScalingGroupConfigs[2].Replicas).To(gomega.Equal(int32(7)))
}

func TestPreserveGrovePodCliqueSetReplicasSkipsCheckpointGatedComponents(t *testing.T) {
	g := gomega.NewGomegaWithT(t)

	desired := &grovev1alpha1.PodCliqueSet{
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{
						Name: "worker",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent: "worker",
						},
						Spec: grovev1alpha1.PodCliqueSpec{Replicas: 0},
					},
				},
				PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
					{Name: "decode", Replicas: ptr.To(int32(0))},
				},
			},
		},
	}
	existing := &grovev1alpha1.PodCliqueSet{
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{Name: "worker", Spec: grovev1alpha1.PodCliqueSpec{Replicas: 5}},
				},
				PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
					{Name: "decode", Replicas: ptr.To(int32(7))},
				},
			},
		},
	}

	preserveGrovePodCliqueSetReplicas(desired, existing, map[string]*checkpoint.CheckpointInfo{
		"worker": {
			Enabled:       true,
			StartupPolicy: v1alpha1.CheckpointStartupPolicyWaitForCheckpoint,
		},
		"decode": {
			Enabled:       true,
			StartupPolicy: v1alpha1.CheckpointStartupPolicyWaitForCheckpoint,
		},
	})

	g.Expect(desired.Spec.Template.Cliques[0].Spec.Replicas).To(gomega.Equal(int32(0)))
	g.Expect(desired.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas).NotTo(gomega.BeNil())
	g.Expect(*desired.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas).To(gomega.Equal(int32(0)))

	preserveGrovePodCliqueSetReplicas(desired, existing, map[string]*checkpoint.CheckpointInfo{
		"worker": {
			Enabled:       true,
			Ready:         true,
			StartupPolicy: v1alpha1.CheckpointStartupPolicyWaitForCheckpoint,
		},
		"decode": {
			Enabled:       true,
			Ready:         true,
			StartupPolicy: v1alpha1.CheckpointStartupPolicyWaitForCheckpoint,
		},
	})

	g.Expect(desired.Spec.Template.Cliques[0].Spec.Replicas).To(gomega.Equal(int32(5)))
	g.Expect(desired.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas).NotTo(gomega.BeNil())
	g.Expect(*desired.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas).To(gomega.Equal(int32(7)))
}

func TestGroveWorkloadRendererRenderKeepsNativeWorkerSelectors(t *testing.T) {
	ctx := context.Background()
	g := gomega.NewGomegaWithT(t)

	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "native-dgd", Namespace: "jsm"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(1))},
			},
		},
	}
	existingPCS := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "native-dgd", Namespace: "jsm"},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{
						Name: "prefill",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:     "prefill",
							commonconsts.KubeLabelDynamoComponentType: commonconsts.ComponentTypePrefill,
						},
					},
				},
			},
		},
	}

	fakeKubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd, existingPCS).
		Build()
	renderer := newGroveWorkloadRenderer(
		fakeKubeClient,
		&configv1alpha1.OperatorConfiguration{},
		&controller_common.RuntimeConfig{},
		nil,
	)
	renderedPCS, err := renderer.Render(ctx, dgd, nil, nil, false)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	renderDGD := renderedPCS.renderDeployment
	prefill := renderDGD.GetComponentByName("prefill")
	if prefill == nil {
		t.Fatal("expected rendered prefill component")
	}
	g.Expect(prefill.ComponentType).To(gomega.Equal(v1beta1.ComponentTypePrefill))
}

func TestDGDRestartReconciler_ComputeStatus(t *testing.T) {
	ctx := context.Background()
	newID := "restart-1"
	oldID := "restart-0"

	tests := []struct {
		name              string
		dgdSpec           v1alpha1.DynamoGraphDeploymentSpec
		dgdStatus         v1alpha1.DynamoGraphDeploymentStatus
		existingResources []client.Object
		groveEnabled      bool
		wantRestartStatus *v1alpha1.RestartStatus
	}{
		{
			name: "no restart requested - returns nil",
		},
		{
			name: "no restart at time - returns nil",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{},
			},
		},
		{
			name: "no restart requested but has completed status - preserves status",
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseCompleted,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: oldID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "no restart requested but has restarting status - returns nil",
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseRestarting,
				},
			},
		},
		{
			name: "restart already processed (completed) - returns existing status",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseCompleted,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "restart already processed (failed) - returns existing status",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseFailed,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseFailed,
			},
		},
		{
			name: "parallel restart - all services complete (DCD pathway)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "parallel restart - services still restarting (DCD pathway)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:   "frontend",
						ComponentType: string(commonconsts.ComponentTypeFrontend),
						Replicas:      ptr.To(int32(1)),
					},
					"decode": {
						ServiceName:   "decode",
						ComponentType: string(commonconsts.ComponentTypeDecode),
						Replicas:      ptr.To(int32(2)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend", "decode"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-decode",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1, // Not yet caught up
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"decode"},
			},
		},
		{
			name: "sequential restart - first service starting",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type:  v1alpha1.RestartStrategyTypeSequential,
						Order: []string{"frontend", "decode"},
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
					"decode": {
						Replicas: ptr.To(int32(2)),
					},
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "sequential restart - first service done, moving to second",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type:  v1alpha1.RestartStrategyTypeSequential,
						Order: []string{"frontend", "decode"},
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
					"decode": {
						Replicas: ptr.To(int32(2)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"decode"},
			},
		},
		{
			name: "sequential restart - all services complete",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeSequential,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "sequential restart - stale in-progress component resets to first service",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type:  v1alpha1.RestartStrategyTypeSequential,
						Order: []string{"frontend", "decode"},
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
					"decode": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"removed"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-removed",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionTrue},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "default strategy (sequential) - no strategy specified",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "parallel restart with empty services - returns completed",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "sequential restart with empty services - returns completed",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeSequential,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "parallel restart - new request with ready resources should NOT complete immediately (race condition fix)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				// No existing restart status - brand new restart request
			},
			existingResources: []client.Object{
				// DCD is READY - simulating state BEFORE restart annotation is applied
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting, // NOT Completed!
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "Grove pathway - parallel restart complete",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
			},
			existingResources: []client.Object{
				&grovev1alpha1.PodCliqueSet{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd",
						Namespace:  "default",
						Generation: 1,
					},
					Status: grovev1alpha1.PodCliqueSetStatus{
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-0-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 1,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:           1,
						UpdatedReplicas:    1,
						ReadyReplicas:      1,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			groveEnabled: true,
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "Grove pathway - sequential restart in progress",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type:  v1alpha1.RestartStrategyTypeSequential,
						Order: []string{"frontend"},
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(2)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
			},
			existingResources: []client.Object{
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-0-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:           2,
						UpdatedReplicas:    1, // Not fully updated
						ReadyReplicas:      1,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			groveEnabled: true,
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "parallel restart - new restart request during ongoing restart resets to all services (DCD pathway)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID, // NEW timestamp
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
					"decode": {
						Replicas: ptr.To(int32(1)),
					},
					"completed": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend", "decode"}, // completed service already done
				},
			},
			existingResources: []client.Object{
				// All services are now ready (simulating state after new restart timestamp is applied)
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionFalse},
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-decode",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionFalse},
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-completed",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionFalse},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"completed", "decode", "frontend"}, // ALL services, sorted
			},
		},
		{
			name: "sequential restart - new restart request during ongoing restart resets to first service (DCD pathway)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID, // NEW timestamp
					Strategy: &v1alpha1.RestartStrategy{
						Type:  v1alpha1.RestartStrategyTypeSequential,
						Order: []string{"frontend", "decode", "worker"},
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
					"decode": {
						Replicas: ptr.To(int32(1)),
					},
					"worker": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"decode"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionTrue},
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-decode",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionTrue},
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-worker",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionTrue},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"}, // Reset to FIRST service
			},
		},
		{
			name: "rolling update in progress + new restart request - superseded",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				RollingUpdate: &v1alpha1.RollingUpdateStatus{
					Phase: v1alpha1.RollingUpdatePhaseInProgress,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseSuperseded,
			},
		},
		{
			name: "rolling update pending + restart already in progress - superseded",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
				RollingUpdate: &v1alpha1.RollingUpdateStatus{
					Phase: v1alpha1.RollingUpdatePhasePending,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseSuperseded,
			},
		},
		{
			name: "rolling update completed + restart request - normal processing",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				RollingUpdate: &v1alpha1.RollingUpdateStatus{
					Phase: v1alpha1.RollingUpdatePhaseCompleted,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "restart already processed as superseded - returns existing status",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseSuperseded,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseSuperseded,
			},
		},
		{
			name: "no restart requested but has superseded status - preserves status",
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseSuperseded,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: oldID,
				Phase:      v1alpha1.RestartPhaseSuperseded,
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

			dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec:   tt.dgdSpec,
				Status: tt.dgdStatus,
			})

			var objects []client.Object
			objects = append(objects, dgd)
			objects = append(objects, tt.existingResources...)

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			recorder := events.NewFakeRecorder(100)
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:   fakeKubeClient,
				Recorder: recorder,
				Config:   &configv1alpha1.OperatorConfiguration{},
				RuntimeConfig: &controller_common.RuntimeConfig{
					Gate: features.Gates{Grove: tt.groveEnabled},
				},
			}

			restartReconciler := newDGDRestartReconciler()
			var resolveProgress restartProgressResolver = newComponentRestartProgressResolver(reconciler.Client).Resolve
			if tt.groveEnabled {
				resolveProgress = newGroveRestartProgressResolver(reconciler.Client).Resolve
			}
			result := restartReconciler.computeRestartStatusWithProgressResolver(ctx, dgd, resolveProgress)

			if tt.wantRestartStatus == nil {
				g.Expect(result).To(gomega.BeNil())
				return
			}

			g.Expect(result).NotTo(gomega.BeNil())
			g.Expect(result).To(gomega.Equal(betaRestartStatus(tt.wantRestartStatus)))
		})
	}
}

func TestComponentWorkloadsReconciler_Reconcile(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                string
		dgdSpec             v1alpha1.DynamoGraphDeploymentSpec
		dgdAnnotations      map[string]string
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
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
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
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-frontend-deployment"},
							Replicas:          2,
							UpdatedReplicas:   2,
							ReadyReplicas:     ptr.To(int32(2)),
							AvailableReplicas: ptr.To(int32(2)),
						},
					},
				}),
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStateSuccessful,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-frontend-deployment"},
						Replicas:          2,
						UpdatedReplicas:   2,
						ReadyReplicas:     ptr.To(int32(2)),
						AvailableReplicas: ptr.To(int32(2)),
					},
				},
			},
		},
		{
			name: "single service - DCD stale observed generation stays pending",
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
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 2,
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "frontend",
							Replicas:    ptr.To(int32(2)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-frontend-deployment"},
							Replicas:          2,
							UpdatedReplicas:   2,
							ReadyReplicas:     ptr.To(int32(2)),
							AvailableReplicas: ptr.To(int32(2)),
						},
					},
				}),
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStatePending,
				Reason:  "some_resources_are_not_ready",
				Message: "Resources not ready: test-dgd-frontend: spec not yet processed: generation=2, observedGeneration=1",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-frontend-deployment"},
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
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
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
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-frontend-deployment"},
							Replicas:          2,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				}),
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStatePending,
				Reason:  "some_resources_are_not_ready",
				Message: "Resources not ready: test-dgd-frontend: Component deployment not ready - Available condition not true",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-frontend-deployment"},
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
			dgdAnnotations: map[string]string{commonconsts.AnnotationCurrentWorkerHashV2: "1b69c0d3"},
			existingDCDs: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
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
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-frontend-deployment"},
							Replicas:          1,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(1)),
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-decode-1b69c0d3",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "decode",
							Replicas:    ptr.To(int32(2)),
							Labels:      map[string]string{commonconsts.KubeLabelDynamoWorkerHash: "1b69c0d3"},
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-decode-1b69c0d3-deployment"},
							Replicas:          2,
							UpdatedReplicas:   2,
							ReadyReplicas:     ptr.To(int32(2)),
							AvailableReplicas: ptr.To(int32(2)),
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-prefill-1b69c0d3",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "prefill",
							Replicas:    ptr.To(int32(3)),
							Labels:      map[string]string{commonconsts.KubeLabelDynamoWorkerHash: "1b69c0d3"},
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-prefill-1b69c0d3-deployment"},
							Replicas:          3,
							UpdatedReplicas:   3,
							ReadyReplicas:     ptr.To(int32(3)),
							AvailableReplicas: ptr.To(int32(3)),
						},
					},
				}),
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStateSuccessful,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-frontend-deployment"},
						Replicas:          1,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(1)),
					},
					"decode": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-decode-1b69c0d3-deployment"},
						Replicas:          2,
						UpdatedReplicas:   2,
						ReadyReplicas:     ptr.To(int32(2)),
						AvailableReplicas: ptr.To(int32(2)),
					},
					"prefill": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-prefill-1b69c0d3-deployment"},
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
			dgdAnnotations: map[string]string{commonconsts.AnnotationCurrentWorkerHashV2: "1b69c0d3"},
			existingDCDs: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
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
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-frontend-deployment"},
							Replicas:          1,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(1)),
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-decode-1b69c0d3",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "decode",
							Replicas:    ptr.To(int32(2)),
							Labels:      map[string]string{commonconsts.KubeLabelDynamoWorkerHash: "1b69c0d3"},
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-decode-1b69c0d3-deployment"},
							Replicas:          2,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-prefill-1b69c0d3",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "prefill",
							Replicas:    ptr.To(int32(3)),
							Labels:      map[string]string{commonconsts.KubeLabelDynamoWorkerHash: "1b69c0d3"},
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-prefill-1b69c0d3-deployment"},
							Replicas:          3,
							UpdatedReplicas:   3,
							ReadyReplicas:     ptr.To(int32(3)),
							AvailableReplicas: ptr.To(int32(3)),
						},
					},
				}),
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStatePending,
				Reason:  "some_resources_are_not_ready",
				Message: "Resources not ready: test-dgd-decode-1b69c0d3: Component deployment not ready - Available condition not true",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-frontend-deployment"},
						Replicas:          1,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(1)),
					},
					"decode": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-decode-1b69c0d3-deployment"},
						Replicas:          2,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(0)),
					},
					"prefill": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-prefill-1b69c0d3-deployment"},
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
			dgdAnnotations: map[string]string{commonconsts.AnnotationCurrentWorkerHashV2: "cabcd5c9"},
			existingDCDs: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
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
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-frontend-deployment"},
							Replicas:          1,
							UpdatedReplicas:   0,
							ReadyReplicas:     ptr.To(int32(0)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-decode-cabcd5c9",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "decode",
							Replicas:    ptr.To(int32(2)),
							Labels:      map[string]string{commonconsts.KubeLabelDynamoWorkerHash: "cabcd5c9"},
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-decode-cabcd5c9-deployment"},
							Replicas:          2,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				}),
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStatePending,
				Reason:  "some_resources_are_not_ready",
				Message: "Resources not ready: test-dgd-decode-cabcd5c9: Component deployment not ready - Available condition not true; test-dgd-frontend: Component deployment not ready - Available condition not true",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-frontend-deployment"},
						Replicas:          1,
						UpdatedReplicas:   0,
						ReadyReplicas:     ptr.To(int32(0)),
						AvailableReplicas: ptr.To(int32(0)),
					},
					"decode": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-decode-cabcd5c9-deployment"},
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

			dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-dgd",
					Namespace:   "default",
					Annotations: tt.dgdAnnotations,
				},
				Spec: tt.dgdSpec,
			})

			var objects []client.Object
			objects = append(objects, dgd)
			objects = append(objects, tt.existingDCDs...)

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			recorder := events.NewFakeRecorder(100)
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:        fakeKubeClient,
				Recorder:      recorder,
				Config:        &configv1alpha1.OperatorConfiguration{},
				RuntimeConfig: &controller_common.RuntimeConfig{},
			}

			result, err := reconciler.newComponentProgram().workloads.Reconcile(
				ctx,
				dgd,
				nil,
				nil,
			)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			g.Expect(result).To(gomega.Equal(tt.wantReconcileResult))
		})
	}
}

func TestDGDGroveTopologyConditionReconciler_Reconcile(t *testing.T) {
	tests := []struct {
		name           string
		dgd            *v1beta1.DynamoGraphDeployment
		pcs            *grovev1alpha1.PodCliqueSet
		groveEnabled   bool
		wantCondition  bool
		wantStatus     metav1.ConditionStatus
		wantReason     string
		wantEventCount int
	}{
		{
			name: "removed topology constraints preserve the previous condition",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {},
					},
				},
				Status: v1alpha1.DynamoGraphDeploymentStatus{
					Conditions: []metav1.Condition{{
						Type:   v1alpha1.ConditionTypeTopologyLevelsAvailable,
						Status: metav1.ConditionTrue,
						Reason: v1alpha1.ConditionReasonAllTopologyLevelsAvailable,
					}},
				},
			}),
			groveEnabled:  true,
			wantCondition: true,
			wantStatus:    metav1.ConditionTrue,
			wantReason:    v1alpha1.ConditionReasonAllTopologyLevelsAvailable,
		},
		{
			name: "topology set, PCS has no topology condition - unknown",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology", PackDomain: v1alpha1.TopologyDomain("rack")},
				},
			}),
			pcs: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Status:     grovev1alpha1.PodCliqueSetStatus{},
			},
			groveEnabled:  true,
			wantCondition: true,
			wantStatus:    metav1.ConditionUnknown,
			wantReason:    v1alpha1.ConditionReasonTopologyConditionPending,
		},
		{
			name: "PCS reports TopologyLevelsUnavailable=True with ClusterTopologyLevelsUnavailable",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology", PackDomain: v1alpha1.TopologyDomain("rack")},
				},
			}),
			pcs: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Status: grovev1alpha1.PodCliqueSetStatus{
					Conditions: []metav1.Condition{
						{
							Type:    groveconstants.ConditionTopologyLevelsUnavailable,
							Status:  metav1.ConditionTrue,
							Reason:  groveconstants.ConditionReasonTopologyLevelsUnavailable,
							Message: "Topology level 'rack' is no longer available",
						},
					},
				},
			},
			groveEnabled:   true,
			wantCondition:  true,
			wantStatus:     metav1.ConditionFalse,
			wantReason:     v1alpha1.ConditionReasonTopologyLevelsUnavailable,
			wantEventCount: 1,
		},
		{
			name: "PCS reports TopologyLevelsUnavailable=True with ClusterTopologyNotFound",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology", PackDomain: v1alpha1.TopologyDomain("rack")},
				},
			}),
			pcs: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Status: grovev1alpha1.PodCliqueSetStatus{
					Conditions: []metav1.Condition{
						{
							Type:    groveconstants.ConditionTopologyLevelsUnavailable,
							Status:  metav1.ConditionTrue,
							Reason:  groveconstants.ConditionReasonClusterTopologyNotFound,
							Message: "ClusterTopology 'default' not found",
						},
					},
				},
			},
			groveEnabled:   true,
			wantCondition:  true,
			wantStatus:     metav1.ConditionFalse,
			wantReason:     v1alpha1.ConditionReasonTopologyDefinitionNotFound,
			wantEventCount: 1,
		},
		{
			name: "PCS reports TopologyLevelsUnavailable=False - all levels available",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology", PackDomain: v1alpha1.TopologyDomain("rack")},
				},
			}),
			pcs: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Status: grovev1alpha1.PodCliqueSetStatus{
					Conditions: []metav1.Condition{
						{
							Type:    groveconstants.ConditionTopologyLevelsUnavailable,
							Status:  metav1.ConditionFalse,
							Reason:  groveconstants.ConditionReasonAllTopologyLevelsAvailable,
							Message: "All topology levels available",
						},
					},
				},
			},
			groveEnabled:  true,
			wantCondition: true,
			wantStatus:    metav1.ConditionTrue,
			wantReason:    v1alpha1.ConditionReasonAllTopologyLevelsAvailable,
		},
		{
			name: "service-only topology constraint triggers condition propagation",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology"},
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {
							TopologyConstraint: &v1alpha1.TopologyConstraint{PackDomain: v1alpha1.TopologyDomain("rack")},
						},
					},
				},
			}),
			pcs: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Status:     grovev1alpha1.PodCliqueSetStatus{},
			},
			groveEnabled:  true,
			wantCondition: true,
			wantStatus:    metav1.ConditionUnknown,
			wantReason:    v1alpha1.ConditionReasonTopologyConditionPending,
		},
		{
			name: "PCS not found yet - no condition added",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology", PackDomain: v1alpha1.TopologyDomain("rack")},
				},
			}),
			pcs:           nil,
			groveEnabled:  true,
			wantCondition: false,
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

			objs := []client.Object{}
			if tt.pcs != nil {
				objs = append(objs, tt.pcs)
			}

			fakeClient := fake.NewClientBuilder().WithScheme(s).WithObjects(objs...).Build()
			reconciler := &DynamoGraphDeploymentReconciler{
				Client: fakeClient,
				RuntimeConfig: &controller_common.RuntimeConfig{
					Gate: features.Gates{Grove: tt.groveEnabled},
				},
			}

			ctx := context.Background()
			originalStatus := tt.dgd.DeepCopy().Status
			programResult := newWorkloadProgramResult(tt.dgd)
			if tt.groveEnabled {
				newDGDGroveTopologyConditionReconciler(reconciler.Client).
					Reconcile(ctx, tt.dgd, &programResult)
			}
			g.Expect(tt.dgd.Status).To(gomega.Equal(originalStatus), "status projection must not mutate request.DGD.Status")

			var topoCond *metav1.Condition
			for i := range programResult.Status.Conditions {
				if programResult.Status.Conditions[i].Type == v1alpha1.ConditionTypeTopologyLevelsAvailable {
					topoCond = &programResult.Status.Conditions[i]
					break
				}
			}

			if !tt.wantCondition {
				g.Expect(topoCond).To(gomega.BeNil(), "expected no TopologyLevelsAvailable condition")
				return
			}

			g.Expect(topoCond).NotTo(gomega.BeNil(), "expected TopologyLevelsAvailable condition to be set")
			g.Expect(topoCond.Status).To(gomega.Equal(tt.wantStatus))
			g.Expect(topoCond.Reason).To(gomega.Equal(tt.wantReason))

			g.Expect(programResult.Events).To(gomega.HaveLen(tt.wantEventCount))
			g.Expect(tt.dgd.Status).To(gomega.Equal(originalStatus), "status projection must remain local until the outer status write")
		})
	}
}

func TestGroveWatchSetup_MapPodCliqueToRequests(t *testing.T) {
	setup := newGroveWatchSetup(nil)

	t.Run("labeled PodClique maps directly to its DGD", func(t *testing.T) {
		requests := setup.mapPodCliqueToRequests(
			context.Background(),
			&grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "graph-0-worker",
					Namespace: "default",
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoGraphDeploymentName: "graph",
					},
				},
			},
		)

		require.Len(t, requests, 1)
		assert.Equal(t, types.NamespacedName{Namespace: "default", Name: "graph"}, requests[0].NamespacedName)
	})

	t.Run("unlabeled or unrelated objects are ignored", func(t *testing.T) {
		assert.Empty(t, setup.mapPodCliqueToRequests(
			context.Background(),
			&grovev1alpha1.PodClique{ObjectMeta: metav1.ObjectMeta{Name: "orphan", Namespace: "default"}},
		))
		assert.Empty(t, setup.mapPodCliqueToRequests(
			context.Background(),
			&corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "not-a-podclique", Namespace: "default"}},
		))
	})
}

func TestGroveWatchSetup_MapPodCliqueScalingGroupToRequests(t *testing.T) {
	// Register Grove types with the scheme so fake client can handle them
	if err := grovev1alpha1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add grovev1alpha1 to scheme: %v", err)
	}

	tests := []struct {
		name         string
		obj          client.Object
		existingPCS  *grovev1alpha1.PodCliqueSet // PCS object that exists in the cluster
		wantRequests int
		wantName     string
		wantNs       string
	}{
		{
			name: "PCSG with PodCliqueSet controller ownerRef returns DGD request",
			obj: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "dynamo-recipe-0-worker",
					Namespace: "mwieczorek-dsv32-trtllm-agg",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: grovev1alpha1.SchemeGroupVersion.String(),
							Kind:       "PodCliqueSet",
							Name:       "dynamo-recipe",
							Controller: ptr.To(true),
						},
					},
				},
			},
			existingPCS: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "dynamo-recipe",
					Namespace: "mwieczorek-dsv32-trtllm-agg",
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoGraphDeploymentName: "dynamo-recipe",
					},
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: v1alpha1.GroupVersion.String(),
							Kind:       "DynamoGraphDeployment",
							Name:       "dynamo-recipe",
							Controller: ptr.To(true),
						},
					},
				},
			},
			wantRequests: 1,
			wantName:     "dynamo-recipe",
			wantNs:       "mwieczorek-dsv32-trtllm-agg",
		},
		{
			name: "PCSG with truncated PCS name resolves to original DGD name",
			obj: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "truncated-pcs-0-worker",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: grovev1alpha1.SchemeGroupVersion.String(),
							Kind:       "PodCliqueSet",
							Name:       "truncated-pcs",
							Controller: ptr.To(true),
						},
					},
				},
			},
			existingPCS: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "truncated-pcs",
					Namespace: "default",
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoGraphDeploymentName: "my-very-long-original-dgd-name",
					},
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: v1alpha1.GroupVersion.String(),
							Kind:       "DynamoGraphDeployment",
							Name:       "my-very-long-original-dgd-name",
							Controller: ptr.To(true),
						},
					},
				},
			},
			wantRequests: 1,
			wantName:     "my-very-long-original-dgd-name",
			wantNs:       "default",
		},
		{
			name: "PCSG with no ownerRef returns no requests",
			obj: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "orphan-pcsg",
					Namespace: "default",
				},
			},
			wantRequests: 0,
		},
		{
			name: "PCSG with non-controller PodCliqueSet ownerRef returns no requests",
			obj: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pcsg-with-non-controller-ref",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: grovev1alpha1.SchemeGroupVersion.String(),
							Kind:       "PodCliqueSet",
							Name:       "some-pcs",
							// Controller flag omitted: metav1.GetControllerOf must ignore this ref.
						},
					},
				},
			},
			wantRequests: 0,
		},
		{
			name: "PCSG with non-PodCliqueSet ownerRef returns no requests",
			obj: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "weird-pcsg",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "Deployment",
							Name:       "not-a-pcs",
						},
					},
				},
			},
			wantRequests: 0,
		},
		{
			name:         "non-PCSG object returns no requests",
			obj:          &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}},
			wantRequests: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			// Build fake client with existing PCS if provided
			builder := fake.NewClientBuilder().WithScheme(scheme.Scheme)
			if tt.existingPCS != nil {
				builder = builder.WithObjects(tt.existingPCS)
			}
			r := &DynamoGraphDeploymentReconciler{
				Client: builder.Build(),
			}
			reqs := newGroveWatchSetup(r.Client).
				mapPodCliqueScalingGroupToRequests(context.Background(), tt.obj)

			g.Expect(reqs).To(gomega.HaveLen(tt.wantRequests))
			if tt.wantRequests == 1 {
				g.Expect(reqs[0].Name).To(gomega.Equal(tt.wantName))
				g.Expect(reqs[0].Namespace).To(gomega.Equal(tt.wantNs))
			}
		})
	}
}

func TestPodCliqueStatusChangeIsSignificant(t *testing.T) {
	base := func() *grovev1alpha1.PodClique {
		return &grovev1alpha1.PodClique{
			Spec: grovev1alpha1.PodCliqueSpec{Replicas: 3},
			Status: grovev1alpha1.PodCliqueStatus{
				Replicas:                          3,
				ReadyReplicas:                     1,
				UpdatedReplicas:                   3,
				ScheduledReplicas:                 1,
				ScheduleGatedReplicas:             0,
				ObservedGeneration:                ptr.To(int64(1)),
				CurrentPodCliqueSetGenerationHash: ptr.To("previous-revision"),
			},
		}
	}

	tests := []struct {
		name   string
		mutate func(pc *grovev1alpha1.PodClique)
		want   bool
	}{
		{
			name:   "no change is filtered",
			mutate: func(pc *grovev1alpha1.PodClique) {},
			want:   false,
		},
		{
			// The regression this predicate change fixes: scheduling advances
			// 1/3 -> 3/3 while ready/updated/replicas and the condition are flat.
			name:   "scheduled-only advance is significant",
			mutate: func(pc *grovev1alpha1.PodClique) { pc.Status.ScheduledReplicas = 3 },
			want:   true,
		},
		{
			name:   "ready change is significant",
			mutate: func(pc *grovev1alpha1.PodClique) { pc.Status.ReadyReplicas = 3 },
			want:   true,
		},
		{
			name:   "updated change is significant",
			mutate: func(pc *grovev1alpha1.PodClique) { pc.Status.UpdatedReplicas = 2 },
			want:   true,
		},
		{
			name:   "replicas change is significant",
			mutate: func(pc *grovev1alpha1.PodClique) { pc.Status.Replicas = 2 },
			want:   true,
		},
		{
			name:   "schedule-gated change is significant",
			mutate: func(pc *grovev1alpha1.PodClique) { pc.Status.ScheduleGatedReplicas = 1 },
			want:   true,
		},
		{
			name:   "spec replicas change is significant",
			mutate: func(pc *grovev1alpha1.PodClique) { pc.Spec.Replicas = 5 },
			want:   true,
		},
		{
			name:   "observedGeneration change is significant",
			mutate: func(pc *grovev1alpha1.PodClique) { pc.Status.ObservedGeneration = ptr.To(int64(2)) },
			want:   true,
		},
		{
			name: "current PCS revision change is significant",
			mutate: func(pc *grovev1alpha1.PodClique) {
				pc.Status.CurrentPodCliqueSetGenerationHash = ptr.To("target-revision")
			},
			want: true,
		},
		{
			name: "update completion change is significant",
			mutate: func(pc *grovev1alpha1.PodClique) {
				updateEndedAt := metav1.Now()
				pc.Status.UpdateProgress = &grovev1alpha1.PodCliqueUpdateProgress{UpdateEndedAt: &updateEndedAt}
			},
			want: true,
		},
		{
			name:   "generation-only change is filtered",
			mutate: func(pc *grovev1alpha1.PodClique) { pc.Generation = 2 },
			want:   false,
		},
		{
			name: "scheduling condition change is significant",
			mutate: func(pc *grovev1alpha1.PodClique) {
				pc.Status.Conditions = []metav1.Condition{{
					Type:               groveconstants.ConditionTypePodCliqueScheduled,
					Status:             metav1.ConditionFalse,
					Reason:             groveconstants.ConditionReasonInsufficientScheduledPods,
					LastTransitionTime: metav1.Now(),
				}}
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oldPC := base()
			newPC := base()
			tt.mutate(newPC)
			assert.Equal(t, tt.want, podCliqueStatusChangeIsSignificant(oldPC, newPC))
		})
	}

	oldPodClique := base()
	oldPodClique.Status.Conditions = []metav1.Condition{{
		Type:    groveconstants.ConditionTypePodCliqueScheduled,
		Status:  metav1.ConditionFalse,
		Reason:  groveconstants.ConditionReasonInsufficientScheduledPods,
		Message: "one node unavailable",
	}}
	newPodClique := oldPodClique.DeepCopy()
	newPodClique.Status.Conditions[0].Message = "two nodes unavailable"
	assert.False(t, podCliqueStatusChangeIsSignificant(oldPodClique, newPodClique))
}

func TestPCSGStatusChangeIsSignificant(t *testing.T) {
	base := func() *grovev1alpha1.PodCliqueScalingGroup {
		return &grovev1alpha1.PodCliqueScalingGroup{
			Spec: grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: 3},
			Status: grovev1alpha1.PodCliqueScalingGroupStatus{
				Replicas:                          3,
				AvailableReplicas:                 1,
				UpdatedReplicas:                   3,
				ScheduledReplicas:                 1,
				ObservedGeneration:                ptr.To(int64(1)),
				CurrentPodCliqueSetGenerationHash: ptr.To("previous-revision"),
			},
		}
	}

	tests := []struct {
		name   string
		mutate func(pcsg *grovev1alpha1.PodCliqueScalingGroup)
		want   bool
	}{
		{
			name:   "no change is filtered",
			mutate: func(pcsg *grovev1alpha1.PodCliqueScalingGroup) {},
			want:   false,
		},
		{
			name:   "scheduled-only advance is significant",
			mutate: func(pcsg *grovev1alpha1.PodCliqueScalingGroup) { pcsg.Status.ScheduledReplicas = 3 },
			want:   true,
		},
		{
			name:   "available change is significant",
			mutate: func(pcsg *grovev1alpha1.PodCliqueScalingGroup) { pcsg.Status.AvailableReplicas = 3 },
			want:   true,
		},
		{
			name:   "updated change is significant",
			mutate: func(pcsg *grovev1alpha1.PodCliqueScalingGroup) { pcsg.Status.UpdatedReplicas = 2 },
			want:   true,
		},
		{
			name:   "replicas change is significant",
			mutate: func(pcsg *grovev1alpha1.PodCliqueScalingGroup) { pcsg.Status.Replicas = 2 },
			want:   true,
		},
		{
			name:   "spec replicas change is significant",
			mutate: func(pcsg *grovev1alpha1.PodCliqueScalingGroup) { pcsg.Spec.Replicas = 5 },
			want:   true,
		},
		{
			name:   "observedGeneration change is significant",
			mutate: func(pcsg *grovev1alpha1.PodCliqueScalingGroup) { pcsg.Status.ObservedGeneration = ptr.To(int64(2)) },
			want:   true,
		},
		{
			name: "current PCS revision change is significant",
			mutate: func(pcsg *grovev1alpha1.PodCliqueScalingGroup) {
				pcsg.Status.CurrentPodCliqueSetGenerationHash = ptr.To("target-revision")
			},
			want: true,
		},
		{
			name: "update completion change is significant",
			mutate: func(pcsg *grovev1alpha1.PodCliqueScalingGroup) {
				updateEndedAt := metav1.Now()
				pcsg.Status.UpdateProgress = &grovev1alpha1.PodCliqueScalingGroupUpdateProgress{UpdateEndedAt: &updateEndedAt}
			},
			want: true,
		},
		{
			name:   "generation-only change is filtered",
			mutate: func(pcsg *grovev1alpha1.PodCliqueScalingGroup) { pcsg.Generation = 2 },
			want:   false,
		},
		{
			name: "MinAvailableBreached condition change is significant",
			mutate: func(pcsg *grovev1alpha1.PodCliqueScalingGroup) {
				pcsg.Status.Conditions = []metav1.Condition{{
					Type:               groveconstants.ConditionTypeMinAvailableBreached,
					Status:             metav1.ConditionFalse,
					Reason:             groveconstants.ConditionReasonInsufficientAvailablePCSGReplicas,
					LastTransitionTime: metav1.Now(),
				}}
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oldPCSG := base()
			newPCSG := base()
			tt.mutate(newPCSG)
			assert.Equal(t, tt.want, pcsgStatusChangeIsSignificant(oldPCSG, newPCSG))
		})
	}

	oldScalingGroup := base()
	oldScalingGroup.Status.Conditions = []metav1.Condition{{
		Type:    groveconstants.ConditionTypeMinAvailableBreached,
		Status:  metav1.ConditionFalse,
		Reason:  groveconstants.ConditionReasonInsufficientAvailablePCSGReplicas,
		Message: "one replica unavailable",
	}}
	newScalingGroup := oldScalingGroup.DeepCopy()
	newScalingGroup.Status.Conditions[0].Message = "two replicas unavailable"
	assert.False(t, pcsgStatusChangeIsSignificant(oldScalingGroup, newScalingGroup))
}

func TestGroveChildEventPredicates(t *testing.T) {
	podClique := &grovev1alpha1.PodClique{}
	podCliquePredicates := podCliqueEventPredicates()
	assert.False(t, podCliquePredicates.Create(event.CreateEvent{Object: podClique}))
	assert.False(t, podCliquePredicates.Delete(event.DeleteEvent{Object: podClique}))
	assert.False(t, podCliquePredicates.Generic(event.GenericEvent{Object: podClique}))

	scalingGroup := &grovev1alpha1.PodCliqueScalingGroup{}
	scalingGroupPredicates := pcsgEventPredicates()
	assert.False(t, scalingGroupPredicates.Create(event.CreateEvent{Object: scalingGroup}))
	assert.False(t, scalingGroupPredicates.Delete(event.DeleteEvent{Object: scalingGroup}))
	assert.False(t, scalingGroupPredicates.Generic(event.GenericEvent{Object: scalingGroup}))
}
