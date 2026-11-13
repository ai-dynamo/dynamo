// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"errors"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	podcontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/pods"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/serviceidentity"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	corev1apply "k8s.io/client-go/applyconfigurations/core/v1"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestDCDServiceIdentityApplyOutcome(t *testing.T) {
	for _, tc := range []struct {
		name    string
		failure error
		pending bool
	}{
		{name: "cached Service version is pending", pending: true, failure: apierrors.NewConflict(schema.GroupResource{Resource: "services"}, "frontend", errors.New("stale resourceVersion"))},
		{name: "denied Service write remains an error", failure: apierrors.NewForbidden(schema.GroupResource{Resource: "services"}, "frontend", errors.New("denied"))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			t.Log("Reconcile a frontend DCD and its owned Service before an operator metadata backfill")
			dcd := &v1beta1.DynamoComponentDeployment{
				Name: "frontend", Namespace: "default", UID: "frontend-uid", Generation: 1,
				Spec: v1beta1.DynamoComponentDeploymentSpec{DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "runtime:old"}}}},
				}},
			}
			scheme := newDynamoGraphDeploymentControllerTestScheme(t)
			require.NoError(t, appsv1.AddToScheme(scheme))
			base := fake.NewClientBuilder().WithScheme(scheme).
				WithIndex(&corev1.Pod{}, podcontroller.SelectorIndex, podcontroller.SelectorIndexValues).
				WithObjects(dcd).WithStatusSubresource(dcd).Build()
			r := &DynamoComponentDeploymentReconciler{
				Client: base, Recorder: events.NewFakeRecorder(100),
				Config: &configv1alpha1.OperatorConfiguration{}, RuntimeConfig: &commoncontroller.RuntimeConfig{},
				DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }},
			}
			request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dcd)}
			_, err := r.Reconcile(ctx, request)
			require.NoError(t, err)
			service := &corev1.Service{}
			require.NoError(t, base.Get(ctx, request.NamespacedName, service))
			delete(service.Labels, consts.KubeLabelDynamoComponentType)
			require.NoError(t, base.Update(ctx, service))
			require.NoError(t, base.Get(ctx, request.NamespacedName, dcd))
			dcd.Status.Component = &v1beta1.ComponentReplicaStatus{ServedModelName: "observed-model", RuntimeNamespace: "observed-runtime"}
			require.NoError(t, base.Status().Update(ctx, dcd))
			previous := dcd.DeepCopy().Status.Component

			t.Log("Refuse only the new identity Apply and observe its full controller status handling")
			failure, attempts := tc.failure, 0
			r.Client = interceptor.NewClient(base, interceptor.Funcs{Apply: func(ctx context.Context, delegated client.WithWatch, configuration runtime.ApplyConfiguration, opts ...client.ApplyOption) error {
				if _, ok := configuration.(*corev1apply.ServiceApplyConfiguration); ok {
					attempts++
					if failure != nil {
						return failure
					}
				}
				return delegated.Apply(ctx, configuration, opts...)
			}})
			recorder := events.NewFakeRecorder(100)
			r.Recorder = recorder
			result, err := r.Reconcile(ctx, request)
			require.Positive(t, attempts, "the test must reach the identity Apply")
			require.Zero(t, result)
			require.NoError(t, base.Get(ctx, request.NamespacedName, dcd))
			available := meta.FindStatusCondition(dcd.Status.Conditions, v1beta1.DynamoComponentDeploymentConditionTypeAvailable)
			require.NotNil(t, available)
			if tc.pending {
				require.NoError(t, err)
				require.Equal(t, metav1.ConditionUnknown, available.Status)
				require.Equal(t, "ServiceIdentityPending", available.Reason)
				require.Equal(t, previous, dcd.Status.Component, "a cache wait preserves observed component projections")
				require.Empty(t, recorder.Events, "expected cache lag must not produce a failure warning")
			} else {
				require.True(t, apierrors.IsForbidden(err), "%v", err)
				require.Equal(t, metav1.ConditionFalse, available.Status)
				require.NotEqual(t, "ServiceIdentityPending", available.Reason)
				require.Empty(t, dcd.Status.Component.ServedModelName)
				require.NotEmpty(t, recorder.Events)
			}

			t.Log("Retry after the Service observation catches up and restore the identity labels")
			failure = nil
			_, err = r.Reconcile(ctx, request)
			require.NoError(t, err)
			require.NoError(t, base.Get(ctx, request.NamespacedName, service))
			require.Equal(t, string(v1beta1.ComponentTypeFrontend), service.Labels[consts.KubeLabelDynamoComponentType])
			require.NoError(t, base.Get(ctx, request.NamespacedName, dcd))
			require.NotEqual(t, "ServiceIdentityPending", meta.FindStatusCondition(dcd.Status.Conditions, v1beta1.DynamoComponentDeploymentConditionTypeAvailable).Reason)
		})
	}
}

func TestGroveServiceIdentityApplyOutcome(t *testing.T) {
	for _, tc := range []struct {
		name         string
		failure      error
		scaleFailure bool
		pending      bool
	}{
		{name: "cached Service version is pending", pending: true, failure: apierrors.NewConflict(schema.GroupResource{Resource: "services"}, "graph-frontend", errors.New("stale resourceVersion"))},
		{name: "denied Service write remains an error", failure: apierrors.NewForbidden(schema.GroupResource{Resource: "services"}, "graph-frontend", errors.New("denied"))},
		{name: "scaling failure outranks a metadata cache wait", scaleFailure: true, failure: apierrors.NewConflict(schema.GroupResource{Resource: "services"}, "graph-frontend", errors.New("stale resourceVersion"))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			t.Log("Converge a Grove frontend before repairing missing Service metadata")
			dgd := &v1beta1.DynamoGraphDeployment{
				Name: "graph", Namespace: "default", UID: "graph-uid", Generation: 1,
				Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
					ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "runtime:old"}}}},
				}}},
			}
			if tc.scaleFailure {
				dgd.Spec.Components[0].Replicas = new(int32(1))
			}
			base := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithIndex(&corev1.Pod{}, podcontroller.ComponentIndex, podcontroller.ComponentIndexValues).
				WithRESTMapper(groveScaleRESTMapper()).WithObjects(dgd).WithStatusSubresource(dgd).Build()
			r := &DynamoGraphDeploymentReconciler{
				Client: base, Recorder: events.NewFakeRecorder(100),
				Config:                &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"}},
				RuntimeConfig:         &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}},
				DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }},
			}
			result, err := r.newGroveProgram().Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			dgd.Status = result.Status
			service := &corev1.Service{}
			serviceKey := client.ObjectKey{Namespace: "default", Name: "graph-frontend"}
			require.NoError(t, base.Get(ctx, serviceKey, service))
			delete(service.Labels, consts.KubeLabelDynamoComponentType)
			require.NoError(t, base.Update(ctx, service))
			previous := dgd.DeepCopy().Status.Components

			t.Log("Refuse the identity Apply, optionally alongside a genuine Grove scaling API failure")
			failure, applyAttempts, scaleAttempts := tc.failure, 0, 0
			scaleFailure := apierrors.NewForbidden(schema.GroupResource{Group: "grove.io", Resource: "podcliques"}, "graph-0-frontend", errors.New("scaling denied"))
			r.Client = interceptor.NewClient(base, interceptor.Funcs{
				Apply: func(ctx context.Context, delegated client.WithWatch, configuration runtime.ApplyConfiguration, opts ...client.ApplyOption) error {
					if _, ok := configuration.(*corev1apply.ServiceApplyConfiguration); ok {
						applyAttempts++
						if failure != nil {
							return failure
						}
					}
					return delegated.Apply(ctx, configuration, opts...)
				},
				Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
					if _, ok := object.(*grovev1alpha1.PodClique); ok && tc.scaleFailure {
						scaleAttempts++
						return scaleFailure
					}
					return delegated.Get(ctx, key, object, opts...)
				},
				SubResourceGet: func(ctx context.Context, delegated client.Client, subresource string, object, target client.Object, opts ...client.SubResourceGetOption) error {
					if subresource == "scale" && tc.scaleFailure {
						scaleAttempts++
						return scaleFailure
					}
					return delegated.SubResource(subresource).Get(ctx, object, target, opts...)
				},
			})
			result, err = r.newGroveProgram().Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.Positive(t, applyAttempts, "the test must reach the identity Apply")
			ready := meta.FindStatusCondition(result.Status.Conditions, "Ready")
			require.NotNil(t, ready)
			if tc.pending {
				require.NoError(t, err)
				require.Equal(t, v1beta1.DGDStatePending, result.Status.State)
				require.Equal(t, "ServiceIdentityPending", ready.Reason)
				require.Contains(t, result.Status.Components, "frontend")
				require.Equal(t, previous["frontend"].Replicas, result.Status.Components["frontend"].Replicas)
			} else {
				require.True(t, apierrors.IsForbidden(err), "%v", err)
				require.Equal(t, v1beta1.DGDStateFailed, result.Status.State)
				require.NotEqual(t, "ServiceIdentityPending", ready.Reason)
				if tc.scaleFailure {
					require.Positive(t, scaleAttempts)
					require.ErrorIs(t, err, scaleFailure)
					require.NotErrorIs(t, err, serviceidentity.ErrCacheObservationPending)
					return
				}
			}

			t.Log("Converge the same graph after the guarded metadata write can succeed")
			failure = nil
			dgd.Status = result.Status
			result, err = r.newGroveProgram().Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.NoError(t, base.Get(ctx, serviceKey, service))
			require.Equal(t, string(v1beta1.ComponentTypeFrontend), service.Labels[consts.KubeLabelDynamoComponentType])
			require.NotEqual(t, "ServiceIdentityPending", meta.FindStatusCondition(result.Status.Conditions, "Ready").Reason)
		})
	}
}
