// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	corev1apply "k8s.io/client-go/applyconfigurations/core/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestLPXServiceIdentityApplyOutcome(t *testing.T) {
	for _, tc := range []struct {
		name    string
		failure error
		pending bool
	}{
		{name: "cached Service version is pending", pending: true, failure: apierrors.NewConflict(schema.GroupResource{Resource: "services"}, "endpoint", errors.New("stale resourceVersion"))},
		{name: "denied Service write remains an error", failure: apierrors.NewForbidden(schema.GroupResource{Resource: "services"}, "endpoint", errors.New("denied"))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			t.Log("Materialize an LPX workload and its Kubernetes discovery Service before scheduler publication")
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			dgd.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			createLPXTestObjects(t, ctx, r.Client, objects...)
			workloads, plans, err := r.resolveWorkloads(ctx, child, dgd)
			require.NoError(t, err)
			_, resources, err := r.renderPodCliqueSet(ctx, child, dgd, workloads, plans, nil)
			require.NoError(t, err)
			require.NoError(t, r.reconcileRuntimeResources(ctx, child, resources))
			service := &corev1.Service{}
			key := client.ObjectKey{Namespace: child.Namespace, Name: dynamo.PCSNameForLPX(child) + "-serve"}
			require.NoError(t, r.Get(ctx, key, service))
			delete(service.Labels, consts.KubeLabelDynamoComponentType)
			require.NoError(t, r.Update(ctx, service))
			before := service.DeepCopy()

			t.Log("Refuse only the identity Apply while the complete LPX controller reconciles")
			failure, attempts := tc.failure, 0
			base := r.Client.(client.WithWatch)
			r.Client = interceptor.NewClient(base, interceptor.Funcs{Apply: func(ctx context.Context, delegated client.WithWatch, configuration runtime.ApplyConfiguration, opts ...client.ApplyOption) error {
				if _, ok := configuration.(*corev1apply.ServiceApplyConfiguration); ok {
					attempts++
					if failure != nil {
						return failure
					}
				}
				return delegated.Apply(ctx, configuration, opts...)
			}})
			request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			_, err = r.Reconcile(ctx, request)
			require.Positive(t, attempts, "the test must reach the identity Apply")
			require.NoError(t, base.Get(ctx, request.NamespacedName, child))
			ready := meta.FindStatusCondition(child.Status.Conditions, "Ready")
			require.NotNil(t, ready)
			if tc.pending {
				require.NoError(t, err)
				require.Equal(t, v1alpha1.LPXReadyReasonPending, ready.Reason)
				require.Contains(t, ready.Message, "cache observation")
			} else {
				require.True(t, apierrors.IsForbidden(err), "%v", err)
				require.Equal(t, v1alpha1.LPXReadyReasonFailed, ready.Reason)
			}
			requests := &lpxv1alpha1.LPUPipelineRequestList{}
			require.NoError(t, base.List(ctx, requests, client.InNamespace(child.Namespace)))
			require.Empty(t, requests.Items, "configuration must converge before scheduler publication")

			t.Log("Retry the fresh observation and repair identity without altering the serving endpoint")
			failure = nil
			_, err = r.Reconcile(ctx, request)
			require.NoError(t, err)
			require.NoError(t, base.Get(ctx, key, service))
			require.Equal(t, dgd.Name, service.Labels[consts.KubeLabelDynamoGraphDeploymentName])
			require.Equal(t, before.Spec, service.Spec)
			require.Equal(t, before.OwnerReferences, service.OwnerReferences)
			require.Contains(t, service.Labels, consts.KubeLabelDynamoComponentType)
			require.NoError(t, base.List(ctx, requests, client.InNamespace(child.Namespace)))
			require.Len(t, requests.Items, len(selected.requests), "the retry resumes scheduler publication")
		})
	}
}
