// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestLPXReplicaUpdatesUseScaleSubresource(t *testing.T) {
	conflict := apierrors.NewConflict(consts.PodCliqueScalingGroupGVR.GroupResource(), "group", errors.New("stale version"))
	for _, tc := range []struct {
		name        string
		replicas    *int32
		seed        int32
		err         error
		wantUpdates int
	}{
		{name: "scale out", replicas: ptr.To(int32(12)), wantUpdates: 1},
		{name: "scale in", replicas: ptr.To(int32(2)), wantUpdates: 1},
		{name: "unchanged", replicas: ptr.To(int32(9))},
		{name: "omitted scale out", seed: 3},
		{name: "omitted scale in", seed: 12},
		{name: "conflict", replicas: ptr.To(int32(12)), err: conflict, wantUpdates: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Observe a Grove group with nine replicas")
			child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			lpx.ServingComponent(source).Replicas = tc.replicas
			r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, source)
			objects := lpxMaterializedObjects(t, r, child, source, selected)
			pcs := findLPXTestPodCliqueSet(t, objects)
			if tc.replicas == nil {
				pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas = ptr.To(tc.seed)
			}
			hash, err := commoncontroller.GetSpecHash(pcs, commoncontroller.WithPreservedListOrder())
			require.NoError(t, err)
			metav1.SetMetaDataAnnotation(&pcs.ObjectMeta, commoncontroller.NvidiaAnnotationHashKey, hash)
			metav1.SetMetaDataAnnotation(&pcs.ObjectMeta, commoncontroller.NvidiaAnnotationGenerationKey, "1")
			group := findLPXTestScalingGroup(t, objects, selected.plan.LPXScalingGroup)
			group.Spec.Replicas = 9
			createLPXTestObjects(t, t.Context(), r.Client, objects...)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(group), group))
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
			beforePCS := pcs.DeepCopy()

			t.Log("Rebuild scheduler intent from explicit replicas or the current native scale")
			selected, rejected := requirePreparedLPX(t, r, t.Context(), child, source)
			require.Nil(t, rejected)
			require.Equal(t, ptr.Deref(tc.replicas, int32(9)), selected.plan.Replicas)
			require.Len(t, selected.requests, int(selected.plan.Replicas))
			updates := 0
			r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
					require.Equal(t, "scale", subresource)
					require.Equal(t, client.ObjectKeyFromObject(group), client.ObjectKeyFromObject(object))
					options := (&client.SubResourceUpdateOptions{}).ApplyOptions(opts)
					scale := options.SubResourceBody.(*autoscalingv1.Scale)
					require.Equal(t, group.ResourceVersion, scale.ResourceVersion)
					require.Equal(t, *tc.replicas, scale.Spec.Replicas)
					updates++
					if tc.err != nil {
						return tc.err
					}
					return delegated.SubResource(subresource).Update(ctx, object, opts...)
				},
			})

			t.Log("Only explicit changes write scale, preserving the observed resource-version precondition")
			_, _, err = r.reconcileWorkload(t.Context(), child, source, selected)
			require.ErrorIs(t, err, tc.err)
			require.Equal(t, tc.wantUpdates, updates)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(group), group))
			want := int32(9)
			if tc.replicas != nil && tc.err == nil {
				want = *tc.replicas
			}
			require.Equal(t, want, group.Spec.Replicas)

			t.Log("Native scale never rewrites the parent PCS seed or its bookkeeping")
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
			require.Equal(t, beforePCS, pcs)
		})
	}
}

func TestLPXScaleDownUpdatesGroveBeforeDeletingStaleRequest(t *testing.T) {
	ctx := t.Context()
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	lpx.ServingComponent(source).Replicas = ptr.To(int32(2))
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
	objects := lpxMaterializedObjects(t, r, child, source, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)

	stale := getLPXRequest(t, ctx, r.Client, child.Namespace, selected.requests[1].requestName)
	stale.Finalizers = []string{"scheduling.lpu.nvidia.com/plan-protection"}
	require.NoError(t, r.Update(ctx, stale))

	liveSource := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(source), liveSource))
	lpx.ServingComponent(liveSource).Replicas = ptr.To(int32(1))
	liveSource.Generation++
	require.NoError(t, r.Update(ctx, liveSource))
	liveChild := &v1alpha1.LPXGraphDeployment{}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), liveChild))
	liveChild.Generation++
	var err error
	liveChild.Spec.InputRevision, err = dynamo.LPXInputRevision(liveSource, "")
	require.NoError(t, err)
	require.NoError(t, r.Update(ctx, liveChild))

	t.Log("Persist the lower Grove scale before request retirement can block reconciliation")
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	_, err = r.Reconcile(ctx, request)
	require.NoError(t, err)
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: child.Namespace, Name: selected.plan.LPXScalingGroup}, group))
	require.Equal(t, int32(1), group.Spec.Replicas)
	stale = getLPXRequest(t, ctx, r.Client, stale.Namespace, stale.Name)
	require.True(t, stale.DeletionTimestamp.IsZero())

	t.Log("Delete the stale request while its scheduler finalizer remains pending")
	_, err = r.Reconcile(ctx, request)
	require.NoError(t, err)
	stale = getLPXRequest(t, ctx, r.Client, stale.Namespace, stale.Name)
	require.False(t, stale.DeletionTimestamp.IsZero())
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(1), group.Spec.Replicas)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(findLPXTestPodCliqueSet(t, objects)), &grovev1alpha1.PodCliqueSet{}))

	t.Log("Reverse the scale-down while the old request still protects its pods")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(source), liveSource))
	lpx.ServingComponent(liveSource).Replicas = ptr.To(int32(2))
	liveSource.Generation++
	require.NoError(t, r.Update(ctx, liveSource))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), liveChild))
	liveChild.Generation++
	liveChild.Spec.InputRevision, err = dynamo.LPXInputRevision(liveSource, "")
	require.NoError(t, err)
	require.NoError(t, r.Update(ctx, liveChild))
	for range 2 {
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
		require.Equal(t, int32(1), group.Spec.Replicas, "Grove must not recreate pods protected by the retiring request")
	}

	t.Log("Allow scale-up and a fresh request once the scheduler finishes cleanup")
	stale = getLPXRequest(t, ctx, r.Client, stale.Namespace, stale.Name)
	retiredUID := stale.UID
	stale.Finalizers = nil
	require.NoError(t, r.Update(ctx, stale))
	for range 4 {
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
	}
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(2), group.Spec.Replicas)
	replacement := getLPXRequest(t, ctx, r.Client, stale.Namespace, stale.Name)
	require.NotEqual(t, retiredUID, replacement.UID)
	require.True(t, replacement.DeletionTimestamp.IsZero())
}

func TestLPXFailedScaleOutPreservesServingEngines(t *testing.T) {
	for _, tc := range []struct {
		name    string
		missing bool
	}{{name: "deadline"}, {name: "missing request", missing: true}} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Observe a completed engine and a newer scheduling batch in their shared PCS")
			ctx := t.Context()
			child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			lpx.ServingComponent(source).Replicas = ptr.To(int32(2))
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, source)
			objects := lpxMaterializedObjects(t, r, child, source, selected)
			createLPXTestObjects(t, ctx, r.Client, objects...)
			publishSelectedLPXForTest(t, ctx, r, child, selected)
			pcs := findLPXTestPodCliqueSet(t, objects)
			serving := getLPXRequest(t, ctx, r.Client, child.Namespace, selected.requests[0].requestName)
			serving.Status = deadlineTestRequest(child, pcs, serving.Name, time.Now(), lpxv1alpha1.RequestPhaseBound).Status
			require.NoError(t, r.Update(ctx, serving))
			pending := getLPXRequest(t, ctx, r.Client, child.Namespace, selected.requests[1].requestName)
			pending.Finalizers = []string{"scheduling.lpu.nvidia.com/plan-protection"}
			require.NoError(t, r.Update(ctx, pending))
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
			attempt := deadlineTestAttempt(child, pending, time.Now().Add(-time.Second))
			if tc.missing {
				attempt.DeadlineAt = nil
				pending.Finalizers = nil
				require.NoError(t, r.Update(ctx, pending))
				require.NoError(t, r.Delete(ctx, pending))
			} else {
				attempt.ExceededAt = ptr.To(metav1.Now())
			}
			child.Status.Placement.LPXAttempt = attempt
			require.NoError(t, r.Status().Update(ctx, child))

			t.Log("Retire only the failed scale-out and keep Grove below the authored scale while cleanup is pending")
			request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			group := findLPXTestScalingGroup(t, objects, selected.plan.LPXScalingGroup)
			for range 3 {
				_, err := r.Reconcile(ctx, request)
				require.NoError(t, err)
				require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), &grovev1alpha1.PodCliqueSet{}))
				require.Equal(t, serving, getLPXRequest(t, ctx, r.Client, serving.Namespace, serving.Name))
				require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
				require.Equal(t, int32(1), group.Spec.Replicas)
			}
			if tc.missing {
				requireLPXRequestNotFound(t, ctx, r.Client, pending.Namespace, pending.Name)
				return
			}

			t.Log("A later edit still waits for the failed request's scheduler finalizer")
			pending = getLPXRequest(t, ctx, r.Client, pending.Namespace, pending.Name)
			require.False(t, pending.DeletionTimestamp.IsZero())
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
			child.Generation++
			require.NoError(t, r.Update(ctx, child))
			_, err := r.Reconcile(ctx, request)
			require.NoError(t, err)
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
			require.Equal(t, int32(1), group.Spec.Replicas)

			t.Log("After cleanup the later edit can retry without replacing the serving engine")
			retiredUID := pending.UID
			pending.Finalizers = nil
			require.NoError(t, r.Update(ctx, pending))
			for range 5 {
				_, err = r.Reconcile(ctx, request)
				require.NoError(t, err)
			}
			replacement := getLPXRequest(t, ctx, r.Client, pending.Namespace, pending.Name)
			require.NotEqual(t, retiredUID, replacement.UID)
			require.True(t, replacement.DeletionTimestamp.IsZero())
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(group), group))
			require.Equal(t, int32(2), group.Spec.Replicas)
			require.Equal(t, serving, getLPXRequest(t, ctx, r.Client, serving.Namespace, serving.Name))
		})
	}
}

func TestLPXEngineOrderAndUnrelatedEditsPreservePublication(t *testing.T) {
	t.Log("Publish a speculative engine using its frozen child identity")
	child, source, registry := newLPXSpecDecodeTestDGD(t)
	source.Spec.Components = append(source.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "frontend:old"}}}},
	}, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "prefill:old"}}}},
	})
	r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, source)
	createLPXTestObjects(t, t.Context(), r.Client, lpxMaterializedObjects(t, r, child, source, selected)...)
	publishSelectedLPXForTest(t, t.Context(), r, child, selected)
	beforeRequests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.NoError(t, r.List(t.Context(), beforeRequests))
	require.NotEmpty(t, beforeRequests.Items)
	beforePCS := renderLPXTestPodCliqueSet(t, t.Context(), r, child, source, selected)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	beforeChild := child.DeepCopy()

	t.Log("Reorder engines, update the frontend and enable an ordinary checkpoint without changing LPX")
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(source), source))
	components := source.Spec.Components
	components[0], components[1] = components[1], components[0]
	source.GetComponentByName("frontend").PodTemplate.Spec.Containers[0].Image = "frontend:next"
	source.GetComponentByName("prefill").Experimental = &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{
		Enabled: true, CheckpointRef: ptr.To("prefill-checkpoint"),
	}}
	source.Generation++
	require.NoError(t, r.Update(t.Context(), source))
	beforeSource := source.DeepCopy()
	require.Empty(t, lpx.ValidateSelectedIntent(source))
	require.NoError(t, dynamo.ValidateLPXSource(child, source))
	afterSelected, rejected := requirePreparedLPX(t, r, t.Context(), child, source)
	require.Nil(t, rejected)
	afterPCS := renderLPXTestPodCliqueSet(t, t.Context(), r, child, source, afterSelected)
	require.Equal(t, beforePCS, afterPCS)
	require.NotContains(t, afterPCS.Annotations, lpx.DGDGenerationAnnotation)
	_, err := reconcileSelectedLPXForTest(t.Context(), r, child, afterSelected)
	require.NoError(t, err)
	afterRequests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.NoError(t, r.List(t.Context(), afterRequests))
	require.Equal(t, beforeRequests, afterRequests)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	require.Equal(t, beforeChild, child)
	require.Equal(t, beforeSource, source)
}

func TestLPXCurrentAttemptSurvivesUnobservedReconciliation(t *testing.T) {
	t.Log("A failed reconciliation may have a current durable attempt and an older result generation")
	source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
	child := newLPXTestDeployment(t, source)
	source.Spec.Scheduling = deadlineTestScheduling()
	child.Status.Placement = &v1beta1.PlacementStatus{LPXAttempt: &v1beta1.LPXAttemptStatus{
		ObservedGeneration: child.Generation,
		DeadlineAt:         ptr.To(metav1.NewTime(time.Now().UTC().Truncate(time.Second).Add(time.Minute))),
		Requests:           []v1beta1.LPXAttemptRequestStatus{{Name: "current-request", UID: "current-uid"}},
	}}
	r := newLPXTestReconciler(t, nil, child, source)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	before := child.Status.Placement.DeepCopy()
	readError := errors.New("temporary request observation failure")
	r.apiReader = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			if _, requests := list.(*lpxv1alpha1.LPUPipelineRequestList); requests {
				return readError
			}
			return delegated.List(ctx, list, opts...)
		},
	})
	result, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err, "active deadlines bound transient errors with a scheduled retry")
	require.Positive(t, result.RequeueAfter)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	require.NotEqual(t, child.Generation, child.Status.ObservedGeneration)
	require.Equal(t, before, child.Status.Placement)
	require.Contains(t, meta.FindStatusCondition(child.Status.Conditions, "Failed").Message, readError.Error())
}

func TestLPXGPUCapacityReportsTheCompleteEngine(t *testing.T) {
	for _, mode := range []string{"lpu-only", "hybrid", "speculative"} {
		t.Run(mode, func(t *testing.T) {
			t.Log("Reconcile the actual LPX workload and publish its per-engine GPU shape")
			var child *v1alpha1.LPXGraphDeployment
			var source *v1beta1.DynamoGraphDeployment
			var registry lpx.ModelRegistry
			var want int64
			switch mode {
			case "lpu-only":
				child, source, registry = newLPXTestDGD(t, lpx.PipelineSingle)
			case "hybrid":
				child, source, registry = newLPXTestDGD(t, lpx.PipelineLPX)
				source.Spec.Components[0].Replicas = ptr.To(int32(3))
				source.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = ptr.To(int32(2))
				pod := &lpx.ServingComponent(source).ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate.Spec
				pod.ResourceClaims = nil
				pod.Containers[0].Resources.Limits = corev1.ResourceList{consts.KubeResourceGPUNvidia: resource.MustParse("4")}
				want = 8
			case "speculative":
				child, source, registry = newLPXSpecDecodeTestDGD(t)
			}
			r := newLPXTestReconciler(t, registry, child, source)
			_, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
			require.NoError(t, err)
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
			require.Contains(t, child.Status.Components, "lpx")
			require.Equal(t, ptr.To(want), child.Status.Components["lpx"].GPUsPerEngine)
			require.Equal(t, ptr.To(want), child.Status.Components["lpx"].GPUsPerReplica)

		})
	}
}

func TestLPXInvalidEditsPreserveExistingWorkload(t *testing.T) {
	for _, scenario := range []struct {
		name, reason, message string
		preserve              bool
	}{
		{name: "intent", reason: "LPXRejected", message: "providerOverride"},
		{name: "selected workload", reason: "LPXRejected", message: "replica count"},
		{name: "render", reason: "LPXReconciliationFailed", message: "model storage volume mount"},
		{name: "invalid source", reason: "LPXReconciliationFailed", message: "source"},
		{name: "transient snapshot", reason: "LPXReconciliationFailed", message: "temporary snapshot timeout", preserve: true},
		{name: "inconsistent snapshot", reason: "LPXReconciliationFailed", message: "immutable LPX build snapshot"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Stage a PCS and discovery endpoint before the API supplies a PCS UID")
			child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			source.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
			r := newLPXTestReconciler(t, registry, child, source)
			request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			_, err := r.Reconcile(t.Context(), request)
			require.NoError(t, err)
			pcs := &grovev1alpha1.PodCliqueSet{}
			pcsKey := client.ObjectKey{Namespace: child.Namespace, Name: dynamo.PCSNameForLPX(child)}
			require.NoError(t, r.Get(t.Context(), pcsKey, pcs))
			pcs.UID = "staged-pcs"
			require.NoError(t, r.Update(t.Context(), pcs))
			endpoint := &corev1.Service{}
			endpointKey := client.ObjectKey{Namespace: child.Namespace, Name: pcsKey.Name + "-serve"}
			require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
			endpoint.UID = "staged-endpoint"
			require.NoError(t, r.Update(t.Context(), endpoint))
			requests := &lpxv1alpha1.LPUPipelineRequestList{}
			require.NoError(t, r.List(t.Context(), requests))
			require.Empty(t, requests.Items)

			beforePCS, beforeEndpoint := pcs.DeepCopy(), endpoint.DeepCopy()

			t.Log("Introduce a current terminal input failure, or a transient snapshot outage")
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(source), source))
			require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
			component := lpx.ServingComponent(source)
			switch scenario.name {
			case "intent":
				source.Spec.ProviderOverride = &v1beta1.ProviderOverride{Target: "PodCliqueSet"}
			case "selected workload":
				component.Replicas = ptr.To(int32(-1))
			case "render":
				component.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.Containers[0].VolumeMounts = nil
			case "invalid source":
				child.OwnerReferences[0].UID = "replaced-source"
			case "transient snapshot", "inconsistent snapshot":
				cause := errors.New("temporary snapshot timeout")
				if !scenario.preserve {
					cause = lpx.ErrBuildSnapshotInconsistent
				}
				r.modelRegistry = &snapshotFailureRegistry{ModelRegistry: registry, err: cause}
			}
			source.Generation++
			require.NoError(t, r.Update(t.Context(), source))
			child.Generation++
			child.Spec.InputRevision, err = dynamo.LPXInputRevision(source, "")
			require.NoError(t, err)
			require.NoError(t, r.Update(t.Context(), child))

			t.Log("Report the actionable failure without destroying existing resources")
			for range 3 {
				_, err = r.Reconcile(t.Context(), request)
				if scenario.reason == "LPXReconciliationFailed" && err != nil {
					require.ErrorContains(t, err, scenario.message)
				} else {
					require.NoError(t, err)
				}
				require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
				require.Contains(t, meta.FindStatusCondition(child.Status.Conditions, "Ready").Message, scenario.message)
			}
			require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
			failure := meta.FindStatusCondition(child.Status.Conditions, "Failed")
			require.NotNil(t, failure)
			require.Equal(t, metav1.ConditionTrue, failure.Status)
			require.Equal(t, scenario.reason, failure.Reason)
			require.Contains(t, failure.Message, scenario.message)
			require.NoError(t, r.Get(t.Context(), pcsKey, pcs))
			require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
			require.Equal(t, beforePCS, pcs)
			require.Equal(t, beforeEndpoint, endpoint)
			require.NoError(t, r.List(t.Context(), requests))
			require.Empty(t, requests.Items)
		})
	}
}

func TestLPXTerminalCleanupPreservesForeignObjectsAndNewerAuthority(t *testing.T) {
	const replacedEndpoint = "replaced endpoint"
	for _, scenario := range []string{"cleanup", "new child generation", replacedEndpoint, "updated PCS", "pending finalizer"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Seed obsolete owned staging beside ordinary and foreign objects at current target names")
			source := newLPXTestSource(lpx.PipelineSingle, "build-v2")
			child := newLPXTestDeployment(t, source)
			owner := []metav1.OwnerReference{*metav1.NewControllerRef(child, v1alpha1.LPXGraphDeploymentGVK)}
			labels := map[string]string{lpxOwnerUIDLabel: string(child.UID)}
			pcs := &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{
				Name: "obsolete-pcs", Namespace: child.Namespace, UID: "old-pcs", Labels: labels, OwnerReferences: owner,
			}, Spec: grovev1alpha1.PodCliqueSetSpec{Replicas: 1}}
			endpoint := &corev1.Service{ObjectMeta: metav1.ObjectMeta{
				Name: "obsolete-endpoint", Namespace: child.Namespace, UID: "old-endpoint", Labels: labels, OwnerReferences: owner,
			}}
			runtimeConfig := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
				Name: "obsolete-runtime", Namespace: child.Namespace, UID: "old-runtime", Labels: labels, OwnerReferences: owner,
			}}
			ordinary := &corev1.Service{ObjectMeta: metav1.ObjectMeta{
				Name: "ordinary-model", Namespace: child.Namespace, UID: "ordinary-service",
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(source, v1beta1.DynamoGraphDeploymentGVK)},
			}}
			foreignPCS, foreignEndpoint := pcs.DeepCopy(), endpoint.DeepCopy()
			foreignRuntime := runtimeConfig.DeepCopy()
			foreignPCS.UID, foreignEndpoint.UID = "foreign-pcs", "foreign-endpoint"
			foreignRuntime.UID = "foreign-runtime"
			foreignPCS.Name = dynamo.PCSNameForLPX(child)
			foreignEndpoint.Name = foreignPCS.Name + "-serve"
			foreignRuntime.Name = foreignPCS.Name + "-runtime"
			foreignPCS.OwnerReferences, foreignEndpoint.OwnerReferences = ordinary.OwnerReferences, ordinary.OwnerReferences
			foreignRuntime.OwnerReferences = ordinary.OwnerReferences
			if scenario == "pending finalizer" {
				pcs.Finalizers = []string{"example.com/staged-cleanup"}
			}
			r := newLPXTestReconciler(t, nil, child, source, pcs, endpoint, runtimeConfig, ordinary, foreignPCS, foreignEndpoint, foreignRuntime)
			for _, object := range []client.Object{ordinary, foreignPCS, foreignEndpoint, foreignRuntime} {
				require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(object), object))
			}

			t.Log("Race observed child or resource authority immediately before terminal deletion")
			deletes := 0
			wrapped := interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					if err := delegated.List(ctx, list, opts...); err != nil {
						return err
					}
					if _, cliques := list.(*grovev1alpha1.PodCliqueSetList); cliques && scenario == "new child generation" {
						current := &v1alpha1.LPXGraphDeployment{}
						require.NoError(t, delegated.Get(ctx, client.ObjectKeyFromObject(child), current))
						current.Generation++
						return delegated.Update(ctx, current)
					}
					return nil
				},
				Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
					deletes++
					preconditions := (&client.DeleteOptions{}).ApplyOptions(opts).Preconditions
					require.NotNil(t, preconditions)
					require.Equal(t, object.GetUID(), *preconditions.UID)
					require.Equal(t, object.GetResourceVersion(), *preconditions.ResourceVersion)
					if scenario == replacedEndpoint && object.GetName() == endpoint.Name || scenario == "updated PCS" && object.GetName() == pcs.Name {
						current := object.DeepCopyObject().(client.Object)
						require.NoError(t, delegated.Get(ctx, client.ObjectKeyFromObject(object), current))
						if scenario == replacedEndpoint {
							current.SetUID("replacement-endpoint")
						}
						current.SetLabels(map[string]string{"example.com/newer": "true"})
						require.NoError(t, delegated.Update(ctx, current))
					}
					return delegated.Delete(ctx, object, opts...)
				},
			})
			r.Client, r.apiReader = wrapped, wrapped
			retiring, err := r.retireInvalidLPXWorkload(t.Context(), child, "invalid runtime")
			switch scenario {
			case "new child generation":
				require.ErrorContains(t, err, "authority changed")
				require.Zero(t, deletes)
				require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), &grovev1alpha1.PodCliqueSet{}))
				require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(endpoint), &corev1.Service{}))
			case replacedEndpoint, "updated PCS":
				require.True(t, apierrors.IsConflict(err), "object resource-version precondition must reject the stale deletion: %v", err)
				object := client.Object(pcs)
				if scenario == replacedEndpoint {
					object = endpoint
				}
				require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(object), object))
				require.Equal(t, "true", object.GetLabels()["example.com/newer"])
			default:
				require.NoError(t, err)
				require.NotNil(t, retiring)
				if scenario == "pending finalizer" {
					require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
					require.False(t, pcs.DeletionTimestamp.IsZero())
					retiring, err = r.retireInvalidLPXWorkload(t.Context(), child, "invalid runtime")
					require.NoError(t, err)
					require.NotNil(t, retiring)
					pcs.Finalizers = nil
					require.NoError(t, r.Update(t.Context(), pcs))
				}
				retiring, err = r.retireInvalidLPXWorkload(t.Context(), child, "invalid runtime")
				require.NoError(t, err)
				require.Nil(t, retiring)
				require.Equal(t, 3, deletes, "cleanup must not repeat deletes after they are accepted")
			}

			t.Log("Ordinary DGD resources and foreign same-name objects remain byte-for-byte unchanged")
			for _, object := range []client.Object{ordinary, foreignPCS, foreignEndpoint, foreignRuntime} {
				before := object.DeepCopyObject().(client.Object)
				require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(object), object))
				require.Equal(t, before, object)
			}
		})
	}
}

func TestLPXEndpointLifecycle(t *testing.T) {
	const uppercaseComponentName = "LPX"
	t.Log("Configure an uppercase LPX component and independently named materialization using Kubernetes discovery")
	source := newLPXTestSource(lpx.PipelineSingle, "test-build")
	source.Spec.Components[0].ComponentName = uppercaseComponentName
	source.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	source.Spec.Components[0].ModelRef = &v1beta1.ModelReference{Name: "test/model"}
	source.Spec.Annotations = map[string]string{"example.com/model-discovery": "enabled"}
	source.Spec.Labels = map[string]string{"example.com/policy": "enabled"}
	child := newLPXTestDeployment(t, source)
	child.Name = "independent-materialization"
	r := newLPXTestReconciler(t, nil, child, source)

	t.Log("Keep the ordinary model Service under the source DGD's ownership")
	modelService := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: dynamo.GenerateServiceName("test/model"), Namespace: source.Namespace,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(source, v1beta1.DynamoGraphDeploymentGVK)},
			Labels:          map[string]string{consts.KubeLabelDynamoGraphDeploymentName: source.Name},
		},
		Spec: corev1.ServiceSpec{ClusterIP: corev1.ClusterIPNone,
			Selector: map[string]string{consts.KubeLabelDynamoBaseModelHash: dynamo.HashModelName("test/model")}},
	}
	require.NoError(t, r.Create(t.Context(), modelService))
	modelKey := client.ObjectKeyFromObject(modelService)
	require.NoError(t, r.Get(t.Context(), modelKey, modelService))
	beforeModelService := modelService.DeepCopy()

	t.Log("Publish the LPX-owned endpoint with its serving-role selector")
	require.NoError(t, r.reconcileEndpoint(t.Context(), child, source))
	service := &corev1.Service{}
	key := client.ObjectKey{Namespace: source.Namespace, Name: dynamo.PCSNameForLPX(child) + "-serve"}
	require.NoError(t, r.Get(t.Context(), key, service))
	require.True(t, metav1.IsControlledBy(service, child))
	require.Equal(t, consts.KubeLabelValueTrue, service.Spec.Selector[dynamo.LPXServingLabel])
	require.Equal(t, dynamo.PCSNameForLPX(child), service.Spec.Selector[grovecommon.LabelPartOfKey])

	t.Log("Converge propagated endpoint metadata without losing identity, content or sync bookkeeping")
	for _, value := range []string{"changed", ""} {
		if value == "" {
			source.Spec.Labels, source.Spec.Annotations = nil, nil
		} else {
			source.Spec.Labels["example.com/policy"] = value
			source.Spec.Annotations["example.com/model-discovery"] = value
		}
		require.NoError(t, r.reconcileEndpoint(t.Context(), child, source))
		updated := &corev1.Service{}
		require.NoError(t, r.Get(t.Context(), key, updated))
		require.Equal(t, value, updated.Labels["example.com/policy"])
		require.Equal(t, value, updated.Annotations["example.com/model-discovery"])
		require.Equal(t, service.Labels[consts.KubeLabelDynamoDiscoveryEnabled], updated.Labels[consts.KubeLabelDynamoDiscoveryEnabled])
		if value == "" {
			require.NotContains(t, updated.Labels, "example.com/policy")
			require.NotContains(t, updated.Annotations, "example.com/model-discovery")
		}
		require.Equal(t, service.Spec, updated.Spec)
		require.Equal(t, service.OwnerReferences, updated.OwnerReferences)
		for _, key := range []string{commoncontroller.NvidiaAnnotationHashKey, commoncontroller.NvidiaAnnotationGenerationKey} {
			require.NotEmpty(t, updated.Annotations[key])
			require.Equal(t, service.Annotations[key], updated.Annotations[key])
		}
		require.NoError(t, r.reconcileEndpoint(t.Context(), child, source))
		unchanged := &corev1.Service{}
		require.NoError(t, r.Get(t.Context(), key, unchanged))
		require.Equal(t, updated.ResourceVersion, unchanged.ResourceVersion)
	}

	t.Log("Remove the LPX-owned discovery endpoint when switching back to non-Kubernetes discovery")
	delete(source.Annotations, consts.KubeAnnotationDynamoDiscoveryBackend)
	require.NoError(t, r.reconcileEndpoint(t.Context(), child, source))
	require.True(t, apierrors.IsNotFound(r.Get(t.Context(), key, service)))

	t.Log("Retire the recreated discovery endpoint when the workload becomes invalid")
	source.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	require.NoError(t, r.reconcileEndpoint(t.Context(), child, source))
	_, err := r.retireInvalidLPXWorkload(t.Context(), child, "invalid source")
	require.NoError(t, err)
	require.True(t, apierrors.IsNotFound(r.Get(t.Context(), key, service)))
	require.NoError(t, r.Get(t.Context(), modelKey, modelService))
	require.Equal(t, beforeModelService, modelService, "LPX cleanup must leave the DGD-owned model Service unchanged")
}

func TestLPXMaterializationUsesOwnerSourceAndChildIdentity(t *testing.T) {
	t.Log("Render a materialization with a different name from its source DGD")
	_, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	source.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	child := newLPXTestDeployment(t, source)
	child.Name = "independent-materialization"
	child.UID = "independent-materialization-uid"
	r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, source)
	objects := lpxMaterializedObjects(t, r, child, source, selected)
	createLPXTestObjects(t, t.Context(), r.Client, objects...)
	publishSelectedLPXForTest(t, t.Context(), r, child, selected)

	t.Log("Route Grove objects to the materialization while retaining the source provenance")
	pcs := objects[0].(*grovev1alpha1.PodCliqueSet)
	require.Equal(t, source.Name, pcs.Labels[consts.KubeLabelDynamoGraphDeploymentName])
	for _, object := range objects[1:] {
		require.Equal(t, []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(child)}}, mapLPXChildToRequests(t.Context(), object))
	}

	t.Log("Publish the runtime configuration and endpoint under the exact LGD owner")
	_, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)

	root := dynamo.PCSNameForLPX(child)
	for _, object := range []client.Object{
		&grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: root, Namespace: child.Namespace}},
		&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: lpx.LPUConfigMapName(root, pcs.Spec.Template.Cliques[0].Annotations[consts.AnnotationExtraResourcesHash]), Namespace: child.Namespace}},
		&corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: root + "-serve", Namespace: child.Namespace}},
	} {
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(object), object))
		require.True(t, metav1.IsControlledBy(object, child))
	}

	t.Log("Keep scheduler-request provenance rooted in the source DGD, not the independently named child")
	requests, err := r.listOwnedLPXRequests(t.Context(), child)
	require.NoError(t, err)
	require.NotEmpty(t, requests)
	for _, request := range requests {
		require.Equal(t, source.Name, request.Labels[consts.KubeLabelDynamoGraphDeploymentName])
	}
}

func TestLPXReadableNamesSurviveServingComponentChanges(t *testing.T) {
	t.Log("Publish a speculative engine with long deployment and component names")
	_, source, registry := newLPXSpecDecodeTestDGD(t)
	source.Name = "test-models-gpt-oss-20b-lp20-b300"
	source.Spec.Components[0].ComponentName = "serving-component-with-name"
	source.Spec.Components[1].ComponentName = "draft-component-with-name"
	source.Spec.Components[1].Replicas = ptr.To(int32(1))
	source.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	child := newLPXTestDeployment(t, source)
	r := newLPXTestReconciler(t, registry, child, source)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	_, err := r.Reconcile(t.Context(), request)
	require.NoError(t, err)
	root := dynamo.PCSNameForLPX(child)
	require.True(t, strings.HasPrefix(root, source.Name+"-"))
	require.Len(t, root, len(source.Name)+len("-ffff"))
	pcs := &grovev1alpha1.PodCliqueSet{}
	pcsKey := client.ObjectKey{Namespace: child.Namespace, Name: root}
	require.NoError(t, r.Get(t.Context(), pcsKey, pcs))
	pcs.UID = "stable-pcs"
	require.NoError(t, r.Update(t.Context(), pcs))
	endpoint := &corev1.Service{}
	endpointKey := client.ObjectKey{Namespace: child.Namespace, Name: root + "-serve"}
	require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
	endpoint.UID = "stable-endpoint"
	require.NoError(t, r.Update(t.Context(), endpoint))
	require.Len(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, 1)
	beforeGroup := pcs.Spec.Template.PodCliqueScalingGroupConfigs[0]
	beforeOwners := endpoint.OwnerReferences

	for _, change := range []string{"rename serving component", "move conductor"} {
		t.Log(change)
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(source), source))
		require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
		if change == "rename serving component" {
			source.Spec.Components[0].ComponentName = "renamed-serving-component"
		} else {
			conductor := *source.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
			source.Spec.Components[0].Roles = slices.DeleteFunc(source.Spec.Components[0].Roles, func(role v1beta1.ComponentRoleSpec) bool {
				return role.Name == v1beta1.ComponentRoleLPXConductor
			})
			source.Spec.Components[1].Roles = append(source.Spec.Components[1].Roles, conductor)
		}
		source.Generation++
		require.NoError(t, r.Update(t.Context(), source))
		child.Generation++
		child.Spec.InputRevision, err = dynamo.LPXInputRevision(source, "")
		require.NoError(t, err)
		require.NoError(t, r.Update(t.Context(), child))

		t.Log("Keep one PCS and endpoint while updating their serving component metadata")
		_, err = r.Reconcile(t.Context(), request)
		require.NoError(t, err)
		require.NoError(t, r.Get(t.Context(), pcsKey, pcs))
		require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
		require.Equal(t, "stable-pcs", string(pcs.UID))
		require.Equal(t, "stable-endpoint", string(endpoint.UID))
		require.Len(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, 1)
		require.Equal(t, beforeGroup.Name, pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].Name)
		require.Equal(t, beforeGroup.CliqueNames, pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].CliqueNames)
		require.Equal(t, beforeOwners, endpoint.OwnerReferences)
		require.Equal(t, source.Name, pcs.Labels[consts.KubeLabelDynamoGraphDeploymentName])
		require.Equal(t, lpx.ServingComponent(source).ComponentName, endpoint.Spec.Selector[consts.KubeLabelDynamoComponent])
		require.Equal(t, root, endpoint.Spec.Selector[grovecommon.LabelPartOfKey])
		require.Equal(t, consts.KubeLabelValueTrue, endpoint.Spec.Selector[dynamo.LPXServingLabel])
		allPCS, allEndpoints := &grovev1alpha1.PodCliqueSetList{}, &corev1.ServiceList{}
		require.NoError(t, r.List(t.Context(), allPCS))
		require.NoError(t, r.List(t.Context(), allEndpoints))
		require.Len(t, allPCS.Items, 1)
		require.Len(t, allEndpoints.Items, 1)

		t.Log("Select only the conductor and share one runtime config across the clique roster")
		require.NotEmpty(t, pcs.Spec.Template.Cliques)
		configHash := pcs.Spec.Template.Cliques[0].Annotations[consts.AnnotationExtraResourcesHash]
		require.NotEmpty(t, configHash)
		cliqueNames := make([]string, 0, len(pcs.Spec.Template.Cliques))
		for _, clique := range pcs.Spec.Template.Cliques {
			cliqueNames = append(cliqueNames, clique.Name)
			require.Equal(t, configHash, clique.Annotations[consts.AnnotationExtraResourcesHash])
			for key, value := range clique.Labels {
				require.Empty(t, validation.IsValidLabelValue(value), key)
			}
			if clique.Name == "cond" {
				for key, value := range endpoint.Spec.Selector {
					if key != grovecommon.LabelPartOfKey {
						require.Equal(t, value, clique.Labels[key], key)
					}
				}
			} else {
				require.NotContains(t, clique.Labels, dynamo.LPXServingLabel)
			}
		}
		require.ElementsMatch(t, []string{"cond", "agt0", "agt1"}, cliqueNames)

		t.Log("Keep runtime Agent addresses aligned with the clique names")
		config := &corev1.ConfigMap{}
		configKey := client.ObjectKey{Namespace: child.Namespace, Name: lpx.LPUConfigMapName(root, configHash)}
		require.NoError(t, r.Get(t.Context(), configKey, config))
		require.True(t, metav1.IsControlledBy(config, child))
		require.Contains(t, config.Data["datacenter.toml"], "${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-agt0-{node}")
		require.Contains(t, config.Data["datacenter.toml"], "${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-agt1-{node}")
	}
}

func TestLPXResourceSyncConvergesContentAndMetadataWithoutAdoption(t *testing.T) {
	t.Log("Publish a runtime ConfigMap under the exact LPX child")
	source := newLPXTestSource(lpx.PipelineSingle, "test-build")
	child := newLPXTestDeployment(t, source)
	r := newLPXTestReconciler(t, nil, child, source)
	desired := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "lpx-runtime", Namespace: child.Namespace,
			Labels: map[string]string{"example.com/policy": "enabled"}, Annotations: map[string]string{"example.com/policy": "enabled"}},
		Data: map[string]string{"config": "initial runtime"},
	}
	require.NoError(t, r.syncLPXResource(t.Context(), child, desired))
	recorded := r.recorder.(*events.FakeRecorder).Events
	require.Len(t, recorded, 1)
	require.Equal(t, "Normal CreateConfigMap Created ConfigMap "+child.Namespace, <-recorded)
	key := client.ObjectKeyFromObject(desired)
	before := &corev1.ConfigMap{}
	require.NoError(t, r.Get(t.Context(), key, before))
	updates := 0
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		Update: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.UpdateOption) error {
			updates++
			return delegated.Update(ctx, object, opts...)
		},
	})

	t.Log("Change runtime content and remove all authored metadata while retaining refreshed bookkeeping")
	desired.Data["config"] = "updated runtime"
	desired.Labels, desired.Annotations = nil, nil
	desired.OwnerReferences = nil
	require.NoError(t, r.syncLPXResource(t.Context(), child, desired))
	require.Len(t, recorded, 1)
	require.Equal(t, "Normal UpdateConfigMap Updated ConfigMap "+child.Namespace, <-recorded)
	require.Equal(t, 1, updates, "content and metadata must converge in one resource update")
	updated := &corev1.ConfigMap{}
	require.NoError(t, r.Get(t.Context(), key, updated))
	require.Equal(t, desired.Data, updated.Data)
	require.Empty(t, updated.Labels)
	require.Len(t, updated.Annotations, 2)
	hash, err := commoncontroller.GetSpecHash(desired)
	require.NoError(t, err)
	require.Equal(t, hash, updated.Annotations[commoncontroller.NvidiaAnnotationHashKey])
	require.NotEqual(t, before.Annotations[commoncontroller.NvidiaAnnotationHashKey], hash)
	require.NotEmpty(t, updated.Annotations[commoncontroller.NvidiaAnnotationGenerationKey])
	require.Equal(t, before.OwnerReferences, updated.OwnerReferences)
	require.NoError(t, r.syncLPXResource(t.Context(), child, desired))
	require.Empty(t, recorded)
	unchanged := &corev1.ConfigMap{}
	require.NoError(t, r.Get(t.Context(), key, unchanged))
	require.Equal(t, updated.ResourceVersion, unchanged.ResourceVersion)

	t.Log("A foreign resource at the same key is never adopted or rewritten")
	updated.OwnerReferences[0].UID = "foreign-child"
	require.NoError(t, r.Update(t.Context(), updated))
	desired.Data["config"] = "must-not-publish"
	require.ErrorContains(t, r.syncLPXResource(t.Context(), child, desired), "refusing to adopt LPX resource")
	require.NoError(t, r.Get(t.Context(), key, unchanged))
	require.Equal(t, updated.ResourceVersion, unchanged.ResourceVersion)
	require.Equal(t, updated.Data, unchanged.Data)
}

func TestLPXDeletesOnlyStaleOwnedRuntimeConfigMaps(t *testing.T) {
	t.Log("Create current, stale, and foreign runtime ConfigMaps")
	source := newLPXTestSource(lpx.PipelineSingle, "test-build")
	child := newLPXTestDeployment(t, source)
	owner := []metav1.OwnerReference{*metav1.NewControllerRef(child, v1alpha1.LPXGraphDeploymentGVK)}
	current := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name: "runtime-current", Namespace: child.Namespace, UID: "current",
		Labels: map[string]string{lpxOwnerUIDLabel: string(child.UID)}, OwnerReferences: owner,
	}}
	stale := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name: "runtime-stale", Namespace: child.Namespace, UID: "stale",
		Labels: map[string]string{lpxOwnerUIDLabel: string(child.UID)}, OwnerReferences: owner,
	}}
	foreign := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name: "runtime-foreign", Namespace: child.Namespace, UID: "foreign",
		Labels:          map[string]string{lpxOwnerUIDLabel: string(child.UID)},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(source, v1beta1.DynamoGraphDeploymentGVK)},
	}}
	r := newLPXTestReconciler(t, nil, child, source, current, stale, foreign)

	t.Log("Delete only the stale ConfigMap owned by the current LPX child")
	require.NoError(t, r.deleteStaleLPXConfigMaps(t.Context(), child, []client.Object{current}))

	t.Log("Preserve the current and foreign ConfigMaps")
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(current), &corev1.ConfigMap{}))
	require.True(t, apierrors.IsNotFound(r.Get(t.Context(), client.ObjectKeyFromObject(stale), &corev1.ConfigMap{})))
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(foreign), &corev1.ConfigMap{}))
}

func TestLPXValidatesIntentBeforeDownloadsOrPublication(t *testing.T) {
	const componentProvider = "component provider"
	for _, scenario := range []struct {
		name, componentName string
		sharedDraft         bool
		missingConductor    string
		messages            []string
	}{
		{name: "invalid intent", messages: []string{
			"spec.providerOverride: Forbidden:",
			"spec.components[0].providerOverride: Forbidden:",
			`LPX agent component requires a "main" runtime container`,
			"does not support checkpointing",
			"spec.topologyConstraint: Forbidden:",
			"spec.components[0].topologyConstraint: Forbidden:",
		}},
		{name: "missing conductor role", missingConductor: "role", messages: []string{"LPX components must declare exactly one conductor role"}},
		{name: "missing conductor template", missingConductor: "podTemplate", messages: []string{"LPX conductor requires an explicit podTemplate"}},
		{name: "shared draft missing conductor role", sharedDraft: true, missingConductor: "role", messages: []string{"LPX components must declare exactly one conductor role"}},
		{name: "shared draft missing conductor template", sharedDraft: true, missingConductor: "podTemplate", messages: []string{"LPX conductor requires an explicit podTemplate"}},
		{name: "disabled Grove", messages: []string{"Grove is disabled"}},
		{name: componentProvider, messages: []string{"requires the Grove workload provider"}},
		{name: "long serving name", componentName: strings.Repeat("serving-", 7) + "engine"},
		{name: "reordered shared draft with long serving name", componentName: strings.Repeat("serving-", 7) + "engine", sharedDraft: true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Reconcile authored LPX intent against cold remote builds")
			source := newLPXTestSource(lpx.PipelineSingle, modelDownloadTestBuildID)
			if scenario.sharedDraft {
				source = newLPXSpecDecodeTestSource()
				source.Spec.Components[0].LPX.BuildID = modelDownloadTestBuildID
				source.Spec.Components[1].LPX.BuildID = modelDownloadTestSecondBuildID
				source.Spec.Components[1].ComponentName = "shared-draft-name-is-not-materialized"
				source.Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{
					{ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend},
					source.Spec.Components[1], source.Spec.Components[0],
				}
			}
			component := lpx.ServingComponent(source)
			if scenario.componentName != "" {
				component.ComponentName = scenario.componentName
			}
			switch scenario.missingConductor {
			case "role":
				component.Roles = slices.DeleteFunc(component.Roles, func(role v1beta1.ComponentRoleSpec) bool {
					return role.Name == v1beta1.ComponentRoleLPXConductor
				})
			case "podTemplate":
				component.ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate = nil
			}
			switch scenario.name {
			case "invalid intent":
				source.Spec.ProviderOverride = &v1beta1.ProviderOverride{
					APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Target: "PodCliqueSet",
				}
				source.Spec.ProviderOverride.Value.Raw = []byte(`{"spec":{"template":{"topologyConstraint":{"topologyName":"fabric","pack":{"required":"rack"}}}}}`)
				component.ProviderOverride = &v1beta1.ProviderOverride{
					APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Target: "PodCliqueTemplateSpec",
				}
				component.ProviderOverride.Value.Raw = []byte(`{"topologyConstraint":{"topologyName":"fabric","pack":{"required":"rack"}}}`)
				component.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate = &corev1.PodTemplateSpec{}
				component.Experimental = &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true}}
				source.Spec.TopologyConstraint = &v1beta1.SpecTopologyConstraint{ClusterTopologyName: "fabric"}
				component.TopologyConstraint = &v1beta1.TopologyConstraint{PackDomain: "rack"}
			case componentProvider:
				delete(source.Annotations, consts.KubeAnnotationLPXSchedulerBackend)
				source.Annotations[consts.KubeAnnotationWorkloadProvider] = consts.WorkloadProviderComponent
			}
			registry, err := lpx.NewModelRegistry("gs://bucket/registry", nil)
			require.NoError(t, err)
			observedRegistry := &fakeModelDownloadRegistry{ModelRegistry: registry}
			child := newLPXTestDeployment(t, source)
			r := newLPXTestReconciler(t, observedRegistry, child, source)
			if source.Spec.ProviderOverride != nil || component.ProviderOverride != nil {
				t.Log("Match the stored provider override after its raw JSON is normalized")
				require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(source), source))
				revision, err := dynamo.LPXInputRevision(source, "")
				require.NoError(t, err)
				child.Spec.InputRevision = revision
				require.NoError(t, r.Update(t.Context(), child))
			}
			if scenario.name == "disabled Grove" {
				r.runtimeConfig.Gate.Grove = false
			}

			t.Log("Reject unsupported intent before downloads and let valid names reach the download gate")
			before := source.DeepCopy()
			_, err = r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
			if scenario.name == componentProvider {
				require.ErrorContains(t, err, "requires the Grove workload provider")
			} else {
				require.NoError(t, err)
			}
			stored := &v1alpha1.LPXGraphDeployment{}
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), stored))
			failed := meta.FindStatusCondition(stored.Status.Conditions, "Failed")
			require.NotNil(t, failed)
			if len(scenario.messages) == 0 {
				require.Equal(t, metav1.ConditionFalse, failed.Status)
				require.Equal(t, modelDownloadPendingReason, failed.Reason)
				wantCalls := []string{modelDownloadTestBuildID}
				if scenario.sharedDraft {
					wantCalls = append(wantCalls, modelDownloadTestSecondBuildID)
				}
				require.Equal(t, wantCalls, observedRegistry.calls)
			} else {
				require.Equal(t, metav1.ConditionTrue, failed.Status)
				if scenario.name == "disabled Grove" {
					require.Equal(t, "LPXUnavailable", failed.Reason)
				} else if scenario.name == componentProvider {
					require.Equal(t, "LPXReconciliationFailed", failed.Reason)
				} else {
					require.Equal(t, "LPXRejected", failed.Reason)
				}
				for _, message := range scenario.messages {
					require.Contains(t, failed.Message, message)
				}
				require.Empty(t, observedRegistry.calls)
			}
			require.Zero(t, observedRegistry.acquireBuildSnapshotCalls)
			pcs := &grovev1alpha1.PodCliqueSetList{}
			requests := &lpxv1alpha1.LPUPipelineRequestList{}
			require.NoError(t, r.List(t.Context(), pcs))
			require.NoError(t, r.List(t.Context(), requests))
			require.Empty(t, pcs.Items)
			require.Empty(t, requests.Items)
			require.Equal(t, before, source)
		})
	}
}

func TestLPXPublicationFencesLiveSourceAndChildMetadata(t *testing.T) {
	t.Log("Prepare an immutable render from the current source and child")
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, source)
	require.NoError(t, r.validateLPXPublicationSource(t.Context(), child))

	t.Log("A source edit racing publication prevents the old render from creating a PCS")
	liveSource := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(source), liveSource))
	liveSource.Annotations[consts.KubeAnnotationEnableMetrics] = "false"
	require.NoError(t, r.Update(t.Context(), liveSource))
	_, _, err := r.reconcileWorkload(t.Context(), child, source, selected)
	require.ErrorContains(t, err, "input revision")
	allPCS := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, r.List(t.Context(), allPCS))
	require.Empty(t, allPCS.Items)

	t.Log("Child publication metadata cannot change behind an unchanged spec generation")
	liveChild := &v1alpha1.LPXGraphDeployment{}
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), liveChild))
	liveChild.Annotations[lpx.DGDGenerationAnnotation] = "999"
	require.NoError(t, r.Update(t.Context(), liveChild))
	require.ErrorContains(t, r.validateLPXPublicationSource(t.Context(), child), "authority changed")
}
