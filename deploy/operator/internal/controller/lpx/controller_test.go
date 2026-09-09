// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

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
	_, err := r.reconcileSelectedLPX(t.Context(), child, selected)
	require.NoError(t, err)
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
	require.Equal(t, child.Annotations[lpx.DGDGenerationAnnotation], afterPCS.Annotations[lpx.DGDGenerationAnnotation])
	_, err = r.reconcileSelectedLPX(t.Context(), child, afterSelected)
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
			var registry *lpx.ModelRegistry
			var want int64
			switch mode {
			case "lpu-only":
				child, source, registry = newLPXTestDGD(t, lpx.PipelineSingle)
			case "hybrid":
				child, source, registry = newLPXTestDGD(t, lpx.PipelineLPX)
				source.Spec.Components[0].Replicas = ptr.To(int32(3))
				source.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLeader).Replicas = ptr.To(int32(2))
				pod := &lpx.ServingComponent(source).ComponentRole(v1beta1.ComponentRoleLeader).PodTemplate.Spec
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

func TestLPXTerminalFailureRetiresUnpublishedWorkload(t *testing.T) {
	for _, scenario := range []struct {
		name, reason, message string
		preserve              bool
	}{
		{name: "intent", reason: "LPXRejected", message: "providerOverride"},
		{name: "name budget", reason: "LPXRejected", message: "spec.components[0].name: Invalid value"},
		{name: "selected workload", reason: "LPXRejected", message: "scaling-group replicas"},
		{name: "render", reason: "LPXReconciliationFailed", message: "model storage volume mount"},
		{name: "invalid source", reason: "LPXRejected", message: "source"},
		{name: "transient snapshot", reason: "LPXReconciliationFailed", message: "temporary snapshot timeout", preserve: true},
		{name: "inconsistent snapshot", reason: "LPXReconciliationFailed", message: "immutable LPX build snapshot"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Stage a real PCS and discovery endpoint before Grove can supply publication identities")
			child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			source.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
			r := newLPXTestReconciler(t, registry, child, source)
			request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			_, err := r.Reconcile(t.Context(), request)
			require.NoError(t, err)
			pcs := &grovev1alpha1.PodCliqueSet{}
			pcsKey := client.ObjectKey{Namespace: child.Namespace, Name: child.Annotations[dynamo.LPXPCSNameAnnotation]}
			require.NoError(t, r.Get(t.Context(), pcsKey, pcs))
			pcs.UID = "staged-pcs"
			require.NoError(t, r.Update(t.Context(), pcs))
			endpoint := &corev1.Service{}
			endpointKey := client.ObjectKey{Namespace: child.Namespace, Name: dynamo.GetDCDResourceName(source, "lpx", "")}
			require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
			endpoint.UID = "staged-endpoint"
			require.NoError(t, r.Update(t.Context(), endpoint))
			requests := &lpxv1alpha1.LPUPipelineRequestList{}
			require.NoError(t, r.List(t.Context(), requests))
			require.Empty(t, requests.Items)

			t.Log("Healthy repeated reconciliation must preserve unpublished workload identities")
			_, err = r.Reconcile(t.Context(), request)
			require.NoError(t, err)
			beforePCS, beforeEndpoint := pcs.DeepCopy(), endpoint.DeepCopy()
			require.NoError(t, r.Get(t.Context(), pcsKey, pcs))
			require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
			require.Equal(t, beforePCS, pcs)
			require.Equal(t, beforeEndpoint, endpoint)

			t.Log("Introduce a current terminal input failure, or a transient snapshot outage")
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(source), source))
			require.NoError(t, r.Get(t.Context(), request.NamespacedName, child))
			component := lpx.ServingComponent(source)
			switch scenario.name {
			case "intent":
				source.Spec.ProviderOverride = &v1beta1.ProviderOverride{Target: "PodCliqueSet"}
			case "name budget":
				component.ComponentName = "serving-engines"
				child.Annotations[dynamo.LPXPCSNameAnnotation] = dynamo.PCSNameForLPX(source)
			case "selected workload":
				component.Replicas = ptr.To(int32(2))
			case "render":
				component.ComponentRole(v1beta1.ComponentRoleWorker).PodTemplate.Spec.Containers[0].VolumeMounts = nil
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

			t.Log("Converge cleanup and retain the original actionable failure on repeated reconciliation")
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
			if scenario.preserve {
				require.NoError(t, r.Get(t.Context(), pcsKey, pcs))
				require.NoError(t, r.Get(t.Context(), endpointKey, endpoint))
				require.Equal(t, beforePCS, pcs)
				require.Equal(t, beforeEndpoint, endpoint)
			} else {
				require.True(t, apierrors.IsNotFound(r.Get(t.Context(), pcsKey, pcs)), "terminal failure retained the staged PCS")
				require.True(t, apierrors.IsNotFound(r.Get(t.Context(), endpointKey, endpoint)), "terminal failure retained the obsolete endpoint")
			}
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
			pcs := &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{
				Name: "obsolete-pcs", Namespace: child.Namespace, UID: "old-pcs", OwnerReferences: owner,
			}, Spec: grovev1alpha1.PodCliqueSetSpec{Replicas: 1}}
			endpoint := &corev1.Service{ObjectMeta: metav1.ObjectMeta{
				Name: "obsolete-endpoint", Namespace: child.Namespace, UID: "old-endpoint", OwnerReferences: owner,
			}}
			ordinary := &corev1.Service{ObjectMeta: metav1.ObjectMeta{
				Name: "ordinary-model", Namespace: child.Namespace, UID: "ordinary-service",
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(source, v1beta1.DynamoGraphDeploymentGVK)},
			}}
			foreignPCS, foreignEndpoint := pcs.DeepCopy(), endpoint.DeepCopy()
			foreignPCS.UID, foreignEndpoint.UID = "foreign-pcs", "foreign-endpoint"
			foreignPCS.Name = child.Annotations[dynamo.LPXPCSNameAnnotation]
			foreignEndpoint.Name = dynamo.GetDCDResourceName(source, "lpx", "")
			foreignPCS.OwnerReferences, foreignEndpoint.OwnerReferences = ordinary.OwnerReferences, ordinary.OwnerReferences
			if scenario == "pending finalizer" {
				pcs.Finalizers = []string{"example.com/staged-cleanup"}
			}
			r := newLPXTestReconciler(t, nil, child, source, pcs, endpoint, ordinary, foreignPCS, foreignEndpoint)
			for _, object := range []client.Object{ordinary, foreignPCS, foreignEndpoint} {
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
				require.Equal(t, 2, deletes, "cleanup must not repeat deletes after they are accepted")
			}

			t.Log("Ordinary DGD resources and foreign same-name objects remain byte-for-byte unchanged")
			for _, object := range []client.Object{ordinary, foreignPCS, foreignEndpoint} {
				before := object.DeepCopyObject().(client.Object)
				require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(object), object))
				require.Equal(t, before, object)
			}
		})
	}
}

func TestLPXEndpointLifecycle(t *testing.T) {
	t.Log("Configure an LPX serving component using Kubernetes discovery")
	source := newLPXTestSource(lpx.PipelineSingle, "test-build")
	source.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	source.Spec.Components[0].ModelRef = &v1beta1.ModelReference{Name: "test/model"}
	source.Spec.Annotations = map[string]string{"example.com/model-discovery": "enabled"}
	source.Spec.Labels = map[string]string{"example.com/policy": "enabled"}
	child := newLPXTestDeployment(t, source)
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
	key := client.ObjectKey{Namespace: source.Namespace, Name: dynamo.GetDCDResourceName(source, "lpx", "")}
	require.NoError(t, r.Get(t.Context(), key, service))
	require.True(t, metav1.IsControlledBy(service, child))
	require.Equal(t, consts.KubeLabelValueTrue, service.Spec.Selector[dynamo.LPXServingLabel])

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
	require.NoError(t, r.Get(t.Context(), modelKey, modelService))
	require.Equal(t, beforeModelService, modelService, "LPX cleanup must leave the DGD-owned model Service unchanged")
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

func TestLPXValidatesIntentBeforeDownloadsOrPublication(t *testing.T) {
	for _, scenario := range []struct {
		name, componentName string
		sharedDraft         bool
		messages            []string
	}{
		{name: "invalid intent", messages: []string{
			"spec.providerOverride: Forbidden:",
			"spec.components[0].providerOverride: Forbidden:",
			`LPX worker component requires a "main" runtime container`,
			"does not support checkpointing",
			"spec.topologyConstraint: Forbidden:",
			"spec.components[0].topologyConstraint: Forbidden:",
		}},
		{name: "disabled Grove", messages: []string{"Grove is disabled"}},
		{name: "component provider", messages: []string{"requires the Grove workload provider"}},
		{name: "too long serving name", componentName: "serving-engines", messages: []string{
			"spec.components[0].name: Invalid value", "combined Grove resource name length 46 exceeds the 45-character limit",
		}},
		{name: "reordered shared draft with too long serving name", componentName: "serving-engines", sharedDraft: true, messages: []string{
			"spec.components[2].name: Invalid value", "combined Grove resource name length 46 exceeds the 45-character limit",
		}},
		{name: "maximum legal serving name", componentName: "serving-engine"},
		{name: "reordered shared draft with maximum legal serving name", componentName: "serving-engine", sharedDraft: true},
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
				component.ComponentRole(v1beta1.ComponentRoleWorker).PodTemplate = &corev1.PodTemplateSpec{}
				component.Experimental = &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true}}
				source.Spec.TopologyConstraint = &v1beta1.SpecTopologyConstraint{ClusterTopologyName: "fabric"}
				component.TopologyConstraint = &v1beta1.TopologyConstraint{PackDomain: "rack"}
			case "component provider":
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
			_, err = r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
			require.NoError(t, err)
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
		})
	}
}

func TestLPXPublicationFencesLiveSourceAndChildMetadata(t *testing.T) {
	t.Log("Prepare an immutable render from the current source and child")
	child, source, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, source)
	pcs := renderLPXTestPodCliqueSet(t, t.Context(), r, child, source, selected)
	require.NoError(t, r.validateLPXPublicationSource(t.Context(), child))

	t.Log("A source edit racing publication prevents the old render from creating a PCS")
	liveSource := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(source), liveSource))
	liveSource.Annotations[consts.KubeAnnotationEnableMetrics] = "false"
	require.NoError(t, r.Update(t.Context(), liveSource))
	_, _, err := r.reconcileGrovePodCliqueSetForLPX(t.Context(), child, nil, pcs, selected)
	require.ErrorContains(t, err, "input revision")
	allPCS := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, r.List(t.Context(), allPCS))
	require.Empty(t, allPCS.Items)

	t.Log("Child publication metadata cannot change behind an unchanged spec generation")
	liveChild := &v1alpha1.LPXGraphDeployment{}
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), liveChild))
	liveChild.Annotations[lpx.DGDGenerationAnnotation] = "999"
	require.NoError(t, r.Update(t.Context(), liveChild))
	require.ErrorContains(t, r.validateLPXPublicationSource(t.Context(), child), "publication metadata changed")
}
