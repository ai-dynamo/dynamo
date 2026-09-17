/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"bytes"
	"context"
	"errors"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

func TestLPXDeploymentPrimaryPredicateTracksOwnerIdentity(t *testing.T) {
	t.Log("Build one current private handoff and its primary-resource predicate")
	deployment := &nvidiacomv1alpha1.LPXGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "graph", Namespace: "workloads", Generation: 1,
			OwnerReferences: []metav1.OwnerReference{{Name: "graph", UID: "source-uid", Controller: ptr.To(true)}},
		},
	}
	filter := lpxDeploymentPrimaryPredicate()

	t.Log("Ignore status-only updates that do not change publication authority")
	statusOnly := deployment.DeepCopy()
	statusOnly.Status.ObservedGeneration = 1
	require.False(t, filter.Update(event.UpdateEvent{ObjectOld: deployment, ObjectNew: statusOnly}))

	t.Log("Wake when the controller owner identity changes without advancing generation")
	ownerChanged := deployment.DeepCopy()
	ownerChanged.OwnerReferences[0].UID = "replacement-source-uid"
	require.True(t, filter.Update(event.UpdateEvent{ObjectOld: deployment, ObjectNew: ownerChanged}))
}

func TestLPXSourceWatchMapsControllerOwnedMaterializations(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1alpha1.AddToScheme(scheme))
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	source := &nvidiacomv1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "workloads", UID: "source-uid"}}
	owned := &nvidiacomv1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{
		Name: "materialization", Namespace: source.Namespace,
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(source, nvidiacomv1beta1.DynamoGraphDeploymentGVK)},
	}}
	unrelated := owned.DeepCopy()
	unrelated.Name = "unrelated"
	unrelated.OwnerReferences[0].UID = "other-source-uid"
	kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(owned, unrelated).
		WithIndex(&nvidiacomv1alpha1.LPXGraphDeployment{}, lpxSourceOwnerIndex, lpxSourceOwnerReferences).Build()
	reconciler := &graphReconciler{Client: kube}

	require.Equal(t, []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(owned)}}, reconciler.mapLPXSourceToRequests(t.Context(), source))

	t.Log("Report failed owner lookups with the exact source identity, without inventing a child name")
	lookupErr := errors.New("owner index unavailable")
	reconciler.Client = interceptor.NewClient(kube, interceptor.Funcs{
		List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
			return lookupErr
		},
	})
	var logs bytes.Buffer
	ctx := ctrl.LoggerInto(t.Context(), zap.New(zap.WriteTo(&logs)))
	require.Empty(t, reconciler.mapLPXSourceToRequests(ctx, source))
	require.Contains(t, logs.String(), lookupErr.Error())
	require.Contains(t, logs.String(), `"name":"source"`)
	require.Contains(t, logs.String(), `"namespace":"workloads"`)
	require.Contains(t, logs.String(), `"sourceUID":"source-uid"`)
}

func TestLPXWorkloadEventPredicates(t *testing.T) {
	for _, test := range []struct {
		name      string
		object    client.Object
		predicate predicate.Predicate
		observe   func(client.Object)
	}{
		{"clique", &grovev1alpha1.PodClique{}, lpxPodCliqueEventPredicates(), func(o client.Object) { o.(*grovev1alpha1.PodClique).Status.ReadyReplicas++ }},
		{"scaling group", &grovev1alpha1.PodCliqueScalingGroup{}, lpxScalingGroupEventPredicates(), func(o client.Object) {
			o.(*grovev1alpha1.PodCliqueScalingGroup).Status.ObservedGeneration = ptr.To(int64(2))
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Observe creation, deletion and all metadata used to fence materialization")
			old := test.object
			old.SetName("grove-child")
			old.SetNamespace("workloads")
			old.SetAnnotations(map[string]string{
				dynamolpx.DeploymentNameAnnotation: "materialization",
				dynamolpx.WorkloadDigestAnnotation: "sha256:workload",
				lpxv1alpha1.PodRoleAnnotation:      lpxv1alpha1.PodRoleAgent,
			})
			require.True(t, test.predicate.Create(event.CreateEvent{Object: old}))
			require.True(t, test.predicate.Delete(event.DeleteEvent{Object: old}))
			require.False(t, test.predicate.Generic(event.GenericEvent{Object: old}))
			require.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "workloads", Name: "materialization"}}},
				mapLPXChildToRequests(t.Context(), old), "Grove children need no source DGD label for routing")

			t.Log("Ordinary readiness traffic does not wake the LPX controller")
			unrelated := old.DeepCopyObject().(client.Object)
			unrelated.SetAnnotations(nil)
			require.False(t, test.predicate.Create(event.CreateEvent{Object: unrelated}))
			observed := unrelated.DeepCopyObject().(client.Object)
			test.observe(observed)
			require.False(t, test.predicate.Update(event.UpdateEvent{ObjectOld: unrelated, ObjectNew: observed}))

			t.Log("All compiled roles and PodClique template-hash convergence remain observable")
			if clique, ok := old.(*grovev1alpha1.PodClique); ok {
				for _, role := range []string{lpxv1alpha1.PodRoleAgent, lpxv1alpha1.PodRoleConductor, lpxv1alpha1.PodRoleCyborgWorker} {
					current := clique.DeepCopy()
					current.Annotations[lpxv1alpha1.PodRoleAnnotation] = role
					require.True(t, test.predicate.Create(event.CreateEvent{Object: current}))
				}
				current := clique.DeepCopy()
				current.Status.CurrentPodTemplateHash = ptr.To("current-template")
				require.True(t, test.predicate.Update(event.UpdateEvent{ObjectOld: clique, ObjectNew: current}))
			}

			t.Log("Wake on role readiness, independently observed group generation and materialization fences")
			for _, change := range []struct {
				name   string
				mutate func(client.Object)
				want   bool
			}{
				{"unchanged", func(client.Object) {}, false},
				{"status noise", func(o client.Object) { o.SetResourceVersion("2") }, false},
				{"status observed", test.observe, true},
				{"stamp lost", func(o client.Object) { o.SetAnnotations(nil) }, true},
				{"workload changed", func(o client.Object) { o.GetAnnotations()[dynamolpx.WorkloadDigestAnnotation] = "sha256:updated" }, true},
				{"labels changed", func(o client.Object) { o.SetLabels(map[string]string{"changed": "true"}) }, true},
				{"owner changed", func(o client.Object) { o.SetOwnerReferences([]metav1.OwnerReference{{UID: "new-owner"}}) }, true},
				{"deleting", func(o client.Object) { now := metav1.Now(); o.SetDeletionTimestamp(&now) }, true},
				{"spec generation changed", func(o client.Object) { o.SetGeneration(o.GetGeneration() + 1) }, true},
			} {
				t.Run(change.name, func(t *testing.T) {
					updated := old.DeepCopyObject().(client.Object)
					change.mutate(updated)
					require.Equal(t, change.want, test.predicate.Update(event.UpdateEvent{ObjectOld: old, ObjectNew: updated}))
					require.Equal(t, change.want, test.predicate.Update(event.UpdateEvent{ObjectOld: updated, ObjectNew: old}))
				})
			}
		})
	}
}

func TestLPXSourceWatchIgnoresOrdinaryChurn(t *testing.T) {
	t.Log("Ordinary capacity and DGD status edits must not rerun LPX compilation")
	source := newLPXTestSource(dynamolpx.PipelineSingle, "build-v2")
	source.Spec.Components = append(source.Spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "prefill", Replicas: ptr.To(int32(1))})
	changed := source.DeepCopy()
	changed.Generation++
	changed.Spec.Components[1].Replicas = ptr.To(int32(4))
	changed.Status.State = nvidiacomv1beta1.DGDStatePending
	predicate := lpxSourcePredicate()
	require.False(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))

	t.Log("Ignore ordinary creation and deletion, but keep LPX creation, deletion and deselection")
	ordinary := source.DeepCopy()
	ordinary.Spec.Components = ordinary.Spec.Components[1:]
	ordinary.Generation++
	require.False(t, predicate.Create(event.CreateEvent{Object: ordinary}))
	require.False(t, predicate.Delete(event.DeleteEvent{Object: ordinary}))
	require.True(t, predicate.Create(event.CreateEvent{Object: source}))
	require.True(t, predicate.Delete(event.DeleteEvent{Object: source}))
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: ordinary}))
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: ordinary, ObjectNew: source}))

	t.Log("LPX input changes and persisted restart selection wake the child")
	changed.Spec.Components[0].ComponentRole(nvidiacomv1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.Containers[0].Image = "new-runtime"
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))
	changed = source.DeepCopy()
	changed.Spec.Restart = &nvidiacomv1beta1.Restart{ID: "restart"}
	require.False(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))
	changed.Status.Restart = &nvidiacomv1beta1.RestartStatus{ObservedID: "restart", Phase: nvidiacomv1beta1.RestartPhaseRestarting, InProgress: []string{"lpx"}}
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))
	changed = source.DeepCopy()
	changed.UID = "replacement"
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))

	t.Log("Inherited scheduler metadata still wakes the child without a new source generation")
	changed = source.DeepCopy()
	changed.Labels = map[string]string{"priorityClassName": "inference"}
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))
	changed = source.DeepCopy()
	changed.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))
}

func BenchmarkLPXSourceWatchStatusChurn(b *testing.B) {
	source := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "inference", Namespace: "workloads", Generation: 1},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "lpx", ComponentType: nvidiacomv1beta1.ComponentTypeLPX,
				LPX: &nvidiacomv1beta1.LPXConfig{BuildID: "model/build"},
				Roles: []nvidiacomv1beta1.ComponentRoleSpec{{Name: nvidiacomv1beta1.ComponentRoleLPXConductor}, {
					Name: nvidiacomv1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "lpu-runtime:test"}}},
					},
				}},
			}},
		},
	}
	updated := source.DeepCopy()
	updated.Status.State = nvidiacomv1beta1.DGDStatePending
	update := event.UpdateEvent{ObjectOld: source, ObjectNew: updated}
	filter := lpxSourcePredicate()
	b.ReportAllocs()
	for b.Loop() {
		if filter.Update(update) {
			b.Fatal("status-only updates must not enqueue reconciliation")
		}
	}
}
