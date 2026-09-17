/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"bytes"
	"context"
	"errors"
	"maps"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	lpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

func TestLPXGraphDeploymentPredicate(t *testing.T) {
	t.Log("Build one current private handoff and its primary-resource predicate")
	deployment := &v1alpha1.LPXGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "graph", Namespace: "workloads", Generation: 1,
			OwnerReferences: []metav1.OwnerReference{{Name: "graph", UID: "source-uid", Controller: ptr.To(true)}},
		},
	}
	filter := lpxGraphDeploymentPredicate()

	t.Log("Ignore status-only updates that do not change publication authority")
	statusOnly := deployment.DeepCopy()
	statusOnly.Status.ObservedGeneration = 1
	require.False(t, filter.Update(event.UpdateEvent{ObjectOld: deployment, ObjectNew: statusOnly}))

	t.Log("Wake when the controller owner identity changes without advancing generation")
	ownerChanged := deployment.DeepCopy()
	ownerChanged.OwnerReferences[0].UID = "replacement-source-uid"
	require.True(t, filter.Update(event.UpdateEvent{ObjectOld: deployment, ObjectNew: ownerChanged}))
}

func TestDGDWatchMapsControllerOwnedDeployments(t *testing.T) {
	t.Log("Index children by their controller owner using the shared test client")
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "workloads", UID: "source-uid"}}
	owned := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{
		Name: "materialization", Namespace: dgd.Namespace,
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)},
	}}
	unrelated := owned.DeepCopy()
	unrelated.Name = "unrelated"
	unrelated.OwnerReferences[0].UID = "other-source-uid"
	kube := newLPXTestClient(t, owned, unrelated)
	reconciler := &graphReconciler{Client: kube}

	require.Equal(t, []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(owned)}}, reconciler.mapDGDToLPXGraphDeployments(t.Context(), dgd))

	t.Log("Report failed owner lookups with the exact source identity, without inventing a child name")
	lookupErr := errors.New("owner index unavailable")
	reconciler.Client = interceptor.NewClient(kube, interceptor.Funcs{
		List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
			return lookupErr
		},
	})
	var logs bytes.Buffer
	ctx := ctrl.LoggerInto(t.Context(), zap.New(zap.WriteTo(&logs)))
	require.Empty(t, reconciler.mapDGDToLPXGraphDeployments(ctx, dgd))
	require.Contains(t, logs.String(), lookupErr.Error())
	require.Contains(t, logs.String(), `"name":"source"`)
	require.Contains(t, logs.String(), `"namespace":"workloads"`)
	require.Contains(t, logs.String(), `"dgdUID":"source-uid"`)
}

func TestGroveEventPredicates(t *testing.T) {
	metadata := metav1.ObjectMeta{
		Name: "grove-child", Namespace: "workloads",
		Annotations: map[string]string{
			lpx.DeploymentNameAnnotation:  "materialization",
			lpx.WorkloadDigestAnnotation:  "sha256:workload",
			lpxv1alpha1.PodRoleAnnotation: lpxv1alpha1.PodRoleAgent,
		},
	}
	for _, test := range []struct {
		name      string
		object    client.Object
		observed  client.Object
		predicate predicate.Predicate
	}{
		{
			"clique", &grovev1alpha1.PodClique{ObjectMeta: metadata},
			&grovev1alpha1.PodClique{ObjectMeta: metadata, Status: grovev1alpha1.PodCliqueStatus{ReadyReplicas: 1}},
			podCliquePredicate(),
		},
		{
			"scaling group", &grovev1alpha1.PodCliqueScalingGroup{ObjectMeta: metadata},
			&grovev1alpha1.PodCliqueScalingGroup{ObjectMeta: metadata, Status: grovev1alpha1.PodCliqueScalingGroupStatus{ObservedGeneration: ptr.To(int64(2))}},
			podCliqueScalingGroupPredicate(),
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Only LPX workload creation, deletion and readiness changes wake the controller")
			require.True(t, test.predicate.Create(event.CreateEvent{Object: test.object}))
			require.True(t, test.predicate.Delete(event.DeleteEvent{Object: test.object}))
			require.False(t, test.predicate.Generic(event.GenericEvent{Object: test.object}))
			require.True(t, test.predicate.Update(event.UpdateEvent{ObjectOld: test.object, ObjectNew: test.observed}))
			require.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "workloads", Name: "materialization"}}},
				mapChildToLPXGraphDeployment(t.Context(), test.object), "Grove children need no DGD label for routing")

			unrelated := test.object.DeepCopyObject().(client.Object)
			unrelated.SetAnnotations(nil)
			observedUnrelated := test.observed.DeepCopyObject().(client.Object)
			observedUnrelated.SetAnnotations(nil)
			require.False(t, test.predicate.Create(event.CreateEvent{Object: unrelated}))
			require.False(t, test.predicate.Delete(event.DeleteEvent{Object: unrelated}))
			require.False(t, test.predicate.Update(event.UpdateEvent{ObjectOld: unrelated, ObjectNew: observedUnrelated}))

			t.Log("Metadata changes wake reconciliation except for resource-version-only noise")
			workloadAnnotations := maps.Clone(metadata.Annotations)
			workloadAnnotations[lpx.WorkloadDigestAnnotation] = "sha256:updated"
			for _, change := range []struct {
				name     string
				metadata metav1.ObjectMeta
				want     bool
			}{
				{"unchanged", metadata, false},
				{"resource version", metav1.ObjectMeta{Annotations: metadata.Annotations, ResourceVersion: "2"}, false},
				{"stamp lost", metav1.ObjectMeta{}, true},
				{"workload", metav1.ObjectMeta{Annotations: workloadAnnotations}, true},
				{"labels", metav1.ObjectMeta{Annotations: metadata.Annotations, Labels: map[string]string{"changed": "true"}}, true},
				{"owner", metav1.ObjectMeta{Annotations: metadata.Annotations, OwnerReferences: []metav1.OwnerReference{{UID: "new-owner"}}}, true},
				{"deleting", metav1.ObjectMeta{Annotations: metadata.Annotations, DeletionTimestamp: ptr.To(metav1.Now())}, true},
				{"generation", metav1.ObjectMeta{Annotations: metadata.Annotations, Generation: 1}, true},
			} {
				t.Run(change.name, func(t *testing.T) {
					updated := test.object.DeepCopyObject().(client.Object)
					updated.SetAnnotations(change.metadata.Annotations)
					updated.SetLabels(change.metadata.Labels)
					updated.SetOwnerReferences(change.metadata.OwnerReferences)
					updated.SetDeletionTimestamp(change.metadata.DeletionTimestamp)
					updated.SetGeneration(change.metadata.Generation)
					updated.SetResourceVersion(change.metadata.ResourceVersion)
					require.Equal(t, change.want, test.predicate.Update(event.UpdateEvent{ObjectOld: test.object, ObjectNew: updated}))
					require.Equal(t, change.want, test.predicate.Update(event.UpdateEvent{ObjectOld: updated, ObjectNew: test.object}))
				})
			}
		})
	}
}

func TestPodCliquePredicateRoles(t *testing.T) {
	t.Log("Observe compiled LPX roles and template-hash convergence")
	filter := podCliquePredicate()
	for _, test := range []struct {
		role string
		want bool
	}{
		{lpxv1alpha1.PodRoleAgent, true},
		{lpxv1alpha1.PodRoleConductor, true},
		{lpxv1alpha1.PodRoleCyborgWorker, true},
		{"ordinary", false},
	} {
		t.Run(test.role, func(t *testing.T) {
			clique := &grovev1alpha1.PodClique{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
				lpx.WorkloadDigestAnnotation:  "sha256:workload",
				lpxv1alpha1.PodRoleAnnotation: test.role,
			}}}
			require.Equal(t, test.want, filter.Create(event.CreateEvent{Object: clique}))
			current := clique.DeepCopy()
			current.Status.CurrentPodTemplateHash = ptr.To("current-template")
			require.Equal(t, test.want, filter.Update(event.UpdateEvent{ObjectOld: clique, ObjectNew: current}))
		})
	}
}

func TestDGDPredicate(t *testing.T) {
	t.Log("Observe LPX input changes, not ordinary capacity and status churn")
	dgd := loadTestDGD(t, lpx.PipelineSingle, "build-v2")
	dgd.Spec.Components = append(dgd.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "prefill", Replicas: ptr.To(int32(1))})
	filter := dgdPredicate()

	ordinary := dgd.DeepCopy()
	ordinary.Spec.Components = ordinary.Spec.Components[1:]
	ordinary.Generation++
	require.False(t, filter.Create(event.CreateEvent{Object: ordinary}))
	require.False(t, filter.Delete(event.DeleteEvent{Object: ordinary}))
	require.True(t, filter.Create(event.CreateEvent{Object: dgd}))
	require.True(t, filter.Delete(event.DeleteEvent{Object: dgd}))

	statusOnly := dgd.DeepCopy()
	statusOnly.Status.State = v1beta1.DGDStatePending
	capacity := dgd.DeepCopy()
	capacity.Generation++
	capacity.Spec.Components[1].Replicas = ptr.To(int32(4))
	capacity.Status.State = v1beta1.DGDStatePending
	image := dgd.DeepCopy()
	image.Generation++
	image.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.Containers[0].Image = "new-runtime"
	componentTopology := dgd.DeepCopy()
	componentTopology.Generation++
	componentTopology.Spec.Components[0].TopologyConstraint = &v1beta1.TopologyConstraint{PackDomain: "rack"}
	graphTopology := dgd.DeepCopy()
	graphTopology.Generation++
	graphTopology.Spec.TopologyConstraint = &v1beta1.SpecTopologyConstraint{ClusterTopologyName: "fabric"}
	restart := dgd.DeepCopy()
	restart.Spec.Restart = &v1beta1.Restart{ID: "restart"}
	selectedRestart := restart.DeepCopy()
	selectedRestart.Status.Restart = &v1beta1.RestartStatus{ObservedID: "restart", Phase: v1beta1.RestartPhaseRestarting, InProgress: []string{"lpx"}}
	replacement := dgd.DeepCopy()
	replacement.UID = "replacement"
	labels := dgd.DeepCopy()
	labels.Labels = map[string]string{"priorityClassName": "inference"}
	annotations := dgd.DeepCopy()
	annotations.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)

	for _, test := range []struct {
		name    string
		changed *v1beta1.DynamoGraphDeployment
		want    bool
	}{
		{"status only", statusOnly, false},
		{"ordinary capacity and status", capacity, false},
		{"LPX deselection", ordinary, true},
		{"runtime image", image, true},
		{"component topology", componentTopology, true},
		{"graph topology", graphTopology, true},
		{"unselected restart", restart, false},
		{"selected restart", selectedRestart, true},
		{"DGD replacement", replacement, true},
		{"inherited labels", labels, true},
		{"inherited annotations", annotations, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			require.Equal(t, test.want, filter.Update(event.UpdateEvent{ObjectOld: dgd, ObjectNew: test.changed}))
			require.Equal(t, test.want, filter.Update(event.UpdateEvent{ObjectOld: test.changed, ObjectNew: dgd}))
		})
	}
}
