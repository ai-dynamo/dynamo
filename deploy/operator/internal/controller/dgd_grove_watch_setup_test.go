// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

func TestGroveWatchSkipsOnlyVerifiedLPXOwners(t *testing.T) {
	t.Log("Distinguish LPX-owned Grove roles from ordinary DGD readiness events")
	const wrongAPIVersion = "unrelated/v1"
	const sourceName = "source-graph"
	scheme := runtime.NewScheme()
	require.NoError(t, grovev1alpha1.AddToScheme(scheme))
	child := &nvidiacomv1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "independent-materialization", UID: "child-uid"}}
	pcs := &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{
		Name: "engine-pcs", Namespace: "workloads", UID: "pcs-uid",
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(child, nvidiacomv1alpha1.LPXGraphDeploymentGVK)},
	}}
	group := &grovev1alpha1.PodCliqueScalingGroup{ObjectMeta: metav1.ObjectMeta{
		Name: "engine-group", Namespace: "workloads", UID: "group-uid",
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
	}}
	clique := &grovev1alpha1.PodClique{ObjectMeta: metav1.ObjectMeta{
		Name: "role", Namespace: "workloads",
		Labels: map[string]string{consts.KubeLabelDynamoGraphDeploymentName: sourceName},
		Annotations: map[string]string{
			dynamolpx.DeploymentNameAnnotation: child.Name,
			lpxv1alpha1.PodRoleAnnotation:      lpxv1alpha1.PodRoleConductor,
			dynamolpx.WorkloadDigestAnnotation: "sha256:workload",
			dynamo.LPXDeploymentUIDAnnotation:  string(child.UID),
		},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(group, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))},
	}}
	ready := clique.DeepCopy()
	ready.Status.ReadyReplicas++
	require.True(t, podCliqueEventPredicates().Update(event.UpdateEvent{ObjectOld: clique, ObjectNew: ready}))

	t.Log("The LPX controller observes role readiness and the DGD observes aggregate child status")
	parent := newGroveWatchSetup(fake.NewClientBuilder().WithScheme(scheme).WithObjects(pcs, group).Build())
	for _, role := range []string{lpxv1alpha1.PodRoleAgent, lpxv1alpha1.PodRoleConductor, lpxv1alpha1.PodRoleCyborgWorker} {
		t.Run(role, func(t *testing.T) {
			observed := ready.DeepCopy()
			observed.Annotations[lpxv1alpha1.PodRoleAnnotation] = role
			require.Empty(t, parent.mapPodCliqueToRequests(t.Context(), observed))
		})
	}

	t.Log("Authorable LPX hints cannot suppress ordinary readiness, nor can incomplete cached ownership")
	for _, test := range []struct {
		name   string
		mutate func(*grovev1alpha1.PodClique, *grovev1alpha1.PodCliqueScalingGroup, *grovev1alpha1.PodCliqueSet)
	}{
		{"ordinary unmarked", func(c *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, _ *grovev1alpha1.PodCliqueSet) {
			c.Annotations = nil
		}},
		{"ordinary direct owner", func(c *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, p *grovev1alpha1.PodCliqueSet) {
			c.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(p, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}
		}},
		{"ordinary grouped owner", func(_ *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, p *grovev1alpha1.PodCliqueSet) {
			p.OwnerReferences[0].Kind = "DynamoGraphDeployment"
		}},
		{"missing group", func(_ *grovev1alpha1.PodClique, g *grovev1alpha1.PodCliqueScalingGroup, _ *grovev1alpha1.PodCliqueSet) {
			g.Name = "not-the-owner"
		}},
		{"replaced group", func(_ *grovev1alpha1.PodClique, g *grovev1alpha1.PodCliqueScalingGroup, _ *grovev1alpha1.PodCliqueSet) {
			g.UID = "new-group-uid"
		}},
		{"wrong group API", func(c *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, _ *grovev1alpha1.PodCliqueSet) {
			c.OwnerReferences[0].APIVersion = wrongAPIVersion
		}},
		{"missing PCS", func(_ *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, p *grovev1alpha1.PodCliqueSet) {
			p.Name = "not-the-owner"
		}},
		{"replaced PCS", func(_ *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, p *grovev1alpha1.PodCliqueSet) {
			p.UID = "new-pcs-uid"
		}},
		{"wrong PCS API", func(_ *grovev1alpha1.PodClique, g *grovev1alpha1.PodCliqueScalingGroup, _ *grovev1alpha1.PodCliqueSet) {
			g.OwnerReferences[0].APIVersion = wrongAPIVersion
		}},
		{"wrong child API", func(_ *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, p *grovev1alpha1.PodCliqueSet) {
			p.OwnerReferences[0].APIVersion = wrongAPIVersion
		}},
		{"different child", func(_ *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, p *grovev1alpha1.PodCliqueSet) {
			p.OwnerReferences[0].Name = "other-graph"
		}},
		{"stale child stamp", func(c *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, _ *grovev1alpha1.PodCliqueSet) {
			c.Annotations[dynamo.LPXDeploymentUIDAnnotation] = "old-child-uid"
		}},
		{"missing child name stamp", func(c *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, _ *grovev1alpha1.PodCliqueSet) {
			delete(c.Annotations, dynamolpx.DeploymentNameAnnotation)
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			currentPCS, currentGroup := pcs.DeepCopy(), group.DeepCopy()
			observed := ready.DeepCopy()
			test.mutate(observed, currentGroup, currentPCS)
			reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(currentPCS, currentGroup).Build()
			require.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: observed.Namespace, Name: sourceName}}}, newGroveWatchSetup(reader).mapPodCliqueToRequests(t.Context(), observed))
		})
	}
}
