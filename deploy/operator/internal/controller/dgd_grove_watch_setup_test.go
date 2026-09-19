// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGroveWatchMapsOnlyDGDControllerOwnedPodCliques(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, grovev1alpha1.AddToScheme(scheme))

	controllerRef := func(apiVersion, kind, name string, uid types.UID) metav1.OwnerReference {
		return metav1.OwnerReference{
			APIVersion: apiVersion,
			Kind:       kind,
			Name:       name,
			UID:        uid,
			Controller: ptr.To(true),
		}
	}

	newObjects := func() (*grovev1alpha1.PodClique, *grovev1alpha1.PodCliqueScalingGroup, *grovev1alpha1.PodCliqueSet) {
		pcs := &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{
			Name:      "engine-pcs",
			Namespace: "workloads",
			UID:       "pcs-uid",
			OwnerReferences: []metav1.OwnerReference{controllerRef(
				nvidiacomv1beta1.GroupVersion.String(), "DynamoGraphDeployment", "graph", "dgd-uid",
			)},
		}}
		group := &grovev1alpha1.PodCliqueScalingGroup{ObjectMeta: metav1.ObjectMeta{
			Name:      "engine-group",
			Namespace: "workloads",
			UID:       "group-uid",
			OwnerReferences: []metav1.OwnerReference{controllerRef(
				grovev1alpha1.SchemeGroupVersion.String(), "PodCliqueSet", pcs.Name, pcs.UID,
			)},
		}}
		clique := &grovev1alpha1.PodClique{ObjectMeta: metav1.ObjectMeta{
			Name:      "role",
			Namespace: "workloads",
			OwnerReferences: []metav1.OwnerReference{controllerRef(
				grovev1alpha1.SchemeGroupVersion.String(), "PodCliqueScalingGroup", group.Name, group.UID,
			)},
		}}
		return clique, group, pcs
	}

	want := []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "workloads", Name: "graph"}}}
	tests := []struct {
		name   string
		mutate func(*grovev1alpha1.PodClique, *grovev1alpha1.PodCliqueScalingGroup, *grovev1alpha1.PodCliqueSet) []client.Object
		want   []ctrl.Request
	}{
		{
			name: "direct PCS ownership",
			mutate: func(clique *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				clique.OwnerReferences = []metav1.OwnerReference{controllerRef(
					grovev1alpha1.SchemeGroupVersion.String(), "PodCliqueSet", pcs.Name, pcs.UID,
				)}
				return []client.Object{pcs}
			},
			want: want,
		},
		{
			name: "PCSG-mediated ownership",
			mutate: func(_ *grovev1alpha1.PodClique, group *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				return []client.Object{group, pcs}
			},
			want: want,
		},
		{
			name: "LGD-owned PCS",
			mutate: func(_ *grovev1alpha1.PodClique, group *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				pcs.OwnerReferences[0] = controllerRef(
					nvidiacomv1alpha1.GroupVersion.String(), "LPXGraphDeployment", "graph", "lgd-uid",
				)
				return []client.Object{group, pcs}
			},
		},
		{
			name: "wrong group owner UID",
			mutate: func(clique *grovev1alpha1.PodClique, group *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				clique.OwnerReferences[0].UID = "old-group-uid"
				return []client.Object{group, pcs}
			},
		},
		{
			name: "missing cached group",
			mutate: func(_ *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				return []client.Object{pcs}
			},
		},
		{
			name: "same-name foreign PCS",
			mutate: func(_ *grovev1alpha1.PodClique, group *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				pcs.UID = "replacement-pcs-uid"
				return []client.Object{group, pcs}
			},
		},
		{
			name: "legacy v1alpha1 DGD owner",
			mutate: func(_ *grovev1alpha1.PodClique, group *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				pcs.OwnerReferences[0].APIVersion = nvidiacomv1alpha1.GroupVersion.String()
				return []client.Object{group, pcs}
			},
			want: want,
		},
		{
			name: "foreign PCS owner API group",
			mutate: func(_ *grovev1alpha1.PodClique, group *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				pcs.OwnerReferences[0].APIVersion = "foreign.example/v1beta1"
				return []client.Object{group, pcs}
			},
		},
		{
			name: "malformed PCS owner API version",
			mutate: func(_ *grovev1alpha1.PodClique, group *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				pcs.OwnerReferences[0].APIVersion = "nvidia.com/"
				return []client.Object{group, pcs}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			clique, group, pcs := newObjects()
			objects := test.mutate(clique, group, pcs)
			reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build()
			require.Equal(t, test.want, newGroveWatchSetup(reader).mapPodCliqueToRequests(t.Context(), clique))
		})
	}
}
