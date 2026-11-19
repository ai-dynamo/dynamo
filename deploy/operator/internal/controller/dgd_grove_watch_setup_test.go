// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"cmp"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	grovev1beta1 "github.com/ai-dynamo/grove/operator/api/core/v1beta1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
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
			Controller: new(true),
		}
	}

	newObjects := func(apiVersion string) (*grovev1alpha1.PodClique, *grovev1alpha1.PodCliqueScalingGroup, *grovev1alpha1.PodCliqueSet) {
		pcs := &grovev1alpha1.PodCliqueSet{
			Name:      "engine-pcs",
			Namespace: "workloads",
			UID:       "pcs-uid",
			OwnerReferences: []metav1.OwnerReference{controllerRef(
				nvidiacomv1beta1.GroupVersion.String(), "DynamoGraphDeployment", "graph", "dgd-uid",
			)}}
		group := &grovev1alpha1.PodCliqueScalingGroup{
			Name:      "engine-group",
			Namespace: "workloads",
			UID:       "group-uid",
			OwnerReferences: []metav1.OwnerReference{controllerRef(
				apiVersion, "PodCliqueSet", pcs.Name, pcs.UID,
			)}}
		clique := &grovev1alpha1.PodClique{
			Name:      "role",
			Namespace: "workloads",
			OwnerReferences: []metav1.OwnerReference{controllerRef(
				apiVersion, "PodCliqueScalingGroup", group.Name, group.UID,
			)}}
		return clique, group, pcs
	}

	want := []ctrl.Request{{Namespace: "workloads", Name: "graph"}}
	tests := []struct {
		name              string
		groupOwnerVersion string
		mutate            func(*grovev1alpha1.PodClique, *grovev1alpha1.PodCliqueScalingGroup, *grovev1alpha1.PodCliqueSet) []client.Object
		want              []ctrl.Request
	}{
		{
			name: "direct PCS ownership",
			mutate: func(clique *grovev1alpha1.PodClique, _ *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				clique.OwnerReferences = []metav1.OwnerReference{controllerRef(
					clique.OwnerReferences[0].APIVersion, "PodCliqueSet", pcs.Name, pcs.UID,
				)}
				return []client.Object{pcs}
			},
			want: want,
		},
		{
			name:              "PCSG-mediated ownership with alpha PCS owner",
			groupOwnerVersion: grovev1alpha1.SchemeGroupVersion.String(),
			mutate: func(_ *grovev1alpha1.PodClique, group *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				return []client.Object{group, pcs}
			},
			want: want,
		},
		{
			name:              "PCSG-mediated ownership with beta PCS owner",
			groupOwnerVersion: grovev1beta1.SchemeGroupVersion.String(),
			mutate: func(_ *grovev1alpha1.PodClique, group *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				return []client.Object{group, pcs}
			},
			want: want,
		},
		{
			name: "foreign PodClique owner API group",
			mutate: func(clique *grovev1alpha1.PodClique, group *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				clique.OwnerReferences[0].APIVersion = "foreign.example/v1beta1"
				return []client.Object{group, pcs}
			},
		},
		{
			name: "foreign scaling group owner API group",
			mutate: func(_ *grovev1alpha1.PodClique, group *grovev1alpha1.PodCliqueScalingGroup, pcs *grovev1alpha1.PodCliqueSet) []client.Object {
				group.OwnerReferences[0].APIVersion = "foreign.example/v1beta1"
				return []client.Object{group, pcs}
			},
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
		for _, apiVersion := range []string{grovev1alpha1.SchemeGroupVersion.String(), grovev1beta1.SchemeGroupVersion.String()} {
			t.Run(test.name+"/"+apiVersion, func(t *testing.T) {
				t.Log("Observe converted Grove children with their original owner-reference versions")
				clique, group, pcs := newObjects(apiVersion)
				group.OwnerReferences[0].APIVersion = cmp.Or(test.groupOwnerVersion, apiVersion)
				objects := test.mutate(clique, group, pcs)
				reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build()
				t.Log("Map readiness updates only through the exact DGD-owned resource hierarchy")
				require.Equal(t, test.want, newGroveWatchSetup(reader).mapPodCliqueToRequests(t.Context(), clique))
			})
		}
	}
}
