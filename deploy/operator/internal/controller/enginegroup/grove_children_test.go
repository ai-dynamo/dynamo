/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"strconv"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGroveWorldBindingIdentity(t *testing.T) {
	cases := []struct {
		name        string
		worldIndex  int32
		groupIndex  string
		cliqueIndex string
		boundUID    string
		want        string
		pending     bool
	}{
		{name: "world zero", groupIndex: "0", cliqueIndex: "0"},
		{name: "nonzero world", worldIndex: 3, groupIndex: "3", cliqueIndex: "3"},
		{name: "backfill an already bound world", cliqueIndex: "0", boundUID: "clique-uid"},
		{name: "conflicting logical index", worldIndex: 3, groupIndex: "0", cliqueIndex: "3", want: "conflicting logical world index"},
		{name: "clique from another world", worldIndex: 3, groupIndex: "3", cliqueIndex: "0", want: "does not belong to world index"},
		{name: "missing native index", groupIndex: "0", want: "does not belong to world index"},
		{name: "noncanonical native index", groupIndex: "0", cliqueIndex: "00", want: "does not belong to world index"},
		{name: "recreated physical clique", groupIndex: "0", cliqueIndex: "0", boundUID: "previous-clique-uid", pending: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("give the logical world a stable UID and a live target independent of its Grove binding")
			scheme := runtime.NewScheme()
			require.NoError(t, api.AddToScheme(scheme))
			require.NoError(t, grove.AddToScheme(scheme))
			group := &api.DynamoGraphDeploymentEngineGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dynamo.EngineGroupNameForComponent("graph", "Worker", tc.worldIndex),
					Namespace: "test", UID: "logical-world-uid",
					Labels:      map[string]string{consts.KubeLabelDynamoComponent: "Worker"},
					Annotations: map[string]string{consts.KubeAnnotationDynamoEngineGroupProfile: "profile"},
				},
				Spec: api.DynamoGraphDeploymentEngineGroupSpec{
					Replicas: 8,
					Policy:   &api.EngineGroupScalingPolicy{MinReplicas: ptr.To(int32(2)), MaxReplicas: ptr.To(int32(16))},
				},
			}
			if tc.groupIndex != "" {
				group.Labels[consts.KubeLabelDynamoEngineGroupWorldIndex] = tc.groupIndex
			}
			if tc.boundUID != "" {
				group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique] = "physical-clique"
				group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID] = tc.boundUID
			}

			t.Log("publish the physical clique with Grove's explicit PCSG replica index and ownership chain")
			pcs := &grove.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: "pcs", Namespace: group.Namespace, UID: "pcs-uid"}}
			world := &grove.PodCliqueScalingGroup{ObjectMeta: metav1.ObjectMeta{
				Name: "physical-world", Namespace: group.Namespace, UID: "pcsg-uid",
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grove.SchemeGroupVersion.WithKind("PodCliqueSet"))},
			}}
			clique := &grove.PodClique{ObjectMeta: metav1.ObjectMeta{
				Name: "physical-clique", Namespace: group.Namespace, UID: "clique-uid",
				Labels: map[string]string{
					consts.KubeLabelDynamoEngineGroup:                  group.Name,
					grovecommon.LabelPodCliqueScalingGroupReplicaIndex: tc.cliqueIndex,
				},
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(world, grove.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))},
			}}
			kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(group, pcs, world, clique).Build()
			reconciler := &GroveChildrenReconciler{Client: kube}
			before := group.DeepCopy()

			t.Log("bind only matching logical and physical identities, without changing the desired target or policy")
			bound, err := reconciler.bindClique(t.Context(), group, pcs, tc.worldIndex)
			stored := &api.DynamoGraphDeploymentEngineGroup{}
			require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(group), stored))
			assert.Equal(t, before.UID, stored.UID)
			assert.Equal(t, before.Spec, stored.Spec)
			assert.Equal(t, before.Status, stored.Status)
			if tc.want != "" {
				require.ErrorContains(t, err, tc.want)
				assert.False(t, bound)
				assert.Equal(t, before.Labels, stored.Labels)
				assert.Equal(t, before.Annotations, stored.Annotations)
				return
			}
			require.NoError(t, err)
			if tc.pending {
				assert.False(t, bound)
				assert.Equal(t, before.Annotations, stored.Annotations)
				return
			}
			assert.True(t, bound)
			assert.Equal(t, strconv.FormatInt(int64(tc.worldIndex), 10), stored.Labels[consts.KubeLabelDynamoEngineGroupWorldIndex])
			assert.Equal(t, clique.Name, stored.Annotations[consts.KubeAnnotationDynamoEngineGroupPodClique])
			assert.Equal(t, string(clique.UID), stored.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])

			t.Log("a repeated binding is a no-op and does not reset the world's durable state")
			version := stored.ResourceVersion
			bound, err = reconciler.bindClique(t.Context(), stored, pcs, tc.worldIndex)
			require.NoError(t, err)
			assert.True(t, bound)
			require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(stored), stored))
			assert.Equal(t, version, stored.ResourceVersion)
		})
	}
}
