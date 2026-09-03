/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// TestDeleteOrphanedElasticEPFollowers covers the review's last gate-off requirement:
// turning the gate from on to off must clean up what it previously synthesized.
//
// Nothing else would. The rollout path prunes worker DCDs by comparing their hash label
// against the current worker generation, and a follower's label still matches, because
// the hash is deliberately gate-independent. A follower left behind is owned by no one:
// generation no longer produces it, so no reconcile will ever touch it again.
func TestDeleteOrphanedElasticEPFollowers(t *testing.T) {
	s := scheme.Scheme
	require.NoError(t, nvidiacomv1beta1.AddToScheme(s))

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "mydgd", Namespace: "default"},
	}
	dcd := func(name string, follower bool) *nvidiacomv1beta1.DynamoComponentDeployment {
		obj := &nvidiacomv1beta1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
				},
			},
		}
		if follower {
			obj.Annotations = map[string]string{
				consts.KubeAnnotationElasticEPFollower: consts.KubeLabelValueTrue,
			}
		}
		return obj
	}

	leader := dcd("mydgd-decode", false)
	orphan := dcd("mydgd-decode-flw", true)
	kept := dcd("mydgd-prefill-flw", true)

	c := fake.NewClientBuilder().WithScheme(s).WithObjects(leader, orphan, kept).Build()
	r := &componentWorkloadsReconciler{syncer: newDGDResourceSyncer(c, nil)}

	t.Log("Generation still produces the leader and one follower, but not the orphan")
	generated := map[string]*nvidiacomv1beta1.DynamoComponentDeployment{
		"decode":      leader,
		"prefill-flw": kept,
	}
	require.NoError(t, r.deleteOrphanedElasticEPFollowers(context.Background(), dgd, generated))

	t.Log("The follower generation no longer produces is deleted")
	err := c.Get(context.Background(), types.NamespacedName{Name: orphan.Name, Namespace: "default"}, &nvidiacomv1beta1.DynamoComponentDeployment{})
	require.True(t, client.IgnoreNotFound(err) == nil && err != nil, "orphaned follower should have been deleted, got err=%v", err)

	t.Log("A follower that is still generated survives")
	require.NoError(t, c.Get(context.Background(), types.NamespacedName{Name: kept.Name, Namespace: "default"}, &nvidiacomv1beta1.DynamoComponentDeployment{}))

	t.Log("A non-follower DCD is never touched, whatever generation says")
	require.NoError(t, c.Get(context.Background(), types.NamespacedName{Name: leader.Name, Namespace: "default"}, &nvidiacomv1beta1.DynamoComponentDeployment{}))
}
