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
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// TestDeleteOrphanedElasticEPFollowers covers the sweep itself: a follower that generation
// no longer produces must be cleaned up.
//
// Nothing else would. The rollout path prunes worker DCDs by comparing their hash label
// against the current worker generation, and a follower's label still matches, because it
// deep-copies its leader and carries that leader's current hash. A follower left behind is
// owned by no one: generation no longer produces it, so no reconcile will ever touch it
// again.
func TestDeleteOrphanedElasticEPFollowers(t *testing.T) {
	s := scheme.Scheme
	require.NoError(t, nvidiacomv1beta1.AddToScheme(s))

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "mydgd", Namespace: "default", UID: "dgd-uid"},
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
		// A real controller reference: the sweep proves ownership before deleting, and a
		// fixture without one would assert that an unowned object is fair game.
		// Ownership itself is covered by TestDeleteOrphanedElasticEPFollowersProvesOwnership.
		require.NoError(t, controllerutil.SetControllerReference(dgd, obj, s))
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

// TestDeleteOrphanedElasticEPFollowersProvesOwnership covers the review's P1: the sweep
// must prove it controls an object before destroying it.
//
// The candidate list is narrowed only by the DGD-name label and the follower annotation.
// Both are ordinary metadata that anything can set, so without an ownership check a
// standalone or foreign-owned DCD carrying those two values is deleted by a DGD that does
// not control it. The delete also needs UID and resourceVersion preconditions: the name is
// reused across worker generations, so between the List and the Delete the object may
// already be a replacement that generation wants to keep.
//
// Mutation check: removing the IsControlledBy guard fails the unowned and foreign
// subtests.
func TestDeleteOrphanedElasticEPFollowersProvesOwnership(t *testing.T) {
	s := scheme.Scheme
	require.NoError(t, nvidiacomv1beta1.AddToScheme(s))

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "mydgd", Namespace: "default", UID: "dgd-uid"},
	}
	other := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "otherdgd", Namespace: "default", UID: "other-uid"},
	}

	for _, tt := range []struct {
		name        string
		owner       *nvidiacomv1beta1.DynamoGraphDeployment // nil = no controller reference
		wantDeleted bool
	}{
		{name: "controlled by this DGD is released", owner: dgd, wantDeleted: true},
		{name: "no controller reference survives", owner: nil, wantDeleted: false},
		{name: "controlled by a different DGD survives", owner: other, wantDeleted: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			follower := &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mydgd-decode-flw",
					Namespace: "default",
					Labels: map[string]string{
						consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
					},
					Annotations: map[string]string{
						consts.KubeAnnotationElasticEPFollower: consts.KubeLabelValueTrue,
					},
				},
				Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						Replicas: ptr.To(int32(0)),
					},
				},
			}
			if tt.owner != nil {
				require.NoError(t, controllerutil.SetControllerReference(tt.owner, follower, s))
			}

			c := fake.NewClientBuilder().WithScheme(s).WithObjects(follower).Build()
			r := &componentWorkloadsReconciler{syncer: newDGDResourceSyncer(c, nil)}

			require.NoError(t, r.deleteOrphanedElasticEPFollowers(
				context.Background(), dgd,
				map[string]*nvidiacomv1beta1.DynamoComponentDeployment{},
			))

			err := c.Get(context.Background(),
				types.NamespacedName{Name: follower.Name, Namespace: "default"},
				&nvidiacomv1beta1.DynamoComponentDeployment{})
			if tt.wantDeleted {
				require.True(t, err != nil && client.IgnoreNotFound(err) == nil,
					"a follower this DGD controls should be released, got err=%v", err)
				return
			}
			require.NoError(t, err,
				"the sweep destroyed an object this DGD does not control; the DGD-name label "+
					"and the follower annotation are both mutable and settable by anyone")
		})
	}
}

// TestDeleteOrphanedElasticEPFollowersRefusesANonEmptyFollower pins the release
// precondition: a follower is removed only when it is provably empty.
//
// Nothing in the operator calls scale_elastic_ep -- there is no engine-control client in
// the tree -- so deleting a follower that still holds ranks leaves the engine committed
// to a DP size whose members are gone. DYN-3838 records the leader surviving at
// restart=0 with inference stopped; DYN-2660 records the orphaned placement group then
// blocking every later scale-up until the pod restarts, which no gate flip undoes.
//
// So an orphaning must leave running capacity alone rather than tear it down, and the
// operator says so.
//
// Mutation check: removing the replicas>0 guard fails the non-empty subtest.
func TestDeleteOrphanedElasticEPFollowersRefusesANonEmptyFollower(t *testing.T) {
	s := scheme.Scheme
	require.NoError(t, nvidiacomv1beta1.AddToScheme(s))

	for _, tt := range []struct {
		name        string
		replicas    *int32
		wantDeleted bool
	}{
		{name: "at rest is released", replicas: ptr.To(int32(0)), wantDeleted: true},
		{name: "nil replicas is released", replicas: nil, wantDeleted: true},
		{name: "scaled up is refused", replicas: ptr.To(int32(3)), wantDeleted: false},
		{name: "a single replica is refused", replicas: ptr.To(int32(1)), wantDeleted: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "mydgd", Namespace: "default", UID: "dgd-uid"},
			}
			follower := &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mydgd-decode-flw",
					Namespace: "default",
					Labels: map[string]string{
						consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
					},
					Annotations: map[string]string{
						consts.KubeAnnotationElasticEPFollower: consts.KubeLabelValueTrue,
					},
				},
				Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						Replicas: tt.replicas,
					},
				},
			}
			// Controlled by this DGD, so this test isolates the emptiness precondition
			// rather than passing because the ownership guard skipped the object.
			require.NoError(t, controllerutil.SetControllerReference(dgd, follower, s))

			c := fake.NewClientBuilder().WithScheme(s).WithObjects(follower).Build()
			r := &componentWorkloadsReconciler{syncer: newDGDResourceSyncer(c, nil)}

			t.Log("Generation no longer produces this follower, so the sweep considers it")
			require.NoError(t, r.deleteOrphanedElasticEPFollowers(
				context.Background(), dgd,
				map[string]*nvidiacomv1beta1.DynamoComponentDeployment{},
			))

			err := c.Get(context.Background(),
				types.NamespacedName{Name: follower.Name, Namespace: "default"},
				&nvidiacomv1beta1.DynamoComponentDeployment{})
			if tt.wantDeleted {
				require.True(t, err != nil && client.IgnoreNotFound(err) == nil,
					"an empty follower should be released, got err=%v", err)
				return
			}
			require.NoError(t, err,
				"a follower with replicas must survive: deleting it strands engine ranks the "+
					"operator has no way to shrink first")
		})
	}
}

// TestPreserveExistingDCDStateKeepsFollowerReplicas covers the defect that made the
// whole feature inert: the operator re-asserted the follower's resting zero on every
// reconcile, so nothing could ever scale it.
//
// synthesizeElasticEPFollowerDCD re-derives the follower from its leader each pass and
// stamps the declared launch width, and SyncResource classifies an externally written
// replica count as a manual change and copies the desired spec over it. Observed on a
// cluster: replicas 1 reverted to 0 within two seconds, with "Manual changes detected on
// DynamoComponentDeployment, will be overwritten" in the operator log. The declared width
// is the value to seed at creation, not to re-assert forever -- the scale client owns it
// after that.
//
// So an external scale is kept rather than snapped back: the operator simply stops
// writing the field.
//
// Mutation check: deleting the follower branch in preserveExistingDCDState fails every
// scaled subtest below.
func TestPreserveExistingDCDStateKeepsFollowerReplicas(t *testing.T) {
	s := scheme.Scheme
	require.NoError(t, nvidiacomv1beta1.AddToScheme(s))

	const ns = "default"
	existingDCD := func(name string, follower bool, replicas int32) *nvidiacomv1beta1.DynamoComponentDeployment {
		obj := &nvidiacomv1beta1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				BackendFramework: "vllm",
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					Replicas: ptr.To(replicas),
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

	tests := []struct {
		name         string
		existing     *nvidiacomv1beta1.DynamoComponentDeployment
		wantReplicas int32
	}{
		{
			// EP16 at TP4 is one leader plus three followers, so the scale client has to
			// be able to hold the follower above zero across reconciles.
			name:         "a scaled follower keeps the count the scale client wrote",
			existing:     existingDCD("mydgd-decode-flw", true, 3),
			wantReplicas: 3,
		},
		{
			name:         "scaling a follower back down is equally preserved",
			existing:     existingDCD("mydgd-decode-flw", true, 1),
			wantReplicas: 1,
		},
		{
			name:         "a follower still at rest stays at zero",
			existing:     existingDCD("mydgd-decode-flw", true, 0),
			wantReplicas: 0,
		},
		{
			// Only the synthesized follower's count is externally owned. A declared
			// component's replica count comes from the DGD and must still be enforced.
			name:         "a non-follower DCD is still driven by generation",
			existing:     existingDCD("mydgd-decode", false, 3),
			wantReplicas: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := fake.NewClientBuilder().WithScheme(s).WithObjects(tt.existing).Build()
			// The live replica count is authoritative and generation must not overwrite
			// it. This holds at either gate position; see the freeze subtests below.
			r := &componentWorkloadsReconciler{syncer: newDGDResourceSyncer(c, nil)}

			t.Log("Generation re-derives the follower and stamps the declared launch width")
			desired := &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:        tt.existing.Name,
					Namespace:   ns,
					Annotations: tt.existing.Annotations,
				},
				Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						Replicas: ptr.To(int32(0)),
					},
				},
			}
			require.NoError(t, r.preserveExistingDCDState(context.Background(), desired))

			require.NotNil(t, desired.Spec.Replicas)
			require.Equal(t, tt.wantReplicas, *desired.Spec.Replicas,
				"the operator must not own a synthesized follower's replica count after creation")
		})
	}

	// A running follower freezes wherever it is, in BOTH directions, whatever the declared
	// width says: scaled up 3 -> 5 stays 5, scaled down 3 -> 1 stays 1.
	//
	// The operator does not drag the count back to the declared width. Dragging it back is
	// the worse failure either way. Downward it deletes pods that may hold live engine
	// ranks, with nothing having called scale_elastic_ep to drain them (DYN-3838,
	// DYN-2660). Upward it re-adds capacity an operator deliberately removed.
	//
	// The reconciler does not read the gate at all, so the freeze is unconditional; these
	// cases assert the behaviour directly rather than per gate position.
	//
	// Mutation check: restoring a `desired`-wins branch for either direction fails the
	// matching subtest below.
	frozenAt := func(t *testing.T, running, declared int32) int32 {
		t.Helper()
		existing := existingDCD("mydgd-decode-flw", true, running)
		c := fake.NewClientBuilder().WithScheme(s).WithObjects(existing).Build()
		r := &componentWorkloadsReconciler{syncer: newDGDResourceSyncer(c, nil)}
		desired := &nvidiacomv1beta1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:        existing.Name,
				Namespace:   ns,
				Annotations: existing.Annotations,
			},
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					Replicas: ptr.To(declared),
				},
			},
		}
		require.NoError(t, r.preserveExistingDCDState(context.Background(), desired))
		require.NotNil(t, desired.Spec.Replicas)
		return *desired.Spec.Replicas
	}

	t.Run("a follower scaled above the declared width stays where it is", func(t *testing.T) {
		require.Equal(t, int32(5), frozenAt(t, 5, 3),
			"shrinking to the declared width would delete two pods that may hold live "+
				"engine ranks, undrained")
	})

	t.Run("a follower scaled below the declared width stays where it is", func(t *testing.T) {
		require.Equal(t, int32(1), frozenAt(t, 1, 3),
			"the count freezes in BOTH directions; growing back to the declared width "+
				"would re-add capacity an operator deliberately removed")
	})

	t.Run("a follower that does not exist yet is seeded at its declared width", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(s).Build()
		r := &componentWorkloadsReconciler{syncer: newDGDResourceSyncer(c, nil)}
		desired := &nvidiacomv1beta1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mydgd-decode-flw",
				Namespace: ns,
				Annotations: map[string]string{
					consts.KubeAnnotationElasticEPFollower: consts.KubeLabelValueTrue,
				},
			},
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					Replicas: ptr.To(int32(0)),
				},
			},
		}
		require.NoError(t, r.preserveExistingDCDState(context.Background(), desired))
		require.Equal(t, int32(0), *desired.Spec.Replicas, "creation must still seed the resting zero")
	})
}
