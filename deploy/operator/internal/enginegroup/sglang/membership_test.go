/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"context"
	"fmt"
	"net/http"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestLegacyGrowthAdapterCommitsCorrelatedGrowth(t *testing.T) {
	ctx := context.Background()
	engineSize := int32(1)
	httpClient := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		switch request.URL.Path {
		case statePath:
			return jsonResponse(fmt.Sprintf(
				`{"is_scaling_elastic_ep":false,"effective_ep_size":%d,"scale_phase":"serving_expanded"}`,
				engineSize,
			)), nil
		case scalePath:
			return jsonResponse(`{"status":"ok","old_ep_size":1,"new_ep_size":2}`), nil
		default:
			return nil, fmt.Errorf("unexpected path %s", request.URL.Path)
		}
	})}
	control, err := NewClient("http://sglang.test:9090", httpClient)
	require.NoError(t, err)
	capacity := &testCapacityObserver{allocations: []enginegroup.CapacityAllocation{testAllocation(0)}}
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	adapter := &LegacyGrowthAdapter{
		Client:             control,
		Capacity:           capacity,
		ProfileFingerprint: "profile-v1",
		Journal:            kubejournal.NewStore(kubeClient, "test", "group", types.UID("group-uid"), "membership"),
	}

	t.Log("establish the already-running EP1 world as topology generation one")
	initial, err := adapter.Observe(ctx, "group-uid", "")
	require.NoError(t, err)
	require.Equal(t, int64(1), initial.CommittedTopology.Generation)
	require.Len(t, initial.CommittedTopology.Replicas, 1)

	plan := testGrowPlan()
	t.Log("preflight the concrete SGLang growth semantics before capacity work")
	preflight, err := adapter.ValidatePlan(ctx, "group-uid", enginegroup.PlanValidationRequest{
		BaseTopology: initial.CommittedTopology,
		Plan:         plan,
		PlanDigest:   "plan-digest",
	})
	require.NoError(t, err)
	require.NotNil(t, preflight.Evidence)

	target := enginegroup.MembershipTarget{
		ControlRevision: 1,
		TransitionID:    "grow-1-2",
		TargetDigest:    "target-digest",
		Validation:      *preflight.Evidence,
		BaseTopology:    initial.CommittedTopology,
		Plan:            plan,
		Joining: []enginegroup.JoiningReplica{{
			ReplicaID:          "replica-1",
			RuntimeIncarnation: "pod-1",
		}},
	}
	targetPreflight, err := adapter.ValidateTarget(ctx, "group-uid", target)
	require.NoError(t, err)
	require.NotNil(t, targetPreflight.Evidence)
	target.Validation = *targetPreflight.Evidence

	t.Log("persist the correlated transition before invoking SGLang")
	require.NoError(t, adapter.Apply(ctx, "group-uid", target))

	t.Log("observe the engine commit and bind generation two to the exact joining incarnation")
	capacity.allocations = append(capacity.allocations, testAllocation(1))
	engineSize = 2
	observed, err := adapter.Observe(ctx, "group-uid", target.TransitionID)
	require.NoError(t, err)
	require.NotNil(t, observed.Transition)
	assert.Equal(t, enginegroup.MembershipTransitionPhaseCommitted, observed.Transition.Phase)
	require.NotNil(t, observed.Transition.ResultTopology)
	assert.Equal(t, int64(2), observed.Transition.ResultTopology.Generation)
	assert.Len(t, observed.Transition.ResultTopology.Replicas, 2)
}

func TestLegacyGrowthAdapterDoesNotRedispatchFromStaleJournal(t *testing.T) {
	tests := []struct {
		name   string
		absent bool
	}{
		{name: "older baseline"},
		{name: "cached absence", absent: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("form the baseline and preserve its cached snapshot")
			ctx := context.Background()
			scaleCalls := 0
			control, err := NewClient("http://sglang.test:9090", &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				if request.URL.Path == statePath {
					return jsonResponse(`{"is_scaling_elastic_ep":false,"effective_ep_size":1,"scale_phase":"serving"}`), nil
				}
				scaleCalls++
				return nil, context.DeadlineExceeded
			})})
			require.NoError(t, err)
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			kubeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
			adapter := &LegacyGrowthAdapter{
				Client: control, Capacity: &testCapacityObserver{allocations: []enginegroup.CapacityAllocation{testAllocation(0)}},
				ProfileFingerprint: "profile-v1",
				Journal:            kubejournal.NewStore(kubeClient, "test", "group", "group-uid", "membership"),
			}
			initial, err := adapter.Observe(ctx, "group-uid", "")
			require.NoError(t, err)
			baseline := &corev1.ConfigMap{}
			require.NoError(t, kubeClient.Get(ctx, client.ObjectKey{Namespace: adapter.Journal.Namespace, Name: adapter.Journal.Name}, baseline))
			target := enginegroup.MembershipTarget{
				ControlRevision: 1, TransitionID: "grow-1-2", TargetDigest: "target-digest",
				BaseTopology: initial.CommittedTopology, Plan: testGrowPlan(),
				Joining: []enginegroup.JoiningReplica{{ReplicaID: "replica-1", RuntimeIncarnation: "pod-1"}},
			}

			t.Log("persist an ambiguous result after dispatching the collective once")
			require.ErrorIs(t, adapter.Apply(ctx, "group-uid", target), context.DeadlineExceeded)
			require.Equal(t, 1, scaleCalls)
			adapter.Journal.Client = &staleJournalClient{Client: kubeClient, baseline: baseline, absent: test.absent}

			t.Log("refuse submission from stale evidence before another backend effect")
			if test.absent {
				_, err = adapter.Observe(ctx, "group-uid", target.TransitionID)
				assert.True(t, apierrors.IsAlreadyExists(err), "%v", err)
			} else {
				_, err = adapter.Observe(ctx, "group-uid", target.TransitionID)
				require.NoError(t, err)
				err = adapter.Apply(ctx, "group-uid", target)
				assert.True(t, apierrors.IsConflict(err), "%v", err)
			}
			assert.Equal(t, 1, scaleCalls)

			t.Log("observe the original Unknown result once the cache catches up")
			adapter.Journal.Client = kubeClient
			observed, err := adapter.Observe(ctx, "group-uid", target.TransitionID)
			require.NoError(t, err)
			require.NotNil(t, observed.Transition)
			assert.Equal(t, enginegroup.MembershipTransitionPhaseUnknown, observed.Transition.Phase)
		})
	}
}

type staleJournalClient struct {
	client.Client
	baseline *corev1.ConfigMap
	absent   bool
}

func (c *staleJournalClient) Get(_ context.Context, key client.ObjectKey, out client.Object, _ ...client.GetOption) error {
	if c.absent {
		return apierrors.NewNotFound(schema.GroupResource{Resource: "configmaps"}, key.Name)
	}
	*out.(*corev1.ConfigMap) = *c.baseline.DeepCopy()
	return nil
}

func TestLegacyGrowthAdapterFailsClosedAfterAmbiguousDispatch(t *testing.T) {
	ctx := context.Background()
	httpClient := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path == statePath {
			return jsonResponse(`{"is_scaling_elastic_ep":false,"effective_ep_size":1,"scale_phase":"serving"}`), nil
		}
		return nil, context.DeadlineExceeded
	})}
	control, err := NewClient("http://sglang.test:9090", httpClient)
	require.NoError(t, err)
	capacity := &testCapacityObserver{allocations: []enginegroup.CapacityAllocation{testAllocation(0)}}
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	adapter := &LegacyGrowthAdapter{
		Client: control, Capacity: capacity, ProfileFingerprint: "profile-v1",
		Journal: kubejournal.NewStore(kubeClient, "test", "group", types.UID("group-uid"), "membership"),
	}
	initial, err := adapter.Observe(ctx, "group-uid", "")
	require.NoError(t, err)
	preflight, err := adapter.ValidatePlan(ctx, "group-uid", enginegroup.PlanValidationRequest{
		BaseTopology: initial.CommittedTopology, Plan: testGrowPlan(), PlanDigest: "plan-digest",
	})
	require.NoError(t, err)
	target := enginegroup.MembershipTarget{
		ControlRevision: 1, TransitionID: "grow-1-2", TargetDigest: "target-digest",
		Validation: *preflight.Evidence, BaseTopology: initial.CommittedTopology, Plan: testGrowPlan(),
		Joining: []enginegroup.JoiningReplica{{ReplicaID: "replica-1", RuntimeIncarnation: "pod-1"}},
	}
	targetEvidence, err := adapter.ValidateTarget(ctx, "group-uid", target)
	require.NoError(t, err)
	target.Validation = *targetEvidence.Evidence

	t.Log("treat a transport timeout after durable intent as an ambiguous outcome")
	require.ErrorIs(t, adapter.Apply(ctx, "group-uid", target), context.DeadlineExceeded)
	observed, err := adapter.Observe(ctx, "group-uid", target.TransitionID)
	require.NoError(t, err)
	require.NotNil(t, observed.Transition)
	assert.Equal(t, enginegroup.MembershipTransitionPhaseUnknown, observed.Transition.Phase)

	t.Log("do not dispatch the same collective a second time while its outcome is unknown")
	require.NoError(t, adapter.Apply(ctx, "group-uid", target))
}

type testCapacityObserver struct {
	allocations []enginegroup.CapacityAllocation
}

func (o *testCapacityObserver) Observe(context.Context, enginegroup.GroupID) (enginegroup.CapacityObservation, error) {
	return enginegroup.CapacityObservation{Allocations: append([]enginegroup.CapacityAllocation(nil), o.allocations...)}, nil
}

func testAllocation(rank int) enginegroup.CapacityAllocation {
	id := fmt.Sprintf("replica-%d", rank)
	uid := fmt.Sprintf("pod-%d", rank)
	return enginegroup.CapacityAllocation{
		Available: true,
		Incarnation: enginegroup.ReplicaIncarnation{
			ReplicaID:          enginegroup.ReplicaID(id),
			SlotID:             enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", rank)),
			RuntimeIncarnation: enginegroup.RuntimeIncarnationID(uid),
			CapacityRefs:       []enginegroup.CapacityRef{{Name: uid, UID: enginegroup.PodUID(uid)}},
		},
	}
}

func testGrowPlan() enginegroup.ResolvedPlan {
	return enginegroup.ResolvedPlan{
		ID:                      "grow-1-2",
		ProfileFingerprint:      "profile-v1",
		ProcessLifecycleOwner:   enginegroup.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      enginegroup.TrafficRequirementKeepServing,
		VerificationRequirement: enginegroup.VerificationRequirementRequired,
		Change: enginegroup.ResolvedChange{
			Kind: enginegroup.PlanKindGrow,
			Grow: &enginegroup.GrowChange{Replicas: []enginegroup.ReplicaTarget{{
				ReplicaID: "replica-1", SlotID: "slot-1", Bootstrap: enginegroup.BootstrapModeJoin,
				NativeMembers: []enginegroup.NativeMemberID{"dp-1"},
			}}},
		},
	}
}
