/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestMembershipCommitsExactGrowthAfterRestart(t *testing.T) {
	t.Log("adopt the already-running DP2 world without creating a bootstrap operation")
	fixture := newMembershipFixture(t, 2, 4)
	initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
	require.NoError(t, err)
	assert.Equal(t, int64(1), initial.CommittedTopology.Generation)
	assert.Nil(t, initial.Transition)

	t.Log("preflight a two-allocation append and persist its exact target before any native request")
	target := fixture.target(t, initial.CommittedTopology, "grow-2-4", 1, 4, enginegroup.PlanKindGrow)
	require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", target))
	assert.Len(t, fixture.engine.requests, 2)
	require.Len(t, fixture.engine.requests[0], 1)
	assert.Equal(t, target.TransitionID, fixture.engine.requests[0][0].OperationID)
	assert.Equal(t, "api-0", fixture.engine.requests[0][0].ExpectedInstanceID)

	t.Log("reconstruct the adapter and recover the same Pending operation from native status")
	restarted := *fixture.adapter
	observation, err := restarted.Observe(t.Context(), "group-uid", target.TransitionID)
	require.NoError(t, err)
	require.NotNil(t, observation.Transition)
	assert.Equal(t, enginegroup.MembershipTransitionPhasePending, observation.Transition.Phase)
	assert.True(t, enginegroup.SameTopology(initial.CommittedTopology, observation.CommittedTopology))

	t.Log("commit the native world, correlate every new process, and retain the immutable result")
	fixture.engine.complete(4)
	observation, err = restarted.Observe(t.Context(), "group-uid", target.TransitionID)
	require.NoError(t, err)
	require.NotNil(t, observation.Transition.ResultTopology)
	assert.Equal(t, enginegroup.MembershipTransitionPhaseCommitted, observation.Transition.Phase)
	assert.Equal(t, int64(2), observation.CommittedTopology.Generation)
	assert.Equal(t, int32(4), observation.CommittedTopology.NativeMemberCount())
	assert.True(t, enginegroup.SameTopology(observation.CommittedTopology, *observation.Transition.ResultTopology))
	assert.Equal(t, target.Joining[1], observation.CommittedTopology.Replicas[3])

	t.Log("a terminal replay uses the durable receipt without dispatching again")
	require.NoError(t, restarted.Apply(t.Context(), "group-uid", target))
	assert.Len(t, fixture.engine.requests[0], 1)
	assert.Len(t, fixture.engine.requests[1], 1)
	assert.Zero(t, fixture.engine.epochs-1)
}

func TestMembershipRepairsPartialFanoutAfterRestart(t *testing.T) {
	t.Log("persist a DP2 baseline and let only rank zero accept the first fan-out")
	fixture := newMembershipFixture(t, 2, 3)
	initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
	require.NoError(t, err)
	target := fixture.target(t, initial.CommittedTopology, "partial-grow", 1, 3, enginegroup.PlanKindGrow)
	fixture.engine.failNext[1] = true
	require.ErrorIs(t, fixture.adapter.Apply(t.Context(), "group-uid", target), context.DeadlineExceeded)
	require.Len(t, fixture.engine.requests[0], 1)
	assert.Empty(t, fixture.engine.requests[1])

	t.Log("native shared Pending status is not mistaken for proof every API received the request")
	restarted := *fixture.adapter
	observed, err := restarted.Observe(t.Context(), "group-uid", target.TransitionID)
	require.NoError(t, err)
	assert.Equal(t, enginegroup.MembershipTransitionPhasePending, observed.Transition.Phase)
	require.NoError(t, restarted.Apply(t.Context(), "group-uid", target))
	require.Len(t, fixture.engine.requests[1], 1)
	assert.Equal(t, target.TransitionID, fixture.engine.requests[1][0].OperationID)
	assert.Equal(t, "api-1", fixture.engine.requests[1][0].ExpectedInstanceID)
	assert.Equal(t, 1, fixture.engine.epochs)

	t.Log("one original native operation completes after the missing participant is repaired")
	fixture.engine.complete(3)
	observed, err = restarted.Observe(t.Context(), "group-uid", target.TransitionID)
	require.NoError(t, err)
	assert.Equal(t, enginegroup.MembershipTransitionPhaseCommitted, observed.Transition.Phase)
}

func TestMembershipKeepsCommitUncertaintySeparateFromFailure(t *testing.T) {
	tests := []struct {
		name  string
		phase Phase
		known bool
		error string
	}{
		{name: "normal commit is temporarily unknown", phase: PhaseCommitting},
		{name: "failed commit is not rejected", phase: PhaseFailed, error: "collective failed"},
		{name: "failed prepare with known base still lacks a no-later-commit guarantee", phase: PhaseFailed, known: true, error: "prepare failed"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("start one durable native resize and then change only authoritative engine evidence")
			fixture := newMembershipFixture(t, 2, 3)
			initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
			require.NoError(t, err)
			target := fixture.target(t, initial.CommittedTopology, "uncertain-grow", 1, 3, enginegroup.PlanKindGrow)
			require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", target))
			fixture.engine.changePhase(test.phase, test.known, test.error)

			t.Log("never turn last-known membership or an HTTP/native failure into rollback authority")
			observed, err := fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
			if !test.known {
				require.ErrorIs(t, err, ErrTopologyUnknown)
				assert.Empty(t, observed.CommittedTopology.Replicas)
			} else {
				require.NoError(t, err)
				require.NotNil(t, observed.Transition)
				assert.Equal(t, enginegroup.MembershipTransitionPhaseUnknown, observed.Transition.Phase)
				assert.Equal(t, "NativeResizeFailed", observed.Transition.Failure.Reason)
			}

			t.Log("when native authority recovers, Pending carries no stale failure and commit remains observable")
			fixture.engine.changePhase(PhasePreparing, true, "")
			observed, err = fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
			require.NoError(t, err)
			assert.Equal(t, enginegroup.MembershipTransitionPhasePending, observed.Transition.Phase)
			assert.Nil(t, observed.Transition.Failure)
			fixture.engine.complete(3)
			observed, err = fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
			require.NoError(t, err)
			assert.Equal(t, enginegroup.MembershipTransitionPhaseCommitted, observed.Transition.Phase)
		})
	}
}

func TestMembershipRejectsConflictingAndCompetingTargets(t *testing.T) {
	t.Log("accept one exact target and retain its complete payload")
	fixture := newMembershipFixture(t, 2, 3)
	initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
	require.NoError(t, err)
	target := fixture.target(t, initial.CommittedTopology, "grow-once", 1, 3, enginegroup.PlanKindGrow)
	require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", target))

	t.Log("reusing the ID and digest with another payload cannot change the accepted plan")
	altered := target
	altered.Plan.VerificationRequirement = enginegroup.VerificationRequirementNone
	require.ErrorContains(t, fixture.adapter.Apply(t.Context(), "group-uid", altered), "different exact payload")

	t.Log("a newer control revision cannot supersede pending native work")
	competing := target
	competing.TransitionID, competing.ControlRevision = "other-operation", 2
	require.ErrorContains(t, fixture.adapter.Apply(t.Context(), "group-uid", competing), "non-terminal")
	assert.Len(t, fixture.engine.requests[0], 1)
	assert.Len(t, fixture.engine.requests[1], 1)
}

func TestMembershipDetectsIdentityEpochAndObservationDrift(t *testing.T) {
	tests := []struct {
		name    string
		mutate  string
		wantErr string
	}{
		{name: "API process restarted", mutate: "instance", wantErr: "pinned API process"},
		{name: "workload process restarted", mutate: "workload", wantErr: "frozen process incarnation"},
		{name: "same operation received another epoch", mutate: "epoch", wantErr: "changed the native epoch"},
		{name: "latest operation superseded externally", mutate: "operation", wantErr: "forgot or superseded"},
		{name: "temporarily skewed per-rank observations", mutate: "skew", wantErr: "do not yet agree"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("start one native resize with immutable process and epoch correlation")
			fixture := newMembershipFixture(t, 2, 3)
			initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
			require.NoError(t, err)
			target := fixture.target(t, initial.CommittedTopology, "bound-grow", 1, 3, enginegroup.PlanKindGrow)
			require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", target))
			_, err = fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
			require.NoError(t, err)

			t.Log("change one identity or observation without granting adoption authority")
			switch test.mutate {
			case "instance":
				fixture.engine.states[0].InstanceID = "restarted-api"
			case "workload":
				fixture.adapter.Processes[0].Membership.Members[0].RuntimeIncarnation = "restarted-container"
			case "epoch":
				for i := range fixture.engine.states {
					fixture.engine.states[i].Epoch = new("different-epoch")
				}
			case "operation":
				for i := range fixture.engine.states {
					fixture.engine.states[i].OperationID = new("another-operation")
				}
			case "skew":
				fixture.engine.states[1].Phase = PhaseCommitting
			}
			_, err = fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
			require.ErrorContains(t, err, test.wantErr)
			assert.Len(t, fixture.engine.requests[0], 1)
		})
	}
}

func TestMembershipShrinksAndRestoresOnlyTheSuffix(t *testing.T) {
	t.Log("observe DP3 and issue planned suffix retirement to every old rank, including the retiree")
	fixture := newMembershipFixture(t, 3, 3)
	initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
	require.NoError(t, err)
	shrink := fixture.target(t, initial.CommittedTopology, "retire-tail", 1, 2, enginegroup.PlanKindRetire)
	require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", shrink))
	assert.Len(t, fixture.engine.requests[2], 1)
	fixture.engine.complete(2)
	shrunk, err := fixture.adapter.Observe(t.Context(), "group-uid", shrink.TransitionID)
	require.NoError(t, err)
	assert.Equal(t, int32(2), shrunk.CommittedTopology.NativeMemberCount())
	assert.Equal(t, int64(2), shrunk.CommittedTopology.Generation)

	t.Log("retaining the terminal receipt no longer requires contacting the released tail API")
	fixture.engine.unreachable[2] = true
	observed, err := fixture.adapter.Observe(t.Context(), "group-uid", shrink.TransitionID)
	require.NoError(t, err)
	assert.True(t, reflect.DeepEqual(shrunk.Transition, observed.Transition))

	t.Log("restore the fully retired tail with a new workload and API incarnation, not survivor masking")
	fixture.adapter.Processes[2].Membership.Members[0].RuntimeIncarnation = "replacement-runtime-2"
	fixture.engine.states[2].InstanceID = "replacement-api-2"
	restore := fixture.target(t, shrunk.CommittedTopology, "restore-tail", 2, 3, enginegroup.PlanKindRestore)
	require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", restore))
	fixture.engine.unreachable[2] = false
	fixture.engine.complete(3)
	restored, err := fixture.adapter.Observe(t.Context(), "group-uid", restore.TransitionID)
	require.NoError(t, err)
	assert.Equal(t, int64(3), restored.CommittedTopology.Generation)
	assert.Equal(t, "replacement-runtime-2", string(restored.CommittedTopology.Replicas[2].Members[0].RuntimeIncarnation))
}

func TestMembershipPreflightHasNoEffects(t *testing.T) {
	tests := []struct {
		name   string
		change string
		reason string
	}{
		{name: "selected non-tail retirement", change: "non-tail", reason: "UnsupportedRetirement"},
		{name: "whole-world retirement", change: "zero", reason: "UnsupportedRetirement"},
		{name: "growth above hard bounds", change: "maximum", reason: "UnsupportedGrowth"},
		{name: "survivor masking", change: "survivors", reason: "UnsupportedOperation"},
		{name: "serving proof omitted", change: "verification", reason: "UnsupportedPlan"},
		{name: "allocation width mismatch", change: "width", reason: "PartialMembershipUnsupported"},
		{name: "unresolved plan digest", change: "digest", reason: "InvalidPlan"},
		{name: "ambiguous change union", change: "union", reason: "InvalidChange"},
		{name: "mismatched change tag", change: "tag", reason: "InvalidChange"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("adopt a world and construct one unsupported declarative operation")
			fixture := newMembershipFixture(t, 2, 4)
			initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
			require.NoError(t, err)
			base := initial.CommittedTopology
			plan := enginegroup.ResolvedPlan{
				ID: "preflight", ProfileFingerprint: fixture.adapter.ProfileFingerprint,
				ProcessLifecycleOwner:   enginegroup.ProcessLifecycleOwnerOrchestrator,
				TrafficRequirement:      enginegroup.TrafficRequirementKeepServing,
				VerificationRequirement: enginegroup.VerificationRequirementRequired,
				Change: enginegroup.ResolvedChange{Kind: enginegroup.PlanKindGrow, Grow: &enginegroup.GrowChange{
					Replicas: []enginegroup.ReplicaTarget{{ReplicaID: "replica-2", SlotID: "slot-2", Bootstrap: enginegroup.BootstrapModeJoin, NativeMembers: []enginegroup.NativeMemberID{"dp-2"}}},
				}},
			}
			digest := "canonical-plan-digest"
			switch test.change {
			case "non-tail":
				plan.Change = enginegroup.ResolvedChange{Kind: enginegroup.PlanKindRetire, Retire: &enginegroup.RetireChange{Replicas: []enginegroup.ReplicaID{"replica-0"}}}
			case "zero":
				plan.Change = enginegroup.ResolvedChange{Kind: enginegroup.PlanKindRetire, Retire: &enginegroup.RetireChange{Replicas: []enginegroup.ReplicaID{"replica-0", "replica-1"}}}
			case "maximum":
				fixture.adapter.MaxDataParallelSize = 2
			case "survivors":
				plan.Change = enginegroup.ResolvedChange{Kind: enginegroup.PlanKindReduceToSurvivors, ReduceToSurvivors: &enginegroup.ReduceToSurvivorsChange{}}
			case "verification":
				plan.VerificationRequirement = enginegroup.VerificationRequirementNone
			case "width":
				base.Replicas[0].Members = append(base.Replicas[0].Members, enginegroup.NativeMemberIncarnation{ID: "dp-2", RuntimeIncarnation: "packed-process"})
			case "digest":
				digest = ""
			case "union":
				plan.Change.Retire = &enginegroup.RetireChange{Replicas: []enginegroup.ReplicaID{"replica-1"}}
			case "tag":
				plan.Change.Kind = enginegroup.PlanKindRetire
			}

			t.Log("return a structured rejection without dispatching or rewriting durable evidence")
			before, _, err := fixture.adapter.load(t.Context())
			require.NoError(t, err)
			result, err := fixture.adapter.ValidatePlan(t.Context(), "group-uid", enginegroup.PlanValidationRequest{BaseTopology: base, Plan: plan, PlanDigest: digest})
			require.NoError(t, err)
			require.NotNil(t, result.Rejection)
			assert.Equal(t, test.reason, result.Rejection.Reason)
			after, _, err := fixture.adapter.load(t.Context())
			require.NoError(t, err)
			assert.Equal(t, before, after)
			assert.Empty(t, fixture.engine.requests)
		})
	}
}

func TestMembershipStaleJournalCannotRedispatch(t *testing.T) {
	for _, absent := range []bool{false, true} {
		t.Run(fmt.Sprintf("stale absence=%t", absent), func(t *testing.T) {
			t.Log("preserve the original baseline snapshot before any request reaches a native API")
			fixture := newMembershipFixture(t, 2, 3)
			initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
			require.NoError(t, err)
			target := fixture.target(t, initial.CommittedTopology, "persisted-before-send", 1, 3, enginegroup.PlanKindGrow)
			originalClient := fixture.adapter.Journal.Client
			baseline := &corev1.ConfigMap{}
			require.NoError(t, originalClient.Get(t.Context(), client.ObjectKey{
				Namespace: fixture.adapter.Journal.Namespace, Name: fixture.adapter.Journal.Name,
			}, baseline))
			fixture.engine.failNext[0], fixture.engine.failNext[1] = true, true
			require.ErrorIs(t, fixture.adapter.Apply(t.Context(), "group-uid", target), context.DeadlineExceeded)

			t.Log("a stale reader cannot overwrite the accepted target and dispatch native effects")
			fixture.adapter.Journal.Client = &staleMembershipClient{Client: originalClient, baseline: baseline, absent: absent}
			err = fixture.adapter.Apply(t.Context(), "group-uid", target)
			if absent {
				assert.True(t, apierrors.IsAlreadyExists(err), "%v", err)
			} else {
				assert.True(t, apierrors.IsConflict(err), "%v", err)
			}
			assert.Empty(t, fixture.engine.requests)

			t.Log("fresh observation recovers the original accepted intent and can safely resume it")
			fixture.adapter.Journal.Client = originalClient
			observed, err := fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
			require.NoError(t, err)
			assert.Equal(t, enginegroup.MembershipTransitionPhasePending, observed.Transition.Phase)
			require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", target))
			assert.Equal(t, 1, fixture.engine.epochs)
		})
	}
}

func TestMembershipDurableBindingValidation(t *testing.T) {
	tests := []struct {
		name   string
		mutate string
	}{
		{name: "incompatible format", mutate: "version"},
		{name: "different profile", mutate: "profile"},
		{name: "missing process identity", mutate: "instances"},
		{name: "changed committed result", mutate: "result"},
		{name: "wrong receipt correlation", mutate: "digest"},
		{name: "lost accepted target", mutate: "target"},
		{name: "journal pretends to own the native phase", mutate: "phase"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("complete a native operation and retain its terminal correlation receipt")
			fixture := newMembershipFixture(t, 2, 3)
			initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
			require.NoError(t, err)
			target := fixture.target(t, initial.CommittedTopology, "durable-grow", 1, 3, enginegroup.PlanKindGrow)
			require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", target))
			fixture.engine.complete(3)
			_, err = fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
			require.NoError(t, err)
			state, snapshot, err := fixture.adapter.load(t.Context())
			require.NoError(t, err)

			t.Log("corrupt one private evidence field; it must not authorize another native request")
			switch test.mutate {
			case "version":
				state.Version++
			case "profile":
				state.ProfileFingerprint = "different-profile"
			case "instances":
				state.Instances[1] = ""
			case "result":
				state.Result.ResultTopology.Generation++
			case "digest":
				state.Result.TargetDigest = "another-target"
			case "target":
				state.Target = nil
			case "phase":
				state.Result.Phase = enginegroup.MembershipTransitionPhasePending
			}
			_, err = fixture.adapter.Journal.Save(t.Context(), snapshot, state)
			require.NoError(t, err)
			require.Error(t, fixture.adapter.Apply(t.Context(), "group-uid", target))
			assert.Len(t, fixture.engine.requests[0], 1)
		})
	}
}

func TestMembershipRetainsLocalRejectionAndScopedAbsence(t *testing.T) {
	t.Log("a future transition has no binding while the initial committed world remains observable")
	fixture := newMembershipFixture(t, 2, 3)
	initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "future-grow")
	require.NoError(t, err)
	assert.Nil(t, initial.Transition)
	target := fixture.target(t, initial.CommittedTopology, "future-grow", 1, 3, enginegroup.PlanKindGrow)

	t.Log("a capability change before dispatch yields a durable local rejection, not a native transaction")
	fixture.adapter.MinDataParallelSize = 2
	require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", target))
	observed, err := fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
	require.NoError(t, err)
	require.NotNil(t, observed.Transition)
	assert.Equal(t, enginegroup.MembershipTransitionPhaseRejected, observed.Transition.Phase)
	assert.Equal(t, "CapabilitiesChanged", observed.Transition.Failure.Reason)
	assert.True(t, enginegroup.SameTopology(initial.CommittedTopology, observed.CommittedTopology))
	assert.Empty(t, fixture.engine.requests)

	t.Log("another requested ID is absent; asking about the rejected ID still returns the immutable result")
	other, err := fixture.adapter.Observe(t.Context(), "group-uid", "another-transition")
	require.NoError(t, err)
	assert.Nil(t, other.Transition)
	restarted := *fixture.adapter
	afterRestart, err := restarted.Observe(t.Context(), "group-uid", target.TransitionID)
	require.NoError(t, err)
	assert.Equal(t, observed.Transition, afterRestart.Transition)
}

func TestMembershipRequiresJoiningProcessEvidenceAfterNativeCommit(t *testing.T) {
	t.Log("new endpoints need not be serving before native prepare starts its joining handshake")
	fixture := newMembershipFixture(t, 2, 3)
	fixture.engine.unreachable[2] = true
	initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
	require.NoError(t, err)
	target := fixture.target(t, initial.CommittedTopology, "joiner-proof", 1, 3, enginegroup.PlanKindGrow)
	require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", target))
	fixture.engine.complete(3)

	t.Log("a completed count without the exact joining process cannot become a committed topology")
	_, err = fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	state, _, err := fixture.adapter.load(t.Context())
	require.NoError(t, err)
	assert.Nil(t, state.Result)
	assert.Equal(t, initial.CommittedTopology, state.Topology)

	t.Log("commit becomes observable only when every frozen joining process supplies correlated evidence")
	fixture.engine.unreachable[2] = false
	observed, err := fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
	require.NoError(t, err)
	assert.Equal(t, enginegroup.MembershipTransitionPhaseCommitted, observed.Transition.Phase)
}

func TestMembershipDoesNotReplayForgottenNativeOperation(t *testing.T) {
	t.Log("observe and persist the native epoch independently of the acceptance fan-out")
	fixture := newMembershipFixture(t, 2, 3)
	initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
	require.NoError(t, err)
	target := fixture.target(t, initial.CommittedTopology, "remembered-grow", 1, 3, enginegroup.PlanKindGrow)
	require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", target))
	_, err = fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
	require.NoError(t, err)

	t.Log("a return to idle is loss of authority, not proof that the accepted collective never existed")
	for i := range fixture.engine.states {
		fixture.engine.states[i] = Status{
			InstanceID: fmt.Sprintf("api-%d", i), Phase: PhaseIdle, EffectiveDataParallelSize: 2, TopologyKnown: true,
		}
	}
	_, err = fixture.adapter.Observe(t.Context(), "group-uid", target.TransitionID)
	require.ErrorContains(t, err, "lost a previously accepted local submission")
	require.ErrorContains(t, fixture.adapter.Apply(t.Context(), "group-uid", target), "lost a previously accepted local submission")
	assert.Len(t, fixture.engine.requests[0], 1)
	assert.Equal(t, 1, fixture.engine.epochs)
}

func TestMembershipReplayKeepsItsOriginalNativePayload(t *testing.T) {
	t.Log("accept one native target with its original per-rank drain budget")
	fixture := newMembershipFixture(t, 2, 3)
	initial, err := fixture.adapter.Observe(t.Context(), "group-uid", "")
	require.NoError(t, err)
	target := fixture.target(t, initial.CommittedTopology, "immutable-native-payload", 1, 3, enginegroup.PlanKindGrow)
	require.NoError(t, fixture.adapter.Apply(t.Context(), "group-uid", target))

	t.Log("a changed operator default must not change requests when replaying an accepted operation")
	restarted := *fixture.adapter
	restarted.DrainTimeout = 1
	require.NoError(t, restarted.Apply(t.Context(), "group-uid", target))
	for rank := range int32(2) {
		require.Len(t, fixture.engine.requests[rank], 2)
		assert.Equal(t, fixture.engine.requests[rank][0], fixture.engine.requests[rank][1])
	}

	t.Log("reassertion also refuses a different physical joiner while the original collective is pending")
	restarted.Processes[2].Membership.Members[0].RuntimeIncarnation = "unexpected-new-joiner"
	require.ErrorContains(t, restarted.Apply(t.Context(), "group-uid", target), "frozen joining process")
	assert.Len(t, fixture.engine.requests[0], 2)
}

type staleMembershipClient struct {
	client.Client
	baseline *corev1.ConfigMap
	absent   bool
}

func (c *staleMembershipClient) Get(_ context.Context, key client.ObjectKey, out client.Object, _ ...client.GetOption) error {
	if c.absent {
		return apierrors.NewNotFound(schema.GroupResource{Resource: "configmaps"}, key.Name)
	}
	*out.(*corev1.ConfigMap) = *c.baseline.DeepCopy()
	return nil
}

type membershipFixture struct {
	adapter *MembershipAdapter
	engine  *nativeMembershipEngine
}

func newMembershipFixture(t *testing.T, initial, maximum int32) *membershipFixture {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	fixture := &membershipFixture{
		adapter: &MembershipAdapter{
			Layout:             ReplicaLayout{NativeMembersPerReplica: 1},
			ProfileFingerprint: "profile-v1", MinDataParallelSize: 1, MaxDataParallelSize: maximum, DrainTimeout: 120,
			Journal: kubejournal.NewStore(fake.NewClientBuilder().WithScheme(scheme).Build(), "test", "group", "group-uid", "vllm-membership"),
		},
		engine: &nativeMembershipEngine{
			requests: make(map[int32][]ScaleRequest), failNext: make(map[int32]bool), unreachable: make(map[int32]bool),
		},
	}
	fixture.engine.journal = fixture.adapter.Journal
	for rank := range maximum {
		fixture.engine.states = append(fixture.engine.states, Status{
			InstanceID: fmt.Sprintf("api-%d", rank), Phase: PhaseIdle, EffectiveDataParallelSize: initial, TopologyKnown: true,
			AsynchronousSubmission: true, TopologyToken: fmt.Sprintf("initial-%d", rank),
		})
		control, err := NewClient(fmt.Sprintf("http://rank-%d.test:8000", rank), &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			return fixture.engine.handle(rank, request)
		})})
		require.NoError(t, err)
		fixture.adapter.Processes = append(fixture.adapter.Processes, Process{
			Rank: rank, Client: control,
			Membership: enginegroup.ReplicaMembership{
				ReplicaID: enginegroup.ReplicaID(fmt.Sprintf("replica-%d", rank)),
				Members: []enginegroup.NativeMemberIncarnation{{
					ID: enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", rank)), RuntimeIncarnation: enginegroup.RuntimeIncarnationID(fmt.Sprintf("runtime-%d", rank)),
				}},
			},
		})
	}
	return fixture
}

func (f *membershipFixture) target(t *testing.T, base enginegroup.MembershipTopology, id string, revision int64, size int32, kind enginegroup.PlanKind) enginegroup.MembershipTarget {
	t.Helper()
	plan := enginegroup.ResolvedPlan{
		ID: id, ProfileFingerprint: f.adapter.ProfileFingerprint,
		ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:    enginegroup.TrafficRequirementKeepServing, VerificationRequirement: enginegroup.VerificationRequirementRequired,
		Change: enginegroup.ResolvedChange{Kind: kind},
	}
	joining := make([]enginegroup.JoiningReplica, 0)
	if kind == enginegroup.PlanKindRetire {
		plan.Change.Retire = &enginegroup.RetireChange{}
		for _, replica := range base.Replicas {
			for _, member := range replica.Members {
				rank, err := NativeRank(member.ID)
				require.NoError(t, err)
				if rank >= size {
					plan.Change.Retire.Replicas = append(plan.Change.Retire.Replicas, replica.ReplicaID)
					break
				}
			}
		}
	} else {
		for rank := base.NativeMemberCount(); rank < size; rank++ {
			process, err := f.adapter.process(rank)
			require.NoError(t, err)
			membership := process.Membership
			membership.Members = append([]enginegroup.NativeMemberIncarnation(nil), membership.Members...)
			joining = appendMemberSubset(joining, membership)
		}
		targets := make([]enginegroup.ReplicaTarget, 0, len(joining))
		for _, membership := range joining {
			rank, err := NativeRank(membership.Members[0].ID)
			require.NoError(t, err)
			ids := make([]enginegroup.NativeMemberID, len(membership.Members))
			for i, member := range membership.Members {
				ids[i] = member.ID
			}
			targets = append(targets, enginegroup.ReplicaTarget{
				ReplicaID: membership.ReplicaID, SlotID: enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", rank/f.adapter.Layout.NativeMembersPerReplica)),
				Bootstrap: enginegroup.BootstrapModeJoin, NativeMembers: ids,
			})
		}
		if kind == enginegroup.PlanKindRestore {
			plan.Change.Restore = &enginegroup.RestoreChange{}
			for _, target := range targets {
				target.Bootstrap = enginegroup.BootstrapModeRestoreFixedSlot
				plan.Change.Restore.Replicas = append(plan.Change.Restore.Replicas, enginegroup.RestorationTarget{ReplicaTarget: target})
			}
		} else {
			plan.Change.Grow = &enginegroup.GrowChange{Replicas: targets}
		}
	}
	preflight, err := f.adapter.ValidatePlan(t.Context(), "group-uid", enginegroup.PlanValidationRequest{
		BaseTopology: base, Plan: plan, PlanDigest: id + "/plan-digest",
	})
	require.NoError(t, err)
	require.NotNil(t, preflight.Evidence, "%#v", preflight.Rejection)
	target := enginegroup.MembershipTarget{
		ControlRevision: revision, TransitionID: id, TargetDigest: id + "/target-digest",
		Validation: *preflight.Evidence, BaseTopology: base, Plan: plan, Joining: joining,
	}
	exact, err := f.adapter.ValidateTarget(t.Context(), "group-uid", target)
	require.NoError(t, err)
	require.NotNil(t, exact.Evidence, "%#v", exact.Rejection)
	target.Validation = *exact.Evidence
	return target
}

type nativeMembershipEngine struct {
	mu          sync.Mutex
	journal     kubejournal.Store
	states      []Status
	requests    map[int32][]ScaleRequest
	failNext    map[int32]bool
	unreachable map[int32]bool
	epochs      int
}

func (e *nativeMembershipEngine) handle(rank int32, request *http.Request) (*http.Response, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.unreachable[rank] {
		return nil, context.DeadlineExceeded
	}
	if request.URL.Path == submitPath {
		var target ScaleRequest
		if err := json.NewDecoder(request.Body).Decode(&target); err != nil {
			return nil, err
		}
		if e.failNext[rank] {
			e.failNext[rank] = false
			return nil, context.DeadlineExceeded
		}
		binding := membershipBinding{}
		if _, err := e.journal.Load(request.Context(), &binding); err != nil || binding.Target == nil || binding.Target.TransitionID != target.OperationID {
			return nil, fmt.Errorf("native effect occurred before its complete target was durable: %v", err)
		}
		if target.ExpectedInstanceID != e.states[rank].InstanceID {
			return &http.Response{StatusCode: http.StatusConflict, Body: io.NopCloser(strings.NewReader("stale instance"))}, nil
		}
		previous := e.states[rank].LocalSubmission
		if previous != nil && previous.Request.OperationID == target.OperationID {
			if previous.Request != target {
				return &http.Response{StatusCode: http.StatusConflict, Body: io.NopCloser(strings.NewReader("conflicting request"))}, nil
			}
		} else if target.BaseTopologyToken != e.states[rank].TopologyToken {
			return &http.Response{StatusCode: http.StatusConflict, Body: io.NopCloser(strings.NewReader("stale base"))}, nil
		}
		e.requests[rank] = append(e.requests[rank], target)
		if e.states[rank].OperationID == nil || *e.states[rank].OperationID != target.OperationID {
			e.epochs++
			for i := range e.states {
				e.states[i].OperationID, e.states[i].Epoch = new(target.OperationID), new(fmt.Sprintf("epoch-%d", e.epochs))
				e.states[i].RequestedDataParallelSize = new(target.NewDataParallelSize)
				e.states[i].Phase, e.states[i].TopologyKnown, e.states[i].Error = PhasePreparing, true, nil
			}
		}
		receipt := SubmissionReceipt{Request: target, FirstRank: rank, LocalSize: 1, Started: true}
		for i := int(rank) + 1; i < len(e.states) && e.states[i].InstanceID == e.states[rank].InstanceID; i++ {
			receipt.LocalSize++
		}
		e.states[rank].LocalSubmission = &receipt
		payload, err := json.Marshal(receipt)
		return &http.Response{StatusCode: http.StatusAccepted, Body: io.NopCloser(strings.NewReader(string(payload)))}, err
	}
	payload, err := json.Marshal(e.states[rank])
	if err != nil {
		return nil, err
	}
	return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(string(payload)))}, nil
}

func (e *nativeMembershipEngine) complete(size int32) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for i := range e.states {
		e.states[i].EffectiveDataParallelSize = size
		e.states[i].Phase, e.states[i].TopologyKnown, e.states[i].Error = PhaseCompleted, true, nil
		e.states[i].TopologyToken = fmt.Sprintf("committed-%d-%s", e.epochs, e.states[i].InstanceID)
		if e.states[i].LocalSubmission != nil {
			e.states[i].LocalSubmission.Finished = true
		}
	}
}

func (e *nativeMembershipEngine) changePhase(phase Phase, known bool, message string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for i := range e.states {
		e.states[i].Phase, e.states[i].TopologyKnown, e.states[i].Error = phase, known, nil
		if message != "" {
			e.states[i].Error = new(message)
		}
	}
}
