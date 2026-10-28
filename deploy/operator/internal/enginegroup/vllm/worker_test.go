//go:build !clustertest

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
	"strings"
	"sync"
	"testing"
	"time"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWorkerRegistrationDoesNotRequireControllerPermission(t *testing.T) {
	t.Log("the independently verified initial world registers before any traffic target")
	fixture := newMembershipFixture(t, 2, 3)
	workers := &workerControlFixture{engine: fixture.engine, registered: map[int32]bool{0: true, 1: true}}
	for i := range fixture.adapter.Processes {
		process := &fixture.adapter.Processes[i]
		process.PodUID = domain.PodUID(fmt.Sprintf("pod-%d", i))
		control, err := NewWorkerClient(fmt.Sprintf("http://rank-%d.test", i), &http.Client{Transport: workers})
		require.NoError(t, err)
		process.Client = control
	}
	groupID := domain.GroupID("group-uid")
	membership, err := fixture.adapter.Observe(t.Context(), groupID, "")
	require.NoError(t, err)
	capacity := &workerTestCapacity{processes: fixture.adapter.Processes, count: 2}
	observedCapacity, err := capacity.Observe(t.Context(), groupID)
	require.NoError(t, err)
	traffic := &WorkerTraffic{Membership: fixture.adapter, Journal: kubejournal.NewStore(fixture.adapter.Journal.Client, "test", "group", "group-uid", "traffic")}
	verifier := &WorkerVerifier{Membership: fixture.adapter}
	initialTraffic, err := traffic.Observe(t.Context(), groupID)
	require.NoError(t, err)
	require.Len(t, initialTraffic.Admitted, 2)
	require.Zero(t, initialTraffic.AppliedRevision)
	status, err := domain.NewGroupStatus(membership.CommittedTopology, observedCapacity, initialTraffic)
	require.NoError(t, err)
	target := fixture.target(t, membership.CommittedTopology, "grow-three", 10, 3, domain.PlanKindGrow)

	t.Log("reconcile until both old ranks have accepted native expansion")
	submitted := false
	for range 30 {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &target.Plan, status)
		require.NoError(t, err)
		status = result.Status
		fixture.engine.mu.Lock()
		submitted = len(fixture.engine.requests[0]) > 0 && len(fixture.engine.requests[1]) > 0
		fixture.engine.mu.Unlock()
		if submitted {
			break
		}
	}
	require.True(t, submitted)
	require.NotNil(t, status.Traffic.Accepted)
	previousRevision := status.Traffic.Accepted.ControlRevision

	t.Log("stop controller reconciliation; native commit and runtime verification register the joiner")
	fixture.engine.complete(3)
	workers.mu.Lock()
	workers.registered[2] = true
	workers.mu.Unlock()
	observed, err := traffic.Observe(t.Context(), groupID)
	require.NoError(t, err)
	assert.Len(t, observed.Admitted, 3)
	assert.Equal(t, previousRevision, observed.AppliedRevision, "registration needs no new controller revision")

	t.Log("observe the already-established proof after the controller restarts")
	for range 20 {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &target.Plan, status)
		require.NoError(t, err)
		status = result.Status
		if status.Transition.Outcome == domain.TransitionOutcomeCompleted {
			break
		}
	}
	require.Equal(t, domain.TransitionOutcomeCompleted, status.Transition.Outcome)
	assert.Zero(t, workers.mutations, "the controller must never grant serving permission or initiate warmup")
}

func TestWorkerServingEvidenceAndUnsupportedFailureWithdrawal(t *testing.T) {
	t.Log("observe runtime-owned verification and registration for one exact native incarnation")
	fixture := newMembershipFixture(t, 2, 2)
	workers := &workerControlFixture{engine: fixture.engine, registered: map[int32]bool{0: true, 1: true}}
	for i := range fixture.adapter.Processes {
		process := &fixture.adapter.Processes[i]
		process.PodUID = domain.PodUID(fmt.Sprintf("pod-%d", i))
		control, err := NewWorkerClient(fmt.Sprintf("http://rank-%d.test", i), &http.Client{Transport: workers})
		require.NoError(t, err)
		process.Client = control
	}
	groupID := domain.GroupID("group-uid")
	membership, err := fixture.adapter.Observe(t.Context(), groupID, "")
	require.NoError(t, err)
	traffic := &WorkerTraffic{Membership: fixture.adapter, Journal: kubejournal.NewStore(fixture.adapter.Journal.Client, "test", "group", "group-uid", "traffic")}
	verifier := &WorkerVerifier{Membership: fixture.adapter}
	verificationStarted := time.Now()
	proof, err := verifier.Verify(t.Context(), groupID, membership.CommittedTopology)
	verificationFinished := time.Now()
	require.NoError(t, err)
	require.NotNil(t, proof.Proof)
	assert.Equal(t, membership.CommittedTopology.Generation, proof.Proof.TopologyGeneration)
	assert.Equal(t, domain.TopologyRuntimeDigest(membership.CommittedTopology), proof.Proof.RuntimeDigest)
	assert.False(t, proof.Proof.ObservedAt.IsZero(), "a serving proof needs its actual observation time")
	assert.False(t, proof.Proof.ObservedAt.Before(verificationStarted))
	assert.False(t, proof.Proof.ObservedAt.After(verificationFinished))

	t.Log("failure-only withdrawal cannot substitute for native graceful retirement")
	result, err := traffic.Apply(t.Context(), groupID, domain.TrafficTarget{
		ControlRevision: 1, TransitionID: "retire", TopologyGeneration: 1,
		Drain: []domain.TrafficDrainTarget{{Membership: membership.CommittedTopology.Replicas[1], Mode: domain.TrafficDrainModeConfirmInactive}},
	})
	require.NoError(t, err)
	require.NotNil(t, result.Rejection)
	assert.Equal(t, "DrainUnsupported", result.Rejection.Reason)

	t.Log("a warming runtime cannot produce proof merely because the resize completed")
	workers.failProbe = true
	_, err = verifier.Verify(t.Context(), groupID, membership.CommittedTopology)
	require.Error(t, err)
	workers.failProbe = false

	t.Log("a stale native epoch cannot be presented as current registration or serving proof")
	workers.staleProof = true
	_, err = traffic.Observe(t.Context(), groupID)
	require.ErrorContains(t, err, "lacks current native serving evidence")
	_, err = verifier.Verify(t.Context(), groupID, membership.CommittedTopology)
	require.Error(t, err)
	workers.staleProof = false

	t.Log("a new native API lifetime invalidates the previous membership and serving proof")
	fixture.engine.mu.Lock()
	fixture.engine.states[0].InstanceID = "replacement"
	fixture.engine.mu.Unlock()
	_, err = verifier.Verify(t.Context(), groupID, membership.CommittedTopology)
	require.Error(t, err)
}

type workerControlFixture struct {
	mu                 sync.Mutex
	engine             *nativeMembershipEngine
	registered         map[int32]bool
	failProbe          bool
	staleProof         bool
	mutations          int
	drainDisabled      bool
	withdrawalPending  bool
	podUIDs            map[int32]domain.PodUID
	pendingWithdrawals map[int32]bool
}

func (w *workerControlFixture) RoundTrip(request *http.Request) (*http.Response, error) {
	var rank int32
	if _, err := fmt.Sscanf(request.URL.Hostname(), "rank-%d.test", &rank); err != nil {
		return nil, err
	}
	path := strings.TrimPrefix(request.URL.Path, "/engine/control")
	if path == statusPath || path == submitPath {
		copy := request.Clone(request.Context())
		copy.URL.Path = path
		return w.engine.handle(rank, copy)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if path != "/engine_group_status" {
		w.mutations++
		return nil, fmt.Errorf("unexpected serving mutation %s", path)
	}
	w.engine.mu.Lock()
	native := w.engine.states[rank]
	binding := nativeBinding(native)
	w.engine.mu.Unlock()
	if w.staleProof {
		epoch := "stale-epoch"
		binding.Epoch = &epoch
	}
	podUID := domain.PodUID(fmt.Sprintf("pod-%d", rank))
	if uid, found := w.podUIDs[rank]; found {
		podUID = uid
	}
	status := workerStatus{
		PodUID:   podUID,
		MemberID: domain.NativeMemberID(fmt.Sprintf("dp-%d", rank)), Binding: &binding,
		ListenerReady: true, Verified: !w.failProbe,
		Admitted:           w.registered[rank] && !w.failProbe,
		InstanceID:         native.InstanceID,
		Retired:            native.Phase == PhaseCompleted && native.TopologyKnown && rank >= native.EffectiveDataParallelSize,
		Withdrawn:          !w.registered[rank] && !w.withdrawalPending && !w.pendingWithdrawals[rank],
		NativeDrainEnabled: !w.drainDisabled,
	}
	if status.Retired {
		status.Binding, status.Verified = nil, false
	}
	payload, err := json.Marshal(status)
	if err != nil {
		return nil, err
	}
	return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(string(payload)))}, nil
}

type workerTestCapacity struct {
	processes []Process
	count     int32
	revision  int64
	releases  []domain.ReleaseFence
	width     int32
}

func (a *workerTestCapacity) Observe(_ context.Context, _ domain.GroupID) (domain.CapacityObservation, error) {
	observation := domain.CapacityObservation{AppliedRevision: a.revision, ReleaseFences: a.releases}
	width := a.width
	if width == 0 {
		width = 1
	}
	for index := int32(0); index < a.count; index++ {
		process := a.processes[index*width]
		members := make([]domain.NativeMemberIncarnation, 0, width)
		for offset := int32(0); offset < width; offset++ {
			members = append(members, a.processes[index*width+offset].Membership.Members...)
		}
		observation.Allocations = append(observation.Allocations, domain.CapacityAllocation{Available: true, Health: domain.AllocationHealthHealthy,
			Incarnation: domain.ReplicaIncarnation{ReplicaID: process.Membership.ReplicaID, SlotID: domain.CapacitySlotID(fmt.Sprintf("slot-%d", index)), Members: members, CapacityRefs: []domain.CapacityRef{{Name: fmt.Sprintf("pod-%d", index), UID: process.PodUID}}},
		})
	}
	return observation, nil
}

func (a *workerTestCapacity) Apply(_ context.Context, _ domain.GroupID, target domain.CapacityTarget) (domain.ApplyResult, error) {
	a.count, a.revision = int32(len(target.Replicas)), target.ControlRevision
	a.releases = target.ReleaseFences
	return domain.ApplyResult{}, nil
}

func TestNativeRetirementRequiresCorrelatedDrainAndRetainsProofAfterRelease(t *testing.T) {
	t.Log("start a three-rank world with runtime-owned registration")
	fixture := newMembershipFixture(t, 3, 3)
	workers := &workerControlFixture{engine: fixture.engine, registered: map[int32]bool{0: true, 1: true, 2: true}}
	for i := range fixture.adapter.Processes {
		process := &fixture.adapter.Processes[i]
		process.PodUID = domain.PodUID(fmt.Sprintf("pod-%d", i))
		control, err := NewWorkerClient(fmt.Sprintf("http://rank-%d.test", i), &http.Client{Transport: workers})
		require.NoError(t, err)
		process.Client = control
	}
	groupID := domain.GroupID("group-uid")
	base, err := fixture.adapter.Observe(t.Context(), groupID, "")
	require.NoError(t, err)
	capacity := &workerTestCapacity{processes: fixture.adapter.Processes, count: 3}
	allocated, err := capacity.Observe(t.Context(), groupID)
	require.NoError(t, err)
	traffic := &WorkerTraffic{Membership: fixture.adapter, Journal: kubejournal.NewStore(fixture.adapter.Journal.Client, "test", "group", "group-uid", "traffic")}
	registered, err := traffic.Observe(t.Context(), groupID)
	require.NoError(t, err)
	status, err := domain.NewGroupStatus(base.CommittedTopology, allocated, registered)
	require.NoError(t, err)
	target := fixture.target(t, base.CommittedTopology, "retire-two", 10, 2, domain.PlanKindRetire)
	target.Plan.TrafficRequirement = domain.TrafficRequirementEngineManagedDrain
	verifier := &WorkerVerifier{Membership: fixture.adapter}

	t.Log("a native-only adapter cannot infer worker drain support")
	preflight, err := fixture.adapter.ValidatePlan(t.Context(), groupID, domain.PlanValidationRequest{
		BaseTopology: base.CommittedTopology, Plan: target.Plan, PlanDigest: "retire-plan",
	})
	require.NoError(t, err)
	require.NotNil(t, preflight.Rejection)
	require.Equal(t, "UnsupportedDrainPolicy", preflight.Rejection.Reason)
	fixture.adapter.ManagedDrainValidator = ValidateWorkerManagedDrain

	t.Log("drain must be enabled on every old process before preparatory work")
	workers.drainDisabled = true
	preflight, err = fixture.adapter.ValidatePlan(t.Context(), groupID, domain.PlanValidationRequest{
		BaseTopology: base.CommittedTopology, Plan: target.Plan, PlanDigest: "retire-plan",
	})
	require.NoError(t, err)
	require.NotNil(t, preflight.Rejection)
	require.Equal(t, "NativeDrainDisabled", preflight.Rejection.Reason)
	workers.drainDisabled = false

	t.Log("restart the coordinator until native retirement is submitted with no terminal drain yet")
	for range 30 {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &target.Plan, status)
		require.NoError(t, err)
		status = result.Status
		if status.Membership.Desired != nil && len(fixture.engine.requests[2]) != 0 {
			break
		}
	}
	require.NotEmpty(t, fixture.engine.requests[2])
	require.Empty(t, status.Traffic.Observed.Drained)
	require.Equal(t, int32(3), capacity.count)

	t.Log("a committed topology without real discovery withdrawal does not authorize physical release")
	fixture.engine.complete(2)
	workers.registered[2] = false
	workers.withdrawalPending = true
	for range 8 {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &target.Plan, status)
		require.NoError(t, err)
		status = result.Status
	}
	require.Equal(t, int32(3), capacity.count)
	require.Empty(t, status.Traffic.Observed.Drained)

	t.Log("an unrelated retiree epoch also cannot become drain evidence")
	workers.withdrawalPending = false
	fixture.engine.states[2].Epoch = new("unrelated")
	observation, err := traffic.Observe(t.Context(), groupID)
	require.NoError(t, err)
	require.Empty(t, observation.Drained)
	fixture.engine.states[2].Epoch = fixture.engine.states[0].Epoch
	observation, err = traffic.Observe(t.Context(), groupID)
	require.NoError(t, err)
	require.Len(t, observation.Drained, 1)

	t.Log("restart the traffic adapter after the retired endpoint disappears; only its persisted proof remains")
	fixture.adapter.Processes = fixture.adapter.Processes[:2]
	traffic = &WorkerTraffic{Membership: fixture.adapter, Journal: traffic.Journal}
	observation, err = traffic.Observe(t.Context(), groupID)
	require.NoError(t, err)
	require.Len(t, observation.Drained, 1)

	t.Log("matching completion and withdrawal permit one release and final serving verification")
	for range 25 {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &target.Plan, status)
		require.NoError(t, err)
		status = result.Status
		if status.Transition.Outcome == domain.TransitionOutcomeCompleted {
			break
		}
	}
	require.Equal(t, domain.TransitionOutcomeCompleted, status.Transition.Outcome)
	require.Equal(t, int32(2), capacity.count)
	require.Zero(t, workers.mutations)

	t.Log("a terminal record survives loss of the retired endpoint without treating absence as new proof")
	observation, err = traffic.Observe(t.Context(), groupID)
	require.NoError(t, err)
	// Completion may already have cleared the retired target; either way no absent endpoint is queried.
	require.Len(t, observation.Admitted, 2)
}
