/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"fmt"
	"math"
	"reflect"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
)

const membershipBindingVersion = 1

// Process binds one API/client lifetime to the contiguous native range it owns.
// Rank is the first global DP rank; Membership freezes every local engine incarnation.
// Client must address this owner, not an individual child engine.
type Process struct {
	Rank       int32
	PodUID     enginegroup.PodUID
	Membership enginegroup.ReplicaMembership
	Client     *Client
}

// ManagedDrainValidator proves a process can drain inside native resize.
// A nil validator disables engine-managed drain; it does not imply support.
type ManagedDrainValidator func(ctx context.Context, groupID enginegroup.GroupID, process Process, instanceID string) (*enginegroup.Failure, error)

// MembershipAdapter maps planned append, tail retirement, and tail restoration to vLLM's correlated external resize API.
// The private binding stores Dynamo generations, exact target payloads, and process-to-API identity correlation;
// native vLLM status remains authoritative for pending, completion, failure, and topology certainty.
// Fault-tolerant masking and independently observed serving health are separate runtime integration requirements.
// The runtime must give this adapter sole resize-writer ownership and profile-derived capability bounds.
type MembershipAdapter struct {
	Processes             []Process
	Layout                ReplicaLayout
	ProfileFingerprint    string
	MinDataParallelSize   int32
	MaxDataParallelSize   int32
	DrainTimeout          int32
	ManagedDrainValidator ManagedDrainValidator
	Journal               kubejournal.Store
}

type membershipBinding struct {
	Version            int                                          `json:"version"`
	ProfileFingerprint string                                       `json:"profileFingerprint"`
	Topology           enginegroup.MembershipTopology               `json:"topology"`
	Instances          []string                                     `json:"instances"`
	TopologyTokens     []string                                     `json:"topologyTokens"`
	NativeOperationID  string                                       `json:"nativeOperationID"`
	NativeEpoch        string                                       `json:"nativeEpoch"`
	Target             *enginegroup.MembershipTarget                `json:"target,omitempty"`
	DrainTimeout       int32                                        `json:"drainTimeout,omitempty"`
	Epoch              string                                       `json:"epoch,omitempty"`
	Result             *enginegroup.MembershipTransitionObservation `json:"result,omitempty"`
	Accepted           []SubmissionReceipt                          `json:"accepted,omitempty"`
}

// Apply durably binds the complete target before dispatching once to every old client.
// Replays observe first and preserve the original target, API instance preconditions, and native epoch.
func (a *MembershipAdapter) Apply(ctx context.Context, groupID enginegroup.GroupID, target enginegroup.MembershipTarget) error {
	// Establish fresh native evidence before interpreting a transport error or replaying an accepted intent.
	observed, err := a.Observe(ctx, groupID, target.TransitionID)
	if err != nil {
		return err
	}
	state, snapshot, err := a.load(ctx)
	if err != nil {
		return err
	}
	if state.Target != nil && state.Target.TransitionID == target.TransitionID {
		if !reflect.DeepEqual(*state.Target, target) {
			return fmt.Errorf("vLLM transition %q was reused with a different exact payload", target.TransitionID)
		}
		if state.Result != nil {
			return nil
		}
		if observed.Transition != nil && observed.Transition.Phase == enginegroup.MembershipTransitionPhaseUnknown {
			return fmt.Errorf("vLLM transition %q has an unresolved native failure", target.TransitionID)
		}
	} else {
		// A newer target may replace only an acknowledged terminal binding, never pending or ambiguous work.
		if state.Target != nil && (state.Result == nil || target.ControlRevision <= state.Target.ControlRevision) {
			return fmt.Errorf("the previous vLLM transition is non-terminal or the target revision is stale")
		}
		if !enginegroup.SameTopology(state.Topology, target.BaseTopology) {
			return fmt.Errorf("vLLM target base no longer matches the committed binding")
		}
		preflight, validationErr := a.ValidateTarget(ctx, groupID, target)
		if validationErr != nil {
			return validationErr
		}
		if target.Validation.TargetDigest != target.TargetDigest {
			return fmt.Errorf("vLLM target lacks exact-target validation evidence")
		}
		state.Target = &target
		state.DrainTimeout = a.DrainTimeout
		state.Epoch, state.Result = "", nil
		state.Accepted = nil
		if preflight.Rejection != nil {
			state.Result = transitionResult(target, enginegroup.MembershipTransitionPhaseRejected)
			state.Result.Failure = preflight.Rejection
		}
	}
	if state.Result == nil {
		if err := a.validateJoining(target); err != nil {
			return fmt.Errorf("vLLM accepted target lost its frozen joining process: %w", err)
		}
	}

	// Compare-and-swap even an unchanged replay: a stale reader must not submit after a newer target was accepted.
	snapshot, err = a.Journal.Save(ctx, snapshot, state)
	if err != nil {
		return err
	}
	if state.Result != nil {
		return nil
	}
	size, failure := a.resolvePlan(target.BaseTopology, target.Plan)
	if failure != nil {
		return fmt.Errorf("validated vLLM plan became unsupported: %s", failure.Message)
	}
	participants := make([]Participant, 0, len(a.Processes))
	for _, process := range a.Processes {
		if process.Rank < int32(len(state.Instances)) {
			participants = append(participants, Participant{Rank: process.Rank, LocalSize: int32(len(process.Membership.Members)),
				ExpectedInstanceID: state.Instances[process.Rank], BaseTopologyToken: state.TopologyTokens[process.Rank], Client: process.Client})
		}
	}
	results, applyErr := (Resize{
		OperationID: target.TransitionID, BaseDataParallelSize: target.BaseTopology.NativeMemberCount(),
		NewDataParallelSize: size, DrainTimeout: state.DrainTimeout, Participants: participants,
	}).Apply(ctx)

	// Persist scoped receipts independently from global epoch progress; another client may still need delivery.
	changed := false
	for _, result := range results {
		if result.Err != nil {
			continue
		}
		added, err := a.recordAcceptance(&state, result.Receipt)
		if err != nil {
			return err
		}
		changed = changed || added
	}
	if changed {
		if _, err := a.Journal.Save(ctx, snapshot, state); err != nil {
			return fmt.Errorf("persist vLLM client acceptance: %w", err)
		}
	}
	return applyErr
}

func (a *MembershipAdapter) validate(groupID enginegroup.GroupID) error {
	// Reject a mismatched world or invalid resolved configuration before journal writes and backend requests.
	if groupID == "" || groupID != enginegroup.GroupID(a.Journal.Owner.UID) {
		return fmt.Errorf("vLLM adapter binding does not belong to the requested Engine Group")
	}
	if a.ProfileFingerprint == "" || a.MinDataParallelSize < 1 || a.MaxDataParallelSize < a.MinDataParallelSize || a.DrainTimeout <= 0 {
		return fmt.Errorf("vLLM adapter requires a profile, positive safe minimum and drain timeout, and valid maximum")
	}
	if _, err := a.Layout.NativeSize(0); err != nil {
		return err
	}
	ranks := make(map[int32]bool, len(a.Processes))
	replicas := make(map[enginegroup.ReplicaID]enginegroup.PodUID, len(a.Processes))
	pods := make(map[enginegroup.PodUID]enginegroup.ReplicaID, len(a.Processes))
	runtimes := make(map[enginegroup.RuntimeIncarnationID]bool, len(a.Processes))
	for _, process := range a.Processes {
		membership := process.Membership
		if process.Rank < 0 || membership.ReplicaID == "" || len(membership.Members) == 0 || process.Client == nil {
			return fmt.Errorf("vLLM runtime binding needs a client and a nonempty native range")
		}
		for offset, member := range membership.Members {
			rank := int64(process.Rank) + int64(offset)
			if rank > math.MaxInt32 || ranks[int32(rank)] || member.ID != enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", rank)) ||
				member.RuntimeIncarnation == "" || runtimes[member.RuntimeIncarnation] {
				return fmt.Errorf("vLLM runtime binding needs disjoint contiguous DP ranges and frozen engine identities")
			}
			ranks[int32(rank)], runtimes[member.RuntimeIncarnation] = true, true
		}
		// Members may share an allocation, but a Pod cannot host unrelated allocations or change their binding.
		if pod, exists := replicas[membership.ReplicaID]; exists && pod != process.PodUID {
			return fmt.Errorf("one vLLM allocation cannot span different Pods in this realization")
		}
		if process.PodUID != "" {
			if replica, exists := pods[process.PodUID]; exists && replica != membership.ReplicaID {
				return fmt.Errorf("one Pod cannot host unrelated vLLM replica allocations")
			}
			pods[process.PodUID] = membership.ReplicaID
		}
		replicas[membership.ReplicaID] = process.PodUID
	}
	return nil
}

func (a *MembershipAdapter) load(ctx context.Context) (membershipBinding, kubejournal.Snapshot, error) {
	state := membershipBinding{}
	snapshot, err := a.Journal.Load(ctx, &state)
	if err != nil || !snapshot.Exists() {
		return state, snapshot, err
	}
	if state.Version != membershipBindingVersion || state.ProfileFingerprint != a.ProfileFingerprint ||
		state.Topology.Generation <= 0 || len(state.Instances) != int(state.Topology.NativeMemberCount()) ||
		len(state.TopologyTokens) != len(state.Instances) {
		return membershipBinding{}, snapshot, fmt.Errorf("vLLM membership binding has an incompatible version, profile, or topology")
	}
	if _, err := membersByRank(state.Topology); err != nil {
		return membershipBinding{}, snapshot, err
	}
	instances := make(map[string]int32, len(state.Instances))
	for rank, instance := range state.Instances {
		process, err := a.process(int32(rank))
		if err != nil {
			return membershipBinding{}, snapshot, err
		}
		owner, exists := instances[instance]
		if !validIdentifier(instance) || !validIdentifier(state.TopologyTokens[rank]) || (exists && owner != process.Rank) {
			return membershipBinding{}, snapshot, fmt.Errorf("vLLM membership binding lacks a native API process identity")
		}
		instances[instance] = process.Rank
	}
	if err := a.validateBinding(state); err != nil {
		return membershipBinding{}, snapshot, err
	}
	return state, snapshot, nil
}

func (a *MembershipAdapter) validateBinding(state membershipBinding) error {
	// The binding owns correlation, not a second native transaction phase machine.
	if (state.NativeOperationID == "") != (state.NativeEpoch == "") {
		return fmt.Errorf("vLLM committed binding has incomplete native operation correlation")
	}
	if state.Target == nil {
		if state.Result != nil || state.Epoch != "" || state.NativeOperationID != "" || state.Topology.Generation != 1 || len(state.Accepted) != 0 {
			return fmt.Errorf("vLLM binding lost its retained operation target or result")
		}
		return nil
	}
	target := *state.Target
	if !validIdentifier(target.TransitionID) || target.ControlRevision <= 0 || target.TargetDigest == "" ||
		target.Validation.TargetDigest != target.TargetDigest || target.BaseTopology.Generation <= 0 || state.DrainTimeout <= 0 {
		return fmt.Errorf("vLLM binding contains incomplete exact-target evidence")
	}
	if _, err := membersByRank(target.BaseTopology); err != nil {
		return err
	}
	if state.Result == nil {
		if _, failure := a.resolvePlan(target.BaseTopology, target.Plan); failure != nil {
			return fmt.Errorf("vLLM pending binding has an unsupported plan: %s", failure.Message)
		}
		if !enginegroup.SameTopology(state.Topology, target.BaseTopology) {
			return fmt.Errorf("vLLM pending binding changed its committed base")
		}
		previous := int32(-1)
		for _, receipt := range state.Accepted {
			if receipt.FirstRank <= previous {
				return fmt.Errorf("vLLM accepted client receipts are duplicated or unordered")
			}
			if err := a.validateAcceptance(state, receipt); err != nil {
				return err
			}
			previous = receipt.FirstRank
		}
		return nil
	}
	if len(state.Accepted) != 0 {
		return fmt.Errorf("vLLM terminal binding retains pending delivery bookkeeping")
	}
	return a.validateTerminalBinding(state)
}

func (a *MembershipAdapter) validateTerminalBinding(state membershipBinding) error {
	// A retained terminal receipt must be exactly the immutable result of the recorded target.
	target := *state.Target
	result := state.Result
	if result.TransitionID != target.TransitionID || result.ControlRevision != target.ControlRevision || result.TargetDigest != target.TargetDigest {
		return fmt.Errorf("vLLM terminal receipt does not correlate with its target")
	}
	switch result.Phase {
	case enginegroup.MembershipTransitionPhaseCommitted:
		if _, failure := a.resolvePlan(target.BaseTopology, target.Plan); failure != nil {
			return fmt.Errorf("vLLM committed binding has an unsupported plan: %s", failure.Message)
		}
		if result.ResultTopology == nil || result.Failure != nil || state.Epoch == "" ||
			state.NativeOperationID != target.TransitionID || state.NativeEpoch != state.Epoch ||
			!enginegroup.SameTopology(*result.ResultTopology, resultTopology(target)) ||
			!enginegroup.SameTopology(state.Topology, *result.ResultTopology) {
			return fmt.Errorf("vLLM committed receipt does not match the exact topology result")
		}
	case enginegroup.MembershipTransitionPhaseRejected:
		if result.ResultTopology != nil || result.Failure == nil ||
			result.Failure.Classification != enginegroup.FailureClassificationTerminal || result.Failure.Reason == "" ||
			state.Epoch != "" || !enginegroup.SameTopology(state.Topology, target.BaseTopology) {
			return fmt.Errorf("vLLM rejection does not prove a locally unsubmitted target on its unchanged base")
		}
	default:
		return fmt.Errorf("vLLM binding must not persist a competing native phase machine")
	}
	return nil
}

func (a *MembershipAdapter) process(rank int32) (Process, error) {
	for _, process := range a.Processes {
		if rank >= process.Rank && int64(rank) < int64(process.Rank)+int64(len(process.Membership.Members)) {
			return process, nil
		}
	}
	return Process{}, fmt.Errorf("vLLM DP rank %d has no runtime endpoint binding", rank)
}

func (a *MembershipAdapter) processForMember(member enginegroup.NativeMemberID) (Process, error) {
	rank, err := nativeRank(member)
	if err != nil {
		return Process{}, err
	}
	return a.process(rank)
}

func (a *MembershipAdapter) capabilityGeneration() string {
	return fmt.Sprintf("vllm-external-elastic-ep/correlated-v1/%d/%d/%d/drain-%t", a.Layout.NativeMembersPerReplica, a.MinDataParallelSize, a.MaxDataParallelSize, a.ManagedDrainValidator != nil)
}

func transitionResult(target enginegroup.MembershipTarget, phase enginegroup.MembershipTransitionPhase) *enginegroup.MembershipTransitionObservation {
	return &enginegroup.MembershipTransitionObservation{
		TransitionID: target.TransitionID, ControlRevision: target.ControlRevision,
		TargetDigest: target.TargetDigest, Phase: phase,
	}
}

var _ enginegroup.MembershipAdapter = (*MembershipAdapter)(nil)
