/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

// NativeBinding is engine-observed evidence, not a Dynamo topology generation.
type NativeBinding struct {
	InstanceID       string  `json:"instance_id"`
	OperationID      *string `json:"operation_id"`
	Epoch            *string `json:"epoch"`
	DataParallelSize int32   `json:"data_parallel_size"`
}

type workerStatus struct {
	PodUID             enginegroup.PodUID         `json:"pod_uid"`
	MemberID           enginegroup.NativeMemberID `json:"member_id"`
	Binding            *NativeBinding             `json:"binding"`
	Verified           bool                       `json:"verified"`
	Admitted           bool                       `json:"admitted"`
	ListenerReady      bool                       `json:"listener_ready"`
	InstanceID         string                     `json:"instance_id"`
	Retired            bool                       `json:"retired"`
	Withdrawn          bool                       `json:"withdrawn"`
	NativeDrainEnabled bool                       `json:"native_drain_enabled"`
}

func (c *Client) workerRequest(ctx context.Context, path string, result any) error {
	// Serving observation remains reachable independently of inference registration.
	if c.routePrefix != "/engine/control" {
		return fmt.Errorf("worker observation requires the Dynamo worker transport")
	}
	ctx, cancel := context.WithTimeout(ctx, c.ObservationTimeout)
	defer cancel()
	payload, err := c.request(ctx, path, nil)
	if err != nil {
		return err
	}
	return json.Unmarshal(payload, result)
}

func nativeBinding(status Status) NativeBinding {
	return NativeBinding{InstanceID: status.InstanceID, OperationID: status.OperationID, Epoch: status.Epoch, DataParallelSize: status.EffectiveDataParallelSize}
}

// ValidateWorkerManagedDrain binds native drain configuration to the exact worker incarnation.
func ValidateWorkerManagedDrain(ctx context.Context, groupID enginegroup.GroupID, process Process, instanceID string) (*enginegroup.Failure, error) {
	// A new or unreachable API server cannot attest to a prior process's drain configuration.
	worker, err := observeWorker(ctx, process)
	if err != nil {
		return nil, err
	}
	if worker.InstanceID != instanceID {
		return nil, fmt.Errorf("native drain configuration belongs to another API instance")
	}
	if !worker.NativeDrainEnabled {
		return terminalFailure("NativeDrainDisabled", "every old vLLM worker must enable native request draining"), nil
	}
	return nil, nil
}

func (a *MembershipAdapter) servingBindings(ctx context.Context, groupID enginegroup.GroupID, topology enginegroup.MembershipTopology) ([]NativeBinding, error) {
	// Correlate runtime-owned serving proof with the engine's exact committed topology.
	if _, err := a.Observe(ctx, groupID, ""); err != nil {
		return nil, err
	}
	state, _, err := a.load(ctx)
	if err != nil {
		return nil, err
	}
	if !enginegroup.SameTopology(state.Topology, topology) || (state.Target != nil && state.Result == nil) {
		return nil, fmt.Errorf("serving evidence requires exact committed membership")
	}
	statuses, err := a.readTopology(ctx, topology, state.Instances)
	if err != nil {
		return nil, err
	}
	bindings := make([]NativeBinding, len(statuses))
	for i, status := range statuses {
		if status.Phase != PhaseIdle && status.Phase != PhaseCompleted {
			return nil, ErrTopologyUnknown
		}
		bindings[i] = nativeBinding(status)
	}
	return bindings, nil
}

// WorkerVerifier observes proof already established by the serving runtime.
// It cannot initiate warmup or grant registration.
type WorkerVerifier struct{ Membership *MembershipAdapter }

// Verify correlates independently produced proof with pre/post native membership evidence.
func (v *WorkerVerifier) Verify(ctx context.Context, groupID enginegroup.GroupID, topology enginegroup.MembershipTopology) (enginegroup.VerificationResult, error) {
	bindings, err := v.Membership.servingBindings(ctx, groupID, topology)
	if err != nil {
		return enginegroup.VerificationResult{}, err
	}
	for rank, binding := range bindings {
		process, err := v.Membership.process(int32(rank))
		if err != nil {
			return enginegroup.VerificationResult{}, err
		}
		status, err := observeWorker(ctx, process)
		if err != nil {
			return enginegroup.VerificationResult{}, err
		}
		if !status.Verified || !reflect.DeepEqual(status.Binding, &binding) {
			return enginegroup.VerificationResult{}, fmt.Errorf("rank %d has not established serving proof for the current topology", rank)
		}
	}
	after, err := v.Membership.servingBindings(ctx, groupID, topology)
	if err != nil {
		return enginegroup.VerificationResult{}, err
	}
	if !reflect.DeepEqual(bindings, after) {
		return enginegroup.VerificationResult{}, fmt.Errorf("native topology changed during serving observation")
	}

	// Timestamp the fresh verification, never a later projection of the persisted proof.
	return enginegroup.VerificationResult{Proof: &enginegroup.ServingProof{
		TopologyGeneration: topology.Generation, RuntimeDigest: enginegroup.TopologyRuntimeDigest(topology),
		ObservedAt: time.Now().UTC(),
	}}, nil
}

func observeWorker(ctx context.Context, process Process) (workerStatus, error) {
	status := workerStatus{}
	if err := process.Client.workerRequest(ctx, "/engine_group_status", &status); err != nil {
		return status, err
	}
	// The resolver proves group ownership; the endpoint proves its exact Pod and native process identity.
	if status.PodUID != process.PodUID ||
		status.MemberID != process.Membership.Members[0].ID || process.PodUID == "" {
		return status, fmt.Errorf("worker serving observation belongs to another incarnation")
	}
	return status, nil
}
