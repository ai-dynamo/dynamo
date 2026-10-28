/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
	"net/http"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	vllmruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/vllm"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// vllmWorkerRuntimeResolver composes physical allocations and the Python worker's native controls.
// Client and HTTP must not be nil. Resolution is read-only, including before first formation.
type vllmWorkerRuntimeResolver struct {
	Client client.Client
	HTTP   *http.Client
}

// Resolve binds physical allocation and native membership to runtime-owned serving observations.
func (p *vllmWorkerRuntimeResolver) Resolve(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup, world WorkloadBinding) (Runtime, error) {
	// Native-server and sidecar deployments need supported remote serving observations.
	if group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] != string(dynamo.BackendFrameworkVLLM) {
		return Runtime{}, fmt.Errorf("%w: the vLLM runtime supports the Python worker only", ErrRuntimeUnavailable)
	}
	capacity, err := resolveVLLMCapacity(ctx, p.Client, group, world)
	if err != nil {
		return Runtime{}, err
	}
	if !dynamo.IsVLLMWorkerLaunch(capacity.Source) {
		return Runtime{}, fmt.Errorf("%w: vLLM managed resizing requires the Dynamo Python worker", ErrRuntimeUnavailable)
	}
	profile, err := dynamo.ResolveVLLMLaunchProfile(capacity.Source)
	if err != nil {
		return Runtime{}, err
	}
	membership, err := ResolveVLLMMembership(ctx, capacity, p.HTTP)
	if err != nil {
		return Runtime{}, err
	}
	membership.DrainTimeout = 120
	membership.ManagedDrainValidator = vllmruntime.ValidateWorkerManagedDrain

	// The traffic journal fences withdrawal revisions; it never grants registration.
	traffic := &vllmruntime.WorkerTraffic{
		Membership: membership,
		Journal:    runtimeJournal(p.Client, group, world.ID, "vllm-traffic"),
	}
	return Runtime{
		Workload: world,
		Profile: api.EngineGroupProfileStatus{
			Backend: "vllm", Fingerprint: profile.Geometry.Fingerprint,
			GPUsPerReplica: profile.Geometry.GPUsPerReplica, PodsPerReplica: profile.Geometry.PodsPerReplica,
			NativeMembersPerReplica: profile.NativeMembersPerReplica, MinSafeServingNativeMembers: membership.MinDataParallelSize,
			MinSupportedReplicas: profile.InitialReplicas, MaxSupportedReplicas: profile.MaximumReplicas,
		},
		Capacity: capacity, Membership: membership, Traffic: traffic,
		Verifier: &vllmruntime.WorkerVerifier{Membership: membership},
		Planner: vllmruntime.ScalePlanner{
			ProfileFingerprint: profile.Geometry.Fingerprint, Minimum: profile.InitialReplicas, Maximum: profile.MaximumReplicas,
			Layout: membership.Layout,
		},
	}, nil
}
