//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"net/http"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVLLMWorkerRuntimeResolvesBeforeBootstrapWithoutNativeRequests(t *testing.T) {
	t.Log("resolve the production provider from the projected Pod cache before APIs or assignments exist")
	kube, group := newVLLMRuntimeTestWorldForTransport(t, true)
	transport := &vllmNoNativeRequests{}
	provider := &runtimeProvider{Client: kube, Workload: &GroveWorkloadProvider{Client: kube}, HTTP: &http.Client{Transport: transport}}
	runtime, err := provider.Resolve(t.Context(), group)
	require.NoError(t, err)
	require.NoError(t, validateEngineGroupRuntime(runtime))
	assert.Empty(t, transport.hosts)
	assert.Equal(t, int32(2), runtime.Profile.MinSupportedReplicas)
	assert.Equal(t, int32(4), runtime.Profile.MaxSupportedReplicas)

	t.Log("bootstrap is a separate lifecycle-fenced capacity effect, not membership observation")
	initial, ok := runtime.Capacity.(InitialCapacityReconciler)
	require.True(t, ok)
	require.NoError(t, initial.ReconcileInitialCapacity(t.Context(), domain.GroupID(group.UID)))
	assert.Empty(t, transport.hosts)

	t.Log("native-only launches cannot silently substitute logical admission for real discovery")
	nativeKube, nativeGroup := newVLLMRuntimeTestWorld(t)
	provider.Client = nativeKube
	_, err = provider.Resolve(t.Context(), nativeGroup)
	require.ErrorIs(t, err, ErrRuntimeUnavailable)

	t.Log("unknown runtimes cannot fall through to vLLM resolution")
	group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] = "unknown"
	_, err = provider.Resolve(t.Context(), group)
	require.ErrorIs(t, err, ErrRuntimeUnavailable)
}

func TestVLLMCapacityBootstrapValidatesEngineMappingBeforeGroveAllocation(t *testing.T) {
	for _, scenario := range []string{"complete", "partial", "wrong-slot-range", "restore"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("validate one allocation's native bootstrap independently of Grove's physical count")
			profile := dynamo.ResolvedVLLMElasticEPProfile{InitialReplicas: 2, NativeMembersPerReplica: 4}
			bootstrap := &domain.CapacityBootstrap{Mode: domain.BootstrapModeJoin, BaseTopologyGeneration: 1,
				NativeMembers: []domain.NativeMemberID{"dp-8", "dp-9", "dp-10", "dp-11"},
			}
			switch scenario {
			case "partial":
				bootstrap.NativeMembers = bootstrap.NativeMembers[:3]
			case "wrong-slot-range":
				bootstrap.NativeMembers = []domain.NativeMemberID{"dp-4", "dp-5", "dp-6", "dp-7"}
			case "restore":
				bootstrap.Mode = domain.BootstrapModeRestoreFixedSlot
			}
			target := domain.CapacityTarget{Replicas: []domain.CapacityReplicaTarget{{ReplicaID: "replica-2", SlotID: "slot-2", Bootstrap: bootstrap}}}
			failure := validateVLLMCapacityBootstrap(target, profile)
			if scenario == "complete" {
				require.Nil(t, failure)
			} else {
				require.NotNil(t, failure)
				require.Equal(t, "UnsupportedVLLMBootstrap", failure.Reason)
			}
		})
	}
}
