/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"strconv"
	"testing"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestSGLangGroveCapacityOwnsProcessRestartFencing(t *testing.T) {
	t.Log("observe the compatibility profile's one native process in its physical allocation")
	capacity, pod := newSGLangGroveCapacityFixture(t)
	observed, err := capacity.Observe(t.Context(), "group-uid")
	require.NoError(t, err)
	require.Len(t, observed.Allocations, 1)
	assert.Equal(t, []domain.NativeMemberIncarnation{{ID: "dp-0", RuntimeIncarnation: domain.RuntimeIncarnationID(string(pod.UID) + "/dp-0")}}, observed.Allocations[0].Incarnation.Members)

	t.Log("a non-engine container restart does not create a new native process incarnation")
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: "telemetry", RestartCount: 1}}
	require.NoError(t, capacity.Grove.Client.Status().Update(t.Context(), pod))
	unchanged, err := capacity.Observe(t.Context(), "group-uid")
	require.NoError(t, err)
	assert.Equal(t, observed.Allocations[0].Incarnation, unchanged.Allocations[0].Incarnation)

	t.Log("a container restart preserves the physical Pod but invalidates the engine-process identity")
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: "main", RestartCount: 1}}
	require.NoError(t, capacity.Grove.Client.Status().Update(t.Context(), pod))
	physical, err := capacity.Grove.Observe(t.Context(), "group-uid")
	require.NoError(t, err)
	require.Len(t, physical.Allocations, 1)
	_, err = capacity.Observe(t.Context(), "group-uid")
	require.ErrorIs(t, err, domain.ErrRecoveryRequired)
}

func TestSGLangPackedCapacitySatisfiesFoundationIdentity(t *testing.T) {
	for _, width := range []int{1, 2, 4} {
		t.Run(strconv.Itoa(width), func(t *testing.T) {
			t.Log("bind every packed native process to a distinct identity under its non-restarting Pod lifetime")
			pod := testSGLangPrimaryPod()
			pod.Spec.Containers[0].Resources.Limits[corev1.ResourceName("nvidia.com/gpu")] = resource.MustParse(strconv.Itoa(width))
			members, err := sglangSlotMembers("slot-0", pod)
			require.NoError(t, err)
			require.Len(t, members, width)
			seen := make(map[domain.RuntimeIncarnationID]bool, width)
			for _, member := range members {
				assert.False(t, seen[member.RuntimeIncarnation])
				seen[member.RuntimeIncarnation] = true
			}

			t.Log("adopt the real capacity projection through the same validation used by the controller")
			membership := domain.ReplicaMembership{ReplicaID: "replica-0", Members: members}
			allocation := domain.CapacityAllocation{Available: true, Health: domain.AllocationHealthHealthy,
				Incarnation: domain.ReplicaIncarnation{ReplicaID: "replica-0", SlotID: "slot-0", Members: members,
					CapacityRefs: []domain.CapacityRef{{Name: pod.Name, UID: domain.PodUID(pod.UID)}},
				}}
			_, err = domain.NewGroupStatus(domain.MembershipTopology{Generation: 1, Replicas: []domain.ReplicaMembership{membership}},
				domain.CapacityObservation{Allocations: []domain.CapacityAllocation{allocation}},
				domain.TrafficObservation{Admitted: []domain.ReplicaMembership{membership}})
			require.NoError(t, err)
		})
	}
}

func TestSGLangGroveCapacityValidatesNativeTargetsBeforeScaling(t *testing.T) {
	for _, scenario := range []string{"wrong-joining-member", "unsupported-recovery", "changed-process"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("pin the existing process and request one joining allocation")
			capacity, _ := newSGLangGroveCapacityFixture(t)
			observed, err := capacity.Observe(t.Context(), "group-uid")
			require.NoError(t, err)
			current := observed.Allocations[0].Incarnation
			target := domain.CapacityTarget{
				ControlRevision: 1, TransitionID: "grow", ProfileFingerprint: "profile", ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator,
				Replicas: []domain.CapacityReplicaTarget{
					{ReplicaID: current.ReplicaID, SlotID: current.SlotID, Incarnation: &current},
					{ReplicaID: "replica-1", SlotID: "slot-1", Bootstrap: &domain.CapacityBootstrap{
						Mode: domain.BootstrapModeJoin, BaseTopologyGeneration: 1, NativeMembers: []domain.NativeMemberID{"dp-1"},
					}},
				},
			}
			switch scenario {
			case "wrong-joining-member":
				target.Replicas[1].Bootstrap.NativeMembers = []domain.NativeMemberID{"dp-9"}
			case "unsupported-recovery":
				target.Replicas[1].Bootstrap.Mode = domain.BootstrapModeRestoreFixedSlot
			case "changed-process":
				current.Members[0].RuntimeIncarnation = "another-process"
			}

			t.Log("reject unsupported bootstrap or require recovery before Grove can receive a count change")
			result, err := capacity.Apply(t.Context(), "group-uid", target)
			if scenario == "changed-process" {
				require.ErrorIs(t, err, domain.ErrRecoveryRequired)
			} else {
				require.NoError(t, err)
				require.NotNil(t, result.Rejection)
				assert.Equal(t, "UnsupportedBootstrap", result.Rejection.Reason)
			}
			clique := &grove.PodClique{}
			require.NoError(t, capacity.Grove.Client.Get(t.Context(), capacity.Grove.Clique, clique))
			assert.Equal(t, int32(1), clique.Spec.Replicas)
		})
	}
}

func newSGLangGroveCapacityFixture(t *testing.T) (*SGLangGroveCapacity, *corev1.Pod) {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, grove.AddToScheme(scheme))
	pod := testSGLangPrimaryPod()
	clique := testSGLangMemberClique(pod)
	clique.Spec.Replicas, clique.Spec.MinAvailable = 1, ptr.To(int32(1))
	kube := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&corev1.Pod{}).WithObjects(pod, clique).Build()
	return &SGLangGroveCapacity{MembersPerReplica: 1, Grove: &grovecapacity.Adapter{
		Client: kube, Clique: client.ObjectKeyFromObject(clique), CliqueUID: clique.UID, GroupName: "group",
		Journal: kubejournal.NewStore(kube, pod.Namespace, "group", "group-uid", "grove-capacity"),
	}}, pod
}
