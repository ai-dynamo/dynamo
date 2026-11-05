//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"fmt"
	"strconv"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/workload"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestVLLMCapacityDoesNotDependOnGrove(t *testing.T) {
	for _, width := range []int32{1, 4} {
		t.Run(strconv.Itoa(int(width)), func(t *testing.T) {
			t.Log("declare native geometry and a physical Pod without any Grove resources or metadata")
			source := dynamo.VLLMProfileGeometrySource{
				Command: []string{"python3", "-m", "dynamo.vllm", "--model", "model"},
				Args: []string{"--elastic-ep-auto-bootstrap", fmt.Sprintf("--elastic-ep-initial-dp-size=%d", 2*width),
					fmt.Sprintf("--data-parallel-size-local=%d", width),
					"--elastic-ep-allocation-index=$(DYNAMO_RANK)", "--data-parallel-address=$(DYNAMO_LEADER_ADDRESS)",
					"--tensor-parallel-size=1", "--distributed-executor-backend=mp", fmt.Sprintf("--elastic-ep-max-dp-size=%d", 3*width),
					"--enable-expert-parallel", "--enable-eplb", "--enable-elastic-ep"},
				InitialReplicas: 2, MainContainerGPUs: int64(width), DedicatedMainGPUAllocation: true, WorkloadRevisionDigest: "native-template",
				Env: []corev1.EnvVar{{Name: "DYN_SYSTEM_PORT", Value: "18000"}, {Name: "DYNAMO_ENGINE_GROUP", Value: "1"},
					{Name: "DYNAMO_POD_UID", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.uid"}}}},
			}
			profile, err := dynamo.ResolveVLLMElasticEPProfile(source)
			require.NoError(t, err)
			dgd := &api.DynamoGraphDeployment{Name: "deployment", Namespace: "test", UID: "dgd-uid"}
			component := &api.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", EngineGroup: &api.ComponentEngineGroupSpec{InitialSize: 2}}
			group := dynamo.RenderVLLMEngineGroup(dgd, component, 0, profile)
			group.UID, group.Finalizers = "group-uid", []string{engineGroupFinalizer}
			pod := &corev1.Pod{Name: "native-allocation", Namespace: "test", UID: "pod-uid", Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyNever}}
			second := pod.DeepCopy()
			second.Name, second.UID = "second-allocation", "second-uid"
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			require.NoError(t, api.AddToScheme(scheme))
			kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(group, pod, second).WithStatusSubresource(pod).Build()
			physical := &physicalCapacityTestProvider{observation: workload.Observation{ReleaseMode: domain.CapacityReleaseExactUID,
				Allocations: []workload.Allocation{
					{ReplicaID: "replica-0", SlotID: "slot-0", Pod: domain.CapacityRef{Name: pod.Name, UID: domain.PodUID(pod.UID)}},
					{ReplicaID: "replica-1", SlotID: "slot-1", Pod: domain.CapacityRef{Name: second.Name, UID: domain.PodUID(second.UID)}},
				}}}
			capacity := &VLLMCapacity{Physical: physical, Client: kube, WorldID: "opaque-world", Group: group, Source: source}

			t.Log("bind every native process to the exact physical allocation and preserve its release guarantee")
			require.NoError(t, capacity.ReconcileAllocations(t.Context()))
			observed, err := capacity.Observe(t.Context(), "group-uid")
			require.NoError(t, err)
			require.Len(t, observed.Allocations, 2)
			assert.Len(t, observed.Allocations[0].Incarnation.Members, int(width))
			assert.Equal(t, domain.CapacityReleaseExactUID, observed.ReleaseMode)
			assert.Equal(t, 1, physical.prepared)

			t.Log("delegate a validated pinned target without any provider-specific resource access")
			current := observed.Allocations[0].Incarnation
			other := observed.Allocations[1].Incarnation
			target := domain.CapacityTarget{ControlRevision: 1, ProfileFingerprint: profile.Geometry.Fingerprint,
				ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator,
				Replicas: []domain.CapacityReplicaTarget{
					{ReplicaID: current.ReplicaID, SlotID: current.SlotID, Incarnation: &current},
					{ReplicaID: other.ReplicaID, SlotID: other.SlotID, Incarnation: &other},
				}}
			result, err := capacity.Apply(t.Context(), "group-uid", target)
			require.NoError(t, err)
			assert.Nil(t, result.Rejection)
			assert.Equal(t, []domain.CapacityTarget{target}, physical.targets)

			t.Log("reject an engine restart before it can reach the physical provider")
			pod.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: "main", RestartCount: 1}}
			require.NoError(t, kube.Status().Update(t.Context(), pod))
			_, err = capacity.Apply(t.Context(), "group-uid", target)
			require.ErrorIs(t, err, domain.ErrRecoveryRequired)
			assert.Len(t, physical.targets, 1)
		})
	}
}
