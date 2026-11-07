//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"net/http"
	"strconv"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/workload"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestSGLangRuntimeDoesNotDependOnGrove(t *testing.T) {
	for _, width := range []int{2, 4} {
		t.Run(strconv.Itoa(width), func(t *testing.T) {
			t.Log("declare packed native bootstrap in a Pod without Grove labels, owners, or API types")
			pod := testSGLangPrimaryPod()
			pod.Labels, pod.OwnerReferences = nil, nil
			pod.Spec.Containers[0].Resources.Limits[corev1.ResourceName("nvidia.com/gpu")] = resource.MustParse(strconv.Itoa(width))
			pod.Spec.Containers[0].Args = []string{"--model-path", "model", "--elastic-ep-join-mode", "auto", "--enable-dp-lm-head",
				"--moe-a2a-backend", "nixl", "--elastic-ep-backend", "mooncake", "--load-balance-method", "round_robin",
				"--elastic-ep-initial-size", strconv.Itoa(width), "--max-ep-size", strconv.Itoa(2 * width),
				"--disable-cuda-graph", "--dist-init-addr", "rendezvous.test:24555"}
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pod).WithStatusSubresource(pod).Build()
			group := &api.DynamoGraphDeploymentEngineGroup{Name: "group", Namespace: pod.Namespace, UID: "group-uid",
				Labels: map[string]string{consts.KubeLabelDynamoEngineGroupRuntime: consts.KubeLabelDynamoEngineGroupSGLang}}
			physical := &physicalCapacityTestProvider{observation: workload.Observation{ReleaseMode: domain.CapacityReleaseExactUID,
				Allocations: []workload.Allocation{{ReplicaID: "replica-0", SlotID: "slot-0", Pod: domain.CapacityRef{Name: pod.Name, UID: domain.PodUID(pod.UID)}, TemplateRevision: "native-template"}}}}
			world := WorkloadBinding{ID: "opaque-world", Capacity: physical}
			provider := &sglangRuntimeResolver{client: kube, httpClient: &http.Client{}}

			t.Log("resolve native controls and every member from the provider-neutral binding")
			resolved, err := provider.Resolve(t.Context(), group, world)
			require.NoError(t, err)
			assert.Equal(t, int32(width), resolved.Profile.NativeMembersPerReplica)
			capacity := resolved.Capacity.(*SGLangCapacity)
			require.NoError(t, capacity.ReconcileInitialCapacity(t.Context(), "group-uid"))
			observed, err := capacity.Observe(t.Context(), "group-uid")
			require.NoError(t, err)
			require.Len(t, observed.Allocations, 1)
			assert.Len(t, observed.Allocations[0].Incarnation.Members, width)
			assert.Equal(t, domain.CapacityReleaseExactUID, observed.ReleaseMode)
			assert.Equal(t, 1, physical.prepared)

			t.Log("delegate a pinned target and refuse process restart before another physical effect")
			current := observed.Allocations[0].Incarnation
			target := domain.CapacityTarget{ControlRevision: 1, Replicas: []domain.CapacityReplicaTarget{{ReplicaID: current.ReplicaID, SlotID: current.SlotID, Incarnation: &current}}}
			result, err := capacity.Apply(t.Context(), "group-uid", target)
			require.NoError(t, err)
			assert.Nil(t, result.Rejection)
			assert.Equal(t, []domain.CapacityTarget{target}, physical.targets)
			pod.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: "main", RestartCount: 1}}
			require.NoError(t, kube.Status().Update(t.Context(), pod))
			_, err = capacity.Apply(t.Context(), "group-uid", target)
			require.ErrorIs(t, err, domain.ErrRecoveryRequired)
			assert.Len(t, physical.targets, 1)
		})
	}
}
