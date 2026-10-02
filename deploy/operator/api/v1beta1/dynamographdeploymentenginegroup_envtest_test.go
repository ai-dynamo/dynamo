/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1beta1

import (
	"context"
	"path/filepath"
	goruntime "runtime"
	"testing"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
)

func TestDynamoGraphDeploymentEngineGroupAPIServerContract(t *testing.T) {
	t.Log("Start an API server with the generated Dynamo CRDs")
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := autoscalingv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	_, sourceFile, _, ok := goruntime.Caller(0)
	if !ok {
		t.Fatal("resolve test source path")
	}
	testEnv := &envtest.Environment{
		Scheme:                scheme,
		CRDDirectoryPaths:     []string{filepath.Join(filepath.Dir(sourceFile), "..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: true,
	}
	config, err := testEnv.Start()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := testEnv.Stop(); err != nil {
			t.Errorf("stop envtest: %v", err)
		}
	})
	kubeClient, err := client.New(config, client.Options{Scheme: scheme})
	if err != nil {
		t.Fatal(err)
	}

	t.Log("Create an isolated namespace and an Engine Group whose create request contains status")
	ctx := context.Background()
	namespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "engine-group-api"}}
	if err := kubeClient.Create(ctx, namespace); err != nil {
		t.Fatal(err)
	}
	group := &DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "group-0", Namespace: namespace.Name},
		Spec: DynamoGraphDeploymentEngineGroupSpec{
			Replicas: 4,
			Policy: &EngineGroupScalingPolicy{
				MinReplicas: ptr.To[int32](2),
				MaxReplicas: ptr.To[int32](8),
			},
		},
		Status: DynamoGraphDeploymentEngineGroupStatus{Replicas: 99},
	}
	if err := kubeClient.Create(ctx, group); err != nil {
		t.Fatal(err)
	}

	t.Log("Verify the status subresource isolates status from the create request")
	key := client.ObjectKeyFromObject(group)
	current := &DynamoGraphDeploymentEngineGroup{}
	if err := kubeClient.Get(ctx, key, current); err != nil {
		t.Fatal(err)
	}
	if current.Status.Replicas != 0 {
		t.Fatalf("status replicas after create = %d, want 0", current.Status.Replicas)
	}

	t.Log("Write allocated, available, active, and representative-selector status")
	current.Status.Replicas = 3
	current.Status.AvailableReplicas = 2
	current.Status.ActiveNativeMemberCount = 2
	current.Status.Selector = "nvidia.com/dynamo-engine-group=group-0,nvidia.com/dynamo-scale-representative=true"
	current.Status.ScaleUnit = EngineGroupScaleUnitReplicas
	if err := kubeClient.Status().Update(ctx, current); err != nil {
		t.Fatal(err)
	}

	t.Log("Read the real scale subresource and verify its logical-replica projection")
	scale := &autoscalingv1.Scale{}
	if err := kubeClient.SubResource("scale").Get(ctx, current, scale); err != nil {
		t.Fatal(err)
	}
	if scale.Spec.Replicas != 4 || scale.Status.Replicas != 3 {
		t.Fatalf("scale replicas = desired %d, current %d; want 4 and 3", scale.Spec.Replicas, scale.Status.Replicas)
	}
	if scale.Status.Selector != current.Status.Selector {
		t.Fatalf("scale selector = %q, want %q", scale.Status.Selector, current.Status.Selector)
	}

	t.Log("Update desired logical replicas through the real scale subresource")
	scale.Spec.Replicas = 6
	if err := kubeClient.SubResource("scale").Update(ctx, current, client.WithSubResourceBody(scale)); err != nil {
		t.Fatal(err)
	}

	t.Log("Verify the scale write changes spec without overwriting status")
	if err := kubeClient.Get(ctx, key, current); err != nil {
		t.Fatal(err)
	}
	if current.Spec.Replicas != 6 {
		t.Fatalf("spec replicas after scale = %d, want 6", current.Spec.Replicas)
	}
	if current.Status.Replicas != 3 || current.Status.Selector == "" {
		t.Fatalf("status after scale = replicas %d, selector %q", current.Status.Replicas, current.Status.Selector)
	}

	t.Log("Reject scale writes outside the statically declared policy bounds")
	for _, tt := range []struct {
		name     string
		replicas int32
	}{
		{name: "below minimum", replicas: 1},
		{name: "above maximum", replicas: 9},
	} {
		t.Run(tt.name, func(t *testing.T) {
			requireScaleTargetRejected(t, ctx, kubeClient, current, tt.replicas)
		})
	}

	t.Log("Create a second group for API-server schema rejection cases")
	invalidTarget := &DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "invalid-status", Namespace: namespace.Name},
		Spec:       DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	if err := kubeClient.Create(ctx, invalidTarget); err != nil {
		t.Fatal(err)
	}

	t.Log("Reject scale-to-zero even when no policy bounds are declared")
	requireScaleTargetRejected(t, ctx, kubeClient, invalidTarget, 0)

	t.Run("packed member status", func(t *testing.T) {
		requirePackedEngineGroupStatusRoundTrip(t, ctx, kubeClient, namespace.Name)
	})
	t.Log("Exercise identity, generation, native-member, and count validation through status")
	invalidCases := []struct {
		name   string
		status DynamoGraphDeploymentEngineGroupStatus
	}{
		{
			name: "release authorization with empty Pod UID",
			status: DynamoGraphDeploymentEngineGroupStatus{ReleaseAuthorizations: []EngineGroupReleaseAuthorization{{
				OperationID:        "retire-1",
				TopologyGeneration: 1,
				ReplicaID:          "replica-0",
				SlotID:             "slot-0",
				CapacityRefs:       []EngineGroupCapacityRef{{Name: "worker-0"}},
			}}},
		},
		{
			name:   "zero topology generation",
			status: DynamoGraphDeploymentEngineGroupStatus{Topology: &EngineGroupTopologyStatus{}},
		},
		{
			name:   "zero traffic topology generation",
			status: DynamoGraphDeploymentEngineGroupStatus{Traffic: &EngineGroupTrafficStatus{}},
		},
		{
			name: "zero release topology generation",
			status: DynamoGraphDeploymentEngineGroupStatus{ReleaseAuthorizations: []EngineGroupReleaseAuthorization{{
				OperationID:  "retire-1",
				ReplicaID:    "replica-0",
				SlotID:       "slot-0",
				CapacityRefs: []EngineGroupCapacityRef{{Name: "worker-0", UID: types.UID("pod-uid-0")}},
			}}},
		},
		{
			name: "empty native member",
			status: DynamoGraphDeploymentEngineGroupStatus{Topology: &EngineGroupTopologyStatus{
				Generation: 1,
				Replicas: []EngineGroupMemberStatus{{
					ReplicaID:     "replica-0",
					NativeMembers: []EngineGroupNativeMemberIncarnationStatus{{ID: "", RuntimeIncarnation: "runtime-0"}},
				}},
			}},
		},
		{
			name:   "negative allocated count",
			status: DynamoGraphDeploymentEngineGroupStatus{Replicas: -1},
		},
		{
			name:   "negative observed generation",
			status: DynamoGraphDeploymentEngineGroupStatus{ObservedGeneration: -1},
		},
		{
			name:   "negative desired assignment generation",
			status: DynamoGraphDeploymentEngineGroupStatus{DesiredAssignmentGeneration: -1},
		},
		{
			name:   "negative desired native-member count",
			status: DynamoGraphDeploymentEngineGroupStatus{DesiredNativeMemberCount: -1},
		},
		{
			name:   "negative active native-member count",
			status: DynamoGraphDeploymentEngineGroupStatus{ActiveNativeMemberCount: -1},
		},
		{
			name: "empty per-member identity",
			status: DynamoGraphDeploymentEngineGroupStatus{ReplicaStates: []EngineGroupReplicaStatus{{
				ReplicaID: "replica-0", SlotID: "slot-0",
				NativeMembers: []EngineGroupNativeMemberStatus{{Membership: EngineGroupReplicaMembershipActive, Traffic: EngineGroupMemberTrafficAdmitted}},
			}}},
		},
		{
			name: "candidate allocation without exact Pod UID",
			status: DynamoGraphDeploymentEngineGroupStatus{ReplicaStates: []EngineGroupReplicaStatus{{
				ReplicaID: "replica-0", SlotID: "slot-0",
				CandidateAllocation: &EngineGroupReplicaAllocationStatus{
					CapacityRefs: []EngineGroupCapacityRef{{Name: "replacement-0"}},
					Availability: EngineGroupReplicaAvailabilityUnknown, Health: EngineGroupAllocationHealthUnknown,
				},
			}}},
		},
	}
	for _, tt := range invalidCases {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Read a fresh resource version for this rejected status write")
			candidate := &DynamoGraphDeploymentEngineGroup{}
			if err := kubeClient.Get(ctx, client.ObjectKeyFromObject(invalidTarget), candidate); err != nil {
				t.Fatal(err)
			}

			t.Log("Attempt the invalid status update through the API server")
			candidate.Status = tt.status
			err := kubeClient.Status().Update(ctx, candidate)
			if !apierrors.IsInvalid(err) {
				t.Fatalf("status update error = %v, want Invalid", err)
			}
		})
	}

	t.Run("immutable DGD creation seed", func(t *testing.T) {
		requireDGDInitialSizeSchema(t, ctx, kubeClient, namespace.Name)
	})
}

func requirePackedEngineGroupStatusRoundTrip(t *testing.T, ctx context.Context, kubeClient client.Client, namespace string) {
	t.Helper()

	t.Log("Round-trip packed EP7 status and a dormant replacement without conflating allocation and membership")
	packed := &DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "packed-status", Namespace: namespace},
		Spec:       DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
	}
	if err := kubeClient.Create(ctx, packed); err != nil {
		t.Fatal(err)
	}
	packed.Status = DynamoGraphDeploymentEngineGroupStatus{
		Replicas: 2, AvailableReplicas: 1,
		DesiredNativeMembers:     []string{"dp-0", "dp-1", "dp-2", "dp-3", "dp-4", "dp-5", "dp-6", "dp-7"},
		DesiredNativeMemberCount: 8,
		ActiveNativeMemberCount:  7,
		Profile: &EngineGroupProfileStatus{
			Backend: "vllm", Fingerprint: "packed-tp1", GPUsPerReplica: 4, PodsPerReplica: 1,
			NativeMembersPerReplica: 4, MinSafeServingNativeMembers: 4,
			MinSupportedReplicas: 1, MaxSupportedReplicas: 16,
		},
		ReplicaStates: []EngineGroupReplicaStatus{{
			ReplicaID: "replica-1", SlotID: "slot-1",
			CurrentAllocation: &EngineGroupReplicaAllocationStatus{
				CapacityRefs: []EngineGroupCapacityRef{{Name: "worker-1", UID: types.UID("pod-uid-1")}},
				Availability: EngineGroupReplicaAvailabilityAvailable,
				Health:       EngineGroupAllocationHealthDegraded,
			},
			CandidateAllocation: &EngineGroupReplicaAllocationStatus{
				CapacityRefs: []EngineGroupCapacityRef{{Name: "replacement-1", UID: types.UID("candidate-uid-1")}},
				Availability: EngineGroupReplicaAvailabilityAvailable,
				Health:       EngineGroupAllocationHealthHealthy,
			},
			NativeMembers: []EngineGroupNativeMemberStatus{
				{ID: "dp-4", Membership: EngineGroupReplicaMembershipActive, Traffic: EngineGroupMemberTrafficAdmitted},
				{ID: "dp-5", Membership: EngineGroupReplicaMembershipMasked, Traffic: EngineGroupMemberTrafficDrained},
				{ID: "dp-6", Membership: EngineGroupReplicaMembershipActive, Traffic: EngineGroupMemberTrafficAdmitted},
				{ID: "dp-7", Membership: EngineGroupReplicaMembershipActive, Traffic: EngineGroupMemberTrafficAdmitted},
			},
		}},
	}
	if err := kubeClient.Status().Update(ctx, packed); err != nil {
		t.Fatal(err)
	}
	if err := kubeClient.Get(ctx, client.ObjectKeyFromObject(packed), packed); err != nil {
		t.Fatal(err)
	}
	if packed.Status.Replicas != 2 || packed.Status.AvailableReplicas != 1 || packed.Status.DesiredNativeMemberCount != 8 || packed.Status.ActiveNativeMemberCount != 7 {
		t.Fatalf("packed status lost independent allocation/member counters: %+v", packed.Status)
	}
	if len(packed.Status.ReplicaStates) != 1 || packed.Status.ReplicaStates[0].CandidateAllocation == nil {
		t.Fatalf("dormant replacement did not survive status round-trip: %+v", packed.Status.ReplicaStates)
	}
	if len(packed.Status.ReplicaStates[0].NativeMembers) != 4 || packed.Status.ReplicaStates[0].NativeMembers[1].Membership != EngineGroupReplicaMembershipMasked {
		t.Fatalf("per-member state did not survive status round-trip: %+v", packed.Status.ReplicaStates[0].NativeMembers)
	}

}

func requireDGDInitialSizeSchema(t *testing.T, ctx context.Context, kubeClient client.Client, namespace string) {
	t.Helper()

	t.Log("Validate the DGD creation seed through CRD rules without enabling its gated workload pathway")
	dgd := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "engine-group-seed", Namespace: namespace},
		Spec: DynamoGraphDeploymentSpec{Components: []DynamoComponentDeploymentSharedSpec{{
			ComponentName: "worker", Replicas: ptr.To[int32](2),
			EngineGroup: &ComponentEngineGroupSpec{InitialSize: 2},
			PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
				Name: "main", Image: "test:1.6.0",
			}}}},
		}}},
	}
	if err := kubeClient.Create(ctx, dgd); err != nil {
		t.Fatal(err)
	}

	t.Log("Change the world count without changing the size seeded into new worlds")
	dgd.Spec.Components[0].Replicas = ptr.To[int32](3)
	if err := kubeClient.Update(ctx, dgd); err != nil {
		t.Fatal(err)
	}

	t.Log("Reject changing or removing the established immutable initialSize")
	for _, remove := range []bool{false, true} {
		candidate := dgd.DeepCopy()
		if remove {
			candidate.Spec.Components[0].EngineGroup = nil
		} else {
			candidate.Spec.Components[0].EngineGroup.InitialSize = 4
		}
		if err := kubeClient.Update(ctx, candidate); !apierrors.IsInvalid(err) {
			t.Fatalf("initialSize update (remove=%v) error = %v, want Invalid", remove, err)
		}
	}
}

func requireScaleTargetRejected(
	t *testing.T,
	ctx context.Context,
	kubeClient client.Client,
	group *DynamoGraphDeploymentEngineGroup,
	replicas int32,
) {
	t.Helper()

	candidate := &autoscalingv1.Scale{}
	if err := kubeClient.SubResource("scale").Get(ctx, group, candidate); err != nil {
		t.Fatal(err)
	}
	candidate.Spec.Replicas = replicas
	if err := kubeClient.SubResource("scale").Update(ctx, group, client.WithSubResourceBody(candidate)); !apierrors.IsInvalid(err) {
		t.Fatalf("scale update error = %v, want Invalid", err)
	}
}
