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
	current.Status.ActiveReplicas = 2
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
					ReplicaID:          "replica-0",
					RuntimeIncarnation: "runtime-0",
					NativeMembers:      []string{""},
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
