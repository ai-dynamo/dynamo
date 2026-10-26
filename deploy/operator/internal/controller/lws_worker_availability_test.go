/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

func TestLWSAvailabilityDrivesWorkerGenerationCleanup(t *testing.T) {
	t.Log("Define replacement readiness and the old groups that must remain available")
	tests := []struct {
		name               string
		desired            int32
		newReady           int32
		wantOld            int32
		hashAlreadyUpdated bool
	}{
		{name: "unready replacement preserves old group", desired: 1, newReady: 0, wantOld: 1},
		{name: "ready replacement retires old group", desired: 1, newReady: 1, wantOld: 0},
		{name: "unready replacements preserve all old groups", desired: 3, newReady: 0, wantOld: 3},
		{name: "one ready replacement retires one old group", desired: 3, newReady: 1, wantOld: 2},
		{name: "partial readiness retires only covered old groups", desired: 3, newReady: 2, wantOld: 1},
		{name: "all ready replacements retire all old groups", desired: 3, newReady: 3, wantOld: 0},
		{name: "cleanup resumes after hash update", desired: 1, newReady: 1, wantOld: 0, hashAlreadyUpdated: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Create a multi-node worker deployment with an old generation")
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      new(tt.desired),
					Multinode:     &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2},
				},
			})
			dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: testOldWorkerHash}
			newHash := betaDGDWorkersSpecHash(t, dgd)
			expectedCurrentHash := testOldWorkerHash
			if tt.hashAlreadyUpdated {
				expectedCurrentHash = newHash
				dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = newHash
			}
			generations := []struct {
				hash  string
				ready int32
			}{
				{hash: testOldWorkerHash, ready: tt.desired},
				{hash: newHash, ready: tt.newReady},
			}
			dcds := make([]*nvidiacomv1beta1.DynamoComponentDeployment, 0, len(generations))

			t.Log("Populate each DCD through the production LWS status mapping")
			for _, generation := range generations {
				dcd := createTestDCD(t, dgd, &nvidiacomv1alpha1.DynamoComponentDeployment{
					Name:      dynamo.GetDCDResourceName(dgd, "worker", generation.hash),
					Namespace: dgd.Namespace,
					Labels: map[string]string{
						consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
						consts.KubeLabelDynamoWorkerHash:          generation.hash,
					},
					Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:   "worker",
							ComponentType: consts.ComponentTypeWorker,
							Replicas:      new(tt.desired),
							Multinode:     &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2},
						},
					},
				})
				lws := &leaderworkersetv1.LeaderWorkerSet{
					Name: dcd.Name,
					Spec: leaderworkersetv1.LeaderWorkerSetSpec{Replicas: new(tt.desired)},
					Status: leaderworkersetv1.LeaderWorkerSetStatus{
						Replicas:        tt.desired,
						UpdatedReplicas: tt.desired,
						ReadyReplicas:   generation.ready,
					},
				}
				original := lws.DeepCopy()
				status := getLeaderWorkerSetReplicasStatus(lws)
				assert.Equal(t, original, lws, "status mapping must not mutate the LWS")
				assert.Nil(t, status.AvailableReplicas, "LWS does not report a separate availability count")
				status.RuntimeNamespace = "test-dgd-" + generation.hash
				dcd.Status.Component = &status
				dcds = append(dcds, dcd)
			}

			t.Log("Calculate rollout targets using readiness when LWS availability is absent")
			r := createTestReconcilerWithStatus(dgd, withObjects(dcds[0], dcds[1]))
			ctx := context.Background()
			rollout, err := r.buildRollingUpdateContext(ctx, dgd)
			require.NoError(t, err)
			assert.Equal(t, tt.wantOld, rollout.OldWorkerReplicaTargetsByComponent["worker"])
			assert.Equal(t, tt.desired, rollout.NewWorkerReplicaTargetsByComponent["worker"])

			t.Log("Persist old-generation scale-down without changing replacement intent")
			require.NoError(t, r.scaleOldWorkerDCDs(ctx, dgd, rollout))
			oldDCD := &nvidiacomv1beta1.DynamoComponentDeployment{}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dcds[0]), oldDCD))
			assert.Equal(t, new(tt.wantOld), oldDCD.Spec.Replicas)
			newDCD := &nvidiacomv1beta1.DynamoComponentDeployment{}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dcds[1]), newDCD))
			assert.Equal(t, new(tt.desired), newDCD.Spec.Replicas)

			t.Log("Keep the old DCD while it still reports live replicas")
			program := &componentProgram{rollout: r}
			programStatus := dgd.DeepCopy().Status
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &programStatus))
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(oldDCD), oldDCD))
			assert.Nil(t, oldDCD.DeletionTimestamp)
			assert.Equal(t, expectedCurrentHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
			if tt.wantOld != 0 {
				return
			}

			t.Log("Wait until the old DCD has observed its scale-to-zero generation")
			oldDCD.Generation = 2
			oldDCD.Status.ObservedGeneration = 1
			oldDCD.Status.Component.Replicas = 0
			oldDCD.Status.Component.ReadyReplicas = new(int32(0))
			oldDCD.Status.Component.AvailableReplicas = nil
			require.NoError(t, r.Update(ctx, oldDCD))
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &programStatus))
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(oldDCD), oldDCD))
			assert.Nil(t, oldDCD.DeletionTimestamp)

			t.Log("Preserve the old generation while native LWS leader and worker pods are still running")
			oldDCD.Status.ObservedGeneration = oldDCD.Generation
			oldDCD.Finalizers = []string{"test.dynamo.nvidia.com/hold-deletion"}
			require.NoError(t, r.Update(ctx, oldDCD))
			oldPod := &corev1.Pod{
				Name: "old-worker", Namespace: dgd.Namespace,
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
					consts.KubeLabelDynamoComponent:           "worker",
					consts.KubeLabelDynamoWorkerHash:          testOldWorkerHash,
					consts.KubeLabelDynamoComponentType:       consts.ComponentTypeWorker,
					dcdWorkloadRoleLabel:                      string(dynamo.RoleWorker),
				},
				Finalizers: []string{"test.dynamo.nvidia.com/hold-deletion"},
				Status:     corev1.PodStatus{Phase: corev1.PodRunning},
			}
			oldLeader := oldPod.DeepCopy()
			oldLeader.Name = "old-leader"
			oldLeader.Labels[dcdWorkloadRoleLabel] = string(dynamo.RoleLeader)
			require.NoError(t, r.Create(ctx, oldLeader))
			require.NoError(t, r.Delete(ctx, oldLeader))
			require.NoError(t, r.Create(ctx, oldPod))
			require.NoError(t, r.Delete(ctx, oldPod))
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &programStatus))
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(oldDCD), oldDCD))
			assert.Nil(t, oldDCD.DeletionTimestamp)

			t.Log("Keep waiting for the leader after its worker disappears")
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(oldPod), oldPod))
			oldPod.Finalizers = nil
			require.NoError(t, r.Update(ctx, oldPod))
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &programStatus))
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(oldDCD), oldDCD))
			assert.Nil(t, oldDCD.DeletionTimestamp)

			t.Log("Delete the drained old DCD after its final pod disappears")
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(oldLeader), oldLeader))
			oldLeader.Finalizers = nil
			require.NoError(t, r.Update(ctx, oldLeader))
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &programStatus))
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(oldDCD), oldDCD))
			require.NotNil(t, oldDCD.DeletionTimestamp)
			assert.Equal(t, expectedCurrentHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])

			t.Log("Keep the old hash and runtime projection until the DCD deletion is observed")
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &programStatus))
			assert.Equal(t, expectedCurrentHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
			rollout, err = r.buildRollingUpdateContext(ctx, dgd)
			require.NoError(t, err)
			assert.True(t, rollout.InProgress())
			oldStatuses, err := r.aggregateOldWorkerComponentStatuses(ctx, dgd, rollout)
			require.NoError(t, err)
			statuses := map[string]nvidiacomv1beta1.ComponentReplicaStatus{"worker": *newDCD.Status.Component}
			mergeWorkerComponentStatuses(statuses, oldStatuses)
			assert.Equal(t, oldDCD.Status.Component.RuntimeNamespace, statuses["worker"].RuntimeNamespace)

			t.Log("Observe deletion and reconcile repeatedly without retaining the old projection")
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(oldDCD), oldDCD))
			oldDCD.Finalizers = nil
			require.NoError(t, r.Update(ctx, oldDCD))
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &programStatus))
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &programStatus))
			assert.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(oldDCD), oldDCD)))
			rollout, err = r.buildRollingUpdateContext(ctx, dgd)
			require.NoError(t, err)
			assert.False(t, rollout.InProgress())
			assert.Equal(t, newHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
			oldStatuses, err = r.aggregateOldWorkerComponentStatuses(ctx, dgd, rollout)
			require.NoError(t, err)
			statuses = map[string]nvidiacomv1beta1.ComponentReplicaStatus{"worker": *newDCD.Status.Component}
			mergeWorkerComponentStatuses(statuses, oldStatuses)
			assert.Equal(t, newDCD.Status.Component.RuntimeNamespace, statuses["worker"].RuntimeNamespace)
			assert.Equal(t, newDCD.Status.Component.ComponentNames, statuses["worker"].ComponentNames)
			assert.Nil(t, programStatus.RollingUpdate, "multinode cleanup must not start a managed rollout")
		})
	}
}

func TestDCDComponentStateAvailabilityFallback(t *testing.T) {
	t.Log("Distinguish missing availability from an explicitly reported zero")
	tests := []struct {
		name          string
		status        *nvidiacomv1beta1.ComponentReplicaStatus
		wantAvailable int32
	}{
		{name: "missing status"},
		{name: "missing replica signals", status: &nvidiacomv1beta1.ComponentReplicaStatus{}},
		{name: "missing availability uses readiness", status: &nvidiacomv1beta1.ComponentReplicaStatus{ReadyReplicas: new(int32(3))}, wantAvailable: 3},
		{name: "zero readiness stays unavailable", status: &nvidiacomv1beta1.ComponentReplicaStatus{ReadyReplicas: new(int32(0))}},
		{name: "explicit zero availability preserves readiness delay", status: &nvidiacomv1beta1.ComponentReplicaStatus{ReadyReplicas: new(int32(3)), AvailableReplicas: new(int32(0))}},
		{name: "reported availability takes precedence", status: &nvidiacomv1beta1.ComponentReplicaStatus{ReadyReplicas: new(int32(3)), AvailableReplicas: new(int32(1))}, wantAvailable: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Read replica signals without changing the public status")
			dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
				Status: nvidiacomv1beta1.DynamoComponentDeploymentStatus{Component: tt.status},
			}
			original := dcd.DeepCopy()
			state := dcdComponentStateFromDCD(dcd)
			assert.Equal(t, tt.wantAvailable, state.Available)
			assert.Equal(t, original, dcd)
		})
	}
}

func TestLWSCleanupWaitsForEveryReplacementWorker(t *testing.T) {
	t.Log("Cover replacement readiness independently of old-generation scale-down")
	tests := []struct {
		name               string
		ready              [2]int32
		newSpec            int32
		recreate           bool
		hashAlreadyUpdated bool
	}{
		{name: "recreate starts replacements after draining without deleting old DCDs", newSpec: 0, recreate: true},
		{name: "transient zero target cannot complete the desired rollout", newSpec: 0},
		{name: "partial replacement readiness retains drained old DCDs", ready: [2]int32{2, 1}, newSpec: 2},
		{name: "excess readiness in one component cannot cover another", ready: [2]int32{3, 1}, newSpec: 2},
		{name: "cleanup after premature hash advancement still waits for readiness", ready: [2]int32{2, 1}, newSpec: 2, hashAlreadyUpdated: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Create two worker components that each require two ready groups")
			components := map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{}
			for _, name := range []string{"prefill", "decode"} {
				annotations := map[string]string{KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "2"}
				if tt.recreate {
					annotations[KubeAnnotationDeploymentStrategy] = string(common.DeploymentStrategyRecreate)
				}
				components[name] = &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      new(int32(2)),
					Multinode:     &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2},
					Annotations:   annotations,
				}
			}
			dgd := createTestDGD("test-dgd", components)
			newHash := betaDGDWorkersSpecHash(t, dgd)
			currentHash := testOldWorkerHash
			if tt.hashAlreadyUpdated {
				currentHash = newHash
			}
			dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: currentHash}
			r := createTestReconcilerWithStatus(dgd)
			program := &componentProgram{rollout: r}
			status := dgd.DeepCopy().Status
			ctx := context.Background()
			oldDCDs := make([]*nvidiacomv1beta1.DynamoComponentDeployment, 0, 2)
			newDCDs := make([]*nvidiacomv1beta1.DynamoComponentDeployment, 0, 2)

			t.Log("Seed old and replacement generations, leaving replacement counts below the DGD requirement")
			for i, name := range []string{"prefill", "decode"} {
				for _, hash := range []string{testOldWorkerHash, newHash} {
					dcd := createTestDCD(t, dgd, &nvidiacomv1alpha1.DynamoComponentDeployment{
						Name: dynamo.GetDCDResourceName(dgd, name, hash), Namespace: dgd.Namespace,
						Labels: map[string]string{
							consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
							consts.KubeLabelDynamoWorkerHash:          hash,
						},
						Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName: name, ComponentType: consts.ComponentTypeWorker,
								Replicas: new(int32(2)), Multinode: &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2},
							},
						},
					})
					dcd.Status.Component = &nvidiacomv1beta1.ComponentReplicaStatus{Replicas: 2, ReadyReplicas: new(int32(2))}
					if hash == newHash {
						dcd.Spec.Replicas = new(tt.newSpec)
						dcd.Status.Component.Replicas = tt.newSpec
						dcd.Status.Component.ReadyReplicas = new(tt.ready[i])
						newDCDs = append(newDCDs, dcd)
					} else {
						oldDCDs = append(oldDCDs, dcd)
					}
					require.NoError(t, r.Create(ctx, dcd))
				}
			}

			t.Log("Scale old replicas to zero before replacement readiness, respecting the chosen strategy")
			rollout, err := r.buildRollingUpdateContext(ctx, dgd)
			require.NoError(t, err)
			assert.Equal(t, map[string]int32{"prefill": 0, "decode": 0}, rollout.OldWorkerReplicaTargetsByComponent)
			if tt.recreate {
				assert.Equal(t, map[string]int32{"prefill": 0, "decode": 0}, rollout.NewWorkerReplicaTargetsByComponent)
			}
			require.NoError(t, r.scaleOldWorkerDCDs(ctx, dgd, rollout))
			for _, dcd := range oldDCDs {
				require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dcd), dcd))
				assert.Equal(t, new(int32(0)), dcd.Spec.Replicas)
				dcd.Status.ObservedGeneration = dcd.Generation
				dcd.Status.Component.Replicas = 0
				dcd.Status.Component.ReadyReplicas = new(int32(0))
				require.NoError(t, r.Update(ctx, dcd))
			}

			t.Log("Allow replacements to start while retaining the drained old DCD objects and current hash")
			rollout, err = r.buildRollingUpdateContext(ctx, dgd)
			require.NoError(t, err)
			assert.Equal(t, map[string]int32{"prefill": 2, "decode": 2}, rollout.NewWorkerReplicaTargetsByComponent)
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &status))
			assert.Equal(t, currentHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
			for _, dcd := range oldDCDs {
				require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dcd), dcd))
				assert.Nil(t, dcd.DeletionTimestamp)
			}

			t.Log("Reach the full DGD desired count for every replacement component before deleting old DCDs")
			for _, dcd := range newDCDs {
				dcd.Spec.Replicas = new(int32(2))
				dcd.Status.Component.Replicas = 2
				dcd.Status.Component.ReadyReplicas = new(int32(2))
				require.NoError(t, r.Update(ctx, dcd))
			}
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &status))
			assert.Equal(t, currentHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2], "deleting DCDs does not yet commit the new hash")
			for _, dcd := range oldDCDs {
				assert.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(dcd), dcd)))
			}

			t.Log("Advance the hash only on the next reconcile after deletion is observed")
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &status))
			assert.Equal(t, newHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
			assert.Nil(t, status.RollingUpdate)
		})
	}
}

func TestWorkerRecreateIgnoresCheckpointCapturePods(t *testing.T) {
	t.Log("Exercise Deployment and LWS rollouts with a capture from the old generation")
	for _, multinode := range []bool{false, true} {
		name := "Deployment"
		if multinode {
			name = "LWS"
		}
		t.Run(name, func(t *testing.T) {
			t.Log("Create drained old DCD and unstarted replacement for a Recreate update")
			component := &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Replicas:      new(int32(1)),
				Annotations: map[string]string{
					KubeAnnotationDeploymentStrategy: string(common.DeploymentStrategyRecreate),
				},
			}
			if multinode {
				component.Multinode = &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2}
			}
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{"worker": component})
			dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: testOldWorkerHash}
			if !multinode {
				dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
					Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
				}
			}
			newHash := betaDGDWorkersSpecHash(t, dgd)
			dcds := make([]*nvidiacomv1beta1.DynamoComponentDeployment, 0, 2)
			for _, hash := range []string{testOldWorkerHash, newHash} {
				dcd := createTestDCD(t, dgd, &nvidiacomv1alpha1.DynamoComponentDeployment{
					Name:       dynamo.GetDCDResourceName(dgd, "worker", hash),
					Namespace:  dgd.Namespace,
					Generation: 1,
					Labels: map[string]string{
						consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
						consts.KubeLabelDynamoWorkerHash:          hash,
					},
					Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: *component.DeepCopy(),
					},
				})
				dcd.Spec.ComponentName = "worker"
				dcd.Spec.Replicas = new(int32(0))
				dcd.Status.ObservedGeneration = 1
				dcd.Status.Component = &nvidiacomv1beta1.ComponentReplicaStatus{ReadyReplicas: new(int32(0))}
				dcds = append(dcds, dcd)
			}
			oldDCD, newDCD := dcds[0], dcds[1]

			t.Log("Index both a serving worker and a checkpoint capture with the same old hash")
			worker := &corev1.Pod{
				Name: "old-worker", Namespace: dgd.Namespace,
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
					consts.KubeLabelDynamoComponent:           "worker",
					consts.KubeLabelDynamoComponentType:       consts.ComponentTypeWorker,
					consts.KubeLabelDynamoWorkerHash:          testOldWorkerHash,
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			}
			if !multinode {
				worker.Labels[consts.KubeLabelDynamoSelector] = oldDCD.Name
			}
			capture := &corev1.Pod{
				Name: "checkpoint-capture", Namespace: dgd.Namespace,
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
					consts.KubeLabelDynamoComponent:           "worker",
					consts.KubeLabelDynamoWorkerHash:          testOldWorkerHash,
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			}
			r := createTestReconcilerWithStatus(dgd, withObjects(oldDCD, newDCD, worker, capture))
			ctx := context.Background()
			pods, err := r.listDGDComponentPods(ctx, dgd, "worker")
			require.NoError(t, err)
			require.Len(t, pods, 2, "capture pods share the DGD/component index")

			t.Log("Capture completion is not a worker event that can wake a stalled rollout")
			captureDone := capture.DeepCopy()
			captureDone.Status.Phase = corev1.PodSucceeded
			assert.False(t, dgdWorkerPodEventPredicate().Update(event.UpdateEvent{ObjectOld: capture, ObjectNew: captureDone}))
			assert.Empty(t, mapDGDWorkerPodToRequests(ctx, captureDone))

			t.Log("The real old worker must still block replacement startup")
			rollout, err := r.buildRollingUpdateContext(ctx, dgd)
			require.NoError(t, err)
			assert.Zero(t, rollout.NewWorkerReplicaTargetsByComponent["worker"])

			t.Log("Start the replacement once the old worker terminates, even while capture is running")
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(worker), worker))
			workerDone := worker.DeepCopy()
			workerDone.Status.Phase = corev1.PodSucceeded
			assert.True(t, dgdWorkerPodEventPredicate().Update(event.UpdateEvent{ObjectOld: worker, ObjectNew: workerDone}))
			require.NoError(t, r.Status().Update(ctx, workerDone))
			rollout, err = r.buildRollingUpdateContext(ctx, dgd)
			require.NoError(t, err)
			require.Equal(t, int32(1), rollout.NewWorkerReplicaTargetsByComponent["worker"])

			t.Log("A ready replacement can retire the drained DCD without waiting for capture")
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(newDCD), newDCD))
			newDCD.Spec.Replicas = new(int32(1))
			newDCD.Status.Component.Replicas = 1
			newDCD.Status.Component.ReadyReplicas = new(int32(1))
			require.NoError(t, r.Update(ctx, newDCD))
			program := &componentProgram{rollout: r}
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &dgd.Status))
			require.True(t, apierrors.IsNotFound(r.Get(ctx, client.ObjectKeyFromObject(oldDCD), oldDCD)))
			assert.Equal(t, testOldWorkerHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])

			t.Log("Observe deletion and complete while the capture remains running")
			require.NoError(t, program.reconcileWorkerRollout(ctx, dgd, &dgd.Status))
			assert.Equal(t, newHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(capture), capture))
			assert.Equal(t, corev1.PodRunning, capture.Status.Phase)
		})
	}
}
