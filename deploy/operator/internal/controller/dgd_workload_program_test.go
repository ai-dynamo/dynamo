/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"errors"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpointjob"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func TestDGDWorkloadProgramSelection(t *testing.T) {
	tests := []struct {
		name        string
		provider    workloadProvider
		wantProgram workloadProgram
	}{
		{
			name:        "component provider selects component program",
			provider:    workloadProviderComponent,
			wantProgram: &componentProgram{},
		},
		{
			name:        "Grove provider selects Grove program",
			provider:    workloadProviderGrove,
			wantProgram: &groveProgram{},
		},
		{
			name: "explicit DisaggregatedSet opt-in selects DS program",
			annotations: map[string]string{
				commonconsts.KubeAnnotationEnableDisaggregatedSet: commonconsts.KubeLabelValueTrue,
			},
			wantProgram: &disaggregatedSetProgram{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build the program composition root")
			reconciler := &DynamoGraphDeploymentReconciler{
				RuntimeConfig: &commonController.RuntimeConfig{},
			}

			t.Log("Select one complete workload program from the durable provider")
			got, err := reconciler.selectWorkloadProgram(tt.provider)
			require.NoError(t, err)

			assert.IsType(t, tt.wantProgram, got)
			if component, ok := got.(*componentProgram); ok {
				assert.NotNil(t, component.sharedResources)
				assert.NotNil(t, component.rollout)
				assert.NotNil(t, component.restart)
				assert.NotNil(t, component.restartProgress)
				assert.NotNil(t, component.workloads)
				assert.NotNil(t, component.scalingAdapters)
			}
			if grove, ok := got.(*groveProgram); ok {
				assert.NotNil(t, grove.sharedResources)
				assert.NotNil(t, grove.rollout)
				assert.NotNil(t, grove.restart)
				assert.NotNil(t, grove.restartProgress)
				assert.NotNil(t, grove.workloads)
				assert.NotNil(t, grove.scalingAdapters)
				assert.NotNil(t, grove.topology)
			}
			if ds, ok := got.(*disaggregatedSetProgram); ok {
				assert.NotNil(t, ds.sharedResources)
				assert.NotNil(t, ds.rollout)
				assert.NotNil(t, ds.restart)
				assert.NotNil(t, ds.workloads)
				assert.NotNil(t, ds.scalingAdapters)
			}
		})
	}
}

func TestSelectedGroveProgramDoesNotFallbackWhenUnavailable(t *testing.T) {
	t.Log("Create a DGD request and an unavailable Grove program")
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Generation: 3},
	}
	program := &groveProgram{gate: features.Gates{}}

	t.Log("Reconcile the durably selected Grove program while Grove is unavailable")
	result, err := program.Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
	require.Error(t, err)
	assert.ErrorIs(t, err, reconcile.TerminalError(nil))

	t.Log("Verify Grove reports provider unavailability without invoking component reconciliation")
	ready := meta.FindStatusCondition(result.Status.Conditions, "Ready")
	require.NotNil(t, ready)
	assert.Equal(t, metav1.ConditionFalse, ready.Status)
	assert.Equal(t, string(reasonSelectedWorkloadProviderUnavailable), ready.Reason)
	assert.Contains(t, ready.Message, "Grove is disabled")
}

func TestNewWorkloadProgramResultCopiesStatus(t *testing.T) {
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		Status: nvidiacomv1beta1.DynamoGraphDeploymentStatus{
			Checkpoints: map[string]nvidiacomv1beta1.ComponentCheckpointStatus{
				"worker": {},
			},
			RollingUpdate: &nvidiacomv1beta1.RollingUpdateStatus{
				Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
			},
		},
	}

	t.Log("Create a status accumulator independent from request.DGD.Status")
	result := newWorkloadProgramResult(dgd)
	result.Status.Checkpoints["decode"] = nvidiacomv1beta1.ComponentCheckpointStatus{}
	result.Status.RollingUpdate.Phase = nvidiacomv1beta1.RollingUpdatePhaseCompleted

	t.Log("Verify status accumulation does not mutate the request object")
	assert.NotContains(t, dgd.Status.Checkpoints, "decode")
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseInProgress, dgd.Status.RollingUpdate.Phase)
}

func TestPersistWorkloadProgramResultEmitsEventsAfterStatusUpdate(t *testing.T) {
	statusUpdateErr := errors.New("status update failed")
	tests := []struct {
		name      string
		updateErr error
		wantEvent bool
	}{
		{
			name:      "successful status update flushes queued events",
			wantEvent: true,
		},
		{
			name:      "failed status update retains queued events without emitting them",
			updateErr: statusUpdateErr,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build an authoritative status result with one queued transition event")
			statusUpdated := false
			kubeClient := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithInterceptorFuncs(interceptor.Funcs{
					SubResourceUpdate: func(
						context.Context,
						client.Client,
						string,
						client.Object,
						...client.SubResourceUpdateOption,
					) error {
						statusUpdated = true
						return tt.updateErr
					},
				}).
				Build()
			recorder := events.NewFakeRecorder(1)
			reconciler := &DynamoGraphDeploymentReconciler{Client: kubeClient, Recorder: recorder}
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
			result := newWorkloadProgramResult(dgd)
			result.Eventf(corev1.EventTypeNormal, "Transition", "transition persisted")

			t.Log("Persist status through the outer controller boundary")
			err := reconciler.persistWorkloadProgramResult(context.Background(), dgd, result)

			t.Log("Verify event publication is strictly ordered after successful status persistence")
			require.True(t, statusUpdated)
			if tt.updateErr != nil {
				require.ErrorIs(t, err, tt.updateErr)
				assert.Empty(t, recorder.Events)
				return
			}
			require.NoError(t, err)
			if tt.wantEvent {
				assert.Len(t, recorder.Events, 1)
			}
		})
	}
}

func TestWorkloadProgramResultOwnsReadyAndObservedGeneration(t *testing.T) {
	t.Run("success installs Ready and advances observed generation", func(t *testing.T) {
		t.Log("Build a successful workload observation")
		result := newWorkloadProgramResult(&nvidiacomv1beta1.DynamoGraphDeployment{})
		workloads := ReconcileResult{
			State:   nvidiacomv1beta1.DGDStateSuccessful,
			Reason:  "all_resources_are_ready",
			Message: "All resources are ready",
		}

		t.Log("Apply the successful observation to authoritative program status")
		result.applyReconcileResult(7, workloads)

		t.Log("Verify the program owns overall state, Ready, and observed generation")
		assert.Equal(t, nvidiacomv1beta1.DGDStateSuccessful, result.Status.State)
		assert.Equal(t, int64(7), result.Status.ObservedGeneration)
		ready := meta.FindStatusCondition(result.Status.Conditions, "Ready")
		require.NotNil(t, ready)
		assert.Equal(t, metav1.ConditionTrue, ready.Status)
		assert.Equal(t, int64(7), ready.ObservedGeneration)
	})

	t.Run("active rolling update keeps an otherwise ready deployment pending", func(t *testing.T) {
		t.Log("Build an otherwise successful workload observation during a rolling update")
		result := newWorkloadProgramResult(&nvidiacomv1beta1.DynamoGraphDeployment{
			Status: nvidiacomv1beta1.DynamoGraphDeploymentStatus{
				RollingUpdate: &nvidiacomv1beta1.RollingUpdateStatus{
					Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
				},
			},
		})
		workloads := ReconcileResult{
			State:   nvidiacomv1beta1.DGDStateSuccessful,
			Reason:  "all_resources_are_ready",
			Message: "All resources are ready",
		}

		t.Log("Apply the workload observation to authoritative program status")
		result.applyReconcileResult(8, workloads)

		t.Log("Verify rollout state owns overall readiness until the transition completes")
		assert.Equal(t, nvidiacomv1beta1.DGDStatePending, result.Status.State)
		assert.Equal(t, int64(8), result.Status.ObservedGeneration)
		ready := meta.FindStatusCondition(result.Status.Conditions, "Ready")
		require.NotNil(t, ready)
		assert.Equal(t, metav1.ConditionFalse, ready.Status)
		assert.Equal(t, "rolling_update_in_progress", ready.Reason)
		assert.Equal(t, int64(8), ready.ObservedGeneration)
	})

	t.Run("failure preserves the last successfully observed generation", func(t *testing.T) {
		t.Log("Build status from the last successful generation")
		result := newWorkloadProgramResult(&nvidiacomv1beta1.DynamoGraphDeployment{
			Status: nvidiacomv1beta1.DynamoGraphDeploymentStatus{ObservedGeneration: 5},
		})
		reconcileErr := errors.New("workload failed")

		t.Log("Install a failure for a newer generation")
		result.Fail(6, reasonFailedToReconcileResources, reconcileErr)

		t.Log("Verify only the Ready condition observes the failed attempt")
		assert.Equal(t, int64(5), result.Status.ObservedGeneration)
		assert.Equal(t, nvidiacomv1beta1.DGDStateFailed, result.Status.State)
		ready := meta.FindStatusCondition(result.Status.Conditions, "Ready")
		require.NotNil(t, ready)
		assert.Equal(t, metav1.ConditionFalse, ready.Status)
		assert.Equal(t, int64(6), ready.ObservedGeneration)
		assert.Equal(t, reconcileErr.Error(), ready.Message)
	})
}

func TestComponentProgram_ReconcilePreservesResultOnError(t *testing.T) {
	t.Log("Inject a component-path API failure before new status is produced")
	reconcileErr := errors.New("reconcile failed")
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithInterceptorFuncs(interceptor.Funcs{
			List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
				return reconcileErr
			},
		}).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        kubeClient,
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &commonController.RuntimeConfig{},
	}
	program := reconciler.newComponentProgram()
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Status: nvidiacomv1beta1.DynamoGraphDeploymentStatus{
			State: nvidiacomv1beta1.DGDStatePending,
			Components: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"worker": {Replicas: 1},
			},
		},
	}
	previous := dgd.DeepCopy().Status

	result, err := program.Reconcile(context.Background(), workloadProgramRequest{DGD: dgd})

	t.Log("Verify the error result preserves prior fields and installs authoritative failure status")
	require.ErrorIs(t, err, reconcileErr)
	assert.Equal(t, previous.Components, result.Status.Components)
	assert.Equal(t, nvidiacomv1beta1.DGDStateFailed, result.Status.State)
	ready := meta.FindStatusCondition(result.Status.Conditions, "Ready")
	require.NotNil(t, ready)
	assert.Equal(t, metav1.ConditionFalse, ready.Status)
	assert.Equal(t, string(reasonFailedToInitializeWorkerHash), ready.Reason)
	assert.Equal(t, previous, dgd.Status)
	reason, ok := workloadProgramFailureReason(err)
	require.True(t, ok)
	assert.Equal(t, reasonFailedToInitializeWorkerHash, reason)
}

func TestComponentProgram_ReconcileRejectsInvalidLegacyGMSClient(t *testing.T) {
	t.Log("Build an already-admitted DGD with an unresolved GMS client")
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker",
				ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name:  commonconsts.MainContainerName,
					Image: "registry.example/runtime:1.1.0",
				}}}},
				Experimental: &nvidiacomv1beta1.ExperimentalSpec{
					GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
						Mode:                  nvidiacomv1beta1.GMSModeIntraPod,
						ExtraClientContainers: []string{"missing-client"},
					},
				},
			}},
		},
	}
	reconciler := createTestDGDReconcilerWithStatus(dgd)
	program := reconciler.newComponentProgram()

	t.Log("Reconcile the legacy object through the composition-first component program")
	result, err := program.Reconcile(context.Background(), workloadProgramRequest{DGD: dgd})
	require.Error(t, err)
	require.ErrorContains(t, err, "gpuMemoryService.extraClientContainers")
	require.ErrorContains(t, err, "missing-client")

	t.Log("Verify the program reports a bounded failure before creating any DCD")
	assert.Equal(t, nvidiacomv1beta1.DGDStateFailed, result.Status.State)
	ready := meta.FindStatusCondition(result.Status.Conditions, "Ready")
	require.NotNil(t, ready)
	assert.Equal(t, metav1.ConditionFalse, ready.Status)
	assert.Equal(t, string(reasonFailedToInitializeWorkerHash), ready.Reason)
	assert.Contains(t, ready.Message, "gpuMemoryService.extraClientContainers")
	assert.Contains(t, ready.Message, "missing-client")
	dcds := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	require.NoError(t, reconciler.Client.List(context.Background(), dcds, client.InNamespace(dgd.Namespace)))
	assert.Empty(t, dcds.Items)
}

func TestGroveProgram_ReconcilePreservesResultOnError(t *testing.T) {
	t.Log("Inject a shared-resource failure before the PodCliqueSet sync")
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: commonconsts.ComponentTypeWorker},
	})
	dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "test-topology"}
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        kubeClient,
		Recorder:      events.NewFakeRecorder(10),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &commonController.RuntimeConfig{Gate: features.Gates{Grove: true}},
	}
	program := reconciler.newGroveProgram()
	dgd.Status = nvidiacomv1beta1.DynamoGraphDeploymentStatus{
		State: nvidiacomv1beta1.DGDStatePending,
		Components: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
			"worker": {Replicas: 1},
		},
	}
	previous := dgd.DeepCopy().Status

	result, err := program.Reconcile(context.Background(), workloadProgramRequest{DGD: dgd})

	t.Log("Verify the failed shared reconciliation returns failure status without mutating request.DGD.Status")
	require.ErrorContains(t, err, "RBAC manager not initialized")
	assert.Equal(t, previous.Components, result.Status.Components)
	assert.Equal(t, nvidiacomv1beta1.DGDStateFailed, result.Status.State)
	ready := meta.FindStatusCondition(result.Status.Conditions, "Ready")
	require.NotNil(t, ready)
	assert.Equal(t, metav1.ConditionFalse, ready.Status)
	assert.Equal(t, string(reasonFailedToReconcileResources), ready.Reason)
	assert.Equal(t, previous, dgd.Status)
}

func TestComponentProgram_ReconcileReturnsPartialRolloutStatusOnLaterError(t *testing.T) {
	t.Log("Build a worker change that starts rollout before shared input reconciliation")
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: commonconsts.ComponentTypeWorker,
			Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "new"}},
		},
	})
	dgd.Annotations = map[string]string{
		commonconsts.AnnotationCurrentWorkerHashV2: "old-worker-hash",
	}
	reconciler := createTestDGDReconcilerWithStatus(dgd)
	program := reconciler.newComponentProgram()

	result, err := program.Reconcile(context.Background(), workloadProgramRequest{DGD: dgd})

	t.Log("Verify rollout status is returned on the later shared-input failure")
	require.ErrorContains(t, err, "RBAC manager not initialized")
	require.NotNil(t, result.Status.RollingUpdate)
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhasePending, result.Status.RollingUpdate.Phase)
	assert.Equal(t, nvidiacomv1beta1.DGDStateFailed, result.Status.State)
	require.Len(t, result.Events, 1)
	assert.Equal(t, "RollingUpdateStarted", result.Events[0].Reason)
	assert.Nil(t, dgd.Status.RollingUpdate)
}

func TestUnsupportedWorkerRolloutEmitsWarningOnlyAfterHashUpdate(t *testing.T) {
	updateErr := errors.New("update failed")
	tests := []struct {
		name      string
		updateErr error
		wantEvent bool
	}{
		{
			name:      "successful hash update emits warning",
			wantEvent: true,
		},
		{
			name:      "failed hash update does not emit warning",
			updateErr: updateErr,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build an unsupported pathway with a changed worker specification")
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: commonconsts.ComponentTypeWorker,
					Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "new"}},
				},
			})
			dgd.Annotations = map[string]string{
				commonconsts.AnnotationCurrentWorkerHashV2: "old-worker-hash",
			}
			kubeClient := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithInterceptorFuncs(interceptor.Funcs{
					Update: func(
						context.Context,
						client.WithWatch,
						client.Object,
						...client.UpdateOption,
					) error {
						return tt.updateErr
					},
				}).
				Build()
			recorder := events.NewFakeRecorder(1)
			reconciler := newDGDWorkerRolloutReconciler(kubeClient, recorder)

			t.Log("Advance the unsupported pathway hash")
			err := reconciler.ReconcileUnsupported(
				context.Background(),
				dgd,
				true,
			)
			require.NoError(t, err)

			t.Log("Verify the warning reflects a successfully persisted primary mutation")
			if tt.wantEvent {
				assert.Len(t, recorder.Events, 1)
				return
			}
			assert.Empty(t, recorder.Events)
		})
	}
}

func TestRecordRestartTransitionQueuesSupersededTransition(t *testing.T) {
	t.Log("Build an active rolling update that supersedes a new restart request")
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Restart: &nvidiacomv1beta1.Restart{ID: "restart-1"},
		},
	}
	result := newWorkloadProgramResult(dgd)
	result.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
	}
	reconciler := newDGDRestartReconciler()

	t.Log("Resolve restart state against the program-owned status accumulator")
	restart := reconciler.Resolve(
		context.Background(),
		dgd,
		&result.Status,
		nil,
	)
	recordRestartTransition(result.Status.Restart, restart.Status, &result)
	result.Status.Restart = restart.Status

	t.Log("Verify status and its transition event remain coupled in the result")
	require.NotNil(t, result.Status.Restart)
	assert.Equal(t, nvidiacomv1beta1.RestartPhaseSuperseded, result.Status.Restart.Phase)
	require.Len(t, result.Events, 1)
	assert.Equal(t, "RestartSuperseded", result.Events[0].Reason)
	assert.Empty(t, dgd.Status.Restart)
}

func TestComponentProgram_ReconcileWorkerRollout(t *testing.T) {
	t.Run("single-node component workload starts a managed rollout", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: commonconsts.ComponentTypeWorker,
				Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "new"}},
			},
		})
		dgd.Annotations = map[string]string{
			commonconsts.AnnotationCurrentWorkerHashV2: "old-worker-hash",
		}
		reconciler := createTestDGDReconcilerWithStatus(dgd)
		program := reconciler.newComponentProgram()
		status := dgd.DeepCopy().Status

		require.NoError(t, program.reconcileWorkerRollout(context.Background(), dgd, &status))

		require.NotNil(t, status.RollingUpdate)
		assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhasePending, status.RollingUpdate.Phase)
		assert.Nil(t, dgd.Status.RollingUpdate)
		assert.Equal(t, "old-worker-hash", dgd.Annotations[commonconsts.AnnotationCurrentWorkerHashV2])
	})

	t.Run("multinode component workload keeps unsupported-path hash behavior", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: commonconsts.ComponentTypeWorker,
				Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "new"}},
				Multinode:     &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2},
			},
		})
		dgd.Annotations = map[string]string{
			commonconsts.AnnotationCurrentWorkerHash: "old-worker-hash",
		}
		reconciler := createTestDGDReconcilerWithStatus(dgd)
		program := reconciler.newComponentProgram()
		status := dgd.DeepCopy().Status

		require.NoError(t, program.reconcileWorkerRollout(context.Background(), dgd, &status))

		assert.Nil(t, status.RollingUpdate)
		assert.Nil(t, dgd.Status.RollingUpdate)
		desired, err := desiredWorkerHashes(dgd)
		require.NoError(t, err)
		assert.True(t, currentWorkerHashesMatchDesired(currentWorkerHashes(dgd), desired))
	})
}

func TestComponentWorkloadsReconciler_PreserveExistingBackendFramework(t *testing.T) {
	tests := []struct {
		name          string
		dcdName       string
		existing      bool
		wantFramework string
	}{
		{
			name:          "existing DCD preserves its immutable stored backend",
			dcdName:       "vllm-disagg-planner-frontend",
			existing:      true,
			wantFramework: "",
		},
		{
			name:          "new DCD keeps its inferred backend",
			dcdName:       "vllm-disagg-planner-vllmdecodeworker-2dad72b9",
			wantFramework: "vllm",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build the desired DCD and any existing immutable API state")
			objects := []client.Object{}
			if tt.existing {
				objects = append(objects, &nvidiacomv1beta1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{Name: tt.dcdName, Namespace: "jsm"},
					Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
							ComponentName: "Frontend",
							ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
						},
					},
				})
			}
			workloads := &componentWorkloadsReconciler{
				syncer: newDGDResourceSyncer(
					fake.NewClientBuilder().
						WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
						WithObjects(objects...).
						Build(),
					nil,
				),
			}
			desired := &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: tt.dcdName, Namespace: "jsm"},
				Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
					BackendFramework: "vllm",
				},
			}

			t.Log("Resolve the backend value that can safely be synchronized")
			require.NoError(t, workloads.preserveExistingBackendFramework(context.Background(), desired))

			t.Log("Verify updates preserve stored state while creates keep the inferred value")
			assert.Equal(t, tt.wantFramework, desired.Spec.BackendFramework)
		})
	}
}

func TestComponentWorkloadsReconciler_ApplyCheckpointStartupPolicy(t *testing.T) {
	workloads := &componentWorkloadsReconciler{}
	tests := []struct {
		name              string
		replicas          int32
		podTemplate       *corev1.PodTemplateSpec
		checkpointInfo    checkpoint.CheckpointInfo
		wantReplicas      int32
		wantStartupPolicy nvidiacomv1beta1.CheckpointStartupPolicy
		wantCandidate     bool
	}{
		{
			name:     "immediate stamps stable restore candidate metadata",
			replicas: 2,
			podTemplate: &corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						snapshotprotocol.CheckpointIDLabel: "stale",
					},
					Annotations: map[string]string{
						snapshotprotocol.CheckpointArtifactVersionAnnotation: "stale",
					},
				},
			},
			checkpointInfo: checkpoint.CheckpointInfo{
				Enabled:        true,
				Exists:         true,
				Ready:          true,
				Hash:           "checkpoint-id",
				CheckpointName: "checkpoint-name",
				StartupPolicy:  nvidiacomv1alpha1.CheckpointStartupPolicyImmediate,
			},
			wantReplicas:      2,
			wantStartupPolicy: nvidiacomv1beta1.CheckpointStartupPolicyImmediate,
			wantCandidate:     true,
		},
		{
			name:     "wait for checkpoint gates replicas until ready",
			replicas: 3,
			checkpointInfo: checkpoint.CheckpointInfo{
				Enabled:        true,
				Exists:         true,
				Ready:          false,
				CheckpointName: "checkpoint-name",
				StartupPolicy:  nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint,
			},
			wantReplicas:      0,
			wantStartupPolicy: nvidiacomv1beta1.CheckpointStartupPolicyWaitForCheckpoint,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a generated DCD and its resolved checkpoint observation")
			dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
				Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						Replicas:    ptr.To(tt.replicas),
						PodTemplate: tt.podTemplate,
					},
				},
			}

			t.Log("Apply the checkpoint startup policy before synchronizing the DCD")
			require.NoError(t, workloads.applyCheckpointStartupPolicy(dcd, &tt.checkpointInfo))

			t.Log("Verify the child checkpoint reference, startup policy, and replica gate")
			require.NotNil(t, dcd.Spec.Experimental)
			require.NotNil(t, dcd.Spec.Experimental.Checkpoint)
			require.NotNil(t, dcd.Spec.Experimental.Checkpoint.CheckpointRef)
			assert.Equal(t, "checkpoint-name", *dcd.Spec.Experimental.Checkpoint.CheckpointRef)
			assert.Nil(t, dcd.Spec.Experimental.Checkpoint.Identity)
			assert.Nil(t, dcd.Spec.Experimental.Checkpoint.Job)
			assert.Equal(t, tt.wantStartupPolicy, dcd.Spec.Experimental.Checkpoint.StartupPolicy)
			assert.Equal(t, tt.wantReplicas, *dcd.Spec.Replicas)
			if !tt.wantCandidate {
				return
			}

			t.Log("Verify immediate startup publishes stable restore-candidate metadata")
			assert.Empty(t, dcd.Spec.PodTemplate.Labels[snapshotprotocol.CheckpointIDLabel])
			assert.Equal(t, commonconsts.KubeLabelValueTrue, dcd.Spec.PodTemplate.Annotations[commonconsts.CheckpointRestoreCandidateAnnotation])
			assert.Equal(t, "checkpoint-name", dcd.Spec.PodTemplate.Annotations[commonconsts.CheckpointNameAnnotation])
			assert.Equal(t, commonconsts.MainContainerName, dcd.Spec.PodTemplate.Annotations[snapshotprotocol.TargetContainersAnnotation])
		})
	}
}
