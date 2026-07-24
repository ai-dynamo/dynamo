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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestDynamoGraphDeploymentReconciler_selectWorkloadProgram(t *testing.T) {
	tests := []struct {
		name         string
		groveEnabled bool
		annotations  map[string]string
		wantProgram  workloadProgram
	}{
		{
			name:        "Grove feature disabled selects component program",
			wantProgram: &componentProgram{},
		},
		{
			name:         "Grove feature enabled selects Grove program",
			groveEnabled: true,
			wantProgram:  &groveProgram{},
		},
		{
			name:         "explicit Grove disable selects component program",
			groveEnabled: true,
			annotations: map[string]string{
				commonconsts.KubeAnnotationEnableGrove: commonconsts.KubeLabelValueFalse,
			},
			wantProgram: &componentProgram{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build the ephemeral state and reconciler selection inputs")
			state := &graphReconcileState{
				DGD: &nvidiacomv1beta1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{Annotations: tt.annotations},
				},
			}
			reconciler := &DynamoGraphDeploymentReconciler{
				RuntimeConfig: &commonController.RuntimeConfig{
					Gate: features.Gates{Grove: tt.groveEnabled},
				},
			}

			t.Log("Select one complete workload program")
			got := reconciler.selectWorkloadProgram(state)

			assert.IsType(t, tt.wantProgram, got)
			if component, ok := got.(*componentProgram); ok {
				assert.Same(t, reconciler, component.reconciler)
			}
			if grove, ok := got.(*groveProgram); ok {
				assert.Same(t, reconciler, grove.reconciler)
			}
		})
	}
}

func TestGroveProgram_ReconcileWorkloadsAdapter(t *testing.T) {
	reconcileErr := errors.New("reconcile failed")
	tests := []struct {
		name         string
		returned     ReconcileResult
		reconcileErr error
		wantState    ReconcileResult
		wantErr      error
	}{
		{
			name: "Grove program records a successful result",
			returned: ReconcileResult{
				State:  nvidiacomv1beta1.DGDStatePending,
				Reason: "grove_pending",
			},
			wantState: ReconcileResult{
				State:  nvidiacomv1beta1.DGDStatePending,
				Reason: "grove_pending",
			},
		},
		{
			name: "Grove program preserves prior state on error",
			returned: ReconcileResult{
				State:  nvidiacomv1beta1.DGDStateSuccessful,
				Reason: "must_not_be_committed",
			},
			reconcileErr: reconcileErr,
			wantState: ReconcileResult{
				State:  nvidiacomv1beta1.DGDStatePending,
				Reason: "existing",
			},
			wantErr: reconcileErr,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build ephemeral inputs that must be passed unchanged to the program")
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
			restartState := &dynamo.RestartState{Timestamp: "restart"}
			checkpointInfos := map[string]*checkpoint.CheckpointInfo{
				"worker": {CheckpointName: "checkpoint"},
			}
			state := &graphReconcileState{
				DGD:             dgd,
				HasMultinode:    true,
				RestartState:    restartState,
				CheckpointInfos: checkpointInfos,
				Result: ReconcileResult{
					State:  nvidiacomv1beta1.DGDStatePending,
					Reason: "existing",
				},
			}
			called := false
			reconcile := func(
				_ context.Context,
				gotDGD *nvidiacomv1beta1.DynamoGraphDeployment,
				gotRestartState *dynamo.RestartState,
				gotCheckpointInfos map[string]*checkpoint.CheckpointInfo,
			) (ReconcileResult, error) {
				called = true
				require.Same(t, dgd, gotDGD)
				require.Same(t, restartState, gotRestartState)
				require.Equal(t, checkpointInfos, gotCheckpointInfos)
				return tt.returned, tt.reconcileErr
			}

			t.Log("Run the selected complete workload program")
			result, err := (&groveProgram{reconcile: reconcile}).reconcileWorkloads(context.Background(), state)

			t.Log("Verify the adapter forwards inputs and outputs unchanged")
			require.True(t, called)
			require.ErrorIs(t, err, tt.wantErr)
			if tt.reconcileErr == nil {
				assert.Equal(t, tt.wantState, result)
			}
		})
	}
}

func TestComponentProgram_ReconcilePreservesResultOnError(t *testing.T) {
	t.Log("Inject a component-path API failure before a result can be committed")
	reconcileErr := errors.New("reconcile failed")
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithInterceptorFuncs(interceptor.Funcs{
			List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
				return reconcileErr
			},
		}).
		Build()
	program := &componentProgram{
		reconciler: &DynamoGraphDeploymentReconciler{Client: kubeClient},
	}
	previous := ReconcileResult{
		State:  nvidiacomv1beta1.DGDStatePending,
		Reason: "existing",
	}
	state := &graphReconcileState{
		DGD: &nvidiacomv1beta1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		},
		Result: previous,
	}

	err := program.Reconcile(context.Background(), state)

	t.Log("Verify failed reconciliation cannot publish a partial result")
	require.ErrorIs(t, err, reconcileErr)
	assert.Equal(t, previous, state.Result)
	reason, ok := workloadProgramFailureReason(err)
	require.True(t, ok)
	assert.Equal(t, reasonFailedToInitializeWorkerHash, reason)
}

func TestGroveProgram_ReconcilePreservesResultOnError(t *testing.T) {
	t.Log("Inject an unsupported-path metadata failure before shared reconciliation")
	reconcileErr := errors.New("reconcile failed")
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: commonconsts.ComponentTypeWorker},
	})
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd).
		WithInterceptorFuncs(interceptor.Funcs{
			Update: func(context.Context, client.WithWatch, client.Object, ...client.UpdateOption) error {
				return reconcileErr
			},
		}).
		Build()
	program := &groveProgram{
		reconciler: &DynamoGraphDeploymentReconciler{
			Client:   kubeClient,
			Recorder: record.NewFakeRecorder(10),
		},
		reconcile: func(
			context.Context,
			*nvidiacomv1beta1.DynamoGraphDeployment,
			*dynamo.RestartState,
			map[string]*checkpoint.CheckpointInfo,
		) (ReconcileResult, error) {
			t.Fatal("Grove workload reconciliation must not run after rollout preparation fails")
			return ReconcileResult{}, nil
		},
	}
	previous := ReconcileResult{
		State:  nvidiacomv1beta1.DGDStatePending,
		Reason: "existing",
	}
	state := &graphReconcileState{DGD: dgd, Result: previous}

	err := program.Reconcile(context.Background(), state)

	t.Log("Verify failed reconciliation cannot publish a partial result")
	require.ErrorIs(t, err, reconcileErr)
	assert.Equal(t, previous, state.Result)
	reason, ok := workloadProgramFailureReason(err)
	require.True(t, ok)
	assert.Equal(t, reasonFailedToInitializeWorkerHash, reason)
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
			commonconsts.AnnotationCurrentWorkerHash: "old-worker-hash",
		}
		reconciler := createTestReconcilerWithStatus(dgd)
		program := &componentProgram{reconciler: reconciler}

		require.NoError(t, program.reconcileWorkerRollout(context.Background(), dgd))

		require.NotNil(t, dgd.Status.RollingUpdate)
		assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhasePending, dgd.Status.RollingUpdate.Phase)
		assert.Equal(t, "old-worker-hash", dgd.Annotations[commonconsts.AnnotationCurrentWorkerHash])
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
		reconciler := createTestReconcilerWithStatus(dgd)
		program := &componentProgram{reconciler: reconciler}

		require.NoError(t, program.reconcileWorkerRollout(context.Background(), dgd))

		assert.Nil(t, dgd.Status.RollingUpdate)
		desired, err := reconciler.desiredWorkerHashes(dgd)
		require.NoError(t, err)
		assert.True(t, currentWorkerHashesMatchDesired(reconciler.currentWorkerHashes(dgd), desired))
	})
}

func TestComponentProgram_PreserveExistingBackendFramework(t *testing.T) {
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
			program := &componentProgram{
				reconciler: &DynamoGraphDeploymentReconciler{
					Client: fake.NewClientBuilder().
						WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
						WithObjects(objects...).
						Build(),
				},
			}
			desired := &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: tt.dcdName, Namespace: "jsm"},
				Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
					BackendFramework: "vllm",
				},
			}

			t.Log("Resolve the backend value that can safely be synchronized")
			require.NoError(t, program.preserveExistingBackendFramework(context.Background(), desired))

			t.Log("Verify updates preserve stored state while creates keep the inferred value")
			assert.Equal(t, tt.wantFramework, desired.Spec.BackendFramework)
		})
	}
}

func TestComponentProgram_ApplyCheckpointStartupPolicy(t *testing.T) {
	program := &componentProgram{}
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
						snapshotprotocol.CheckpointStatusAnnotation: "stale",
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
			require.NoError(t, program.applyCheckpointStartupPolicy(dcd, &tt.checkpointInfo))

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
