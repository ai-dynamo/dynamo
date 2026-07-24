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
		})
	}
}

func TestGroveProgram_ReconcileAdapter(t *testing.T) {
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
			err := (&groveProgram{reconcile: reconcile}).Reconcile(context.Background(), state)

			t.Log("Verify state is committed only after successful reconciliation")
			require.True(t, called)
			require.ErrorIs(t, err, tt.wantErr)
			assert.Equal(t, tt.wantState, state.Result)
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
}

func TestComponentProgram_PreserveExistingBackendFramework(t *testing.T) {
	ctx := context.Background()
	existing := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-disagg-planner-frontend",
			Namespace: "jsm",
		},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "Frontend",
				ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
			},
		},
	}
	program := &componentProgram{
		reconciler: &DynamoGraphDeploymentReconciler{
			Client: fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithObjects(existing).
				Build(),
		},
	}

	t.Log("Preserve the immutable stored backend when updating an existing DCD")
	desiredExisting := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      existing.Name,
			Namespace: existing.Namespace,
		},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
		},
	}
	require.NoError(t, program.preserveExistingBackendFramework(ctx, desiredExisting))
	assert.Empty(t, desiredExisting.Spec.BackendFramework)

	t.Log("Keep the inferred backend when creating a new DCD")
	desiredNew := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-disagg-planner-vllmdecodeworker-2dad72b9",
			Namespace: "jsm",
		},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
		},
	}
	require.NoError(t, program.preserveExistingBackendFramework(ctx, desiredNew))
	assert.Equal(t, "vllm", desiredNew.Spec.BackendFramework)
}

func TestComponentProgram_ApplyCheckpointStartupPolicy(t *testing.T) {
	program := &componentProgram{}

	t.Run("immediate stamps stable restore candidate metadata", func(t *testing.T) {
		dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					Replicas: ptr.To(int32(2)),
					PodTemplate: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								snapshotprotocol.CheckpointIDLabel: "stale",
							},
							Annotations: map[string]string{
								snapshotprotocol.CheckpointStatusAnnotation: "stale",
							},
						},
					},
				},
			},
		}
		info := &checkpoint.CheckpointInfo{
			Enabled:        true,
			Exists:         true,
			Ready:          true,
			Hash:           "checkpoint-id",
			CheckpointName: "checkpoint-name",
			StartupPolicy:  nvidiacomv1alpha1.CheckpointStartupPolicyImmediate,
		}

		require.NoError(t, program.applyCheckpointStartupPolicy(dcd, info))

		require.NotNil(t, dcd.Spec.Experimental)
		require.NotNil(t, dcd.Spec.Experimental.Checkpoint)
		require.NotNil(t, dcd.Spec.Experimental.Checkpoint.CheckpointRef)
		assert.Equal(t, "checkpoint-name", *dcd.Spec.Experimental.Checkpoint.CheckpointRef)
		assert.Nil(t, dcd.Spec.Experimental.Checkpoint.Identity)
		assert.Nil(t, dcd.Spec.Experimental.Checkpoint.Job)
		assert.Equal(t, nvidiacomv1beta1.CheckpointStartupPolicyImmediate, dcd.Spec.Experimental.Checkpoint.StartupPolicy)
		assert.Equal(t, int32(2), *dcd.Spec.Replicas)
		assert.Empty(t, dcd.Spec.PodTemplate.Labels[snapshotprotocol.CheckpointIDLabel])
		assert.Equal(t, commonconsts.KubeLabelValueTrue, dcd.Spec.PodTemplate.Annotations[commonconsts.CheckpointRestoreCandidateAnnotation])
		assert.Equal(t, "checkpoint-name", dcd.Spec.PodTemplate.Annotations[commonconsts.CheckpointNameAnnotation])
		assert.Equal(t, commonconsts.MainContainerName, dcd.Spec.PodTemplate.Annotations[snapshotprotocol.TargetContainersAnnotation])
	})

	t.Run("wait for checkpoint gates replicas until ready", func(t *testing.T) {
		dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					Replicas: ptr.To(int32(3)),
				},
			},
		}
		info := &checkpoint.CheckpointInfo{
			Enabled:        true,
			Exists:         true,
			Ready:          false,
			CheckpointName: "checkpoint-name",
			StartupPolicy:  nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint,
		}

		require.NoError(t, program.applyCheckpointStartupPolicy(dcd, info))

		require.NotNil(t, dcd.Spec.Experimental)
		require.NotNil(t, dcd.Spec.Experimental.Checkpoint)
		require.NotNil(t, dcd.Spec.Experimental.Checkpoint.CheckpointRef)
		assert.Equal(t, "checkpoint-name", *dcd.Spec.Experimental.Checkpoint.CheckpointRef)
		assert.Equal(t, nvidiacomv1beta1.CheckpointStartupPolicyWaitForCheckpoint, dcd.Spec.Experimental.Checkpoint.StartupPolicy)
		assert.Equal(t, int32(0), *dcd.Spec.Replicas)
	})
}
