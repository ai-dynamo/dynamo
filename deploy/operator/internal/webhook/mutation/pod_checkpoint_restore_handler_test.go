/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package mutation

import (
	"context"
	"encoding/json"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func TestPodCheckpointRestoreMutatorHandle(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, nvidiacomv1alpha1.AddToScheme(scheme))

	readyCheckpoint := &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-checkpoint",
			Namespace: "default",
			Labels: map[string]string{
				snapshotprotocol.CheckpointIDLabel: "checkpoint-123",
			},
			Annotations: map[string]string{
				snapshotprotocol.CheckpointArtifactVersionAnnotation: "2",
			},
		},
		Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
			Phase: nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
		},
	}
	notReadyCheckpoint := readyCheckpoint.DeepCopy()
	notReadyCheckpoint.Name = "pending-checkpoint"
	notReadyCheckpoint.Labels = map[string]string{snapshotprotocol.CheckpointIDLabel: "checkpoint-456"}
	notReadyCheckpoint.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseCreating
	automaticCheckpoint := readyCheckpoint.DeepCopy()
	automaticCheckpoint.Name = "automatic-checkpoint"
	automaticCheckpoint.UID = types.UID("automatic-uid")
	automaticCheckpoint.Generation = 2
	automaticCheckpoint.Annotations[consts.CheckpointAutoAnnotation] = consts.KubeLabelValueTrue
	automaticBinding, err := checkpoint.AutomaticCheckpointBinding(automaticCheckpoint)
	require.NoError(t, err)
	staleAutomaticCheckpoint := automaticCheckpoint.DeepCopy()
	staleAutomaticCheckpoint.Generation--
	staleAutomaticBinding, err := checkpoint.AutomaticCheckpointBinding(staleAutomaticCheckpoint)
	require.NoError(t, err)
	replacementCheckpoint := readyCheckpoint.DeepCopy()
	replacementCheckpoint.Name = "replacement-checkpoint"
	replacementCheckpoint.UID = types.UID("replacement-uid")
	replacementCheckpoint.Generation = 1
	replacementCheckpoint.Annotations[consts.CheckpointAutoAnnotation] = consts.KubeLabelValueTrue
	originalReplacementCheckpoint := replacementCheckpoint.DeepCopy()
	originalReplacementCheckpoint.UID = types.UID("original-uid")
	originalReplacementBinding, err := checkpoint.AutomaticCheckpointBinding(originalReplacementCheckpoint)
	require.NoError(t, err)

	mutator := NewPodCheckpointRestoreMutator(
		fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(readyCheckpoint, notReadyCheckpoint, automaticCheckpoint, replacementCheckpoint).
			Build(),
		&configv1alpha1.OperatorConfiguration{
			Checkpoint: configv1alpha1.CheckpointConfiguration{
				Enabled: true,
				Storage: configv1alpha1.CheckpointStorageConfiguration{
					Type: snapshotprotocol.StorageTypePVC,
					PVC: configv1alpha1.CheckpointPVCConfig{
						PVCName:  "snapshot-pvc",
						BasePath: "/checkpoints",
					},
				},
			},
		},
	)
	mutator.scheme = scheme
	ctx := features.WithGate(context.Background(), features.Gates{Checkpoint: true})

	t.Run("ready checkpoint restore-shapes pod create", func(t *testing.T) {
		pod := checkpointCandidatePod("worker-checkpoint")
		req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Namespace: "default",
			Object:    runtime.RawExtension{Raw: mustMarshalPod(t, pod)},
		}}

		resp := mutator.Handle(ctx, req)
		require.True(t, resp.Allowed)
		require.NotEmpty(t, resp.Patches)

		patchesByPath := map[string]any{}
		for _, patch := range resp.Patches {
			patchesByPath[patch.Path] = patch.Value
		}
		assert.Equal(t, "checkpoint-123", patchesByPath["/metadata/labels/nvidia.com~1snapshot-checkpoint-id"])
		assert.Equal(t, "true", patchesByPath["/metadata/labels/nvidia.com~1snapshot-is-restore-target"])
		assert.Equal(t, "2", patchesByPath["/metadata/annotations/nvidia.com~1snapshot-artifact-version"])
		assert.NotContains(t, patchesByPath, "/metadata/annotations/nvidia.com~1snapshot-target-containers")
		assert.Contains(t, patchesByPath, "/spec/volumes")
		for _, patch := range resp.Patches {
			assert.NotContains(t, patch.Path, "/command")
			assert.NotContains(t, patch.Path, "/args")
		}
		envPatch, ok := patchesByPath["/spec/containers/0/env"].([]any)
		require.True(t, ok, "expected env patch, got %#v", patchesByPath)
		assert.Contains(t, envPatch, map[string]any{
			"name":  "DYN_SNAPSHOT_RESTORE_STANDBY",
			"value": "1",
		})

		t.Run("automatic checkpoint with matching binding restore-shapes pod", func(t *testing.T) {
			pod := checkpointCandidatePod("automatic-checkpoint")
			pod.Annotations[consts.CheckpointBindingAnnotation] = automaticBinding
			req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
				Operation: admissionv1.Create,
				Namespace: "default",
				Object:    runtime.RawExtension{Raw: mustMarshalPod(t, pod)},
			}}

			resp := mutator.Handle(ctx, req)
			require.True(t, resp.Allowed)
			require.NotEmpty(t, resp.Patches)
			for _, patch := range resp.Patches {
				assert.NotEqual(t,
					"/metadata/annotations/nvidia.com~1dynamo-checkpoint-name",
					patch.Path,
				)
				assert.NotEqual(t,
					"/metadata/annotations/nvidia.com~1dynamo-checkpoint-binding",
					patch.Path,
				)
			}
		})

		for _, tt := range []struct {
			name           string
			checkpointName string
			binding        string
		}{
			{
				name:           "DCD candidate rejects changed checkpoint generation",
				checkpointName: "automatic-checkpoint",
				binding:        staleAutomaticBinding,
			},
			{
				name:           "Grove candidate rejects same-name checkpoint replacement",
				checkpointName: "replacement-checkpoint",
				binding:        originalReplacementBinding,
			},
			{
				name:           "automatic candidate rejects missing binding",
				checkpointName: "automatic-checkpoint",
			},
			{
				name:           "bound candidate rejects missing checkpoint",
				checkpointName: "deleted-checkpoint",
				binding:        "deleted-uid/1",
			},
			{
				name:           "bound candidate rejects unmarked checkpoint",
				checkpointName: "worker-checkpoint",
				binding:        "worker-uid/1",
			},
		} {
			t.Run(tt.name, func(t *testing.T) {
				pod := checkpointCandidatePod(tt.checkpointName)
				if tt.binding != "" {
					pod.Annotations[consts.CheckpointBindingAnnotation] = tt.binding
				}
				req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
					Operation: admissionv1.Create,
					Namespace: "default",
					Object:    runtime.RawExtension{Raw: mustMarshalPod(t, pod)},
				}}

				resp := mutator.Handle(ctx, req)
				require.False(t, resp.Allowed)
				assert.Empty(t, resp.Patches)
			})
		}
	})

	t.Run("bound already-shaped restore targets verify checkpoint provenance", func(t *testing.T) {
		cases := []struct {
			name           string
			checkpointName string
			binding        string
		}{
			{name: "missing", checkpointName: "deleted-checkpoint", binding: "deleted-uid/1"},
			{name: "unmarked", checkpointName: "worker-checkpoint", binding: "worker-uid/1"},
			{name: "replaced", checkpointName: "replacement-checkpoint", binding: originalReplacementBinding},
			{name: "stale", checkpointName: "automatic-checkpoint", binding: staleAutomaticBinding},
		}

		for _, provider := range []string{"DCD", "Grove"} {
			for _, tt := range cases {
				t.Run(provider+" "+tt.name, func(t *testing.T) {
					t.Log("Build an already-shaped bound restore target")
					pod := checkpointCandidatePod(tt.checkpointName)
					pod.Labels[snapshotprotocol.CheckpointIDLabel] = "checkpoint-123"
					pod.Labels[snapshotprotocol.RestoreTargetLabel] = consts.KubeLabelValueTrue
					pod.Annotations[consts.CheckpointBindingAnnotation] = tt.binding
					delete(pod.Annotations, consts.CheckpointRestoreCandidateAnnotation)
					pod.Labels["test.nvidia.com/workload-provider"] = provider

					t.Log("Reject before the already-shaped return")
					resp := mutator.Handle(ctx, admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
						Operation: admissionv1.Create,
						Namespace: "default",
						Object:    runtime.RawExtension{Raw: mustMarshalPod(t, pod)},
					}})
					require.False(t, resp.Allowed)
					assert.Empty(t, resp.Patches)
				})
			}
		}
	})

	t.Run("matching bound shaped target and true checkpoint source remain exempt", func(t *testing.T) {
		t.Log("Allow a shaped restore target only after its exact binding verifies")
		target := checkpointCandidatePod("automatic-checkpoint")
		target.Labels[snapshotprotocol.CheckpointIDLabel] = "checkpoint-123"
		target.Labels[snapshotprotocol.RestoreTargetLabel] = consts.KubeLabelValueTrue
		target.Annotations[consts.CheckpointBindingAnnotation] = automaticBinding
		targetResp := mutator.Handle(ctx, admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Namespace: "default",
			Object:    runtime.RawExtension{Raw: mustMarshalPod(t, target)},
		}})
		require.True(t, targetResp.Allowed)
		assert.Empty(t, targetResp.Patches)

		t.Log("Keep true checkpoint source Pods exempt from restore binding checks")
		source := checkpointCandidatePod("deleted-checkpoint")
		source.Labels[snapshotprotocol.CheckpointIDLabel] = "checkpoint-123"
		source.Labels[snapshotprotocol.CheckpointSourceLabel] = consts.KubeLabelValueTrue
		source.Annotations[consts.CheckpointBindingAnnotation] = "stale/1"
		sourceResp := mutator.Handle(ctx, admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Namespace: "default",
			Object:    runtime.RawExtension{Raw: mustMarshalPod(t, source)},
		}})
		require.True(t, sourceResp.Allowed)
		assert.Empty(t, sourceResp.Patches)

		t.Log("Keep an unbound already-shaped non-candidate Pod unchanged")
		unbound := checkpointCandidatePod("worker-checkpoint")
		unbound.Labels[snapshotprotocol.CheckpointIDLabel] = "checkpoint-123"
		unbound.Labels[snapshotprotocol.RestoreTargetLabel] = consts.KubeLabelValueTrue
		delete(unbound.Annotations, consts.CheckpointRestoreCandidateAnnotation)
		unboundResp := mutator.Handle(ctx, admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Namespace: "default",
			Object:    runtime.RawExtension{Raw: mustMarshalPod(t, unbound)},
		}})
		require.True(t, unboundResp.Allowed)
		assert.Empty(t, unboundResp.Patches)
	})

	t.Run("not ready checkpoint leaves pod unchanged", func(t *testing.T) {
		pod := checkpointCandidatePod("pending-checkpoint")
		req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Namespace: "default",
			Object:    runtime.RawExtension{Raw: mustMarshalPod(t, pod)},
		}}

		resp := mutator.Handle(ctx, req)
		require.True(t, resp.Allowed)
		assert.Empty(t, resp.Patches)
	})

	t.Run("update leaves pod unchanged", func(t *testing.T) {
		pod := checkpointCandidatePod("worker-checkpoint")
		req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Update,
			Namespace: "default",
			Object:    runtime.RawExtension{Raw: mustMarshalPod(t, pod)},
		}}

		resp := mutator.Handle(ctx, req)
		require.True(t, resp.Allowed)
		assert.Empty(t, resp.Patches)
	})

	t.Run("arbitrary annotated pod without operator stamp is ignored", func(t *testing.T) {
		pod := checkpointCandidatePod("worker-checkpoint")
		delete(pod.Labels, consts.KubeLabelDynamoComponent)
		req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Namespace: "default",
			Object:    runtime.RawExtension{Raw: mustMarshalPod(t, pod)},
		}}

		resp := mutator.Handle(ctx, req)
		require.True(t, resp.Allowed)
		assert.Empty(t, resp.Patches)
	})
}

func checkpointCandidatePod(checkpointName string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-0",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoComponent: "worker",
				consts.KubeLabelDynamoNamespace: "default-worker",
				consts.KubeLabelDynamoSelector:  "worker",
			},
			Annotations: map[string]string{
				consts.CheckpointRestoreCandidateAnnotation: consts.KubeLabelValueTrue,
				consts.CheckpointNameAnnotation:             checkpointName,
				snapshotprotocol.TargetContainersAnnotation: consts.MainContainerName,
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:    consts.MainContainerName,
				Image:   "worker:latest",
				Command: []string{"python3", "-m", "dynamo.vllm"},
			}},
		},
	}
}

func mustMarshalPod(t *testing.T, pod *corev1.Pod) []byte {
	t.Helper()
	raw, err := json.Marshal(pod)
	require.NoError(t, err)
	return raw
}
