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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
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

	mutator := NewPodCheckpointRestoreMutator(
		fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(readyCheckpoint, notReadyCheckpoint).
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

	t.Run("ready checkpoint restore-shapes pod create", func(t *testing.T) {
		pod := checkpointCandidatePod("worker-checkpoint")
		req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Namespace: "default",
			Object:    runtime.RawExtension{Raw: mustMarshalPod(t, pod)},
		}}

		resp := mutator.Handle(context.Background(), req)
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
		assert.Equal(t, "sleep", patchesByPath["/spec/containers/0/command/0"])
		assert.Equal(t, "infinity", patchesByPath["/spec/containers/0/command/1"])
	})

	t.Run("not ready checkpoint leaves pod unchanged", func(t *testing.T) {
		pod := checkpointCandidatePod("pending-checkpoint")
		req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Namespace: "default",
			Object:    runtime.RawExtension{Raw: mustMarshalPod(t, pod)},
		}}

		resp := mutator.Handle(context.Background(), req)
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

		resp := mutator.Handle(context.Background(), req)
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
