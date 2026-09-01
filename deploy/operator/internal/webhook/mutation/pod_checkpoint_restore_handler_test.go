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
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpointjob"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	podcontract "github.com/ai-dynamo/snapshot/api/podcontract"
	snapshotv1alpha1 "github.com/ai-dynamo/snapshot/api/v1alpha1"
	jsonpatch "github.com/evanphx/json-patch/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
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

	webhookClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(readyCheckpoint, notReadyCheckpoint).
		Build()
	mutator := NewPodCheckpointRestoreMutator(
		webhookClient,
		webhookClient,
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
			"name":  podcontract.RestoreStandbyModeEnv,
			"value": "1",
		})
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

func TestPodCheckpointRestoreMutatorNativeRestore(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, nvidiacomv1alpha1.AddToScheme(scheme))
	require.NoError(t, snapshotv1alpha1.AddToScheme(scheme))

	snapshot := nativeRestoreTestSnapshot()
	staleSnapshot := snapshot.DeepCopy()
	staleSnapshot.UID = types.UID("stale-snapshot-uid")
	staleSnapshot.Status.BoundPodSnapshotContentName = ptr.To("stale-content")
	webhookClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(staleSnapshot).Build()
	apiReader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(snapshot).Build()
	mutator := NewPodCheckpointRestoreMutator(
		webhookClient,
		apiReader,
		&configv1alpha1.OperatorConfiguration{
			Checkpoint: configv1alpha1.CheckpointConfiguration{Enabled: true},
		},
	)
	mutator.scheme = scheme
	ctx := features.WithGate(context.Background(), features.Gates{Checkpoint: true})

	t.Run("shapes one captured source into two engine destinations", func(t *testing.T) {
		t.Log("Given a native restore candidate pinned to a Ready compatible PodSnapshot")
		pod := nativeRestoreCandidatePod(snapshot)
		original := mustMarshalPod(t, pod)
		req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Namespace: pod.Namespace,
			Object:    runtime.RawExtension{Raw: original},
		}}

		t.Log("When the Dynamo admission webhook shapes the Pod")
		resp := mutator.Handle(ctx, req)

		t.Log("Then only the standalone Snapshot wire contract and Dynamo standby behavior remain")
		require.True(t, resp.Allowed)
		shaped := applyAdmissionPatches(t, original, resp)
		assert.Equal(t, snapshot.Name, shaped.Annotations[podcontract.RestoreFromAnnotation])
		assert.Equal(t, "main=engine-0,main=engine-1", shaped.Annotations[podcontract.RestoreContainerMapAnnotation])
		assert.NotContains(t, shaped.Annotations, consts.CheckpointRestoreCandidateAnnotation)
		assert.NotContains(t, shaped.Annotations, consts.RestoreCandidateTargetContainersAnnotation)
		assert.NotContains(t, shaped.Annotations, snapshotprotocol.TargetContainersAnnotation)
		require.Len(t, shaped.Spec.Volumes, 1)
		assert.Equal(t, podcontract.SnapshotControlVolumeName, shaped.Spec.Volumes[0].Name)
		for _, container := range shaped.Spec.Containers {
			assert.Equal(t, container.Name, container.VolumeMounts[0].SubPath)
			assert.Equal(t, podcontract.SnapshotControlMountPath, container.VolumeMounts[0].MountPath)
			assert.Contains(t, container.Env, corev1.EnvVar{Name: podcontract.RestoreStandbyModeEnv, Value: "1"})
			require.NotNil(t, container.StartupProbe)
			require.NotNil(t, container.StartupProbe.Exec)
			assert.Equal(t, []string{"cat", "/snapshot-control/restore-complete"}, container.StartupProbe.Exec.Command)
		}
	})

	t.Run("denies stale or unsafe native candidates", func(t *testing.T) {
		tests := []struct {
			name    string
			mutate  func(*corev1.Pod)
			wantErr string
		}{
			{
				name: "deleted and recreated snapshot",
				mutate: func(pod *corev1.Pod) {
					pod.Annotations[consts.SnapshotCandidateUIDAnnotation] = "stale-snapshot-uid"
					pod.Annotations[consts.SnapshotCandidateContentAnnotation] = "stale-content"
				},
				wantErr: "UID changed",
			},
			{
				name: "worker generation mismatch",
				mutate: func(pod *corev1.Pod) {
					pod.Labels[consts.KubeLabelDynamoWorkerHash] = "worker-v2"
				},
				wantErr: "does not match expected hash",
			},
			{
				name: "missing operator stamp",
				mutate: func(pod *corev1.Pod) {
					delete(pod.Labels, consts.KubeLabelDynamoComponent)
				},
				wantErr: "not operator-stamped",
			},
			{
				name: "unsupported workload entrypoint",
				mutate: func(pod *corev1.Pod) {
					pod.Spec.Containers[1].Command = []string{"serve-model"}
				},
				wantErr: "must directly invoke python -m",
			},
			{
				name: "legacy restore metadata conflict",
				mutate: func(pod *corev1.Pod) {
					pod.Labels[snapshotprotocol.CheckpointIDLabel] = "legacy-checkpoint"
				},
				wantErr: "conflicts with legacy checkpoint metadata",
			},
			{
				name: "retired Snapshot target annotation without Dynamo target metadata",
				mutate: func(pod *corev1.Pod) {
					delete(pod.Annotations, consts.RestoreCandidateTargetContainersAnnotation)
					pod.Annotations[snapshotprotocol.TargetContainersAnnotation] = "engine-0,engine-1"
				},
				wantErr: "missing required nvidia.com/dynamo-restore-target-containers annotation",
			},
		}

		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				t.Log("Given a native restore candidate that cannot be proven safe")
				pod := nativeRestoreCandidatePod(snapshot)
				test.mutate(pod)
				req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
					Operation: admissionv1.Create,
					Namespace: pod.Namespace,
					Object:    runtime.RawExtension{Raw: mustMarshalPod(t, pod)},
				}}

				t.Log("When admission repeats native snapshot validation")
				resp := mutator.Handle(ctx, req)

				t.Log("Then the Pod is denied instead of cold-starting unshaped")
				assert.False(t, resp.Allowed)
				require.NotNil(t, resp.Result)
				assert.Contains(t, resp.Result.Message, test.wantErr)
			})
		}
	})
}

func TestUsesSupportedDynamoRestoreEntrypoint(t *testing.T) {
	tests := []struct {
		name      string
		command   []string
		args      []string
		supported bool
	}{
		{
			name:      "vLLM module in command",
			command:   []string{"python3", "-m", "dynamo.vllm"},
			supported: true,
		},
		{
			name:      "SGLang module split across command and args",
			command:   []string{"python"},
			args:      []string{"-m", "dynamo.sglang", "--model", "test"},
			supported: true,
		},
		{
			name:      "TensorRT-LLM module with versioned Python path",
			command:   []string{"/usr/bin/python3.11", "-m", "dynamo.trtllm"},
			supported: true,
		},
		{
			name:      "vLLM module after operand-free interpreter flags",
			command:   []string{"python3", "-u", "-O", "-m", "dynamo.vllm"},
			supported: true,
		},
		{
			name:    "shell wrapper",
			command: []string{"/bin/sh", "-c"},
			args:    []string{"python3 -m dynamo.vllm"},
		},
		{
			name:    "custom wrapper with module arguments",
			command: []string{"serve-model"},
			args:    []string{"-m", "dynamo.vllm"},
		},
		{
			name:    "unsupported Dynamo module",
			command: []string{"python3", "-m", "dynamo.frontend"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Given a restore destination with an explicit container entrypoint")
			container := &corev1.Container{Command: test.command, Args: test.args}

			t.Log("Then only a direct invocation of a standby-aware engine is accepted")
			assert.Equal(t, test.supported, usesSupportedDynamoRestoreEntrypoint(container))
		})
	}
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

func nativeRestoreTestSnapshot() *snapshotv1alpha1.PodSnapshot {
	return &snapshotv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-snapshot",
			Namespace: "default",
			UID:       types.UID("snapshot-uid"),
			Annotations: map[string]string{
				consts.SnapshotCompatibilityVersionAnnotation: consts.SnapshotCompatibilityVersion,
				consts.SnapshotWorkerHashAnnotation:           "worker-v1",
				consts.SnapshotGMSModeAnnotation:              consts.SnapshotGMSModeDisabled,
			},
		},
		Spec: snapshotv1alpha1.PodSnapshotSpec{
			Source: snapshotv1alpha1.PodSnapshotSource{
				PodRef: snapshotv1alpha1.PodReference{
					Name:       "capture-worker",
					Containers: []string{"main"},
				},
			},
		},
		Status: snapshotv1alpha1.PodSnapshotStatus{
			BoundPodSnapshotContentName: ptr.To("content-a"),
			Conditions: []metav1.Condition{{
				Type:   snapshotv1alpha1.PodSnapshotConditionReady,
				Status: metav1.ConditionTrue,
			}},
		},
	}
}

func nativeRestoreCandidatePod(snapshot *snapshotv1alpha1.PodSnapshot) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-0",
			Namespace: snapshot.Namespace,
			Labels: map[string]string{
				consts.KubeLabelDynamoComponent:  "worker",
				consts.KubeLabelDynamoNamespace:  "default-worker",
				consts.KubeLabelDynamoSelector:   "worker",
				consts.KubeLabelDynamoWorkerHash: "worker-v1",
			},
			Annotations: map[string]string{
				consts.CheckpointRestoreCandidateAnnotation:       consts.KubeLabelValueTrue,
				consts.CheckpointNameAnnotation:                   snapshot.Name,
				consts.CheckpointSourceKindAnnotation:             consts.CheckpointSourceKindSnapshot,
				consts.SnapshotCandidateUIDAnnotation:             string(snapshot.UID),
				consts.SnapshotCandidateContentAnnotation:         "content-a",
				consts.SnapshotCandidateGMSModeAnnotation:         consts.SnapshotGMSModeDisabled,
				consts.SnapshotCandidateVersionAnnotation:         consts.SnapshotCompatibilityVersion,
				consts.RestoreCandidateTargetContainersAnnotation: "engine-0,engine-1",
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "engine-0", Image: "worker:latest", Command: []string{"python3", "-m", "dynamo.vllm"}},
				{Name: "engine-1", Image: "worker:latest", Command: []string{"python3", "-m", "dynamo.vllm"}},
			},
		},
	}
}

func applyAdmissionPatches(t *testing.T, original []byte, response admission.Response) *corev1.Pod {
	t.Helper()
	rawPatch, err := json.Marshal(response.Patches)
	require.NoError(t, err)
	patch, err := jsonpatch.DecodePatch(rawPatch)
	require.NoError(t, err)
	shapedRaw, err := patch.Apply(original)
	require.NoError(t, err)
	shaped := &corev1.Pod{}
	require.NoError(t, json.Unmarshal(shapedRaw, shaped))
	return shaped
}

func mustMarshalPod(t *testing.T, pod *corev1.Pod) []byte {
	t.Helper()
	raw, err := json.Marshal(pod)
	require.NoError(t, err)
	return raw
}
