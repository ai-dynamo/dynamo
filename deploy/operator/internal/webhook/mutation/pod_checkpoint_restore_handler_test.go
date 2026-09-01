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
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func TestPodCheckpointRestoreMutatorNativeRestore(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, nvidiacomv1alpha1.AddToScheme(scheme))
	require.NoError(t, snapshotv1alpha1.AddToScheme(scheme))

	snapshot := nativeRestoreTestSnapshot()
	apiReader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(snapshot).Build()
	mutator := NewPodCheckpointRestoreMutator(
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

	t.Run("preserves workload health semantics in the restore startup gate", func(t *testing.T) {
		t.Log("Given a native restore candidate whose liveness probe defines engine startup")
		pod := nativeRestoreCandidatePod(snapshot)
		pod.Spec.Containers[0].LivenessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{Path: "/live", Port: intstr.FromString("system")},
			},
			PeriodSeconds:    5,
			TimeoutSeconds:   4,
			FailureThreshold: 1,
		}
		original := mustMarshalPod(t, pod)
		req := admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Namespace: pod.Namespace,
			Object:    runtime.RawExtension{Raw: original},
		}}

		t.Log("When the Dynamo admission webhook shapes the Pod")
		resp := mutator.Handle(ctx, req)

		t.Log("Then startup waits for the engine health endpoint before enabling liveness")
		require.True(t, resp.Allowed)
		shaped := applyAdmissionPatches(t, original, resp)
		startup := shaped.Spec.Containers[0].StartupProbe
		require.NotNil(t, startup)
		require.NotNil(t, startup.HTTPGet)
		assert.Equal(t, "/live", startup.HTTPGet.Path)
		assert.Equal(t, int32(1), startup.PeriodSeconds)
		assert.Equal(t, int32(1800), startup.FailureThreshold)
		assert.Equal(t, int32(1), startup.SuccessThreshold)
		assert.Equal(t, pod.Spec.Containers[0].LivenessProbe, shaped.Spec.Containers[0].LivenessProbe)
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
				name: "missing restore target metadata",
				mutate: func(pod *corev1.Pod) {
					delete(pod.Annotations, consts.RestoreCandidateTargetContainersAnnotation)
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
