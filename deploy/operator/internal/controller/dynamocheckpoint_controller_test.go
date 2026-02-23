/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controller_common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testHash      = "abc123def4567890"
	testNamespace = "default"
)

func checkpointTestScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	_ = nvidiacomv1alpha1.AddToScheme(s)
	_ = corev1.AddToScheme(s)
	_ = batchv1.AddToScheme(s)
	return s
}

func checkpointTestConfig() controller_common.Config {
	return controller_common.Config{
		Checkpoint: controller_common.CheckpointConfig{
			Enabled:                    true,
			ReadyForCheckpointFilePath: "/tmp/ready-for-checkpoint",
			Storage: controller_common.CheckpointStorageConfig{
				Type: controller_common.CheckpointStorageTypePVC,
				PVC: controller_common.CheckpointPVCConfig{
					PVCName:  "chrek-pvc",
					BasePath: "/checkpoints",
				},
			},
		},
	}
}

func makeCheckpointReconciler(s *runtime.Scheme, objs ...client.Object) *CheckpointReconciler {
	return &CheckpointReconciler{
		Client:   fake.NewClientBuilder().WithScheme(s).WithObjects(objs...).WithStatusSubresource(&nvidiacomv1alpha1.DynamoCheckpoint{}).Build(),
		Config:   checkpointTestConfig(),
		Recorder: record.NewFakeRecorder(10),
	}
}

func makeTestCheckpoint(name string, phase nvidiacomv1alpha1.DynamoCheckpointPhase) *nvidiacomv1alpha1.DynamoCheckpoint {
	return &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: testNamespace},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "meta-llama/Llama-2-7b-hf",
				BackendFramework: "vllm",
			},
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:    "main",
							Image:   "test-image:latest",
							Command: []string{"python3", "-m", "dynamo.vllm"},
							Env:     []corev1.EnvVar{{Name: "HF_TOKEN", Value: "secret"}},
						}},
					},
				},
			},
		},
		Status: nvidiacomv1alpha1.DynamoCheckpointStatus{Phase: phase},
	}
}

func TestBuildCheckpointJob(t *testing.T) {
	s := checkpointTestScheme()
	ckpt := makeTestCheckpoint("test-ckpt", nvidiacomv1alpha1.DynamoCheckpointPhasePending)
	ckpt.Status.IdentityHash = testHash

	r := makeCheckpointReconciler(s, ckpt)
	job := r.buildCheckpointJob(ckpt, "checkpoint-test-ckpt")
	podSpec := job.Spec.Template.Spec
	main := podSpec.Containers[0]

	// Job and pod template labels
	assert.Equal(t, testHash, job.Labels[consts.KubeLabelCheckpointHash])
	assert.Equal(t, "true", job.Spec.Template.Labels[consts.KubeLabelIsCheckpointSource])
	assert.Equal(t, testHash, job.Spec.Template.Labels[consts.KubeLabelCheckpointHash])

	// Env vars (checkpoint-specific + user-provided preserved)
	envMap := make(map[string]string, len(main.Env))
	for _, e := range main.Env {
		envMap[e.Name] = e.Value
	}
	assert.Equal(t, "/tmp/ready-for-checkpoint", envMap[consts.EnvReadyForCheckpointFile])
	assert.Equal(t, testHash, envMap[consts.EnvCheckpointHash])
	assert.Equal(t, "/checkpoints/"+testHash, envMap[consts.EnvCheckpointLocation])
	assert.Equal(t, "pvc", envMap[consts.EnvCheckpointStorageType])
	assert.Equal(t, "secret", envMap["HF_TOKEN"])

	// Seccomp profile
	require.NotNil(t, podSpec.SecurityContext)
	require.NotNil(t, podSpec.SecurityContext.SeccompProfile)
	assert.Equal(t, corev1.SeccompProfileTypeLocalhost, podSpec.SecurityContext.SeccompProfile.Type)
	assert.Equal(t, consts.SeccompProfilePath, *podSpec.SecurityContext.SeccompProfile.LocalhostProfile)

	// Probes: readiness set, liveness/startup cleared
	require.NotNil(t, main.ReadinessProbe)
	assert.Equal(t, []string{"cat", "/tmp/ready-for-checkpoint"}, main.ReadinessProbe.Exec.Command)
	assert.Nil(t, main.LivenessProbe)
	assert.Nil(t, main.StartupProbe)

	// Checkpoint PVC volume + mount
	volNames := make(map[string]bool)
	for _, v := range podSpec.Volumes {
		volNames[v.Name] = true
		if v.Name == consts.CheckpointVolumeName {
			require.NotNil(t, v.PersistentVolumeClaim)
			assert.Equal(t, "chrek-pvc", v.PersistentVolumeClaim.ClaimName)
		}
		if v.Name == consts.PodInfoVolumeName {
			require.NotNil(t, v.DownwardAPI)
		}
	}
	assert.True(t, volNames[consts.CheckpointVolumeName])
	assert.True(t, volNames[consts.PodInfoVolumeName])

	mountPaths := make(map[string]string)
	for _, m := range main.VolumeMounts {
		mountPaths[m.Name] = m.MountPath
	}
	assert.Equal(t, "/checkpoints", mountPaths[consts.CheckpointVolumeName])
	assert.Equal(t, consts.PodInfoMountPath, mountPaths[consts.PodInfoVolumeName])

	// Restart policy, user image/command preserved
	assert.Equal(t, corev1.RestartPolicyNever, podSpec.RestartPolicy)
	assert.Equal(t, "test-image:latest", main.Image)
	assert.Equal(t, []string{"python3", "-m", "dynamo.vllm"}, main.Command)

	// Default deadlines
	assert.Equal(t, int64(3600), *job.Spec.ActiveDeadlineSeconds)
	assert.Equal(t, int32(3), *job.Spec.BackoffLimit)
	assert.Equal(t, int32(300), *job.Spec.TTLSecondsAfterFinished)

	// Custom deadlines override defaults
	deadline := int64(7200)
	backoff := int32(5)
	ttl := int32(600)
	ckpt.Spec.Job.ActiveDeadlineSeconds = &deadline
	ckpt.Spec.Job.BackoffLimit = &backoff
	ckpt.Spec.Job.TTLSecondsAfterFinished = &ttl
	job = r.buildCheckpointJob(ckpt, "checkpoint-test-ckpt")
	assert.Equal(t, int64(7200), *job.Spec.ActiveDeadlineSeconds)
	assert.Equal(t, int32(5), *job.Spec.BackoffLimit)
	assert.Equal(t, int32(600), *job.Spec.TTLSecondsAfterFinished)
}

func TestCheckpointReconciler_Reconcile(t *testing.T) {
	s := checkpointTestScheme()
	ctx := context.Background()

	t.Run("not found returns no error", func(t *testing.T) {
		r := makeCheckpointReconciler(s)
		result, err := r.Reconcile(ctx, ctrl.Request{
			NamespacedName: types.NamespacedName{Name: "nonexistent", Namespace: testNamespace},
		})
		require.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})

	t.Run("new CR computes hash and sets Pending", func(t *testing.T) {
		ckpt := makeTestCheckpoint("new-ckpt", "")
		r := makeCheckpointReconciler(s, ckpt)

		_, err := r.Reconcile(ctx, ctrl.Request{
			NamespacedName: types.NamespacedName{Name: "new-ckpt", Namespace: testNamespace},
		})
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: "new-ckpt", Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhasePending, updated.Status.Phase)
		assert.Len(t, updated.Status.IdentityHash, 16)
	})

	t.Run("Ready phase is a no-op", func(t *testing.T) {
		ckpt := makeTestCheckpoint("ready-ckpt", nvidiacomv1alpha1.DynamoCheckpointPhaseReady)
		ckpt.Status.IdentityHash = testHash
		r := makeCheckpointReconciler(s, ckpt)

		result, err := r.Reconcile(ctx, ctrl.Request{
			NamespacedName: types.NamespacedName{Name: "ready-ckpt", Namespace: testNamespace},
		})
		require.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})

	t.Run("unknown phase resets to Pending", func(t *testing.T) {
		ckpt := makeTestCheckpoint("unknown-ckpt", "SomeUnknownPhase")
		ckpt.Status.IdentityHash = testHash
		r := makeCheckpointReconciler(s, ckpt)

		_, err := r.Reconcile(ctx, ctrl.Request{
			NamespacedName: types.NamespacedName{Name: "unknown-ckpt", Namespace: testNamespace},
		})
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: "unknown-ckpt", Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhasePending, updated.Status.Phase)
	})
}

func TestCheckpointReconciler_HandleCreating(t *testing.T) {
	s := checkpointTestScheme()
	ctx := context.Background()

	// Helper to create a checkpoint CR in Creating phase with a named job
	makeCreatingCkpt := func(name, jobName string) *nvidiacomv1alpha1.DynamoCheckpoint {
		ckpt := makeTestCheckpoint(name, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating)
		ckpt.Status.IdentityHash = testHash
		ckpt.Status.JobName = jobName
		return ckpt
	}

	t.Run("succeeded job transitions to Ready", func(t *testing.T) {
		ckpt := makeCreatingCkpt("ckpt-ok", "job-ok")
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{Name: "job-ok", Namespace: testNamespace},
			Status:     batchv1.JobStatus{Succeeded: 1},
		}

		r := makeCheckpointReconciler(s, ckpt, job)
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: "ckpt-ok", Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseReady, updated.Status.Phase)
		assert.Equal(t, "/checkpoints/"+testHash, updated.Status.Location)
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointStorageType("pvc"), updated.Status.StorageType)
		assert.NotNil(t, updated.Status.CreatedAt)
	})

	t.Run("failed job transitions to Failed", func(t *testing.T) {
		ckpt := makeCreatingCkpt("ckpt-fail", "job-fail")
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{Name: "job-fail", Namespace: testNamespace},
			Status: batchv1.JobStatus{
				Conditions: []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue}},
			},
		}

		r := makeCheckpointReconciler(s, ckpt, job)
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: "ckpt-fail", Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
	})

	t.Run("running job keeps Creating phase", func(t *testing.T) {
		ckpt := makeCreatingCkpt("ckpt-run", "job-run")
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{Name: "job-run", Namespace: testNamespace},
			Status:     batchv1.JobStatus{Active: 1},
		}

		r := makeCheckpointReconciler(s, ckpt, job)
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: "ckpt-run", Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, updated.Status.Phase)
	})

	t.Run("deleted job resets to Pending", func(t *testing.T) {
		ckpt := makeCreatingCkpt("ckpt-del", "job-deleted")
		r := makeCheckpointReconciler(s, ckpt) // no job object

		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: "ckpt-del", Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhasePending, updated.Status.Phase)
		assert.Empty(t, updated.Status.JobName)
	})
}
