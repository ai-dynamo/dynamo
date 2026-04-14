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
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const testNamespace = "default"
const friendlyCheckpointName = "friendly-checkpoint"

var checkpointTestIdentity = nvidiacomv1alpha1.DynamoCheckpointIdentity{
	Model:            "meta-llama/Llama-2-7b-hf",
	BackendFramework: "vllm",
}

var testHash = func() string {
	hash, err := checkpoint.ComputeIdentityHash(checkpointTestIdentity)
	if err != nil {
		panic(err)
	}
	return hash
}()

var defaultCheckpointJobName = snapshotprotocol.GetCheckpointJobName(testHash, snapshotprotocol.DefaultCheckpointArtifactVersion)

func checkpointTestScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	_ = nvidiacomv1alpha1.AddToScheme(s)
	_ = corev1.AddToScheme(s)
	_ = batchv1.AddToScheme(s)
	_ = coordinationv1.AddToScheme(s)
	_ = rbacv1.AddToScheme(s)
	return s
}

func checkpointTestConfig() *configv1alpha1.OperatorConfiguration {
	return &configv1alpha1.OperatorConfiguration{
		Checkpoint: configv1alpha1.CheckpointConfiguration{
			Enabled:                    true,
			ReadyForCheckpointFilePath: "/tmp/ready-for-checkpoint",
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

func makeTestCheckpoint(phase nvidiacomv1alpha1.DynamoCheckpointPhase) *nvidiacomv1alpha1.DynamoCheckpoint {
	runAsUser := int64(1234)
	fsGroup := int64(4321)
	return &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: testHash, Namespace: testNamespace},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Identity: checkpointTestIdentity,
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						SecurityContext: &corev1.PodSecurityContext{
							RunAsUser: &runAsUser,
							FSGroup:   &fsGroup,
						},
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

func makeCheckpointLease(name string, renewTime time.Time, durationSeconds int32) *coordinationv1.Lease {
	renewMicroTime := metav1.NewMicroTime(renewTime)
	return &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: testNamespace},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       ptr.To("snapshot-agent/test"),
			LeaseDurationSeconds: &durationSeconds,
			AcquireTime:          &renewMicroTime,
			RenewTime:            &renewMicroTime,
		},
	}
}

func TestBuildCheckpointJob(t *testing.T) {
	s := checkpointTestScheme()
	ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhasePending)
	ckpt.Spec.Job.PodTemplateSpec.Labels = map[string]string{
		consts.KubeLabelDynamoNamespace:  "manual-checkpoint",
		consts.KubeLabelDynamoWorkerHash: "worker-1234",
	}

	r := makeCheckpointReconciler(s, ckpt)
	job, err := buildCheckpointJob(r.Config, ckpt, defaultCheckpointJobName)
	require.NoError(t, err)
	podSpec := job.Spec.Template.Spec
	main := podSpec.Containers[0]

	// Job and pod template labels
	assert.Equal(t, testHash, job.Labels[snapshotprotocol.CheckpointIDLabel])
	assert.Equal(t, "true", job.Spec.Template.Labels[snapshotprotocol.CheckpointSourceLabel])
	assert.Equal(t, testHash, job.Spec.Template.Labels[snapshotprotocol.CheckpointIDLabel])

	// Env vars (checkpoint-specific + user-provided preserved)
	envMap := make(map[string]string, len(main.Env))
	for _, e := range main.Env {
		envMap[e.Name] = e.Value
	}
	assert.Equal(t, "/tmp/ready-for-checkpoint", envMap[consts.EnvReadyForCheckpointFile])
	assert.Equal(t, "manual-checkpoint", envMap[consts.DynamoNamespaceEnvVar])
	assert.Equal(t, consts.ComponentTypeWorker, envMap[consts.DynamoComponentEnvVar])
	assert.Equal(t, "worker-1234", envMap[consts.DynamoNamespaceWorkerSuffixEnvVar])
	assert.Equal(t, "kubernetes", envMap[consts.DynamoDiscoveryBackendEnvVar])
	assert.Equal(t, "9090", envMap["DYN_SYSTEM_PORT"])
	assert.Equal(t, "true", envMap["DYN_SYSTEM_ENABLED"])
	assert.Equal(t, "secret", envMap["HF_TOKEN"])

	var podNameEnv *corev1.EnvVar
	for i := range main.Env {
		if main.Env[i].Name == "POD_NAME" {
			podNameEnv = &main.Env[i]
			break
		}
	}
	require.NotNil(t, podNameEnv)
	require.NotNil(t, podNameEnv.ValueFrom)
	require.NotNil(t, podNameEnv.ValueFrom.FieldRef)
	assert.Equal(t, "metadata.name", podNameEnv.ValueFrom.FieldRef.FieldPath)

	// Seccomp profile
	require.NotNil(t, podSpec.SecurityContext)
	require.NotNil(t, podSpec.SecurityContext.SeccompProfile)
	assert.Equal(t, corev1.SeccompProfileTypeLocalhost, podSpec.SecurityContext.SeccompProfile.Type)
	assert.Equal(t, snapshotprotocol.DefaultSeccompLocalhostProfile, *podSpec.SecurityContext.SeccompProfile.LocalhostProfile)
	require.NotNil(t, podSpec.SecurityContext.RunAsUser)
	assert.Equal(t, int64(1234), *podSpec.SecurityContext.RunAsUser)
	require.NotNil(t, podSpec.SecurityContext.FSGroup)
	assert.Equal(t, int64(4321), *podSpec.SecurityContext.FSGroup)

	// Probes: readiness set, liveness/startup cleared
	require.NotNil(t, main.ReadinessProbe)
	assert.Equal(t, []string{"cat", "/tmp/ready-for-checkpoint"}, main.ReadinessProbe.Exec.Command)
	assert.Nil(t, main.LivenessProbe)
	assert.Nil(t, main.StartupProbe)

	// Checkpoint jobs still mount podinfo for Kubernetes discovery, but not checkpoint storage.
	volNames := make(map[string]bool)
	for _, v := range podSpec.Volumes {
		volNames[v.Name] = true
	}
	assert.False(t, volNames[snapshotprotocol.CheckpointVolumeName])
	assert.True(t, volNames[consts.PodInfoVolumeName])

	mountPaths := make(map[string]string)
	for _, m := range main.VolumeMounts {
		mountPaths[m.Name] = m.MountPath
	}
	_, hasCheckpointMount := mountPaths[snapshotprotocol.CheckpointVolumeName]
	assert.False(t, hasCheckpointMount)
	assert.Equal(t, consts.PodInfoMountPath, mountPaths[consts.PodInfoVolumeName])
	assert.Equal(t, consts.DefaultSharedMemoryMountPath, mountPaths[consts.KubeValueNameSharedMemory])

	foundSharedMemoryVolume := false
	for _, v := range podSpec.Volumes {
		if v.Name != consts.KubeValueNameSharedMemory {
			continue
		}
		foundSharedMemoryVolume = true
		require.NotNil(t, v.EmptyDir)
		assert.Equal(t, corev1.StorageMediumMemory, v.EmptyDir.Medium)
		require.NotNil(t, v.EmptyDir.SizeLimit)
		assert.Equal(t, resource.MustParse(consts.DefaultSharedMemorySize), *v.EmptyDir.SizeLimit)
	}
	require.True(t, foundSharedMemoryVolume, "shared-memory volume not found: "+consts.KubeValueNameSharedMemory)

	// Restart policy, user image/command preserved
	assert.Equal(t, corev1.RestartPolicyNever, podSpec.RestartPolicy)
	assert.Equal(t, "test-image:latest", main.Image)
	assert.Equal(t, []string{"python3", "-m", "dynamo.vllm"}, main.Command)

	// Default deadlines
	assert.Equal(t, int64(3600), *job.Spec.ActiveDeadlineSeconds)
	assert.Equal(t, int32(0), *job.Spec.BackoffLimit)
	assert.Equal(t, int32(300), *job.Spec.TTLSecondsAfterFinished)

	// Custom active deadlines override defaults, but checkpoint jobs never retry and keep a fixed TTL.
	deadline := int64(7200)
	backoff := int32(5)
	ckpt.Spec.Job.ActiveDeadlineSeconds = &deadline
	ckpt.Spec.Job.BackoffLimit = &backoff //nolint:staticcheck // Compatibility test: deprecated field must remain ignored by checkpoint Jobs.
	job, err = buildCheckpointJob(r.Config, ckpt, defaultCheckpointJobName)
	require.NoError(t, err)
	assert.Equal(t, int64(7200), *job.Spec.ActiveDeadlineSeconds)
	assert.Equal(t, int32(0), *job.Spec.BackoffLimit)
	assert.Equal(t, int32(300), *job.Spec.TTLSecondsAfterFinished)

	ckpt.Spec.Job.PodTemplateSpec.Spec.Containers[0].Resources = corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("2"),
		},
	}
	job, err = buildCheckpointJob(r.Config, ckpt, defaultCheckpointJobName)
	require.NoError(t, err)
	assert.Equal(t, []string{"cuda-checkpoint"}, job.Spec.Template.Spec.Containers[0].Command)
	assert.Equal(t, []string{"--launch-job", "python3", "-m", "dynamo.vllm"}, job.Spec.Template.Spec.Containers[0].Args)
}

func TestBuildCheckpointJobInjectsStandardEnvVars(t *testing.T) {
	s := checkpointTestScheme()
	ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhasePending)
	ckpt.Spec.Job.PodTemplateSpec.Spec.Containers[0].Env = append(
		ckpt.Spec.Job.PodTemplateSpec.Spec.Containers[0].Env,
		corev1.EnvVar{Name: "NATS_SERVER", Value: "nats://custom:4222"},
		corev1.EnvVar{Name: "DYN_SYSTEM_PORT", Value: "10090"},
	)

	r := makeCheckpointReconciler(s, ckpt)
	r.Config.Infrastructure = configv1alpha1.InfrastructureConfiguration{
		NATSAddress:        "nats://platform:4222",
		ETCDAddress:        "http://etcd:2379",
		ModelExpressURL:    "http://model-express:8000",
		PrometheusEndpoint: "http://prometheus:9090",
	}

	customShmSize := resource.MustParse("16Gi")
	ckpt.Spec.Job.SharedMemory = &nvidiacomv1alpha1.SharedMemorySpec{Size: customShmSize}
	job, err := buildCheckpointJob(r.Config, ckpt, defaultCheckpointJobName)
	require.NoError(t, err)
	foundCustomShmVolume := false
	for _, v := range job.Spec.Template.Spec.Volumes {
		if v.Name == consts.KubeValueNameSharedMemory {
			foundCustomShmVolume = true
			require.NotNil(t, v.EmptyDir)
			require.NotNil(t, v.EmptyDir.SizeLimit)
			assert.Equal(t, customShmSize, *v.EmptyDir.SizeLimit)
		}
	}
	require.True(t, foundCustomShmVolume, "shared-memory volume not found: "+consts.KubeValueNameSharedMemory)
	main := job.Spec.Template.Spec.Containers[0]

	envMap := make(map[string]string, len(main.Env))
	for _, e := range main.Env {
		envMap[e.Name] = e.Value
	}

	assert.Equal(t, "nats://custom:4222", envMap["NATS_SERVER"])
	assert.Equal(t, "10090", envMap["DYN_SYSTEM_PORT"])
	assert.Equal(t, "http://etcd:2379", envMap["ETCD_ENDPOINTS"])
	assert.Equal(t, "http://model-express:8000", envMap["MODEL_EXPRESS_URL"])
	assert.Equal(t, "http://prometheus:9090", envMap["PROMETHEUS_ENDPOINT"])
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
		ckpt := makeTestCheckpoint("")
		r := makeCheckpointReconciler(s, ckpt)

		_, err := r.Reconcile(ctx, ctrl.Request{
			NamespacedName: types.NamespacedName{Name: testHash, Namespace: testNamespace},
		})
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhasePending, updated.Status.Phase)
		assert.Equal(t, testHash, updated.Status.IdentityHash)
		assert.Empty(t, updated.Status.Message)
		assert.Equal(t, testHash, updated.Labels[snapshotprotocol.CheckpointIDLabel])
	})

	t.Run("Ready phase is a no-op", func(t *testing.T) {
		ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhaseReady)
		r := makeCheckpointReconciler(s, ckpt)

		result, err := r.Reconcile(ctx, ctrl.Request{
			NamespacedName: types.NamespacedName{Name: ckpt.Name, Namespace: testNamespace},
		})
		require.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})

	t.Run("human-readable checkpoint name backfills hash state", func(t *testing.T) {
		ckpt := makeTestCheckpoint("")
		ckpt.Name = friendlyCheckpointName
		r := makeCheckpointReconciler(s, ckpt)

		_, err := r.Reconcile(ctx, ctrl.Request{
			NamespacedName: types.NamespacedName{Name: friendlyCheckpointName, Namespace: testNamespace},
		})
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: friendlyCheckpointName, Namespace: testNamespace}, updated))
		assert.Equal(t, testHash, updated.Labels[snapshotprotocol.CheckpointIDLabel])
		assert.Equal(t, testHash, updated.Status.IdentityHash)
	})

	t.Run("unknown phase resets to Pending", func(t *testing.T) {
		ckpt := makeTestCheckpoint("SomeUnknownPhase")
		r := makeCheckpointReconciler(s, ckpt)

		_, err := r.Reconcile(ctx, ctrl.Request{
			NamespacedName: types.NamespacedName{Name: testHash, Namespace: testNamespace},
		})
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, updated.Status.Phase)
		assert.Equal(t, defaultCheckpointJobName, updated.Status.JobName)
	})

	t.Run("artifact version bump starts a new checkpoint job", func(t *testing.T) {
		ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhaseReady)
		ckpt.Status.IdentityHash = testHash
		ckpt.Status.JobName = defaultCheckpointJobName
		ckpt.Annotations = map[string]string{snapshotprotocol.CheckpointArtifactVersionAnnotation: "2"}
		r := makeCheckpointReconciler(s, ckpt)

		_, err := r.Reconcile(ctx, ctrl.Request{
			NamespacedName: types.NamespacedName{Name: ckpt.Name, Namespace: testNamespace},
		})
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: ckpt.Name, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, updated.Status.Phase)
		assert.Equal(t, "checkpoint-job-"+testHash+"-2", updated.Status.JobName)
	})

	t.Run("duplicate identity hash is rejected even with a readable name", func(t *testing.T) {
		primary := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhaseReady)
		primary.Name = "friendly-primary"
		primary.Status.IdentityHash = testHash
		primary.Status.JobName = defaultCheckpointJobName
		duplicate := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhaseReady)
		duplicate.Name = "friendly-duplicate"
		duplicate.Status.IdentityHash = testHash
		duplicate.Status.JobName = "checkpoint-job-" + testHash + "-2"

		r := makeCheckpointReconciler(s, primary, duplicate)
		_, err := r.Reconcile(ctx, ctrl.Request{
			NamespacedName: types.NamespacedName{Name: duplicate.Name, Namespace: testNamespace},
		})
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: duplicate.Name, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
		assert.Contains(t, updated.Status.Message, primary.Name)
	})
}

func TestCheckpointReconciler_HandleCreating(t *testing.T) {
	s := checkpointTestScheme()
	ctx := context.Background()

	// Helper to create a checkpoint CR in Creating phase with a named job
	makeCreatingCkpt := func(name, jobName string) *nvidiacomv1alpha1.DynamoCheckpoint {
		ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhaseCreating)
		if name != "" {
			ckpt.Name = name
		}
		ckpt.Status.IdentityHash = testHash
		ckpt.Status.JobName = jobName
		return ckpt
	}

	t.Run("succeeded job transitions to Ready", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name:        defaultCheckpointJobName,
				Namespace:   testNamespace,
				Annotations: map[string]string{snapshotprotocol.CheckpointStatusAnnotation: snapshotprotocol.CheckpointStatusCompleted},
			},
			Status: batchv1.JobStatus{
				Succeeded: 1,
				Conditions: []batchv1.JobCondition{
					{Type: batchv1.JobComplete, Status: corev1.ConditionTrue, LastTransitionTime: metav1.Now()},
				},
			},
		}

		r := makeCheckpointReconciler(s, ckpt, job)
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseReady, updated.Status.Phase)
		assert.NotNil(t, updated.Status.CreatedAt)
	})

	t.Run("failed job transitions to Failed", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, "job-fail")
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
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
	})

	t.Run("completed job without completion annotation waits while lease is active", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, "job-missing-status-active-lease")
		completionTime := metav1.NewTime(time.Now())
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{Name: "job-missing-status-active-lease", Namespace: testNamespace},
			Status: batchv1.JobStatus{
				Succeeded:      1,
				CompletionTime: &completionTime,
				Conditions: []batchv1.JobCondition{
					{Type: batchv1.JobComplete, Status: corev1.ConditionTrue, LastTransitionTime: completionTime},
				},
			},
		}
		lease := makeCheckpointLease("job-missing-status-active-lease", time.Now(), 30)

		r := makeCheckpointReconciler(s, ckpt, job, lease)
		result, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)
		assert.Equal(t, time.Second, result.RequeueAfter)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, updated.Status.Phase)
	})

	t.Run("completed job without completion annotation transitions to Failed once lease expires", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, "job-missing-status")
		completionTime := metav1.NewTime(time.Now().Add(-time.Minute))
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{Name: "job-missing-status", Namespace: testNamespace},
			Status: batchv1.JobStatus{
				Succeeded:      1,
				CompletionTime: &completionTime,
				Conditions: []batchv1.JobCondition{
					{Type: batchv1.JobComplete, Status: corev1.ConditionTrue, LastTransitionTime: completionTime},
				},
			},
		}
		r := makeCheckpointReconciler(s, ckpt, job)
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
		assert.Contains(t, updated.Status.Message, "without snapshot-agent completion confirmation")
	})

	t.Run("completed job with failed completion annotation transitions to Failed", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, "job-agent-failed")
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "job-agent-failed",
				Namespace:   testNamespace,
				Annotations: map[string]string{snapshotprotocol.CheckpointStatusAnnotation: snapshotprotocol.CheckpointStatusFailed},
			},
			Status: batchv1.JobStatus{
				Succeeded: 1,
				Conditions: []batchv1.JobCondition{
					{Type: batchv1.JobComplete, Status: corev1.ConditionTrue, LastTransitionTime: metav1.Now()},
				},
			},
		}

		r := makeCheckpointReconciler(s, ckpt, job)
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
		assert.Contains(t, updated.Status.Message, "snapshot-agent reported checkpoint failure")
	})

	t.Run("running job with failed checkpoint annotation transitions to Failed", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, "job-running-agent-failed")
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "job-running-agent-failed",
				Namespace:   testNamespace,
				Annotations: map[string]string{snapshotprotocol.CheckpointStatusAnnotation: snapshotprotocol.CheckpointStatusFailed},
			},
			Status: batchv1.JobStatus{Active: 1},
		}

		r := makeCheckpointReconciler(s, ckpt, job)
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
		assert.Equal(t, "Checkpoint job failed", updated.Status.Message)
	})

	t.Run("running job keeps Creating phase", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, "job-run")
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{Name: "job-run", Namespace: testNamespace},
			Status:     batchv1.JobStatus{Active: 1},
		}

		r := makeCheckpointReconciler(s, ckpt, job)
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, updated.Status.Phase)
	})

	t.Run("in-flight version changes do not relabel the running job's artifact", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		ckpt.Annotations = map[string]string{snapshotprotocol.CheckpointArtifactVersionAnnotation: "2"}
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name:        defaultCheckpointJobName,
				Namespace:   testNamespace,
				Annotations: map[string]string{snapshotprotocol.CheckpointStatusAnnotation: snapshotprotocol.CheckpointStatusCompleted},
			},
			Status: batchv1.JobStatus{
				Succeeded: 1,
				Conditions: []batchv1.JobCondition{
					{Type: batchv1.JobComplete, Status: corev1.ConditionTrue, LastTransitionTime: metav1.Now()},
				},
			},
		}

		r := makeCheckpointReconciler(s, ckpt, job)
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseReady, updated.Status.Phase)
	})

	t.Run("succeeded count without complete condition keeps Creating phase", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, "job-succeeded-not-complete")
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{Name: "job-succeeded-not-complete", Namespace: testNamespace},
			Status:     batchv1.JobStatus{Succeeded: 1},
		}

		r := makeCheckpointReconciler(s, ckpt, job)
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, updated.Status.Phase)
	})

	t.Run("deleted job transitions to Failed without retrying", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, "job-deleted")
		r := makeCheckpointReconciler(s, ckpt) // no job object

		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
		assert.Equal(t, "job-deleted", updated.Status.JobName)
		assert.Equal(t, "checkpoint job was deleted", updated.Status.Message)
	})

}
