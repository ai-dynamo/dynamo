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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
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
	_ = resourcev1.AddToScheme(s)
	return s
}

func checkpointTestConfig() *configv1alpha1.OperatorConfiguration {
	return &configv1alpha1.OperatorConfiguration{
		Checkpoint: configv1alpha1.CheckpointConfiguration{
			Enabled: true,
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

func requireCheckpointContainer(t *testing.T, containers []corev1.Container, name string) *corev1.Container {
	t.Helper()
	if container := findCheckpointContainer(containers, name); container != nil {
		return container
	}
	t.Fatalf("container %q not found", name)
	return nil
}

func findCheckpointContainer(containers []corev1.Container, name string) *corev1.Container {
	for i := range containers {
		if containers[i].Name == name {
			return &containers[i]
		}
	}
	return nil
}

func TestBuildCheckpointJob(t *testing.T) {
	s := checkpointTestScheme()
	ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhasePending)
	ckpt.Spec.Job.PodTemplateSpec.Labels = map[string]string{
		consts.KubeLabelDynamoNamespace:  "manual-checkpoint",
		consts.KubeLabelDynamoWorkerHash: "worker-1234",
	}

	r := makeCheckpointReconciler(s, ckpt)
	job, err := buildCheckpointJob(context.Background(), nil, r.Config, ckpt, defaultCheckpointJobName)
	require.NoError(t, err)
	podSpec := job.Spec.Template.Spec
	main := podSpec.Containers[0]

	// Job and pod template labels
	assert.Equal(t, testHash, job.Labels[snapshotprotocol.CheckpointIDLabel])
	assert.Equal(t, "true", job.Spec.Template.Labels[snapshotprotocol.CheckpointSourceLabel])
	assert.Equal(t, testHash, job.Spec.Template.Labels[snapshotprotocol.CheckpointIDLabel])
	// The owner label on the Job pod template is what mapSourcePodToCheckpoint reads to drive the
	// source-pod watch (LLD §5); without it the watch never enqueues and capture hangs silently.
	assert.Equal(t, ckpt.Name, job.Spec.Template.Labels[consts.SnapshotOwnerLabel])

	// Env vars: checkpoint-specific env is added, and the caller-provided
	// workload env is preserved. Dynamo-specific runtime env is expected to be
	// present in the prepared pod template for auto-created checkpoints.
	assert.Contains(t, main.Env, corev1.EnvVar{Name: snapshotprotocol.SnapshotControlDirEnv, Value: snapshotprotocol.SnapshotControlMountPath})
	assert.Contains(t, main.Env, corev1.EnvVar{Name: "HF_TOKEN", Value: "secret"})

	var podNameEnv *corev1.EnvVar
	for i := range main.Env {
		if main.Env[i].Name == "POD_NAME" {
			podNameEnv = &main.Env[i]
			break
		}
	}

	assert.Nil(t, podNameEnv)

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
	assert.Equal(t, []string{"cat", "/snapshot-control/ready-for-snapshot"}, main.ReadinessProbe.Exec.Command)
	assert.Nil(t, main.LivenessProbe)
	assert.Nil(t, main.StartupProbe)

	// Checkpoint jobs mount snapshot-control, but not checkpoint storage.
	volNames := make(map[string]bool)
	for _, v := range podSpec.Volumes {
		volNames[v.Name] = true
	}
	assert.False(t, volNames[snapshotprotocol.CheckpointVolumeName])
	assert.True(t, volNames[snapshotprotocol.SnapshotControlVolumeName])
	assert.Empty(t, podSpec.ServiceAccountName)

	for _, mount := range main.VolumeMounts {
		assert.NotEqual(t, snapshotprotocol.CheckpointVolumeName, mount.Name)
	}
	assert.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: consts.KubeValueNameSharedMemory, MountPath: consts.DefaultSharedMemoryMountPath})
	assert.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.SnapshotControlVolumeName, MountPath: snapshotprotocol.SnapshotControlMountPath, SubPath: consts.MainContainerName})

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
	job, err = buildCheckpointJob(context.Background(), nil, r.Config, ckpt, defaultCheckpointJobName)
	require.NoError(t, err)
	assert.Equal(t, int64(7200), *job.Spec.ActiveDeadlineSeconds)
	assert.Equal(t, int32(0), *job.Spec.BackoffLimit)
	assert.Equal(t, int32(300), *job.Spec.TTLSecondsAfterFinished)

	// Deprecated identity fields no longer control checkpoint launch wrapping.
	ckpt.Spec.Identity.TensorParallelSize = 2
	job, err = buildCheckpointJob(context.Background(), nil, r.Config, ckpt, defaultCheckpointJobName)
	require.NoError(t, err)
	assert.Equal(t, []string{"python3", "-m", "dynamo.vllm"}, job.Spec.Template.Spec.Containers[0].Command)

	// Multi-GPU: wrapping decision uses target-container GPU resources.
	ckpt.Spec.Job.PodTemplateSpec.Spec.Containers[0].Resources = corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("2"),
		},
	}
	job, err = buildCheckpointJob(context.Background(), nil, r.Config, ckpt, defaultCheckpointJobName)
	require.NoError(t, err)
	assert.Equal(t, []string{"cuda-checkpoint"}, job.Spec.Template.Spec.Containers[0].Command)
	assert.Equal(t, []string{"--launch-job", "python3", "-m", "dynamo.vllm"}, job.Spec.Template.Spec.Containers[0].Args)
}

func TestBuildCheckpointJobWrapsWithCudaCheckpointForMultiGPU(t *testing.T) {
	s := checkpointTestScheme()
	ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhasePending)
	ckpt.Spec.Job.PodTemplateSpec.Spec.Containers = []corev1.Container{
		{
			Name:    consts.MainContainerName,
			Image:   "test-image:latest",
			Command: []string{"python3", "-m", "dynamo.vllm"},
			Env:     []corev1.EnvVar{{Name: "HF_TOKEN", Value: "secret"}},
			Resources: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{
					corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("2"),
				},
			},
		},
		{
			Name:    "sidecar",
			Image:   "sidecar:latest",
			Command: []string{"sleep"},
			Args:    []string{"infinity"},
		},
	}

	r := makeCheckpointReconciler(s, ckpt)
	job, err := buildCheckpointJob(context.Background(), nil, r.Config, ckpt, defaultCheckpointJobName)
	require.NoError(t, err)

	main := &job.Spec.Template.Spec.Containers[0]
	assert.Equal(t, []string{"cuda-checkpoint"}, main.Command)
	assert.Equal(t, []string{"--launch-job", "python3", "-m", "dynamo.vllm"}, main.Args)
	require.NotNil(t, main.ReadinessProbe)
	assert.Equal(t, []string{"cat", "/snapshot-control/ready-for-snapshot"}, main.ReadinessProbe.Exec.Command)
	assert.Nil(t, main.LivenessProbe)
	assert.Nil(t, main.StartupProbe)

	assert.Contains(t, main.Env, corev1.EnvVar{Name: snapshotprotocol.SnapshotControlDirEnv, Value: snapshotprotocol.SnapshotControlMountPath})
	assert.Contains(t, main.Env, corev1.EnvVar{Name: "HF_TOKEN", Value: "secret"})

	sidecar := requireCheckpointContainer(t, job.Spec.Template.Spec.Containers, "sidecar")
	assert.Equal(t, []string{"sleep"}, sidecar.Command)
	assert.Equal(t, []string{"infinity"}, sidecar.Args)
	assert.Nil(t, sidecar.ReadinessProbe)
	assert.Nil(t, sidecar.LivenessProbe)
	assert.Nil(t, sidecar.StartupProbe)
	for _, env := range sidecar.Env {
		assert.NotEqual(t, snapshotprotocol.SnapshotControlDirEnv, env.Name)
	}
}

func TestBuildCheckpointJobDRAResourceClaimsForCudaCheckpoint(t *testing.T) {
	tests := []struct {
		name          string
		resourceClaim bool
		missing       bool
		deviceClass   string
		gmsClass      string
		allocation    resourcev1.DeviceAllocationMode
		count         int64
		wantWrap      bool
		wantErr       string
	}{
		{
			name:        "resource claim template exact count",
			deviceClass: dra.DefaultDeviceClassName,
			allocation:  resourcev1.DeviceAllocationModeExactCount,
			count:       2,
			wantWrap:    true,
		},
		{
			name:          "resource claim exact count",
			resourceClaim: true,
			deviceClass:   dra.DefaultDeviceClassName,
			allocation:    resourcev1.DeviceAllocationModeExactCount,
			count:         2,
			wantWrap:      true,
		},
		{
			name:        "allocation mode all",
			deviceClass: dra.DefaultDeviceClassName,
			allocation:  resourcev1.DeviceAllocationModeAll,
			wantWrap:    true,
		},
		{
			name:        "custom configured device class",
			deviceClass: "gpu.nvidia.com/h100",
			gmsClass:    "gpu.nvidia.com/h100",
			allocation:  resourcev1.DeviceAllocationModeExactCount,
			count:       2,
			wantWrap:    true,
		},
		{
			name:        "unconfigured device class",
			deviceClass: "gpu.nvidia.com/h100",
			allocation:  resourcev1.DeviceAllocationModeExactCount,
			count:       2,
			wantWrap:    false,
		},
		{
			name:    "missing template",
			missing: true,
			wantErr: "failed to get ResourceClaimTemplate default/checkpoint-gpu for checkpoint GPU count",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := checkpointTestScheme()
			ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhasePending)
			if tt.gmsClass != "" {
				ckpt.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled:         true,
					DeviceClassName: tt.gmsClass,
				}
			}
			podClaim := corev1.PodResourceClaim{Name: "gpu"}
			if tt.resourceClaim {
				podClaim.ResourceClaimName = ptr.To("checkpoint-gpu")
			} else {
				podClaim.ResourceClaimTemplateName = ptr.To("checkpoint-gpu")
			}
			ckpt.Spec.Job.PodTemplateSpec.Spec.ResourceClaims = []corev1.PodResourceClaim{podClaim}
			ckpt.Spec.Job.PodTemplateSpec.Spec.Containers[0].Resources = corev1.ResourceRequirements{
				Claims: []corev1.ResourceClaim{{Name: "gpu"}},
			}

			objects := []client.Object{ckpt}
			if !tt.missing {
				request := resourcev1.DeviceRequest{
					Name: "gpus",
					Exactly: &resourcev1.ExactDeviceRequest{
						DeviceClassName: tt.deviceClass,
						AllocationMode:  tt.allocation,
						Count:           tt.count,
					},
				}
				claimSpec := resourcev1.ResourceClaimSpec{Devices: resourcev1.DeviceClaim{Requests: []resourcev1.DeviceRequest{request}}}
				if tt.resourceClaim {
					objects = append(objects, &resourcev1.ResourceClaim{
						ObjectMeta: metav1.ObjectMeta{Name: "checkpoint-gpu", Namespace: testNamespace},
						Spec:       claimSpec,
					})
				} else {
					objects = append(objects, &resourcev1.ResourceClaimTemplate{
						ObjectMeta: metav1.ObjectMeta{Name: "checkpoint-gpu", Namespace: testNamespace},
						Spec:       resourcev1.ResourceClaimTemplateSpec{Spec: claimSpec},
					})
				}
			}

			r := makeCheckpointReconciler(s, objects...)
			job, err := buildCheckpointJob(context.Background(), r.Client, r.Config, ckpt, defaultCheckpointJobName)
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				return
			}
			require.NoError(t, err)

			main := &job.Spec.Template.Spec.Containers[0]
			if tt.wantWrap {
				assert.Equal(t, []string{"cuda-checkpoint"}, main.Command)
				assert.Equal(t, []string{"--launch-job", "python3", "-m", "dynamo.vllm"}, main.Args)
			} else {
				assert.Equal(t, []string{"python3", "-m", "dynamo.vllm"}, main.Command)
				assert.Empty(t, main.Args)
			}
		})
	}
}

func TestBuildCheckpointJobUsesTargetContainerName(t *testing.T) {
	s := checkpointTestScheme()
	ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhasePending)
	ckpt.Spec.Job.TargetContainerName = "worker"
	ckpt.Spec.Job.PodTemplateSpec.Spec.Containers = []corev1.Container{
		{
			Name:    consts.MainContainerName,
			Image:   "main:latest",
			Command: []string{"python3", "-m", "main"},
		},
		{
			Name:    "worker",
			Image:   "worker:latest",
			Command: []string{"python3", "-m", "worker"},
			Env:     []corev1.EnvVar{{Name: "USER_ENV", Value: "1"}},
		},
	}

	r := makeCheckpointReconciler(s, ckpt)
	job, err := buildCheckpointJob(context.Background(), nil, r.Config, ckpt, defaultCheckpointJobName)
	require.NoError(t, err)

	assert.Equal(t, "worker", job.Spec.Template.Annotations[snapshotprotocol.TargetContainersAnnotation])
	main := requireCheckpointContainer(t, job.Spec.Template.Spec.Containers, consts.MainContainerName)
	target := requireCheckpointContainer(t, job.Spec.Template.Spec.Containers, "worker")

	assert.Nil(t, main.ReadinessProbe)
	for _, env := range main.Env {
		assert.NotEqual(t, snapshotprotocol.SnapshotControlDirEnv, env.Name)
	}
	for _, mount := range main.VolumeMounts {
		assert.NotEqual(t, snapshotprotocol.SnapshotControlVolumeName, mount.Name)
	}

	require.NotNil(t, target.ReadinessProbe)
	assert.Contains(t, target.Env, corev1.EnvVar{Name: snapshotprotocol.SnapshotControlDirEnv, Value: snapshotprotocol.SnapshotControlMountPath})
	assert.Contains(t, target.Env, corev1.EnvVar{Name: "USER_ENV", Value: "1"})
	assert.Contains(t, target.VolumeMounts, corev1.VolumeMount{Name: snapshotprotocol.SnapshotControlVolumeName, MountPath: snapshotprotocol.SnapshotControlMountPath, SubPath: "worker"})
}

func TestBuildCheckpointJobPreservesPreparedEnvAndSharedMemory(t *testing.T) {
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
	job, err := buildCheckpointJob(context.Background(), nil, r.Config, ckpt, defaultCheckpointJobName)
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

	assert.Contains(t, main.Env, corev1.EnvVar{Name: "NATS_SERVER", Value: "nats://custom:4222"})
	assert.Contains(t, main.Env, corev1.EnvVar{Name: "DYN_SYSTEM_PORT", Value: "10090"})
	for _, env := range main.Env {
		assert.NotEqual(t, "ETCD_ENDPOINTS", env.Name)
		assert.NotEqual(t, "MODEL_EXPRESS_URL", env.Name)
		assert.NotEqual(t, "PROMETHEUS_ENDPOINT", env.Name)
	}
}

func TestCheckpointReconciler_handlePendingFailsUnpreparedGMSCheckpoint(t *testing.T) {
	t.Setenv(consts.DynamoOperatorAllowGMSSnapshotEnvVar, "1")
	s := checkpointTestScheme()
	ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhasePending)
	ckpt.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
		Enabled: true,
	}

	r := makeCheckpointReconciler(s, ckpt)
	result, err := r.handlePending(context.Background(), ckpt)
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
	require.NoError(t, r.Get(context.Background(), types.NamespacedName{Name: ckpt.Name, Namespace: ckpt.Namespace}, updated))
	assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
	assert.Contains(t, updated.Status.Message, "gpuMemoryService checkpoint pod template is missing pod resource claim")
	condition := meta.FindStatusCondition(updated.Status.Conditions, string(nvidiacomv1alpha1.DynamoCheckpointConditionJobCreated))
	require.NotNil(t, condition)
	assert.Equal(t, metav1.ConditionFalse, condition.Status)
	assert.Equal(t, "GMSPodTemplateNotPrepared", condition.Reason)

	jobs := &batchv1.JobList{}
	require.NoError(t, r.List(context.Background(), jobs, client.InNamespace(testNamespace)))
	assert.Empty(t, jobs.Items)
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
		assert.Equal(t, testHash, updated.Status.CheckpointID)
		assert.Equal(t, testHash, updated.Status.IdentityHash)
		assert.Empty(t, updated.Status.Message)
		assert.Equal(t, testHash, updated.Labels[snapshotprotocol.CheckpointIDLabel])
	})

	t.Run("GMS snapshot fails when gate is disabled", func(t *testing.T) {
		t.Setenv(consts.DynamoOperatorAllowGMSSnapshotEnvVar, "")
		ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhasePending)
		ckpt.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true}
		r := makeCheckpointReconciler(s, ckpt)

		result, err := r.Reconcile(ctx, ctrl.Request{
			NamespacedName: types.NamespacedName{Name: ckpt.Name, Namespace: testNamespace},
		})
		require.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: ckpt.Name, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
		assert.Contains(t, updated.Status.Message, "GMS + Snapshot is temporarily disabled")

		jobs := &batchv1.JobList{}
		require.NoError(t, r.List(ctx, jobs, client.InNamespace(testNamespace)))
		assert.Empty(t, jobs.Items)
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
		assert.Equal(t, testHash, updated.Status.CheckpointID)
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

func TestCheckpointReconciler_FinalizeResourceCleansRetainedAutoCheckpointOnCRDelete(t *testing.T) {
	ctx := context.Background()
	s := checkpointTestScheme()

	cfg := checkpointTestConfig()
	cfg.Checkpoint.Storage = configv1alpha1.CheckpointStorageConfiguration{
		Type: snapshotprotocol.StorageTypePVC,
		PVC: configv1alpha1.CheckpointPVCConfig{
			PVCName:  "snapshot-pvc",
			BasePath: "/checkpoints",
		},
	}

	t.Run("creates cleanup job and keeps finalizer pending", func(t *testing.T) {
		ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhaseReady)
		ckpt.Labels = map[string]string{snapshotprotocol.CheckpointIDLabel: testHash}
		ckpt.Annotations = map[string]string{
			consts.CheckpointAutoAnnotation:           consts.KubeLabelValueTrue,
			consts.CheckpointDeletionPolicyAnnotation: string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain),
		}
		r := makeCheckpointReconciler(s, ckpt)
		r.Config = cfg

		err := r.FinalizeResource(ctx, ckpt)
		require.ErrorIs(t, err, errCheckpointCleanupPending)

		current := &batchv1.Job{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{
			Name:      "checkpoint-cleanup-" + testHash,
			Namespace: testNamespace,
		}, current))
		assert.Equal(t, testHash, current.Labels[snapshotprotocol.CheckpointIDLabel])
	})

	t.Run("running cleanup job keeps finalizer pending", func(t *testing.T) {
		ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhaseReady)
		ckpt.Labels = map[string]string{snapshotprotocol.CheckpointIDLabel: testHash}
		ckpt.Annotations = map[string]string{
			consts.CheckpointAutoAnnotation:           consts.KubeLabelValueTrue,
			consts.CheckpointDeletionPolicyAnnotation: string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain),
		}
		job, err := buildCheckpointCleanupJob(cfg, ckpt, testHash, snapshotprotocol.Storage{
			Type:     snapshotprotocol.StorageTypePVC,
			PVCName:  "snapshot-pvc",
			BasePath: "/checkpoints",
		})
		require.NoError(t, err)
		r := makeCheckpointReconciler(s, ckpt, job, newOwnedPod(podNameFromJob(job.Name), job))
		r.Config = cfg

		err = r.FinalizeResource(ctx, ckpt)
		require.ErrorIs(t, err, errCheckpointCleanupPending)
	})

	t.Run("failed cleanup job is deleted for retry", func(t *testing.T) {
		ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhaseReady)
		ckpt.Labels = map[string]string{snapshotprotocol.CheckpointIDLabel: testHash}
		ckpt.Annotations = map[string]string{
			consts.CheckpointAutoAnnotation:           consts.KubeLabelValueTrue,
			consts.CheckpointDeletionPolicyAnnotation: string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain),
		}
		job, err := buildCheckpointCleanupJob(cfg, ckpt, testHash, snapshotprotocol.Storage{
			Type:     snapshotprotocol.StorageTypePVC,
			PVCName:  "snapshot-pvc",
			BasePath: "/checkpoints",
		})
		require.NoError(t, err)
		job.Status.Conditions = []batchv1.JobCondition{{
			Type:    batchv1.JobFailed,
			Status:  corev1.ConditionTrue,
			Message: "boom",
		}}
		r := makeCheckpointReconciler(s, ckpt, job, newOwnedPod(podNameFromJob(job.Name), job))
		r.Config = cfg

		err = r.FinalizeResource(ctx, ckpt)
		require.ErrorIs(t, err, errCheckpointCleanupPending)
		current := &batchv1.Job{}
		err = r.Get(ctx, types.NamespacedName{Name: job.Name, Namespace: job.Namespace}, current)
		require.True(t, apierrors.IsNotFound(err), "expected failed cleanup job to be deleted, got %v", err)
	})

	t.Run("completed cleanup job is removed and finalizer may finish", func(t *testing.T) {
		ckpt := makeTestCheckpoint(nvidiacomv1alpha1.DynamoCheckpointPhaseReady)
		ckpt.Labels = map[string]string{snapshotprotocol.CheckpointIDLabel: testHash}
		ckpt.Annotations = map[string]string{
			consts.CheckpointAutoAnnotation:           consts.KubeLabelValueTrue,
			consts.CheckpointDeletionPolicyAnnotation: string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain),
		}
		job, err := buildCheckpointCleanupJob(cfg, ckpt, testHash, snapshotprotocol.Storage{
			Type:     snapshotprotocol.StorageTypePVC,
			PVCName:  "snapshot-pvc",
			BasePath: "/checkpoints",
		})
		require.NoError(t, err)
		job.Status.Conditions = []batchv1.JobCondition{{
			Type:   batchv1.JobComplete,
			Status: corev1.ConditionTrue,
		}}
		r := makeCheckpointReconciler(s, ckpt, job, newOwnedPod(podNameFromJob(job.Name), job))
		r.Config = cfg

		require.NoError(t, r.FinalizeResource(ctx, ckpt))
		current := &batchv1.Job{}
		err = r.Get(ctx, types.NamespacedName{Name: job.Name, Namespace: job.Namespace}, current)
		require.True(t, apierrors.IsNotFound(err), "expected completed cleanup job to be removed, got %v", err)
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
		ckpt.UID = types.UID("ckpt-uid") // required for PodSnapshot ownership
		ckpt.Status.IdentityHash = testHash
		ckpt.Status.JobName = jobName
		return ckpt
	}

	t.Run("waits without creating a PodSnapshot until the source pod exists", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, "job-no-pod")
		job := &batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: "job-no-pod", Namespace: testNamespace}}

		r := makeCheckpointReconciler(s, ckpt, job) // no Job-owned pod yet
		result, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)
		// Source pod not yet created: no time-based poll — the scoped pod watch re-enqueues. The Job
		// here has no deadline, so the result is empty (watch-driven).
		assert.Zero(t, result.RequeueAfter)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, updated.Status.Phase)

		var snaps nvidiacomv1alpha1.PodSnapshotList
		require.NoError(t, r.List(ctx, &snaps, client.InNamespace(testNamespace)))
		assert.Empty(t, snaps.Items)
	})

	t.Run("creates the PodSnapshot once the source pod exists", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		job := newCheckpointJob(defaultCheckpointJobName)
		r := makeCheckpointReconciler(s, ckpt, job, newOwnedPod("worker-0", job))

		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		snap := &nvidiacomv1alpha1.PodSnapshot{}
		require.NoError(t, r.Get(ctx,
			types.NamespacedName{Name: podSnapshotName(ckpt), Namespace: testNamespace}, snap))
		assert.Equal(t, ckpt.Name, snap.Labels[consts.SnapshotOwnerLabel])
		assert.Equal(t, "worker-0", snap.Spec.Source.PodRef.Name)
		assert.True(t, metav1.IsControlledBy(snap, ckpt))

		// The created snapshot's name is recorded in status as the authoritative pointer.
		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, snap.Name, updated.Status.PodSnapshotName)
	})

	// ownedSnapshot returns a PodSnapshot owned by ckpt and bound to a PodSnapshotContent,
	// carrying the given terminal condition (empty type leaves it Pending).
	ownedSnapshot := func(ckpt *nvidiacomv1alpha1.DynamoCheckpoint, condType string) *nvidiacomv1alpha1.PodSnapshot {
		bound := "podsnapshotcontent-" + testHash
		snap := &nvidiacomv1alpha1.PodSnapshot{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podSnapshotName(ckpt),
				Namespace: testNamespace,
				Labels:    map[string]string{consts.SnapshotOwnerLabel: ckpt.Name},
				OwnerReferences: []metav1.OwnerReference{{
					APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
					Kind:       "DynamoCheckpoint",
					Name:       ckpt.Name,
					UID:        ckpt.UID,
					Controller: ptr.To(true),
				}},
			},
			Spec: nvidiacomv1alpha1.PodSnapshotSpec{
				Source: nvidiacomv1alpha1.PodSnapshotSource{PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0"}},
			},
			Status: nvidiacomv1alpha1.PodSnapshotStatus{BoundPodSnapshotContentName: &bound},
		}
		if condType != "" {
			snap.Status.Conditions = []metav1.Condition{{
				Type:    condType,
				Status:  metav1.ConditionTrue,
				Reason:  "Test",
				Message: condType + " from agent",
			}}
		}
		return snap
	}

	t.Run("PodSnapshot Ready transitions checkpoint to Ready", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		job := newCheckpointJob(defaultCheckpointJobName)
		snap := ownedSnapshot(ckpt, nvidiacomv1alpha1.PodSnapshotConditionReady)

		r := makeCheckpointReconciler(s, ckpt, job, snap, newOwnedPod(podNameFromJob(job.Name), job))
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseReady, updated.Status.Phase)
		assert.Equal(t, testHash, updated.Status.CheckpointID)
		assert.NotNil(t, updated.Status.CreatedAt)
	})

	t.Run("PodSnapshot Failed transitions checkpoint to Failed", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		job := newCheckpointJob(defaultCheckpointJobName)
		snap := ownedSnapshot(ckpt, nvidiacomv1alpha1.PodSnapshotConditionFailed)

		r := makeCheckpointReconciler(s, ckpt, job, snap, newOwnedPod(podNameFromJob(job.Name), job))
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
		assert.Contains(t, updated.Status.Message, "from agent")
	})

	t.Run("unbound PodSnapshot Failed transitions checkpoint to Failed", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		job := newCheckpointJob(defaultCheckpointJobName)
		snap := ownedSnapshot(ckpt, nvidiacomv1alpha1.PodSnapshotConditionFailed)
		snap.Status.BoundPodSnapshotContentName = nil // failed before binding

		r := makeCheckpointReconciler(s, ckpt, job, snap, newOwnedPod(podNameFromJob(job.Name), job))
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
	})

	t.Run("failed Job while PodSnapshot non-terminal transitions to Failed", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		job := newCheckpointJob(defaultCheckpointJobName)
		job.Status = batchv1.JobStatus{
			Conditions: []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue, Message: "deadline"}},
		}
		snap := ownedSnapshot(ckpt, "")

		r := makeCheckpointReconciler(s, ckpt, job, snap, newOwnedPod(podNameFromJob(job.Name), job))
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
	})

	t.Run("Job DeadlineExceeded (k8s-set JobFailed) transitions to Failed", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		job := newCheckpointJob(defaultCheckpointJobName)
		// k8s enforces ActiveDeadlineSeconds and, on expiry, sets a JobFailed condition with reason
		// DeadlineExceeded; the operator just observes that condition.
		job.Status.Conditions = []batchv1.JobCondition{{
			Type:    batchv1.JobFailed,
			Status:  corev1.ConditionTrue,
			Reason:  "DeadlineExceeded",
			Message: "Job was active longer than specified deadline",
		}}
		snap := ownedSnapshot(ckpt, "")

		r := makeCheckpointReconciler(s, ckpt, job, snap, newOwnedPod(podNameFromJob(job.Name), job))
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
		assert.Contains(t, updated.Status.Message, "deadline")
	})

	t.Run("PodSnapshot not yet found requeues without changing phase", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		job := newCheckpointJob(defaultCheckpointJobName)

		r := makeCheckpointReconciler(s, ckpt, job, newOwnedPod(podNameFromJob(job.Name), job))
		// createPodSnapshot creates the PodSnapshot and records status.podSnapshotName; the snapshot
		// is not yet terminal, so the checkpoint stays Creating.
		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating, updated.Status.Phase)
	})

	t.Run("source pod absent with failed Job transitions to Failed", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		job := newCheckpointJob(defaultCheckpointJobName)
		job.Status = batchv1.JobStatus{
			Conditions: []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue, Message: "unschedulable"}},
		}
		r := makeCheckpointReconciler(s, ckpt, job) // no source pod

		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
	})

	t.Run("no snapshot with source pod and failed Job transitions to Failed", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		job := newCheckpointJob(defaultCheckpointJobName)
		job.Status = batchv1.JobStatus{
			Conditions: []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue, Message: "boom"}},
		}
		r := makeCheckpointReconciler(s, ckpt, job, newOwnedPod(podNameFromJob(job.Name), job))

		_, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)

		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		// The hang guard fails the checkpoint before creating a doomed snapshot, even with a pod present.
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointPhaseFailed, updated.Status.Phase)
	})

	t.Run("no snapshot with source pod and healthy Job creates the snapshot, no requeue", func(t *testing.T) {
		ckpt := makeCreatingCkpt(testHash, defaultCheckpointJobName)
		job := newCheckpointJob(defaultCheckpointJobName)
		r := makeCheckpointReconciler(s, ckpt, job, newOwnedPod(podNameFromJob(job.Name), job))

		result, err := r.handleCreating(ctx, ckpt)
		require.NoError(t, err)
		assert.Zero(t, result.RequeueAfter, "watch-driven: Owns(&PodSnapshot) re-enqueues for observation")

		// A missing snapshot is recreated (not treated as terminal), and its name is recorded.
		snap := &nvidiacomv1alpha1.PodSnapshot{}
		require.NoError(t, r.Get(ctx,
			types.NamespacedName{Name: podSnapshotName(ckpt), Namespace: testNamespace}, snap))
		updated := &nvidiacomv1alpha1.DynamoCheckpoint{}
		require.NoError(t, r.Get(ctx, types.NamespacedName{Name: testHash, Namespace: testNamespace}, updated))
		assert.Equal(t, snap.Name, updated.Status.PodSnapshotName)
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
