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

package checkpoint

import (
	"context"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controller_common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testHash      = "abc123def4567890"
	testNamespace = "default"
)

func testPVCConfig() *controller_common.CheckpointConfig {
	return &controller_common.CheckpointConfig{
		Enabled: true,
		Storage: controller_common.CheckpointStorageConfig{
			Type: controller_common.CheckpointStorageTypePVC,
			PVC: controller_common.CheckpointPVCConfig{
				PVCName:  "chrek-pvc",
				BasePath: "/checkpoints",
			},
		},
	}
}

func testIdentity() nvidiacomv1alpha1.DynamoCheckpointIdentity {
	return nvidiacomv1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
}

func testPodSpec() *corev1.PodSpec {
	return &corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:    consts.MainContainerName,
			Image:   "test-image:latest",
			Command: []string{"python3"},
			Args:    []string{"-m", "dynamo.vllm"},
		}},
	}
}

func testScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	_ = nvidiacomv1alpha1.AddToScheme(s)
	_ = corev1.AddToScheme(s)
	return s
}

func testInfo() *CheckpointInfo {
	return &CheckpointInfo{Enabled: true, Hash: testHash}
}

// --- Helper function tests ---

func TestHelpers(t *testing.T) {
	// GetPVCBasePath
	assert.Equal(t, "", GetPVCBasePath(nil))
	assert.Equal(t, "/checkpoints", GetPVCBasePath(testPVCConfig()))

	// getCheckpointInfoFromCheckpoint — ready
	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "ckpt-abc"},
		Spec:       nvidiacomv1alpha1.DynamoCheckpointSpec{Identity: testIdentity()},
		Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
			Phase: nvidiacomv1alpha1.DynamoCheckpointPhaseReady, IdentityHash: testHash,
			Location: "/checkpoints/" + testHash, StorageType: "pvc",
		},
	}
	info := getCheckpointInfoFromCheckpoint(ckpt)
	assert.True(t, info.Enabled)
	assert.True(t, info.Ready)
	assert.Equal(t, testHash, info.Hash)
	assert.Equal(t, "/checkpoints/"+testHash, info.Location)

	// getCheckpointInfoFromCheckpoint — not ready
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseCreating
	info = getCheckpointInfoFromCheckpoint(ckpt)
	assert.False(t, info.Ready)
}

// --- Injection idempotency tests ---

func TestInjectionIdempotency(t *testing.T) {
	// Volume injection is idempotent
	podSpec := &corev1.PodSpec{Volumes: []corev1.Volume{{Name: consts.CheckpointVolumeName}, {Name: consts.PodInfoVolumeName}}}
	InjectCheckpointVolume(podSpec, "chrek-pvc")
	InjectPodInfoVolume(podSpec)
	assert.Len(t, podSpec.Volumes, 2)

	// Mount injection is idempotent
	container := &corev1.Container{VolumeMounts: []corev1.VolumeMount{
		{Name: consts.CheckpointVolumeName}, {Name: consts.PodInfoVolumeName},
	}}
	InjectCheckpointVolumeMount(container, "/checkpoints")
	InjectPodInfoVolumeMount(container)
	assert.Len(t, container.VolumeMounts, 2)
}

// --- InjectCheckpointEnvVars tests ---

func TestInjectCheckpointEnvVars(t *testing.T) {
	t.Run("PVC storage injects PATH and HASH", func(t *testing.T) {
		container := &corev1.Container{}
		InjectCheckpointEnvVars(container, testInfo(), testPVCConfig())

		envMap := make(map[string]string, len(container.Env))
		for _, e := range container.Env {
			envMap[e.Name] = e.Value
		}
		assert.Equal(t, "/checkpoints", envMap[consts.EnvCheckpointPath])
		assert.Equal(t, testHash, envMap[consts.EnvCheckpointHash])
		_, hasLocation := envMap[consts.EnvCheckpointLocation]
		assert.False(t, hasLocation)
	})

	t.Run("S3 storage injects LOCATION and HASH", func(t *testing.T) {
		container := &corev1.Container{}
		info := &CheckpointInfo{Enabled: true, Hash: testHash, Location: "s3://bucket/" + testHash + ".tar"}
		config := &controller_common.CheckpointConfig{
			Storage: controller_common.CheckpointStorageConfig{
				Type: controller_common.CheckpointStorageTypeS3,
				S3:   controller_common.CheckpointS3Config{URI: "s3://bucket"},
			},
		}
		InjectCheckpointEnvVars(container, info, config)

		envMap := make(map[string]string, len(container.Env))
		for _, e := range container.Env {
			envMap[e.Name] = e.Value
		}
		assert.Equal(t, "s3://bucket/"+testHash+".tar", envMap[consts.EnvCheckpointLocation])
		assert.Equal(t, testHash, envMap[consts.EnvCheckpointHash])
	})

	t.Run("disabled is a no-op", func(t *testing.T) {
		container := &corev1.Container{}
		InjectCheckpointEnvVars(container, &CheckpointInfo{Enabled: false}, testPVCConfig())
		assert.Empty(t, container.Env)
	})

	t.Run("preserves existing env vars", func(t *testing.T) {
		container := &corev1.Container{Env: []corev1.EnvVar{{Name: "EXISTING", Value: "keep"}}}
		InjectCheckpointEnvVars(container, testInfo(), testPVCConfig())

		envMap := make(map[string]string, len(container.Env))
		for _, e := range container.Env {
			envMap[e.Name] = e.Value
		}
		assert.Equal(t, "keep", envMap["EXISTING"])
		assert.Equal(t, testHash, envMap[consts.EnvCheckpointHash])
	})
}

// --- InjectCheckpointLabelsFromConfig tests ---

func TestInjectCheckpointLabelsFromConfig(t *testing.T) {
	// Disabled/nil configs are no-ops
	for _, cfg := range []*nvidiacomv1alpha1.ServiceCheckpointConfig{nil, {Enabled: false}} {
		labels := map[string]string{"existing": "value"}
		result, err := InjectCheckpointLabelsFromConfig(labels, cfg)
		require.NoError(t, err)
		assert.Equal(t, map[string]string{"existing": "value"}, result)
	}

	// Enabled with identity adds hash label
	identity := testIdentity()
	result, err := InjectCheckpointLabelsFromConfig(nil, &nvidiacomv1alpha1.ServiceCheckpointConfig{
		Enabled: true, Identity: &identity,
	})
	require.NoError(t, err)
	hash, ok := result[consts.KubeLabelCheckpointHash]
	assert.True(t, ok)
	assert.Len(t, hash, 16)

	// Enabled without identity does not add hash
	result, err = InjectCheckpointLabelsFromConfig(map[string]string{}, &nvidiacomv1alpha1.ServiceCheckpointConfig{Enabled: true})
	require.NoError(t, err)
	_, ok = result[consts.KubeLabelCheckpointHash]
	assert.False(t, ok)
}

// --- InjectCheckpointIntoPodSpec tests ---

func TestInjectCheckpointIntoPodSpec(t *testing.T) {
	t.Run("nil or disabled info is a no-op", func(t *testing.T) {
		for _, info := range []*CheckpointInfo{nil, {Enabled: false}} {
			podSpec := testPodSpec()
			require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, info, testPVCConfig()))
			assert.Equal(t, []string{"python3"}, podSpec.Containers[0].Command)
		}
	})

	t.Run("ready checkpoint overrides command to sleep infinity", func(t *testing.T) {
		podSpec := testPodSpec()
		info := &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash}
		require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, info, testPVCConfig()))
		assert.Equal(t, []string{"sleep", "infinity"}, podSpec.Containers[0].Command)
		assert.Nil(t, podSpec.Containers[0].Args)
	})

	t.Run("not-ready checkpoint preserves original command", func(t *testing.T) {
		podSpec := testPodSpec()
		require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, testInfo(), testPVCConfig()))
		assert.Equal(t, []string{"python3"}, podSpec.Containers[0].Command)
	})

	t.Run("sets seccomp profile", func(t *testing.T) {
		podSpec := testPodSpec()
		require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, testInfo(), testPVCConfig()))
		require.NotNil(t, podSpec.SecurityContext)
		require.NotNil(t, podSpec.SecurityContext.SeccompProfile)
		assert.Equal(t, corev1.SeccompProfileTypeLocalhost, podSpec.SecurityContext.SeccompProfile.Type)
		assert.Equal(t, consts.SeccompProfilePath, *podSpec.SecurityContext.SeccompProfile.LocalhostProfile)
	})

	t.Run("preserves existing security context", func(t *testing.T) {
		podSpec := testPodSpec()
		podSpec.SecurityContext = &corev1.PodSecurityContext{RunAsUser: ptr.To(int64(1000))}
		require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, testInfo(), testPVCConfig()))
		assert.Equal(t, int64(1000), *podSpec.SecurityContext.RunAsUser)
		require.NotNil(t, podSpec.SecurityContext.SeccompProfile)
	})

	t.Run("PVC storage injects volumes, mounts, and env vars", func(t *testing.T) {
		podSpec := testPodSpec()
		require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, testInfo(), testPVCConfig()))

		// Volumes
		volNames := make(map[string]bool)
		for _, v := range podSpec.Volumes {
			volNames[v.Name] = true
			if v.Name == consts.CheckpointVolumeName {
				assert.Equal(t, "chrek-pvc", v.PersistentVolumeClaim.ClaimName)
			}
		}
		assert.True(t, volNames[consts.CheckpointVolumeName])
		assert.True(t, volNames[consts.PodInfoVolumeName])

		// Mounts
		mountPaths := make(map[string]string)
		for _, m := range podSpec.Containers[0].VolumeMounts {
			mountPaths[m.Name] = m.MountPath
		}
		assert.Equal(t, "/checkpoints", mountPaths[consts.CheckpointVolumeName])
		assert.Equal(t, consts.PodInfoMountPath, mountPaths[consts.PodInfoVolumeName])

		// Env
		envMap := make(map[string]string, len(podSpec.Containers[0].Env))
		for _, e := range podSpec.Containers[0].Env {
			envMap[e.Name] = e.Value
		}
		assert.Equal(t, "/checkpoints", envMap[consts.EnvCheckpointPath])
		assert.Equal(t, testHash, envMap[consts.EnvCheckpointHash])
	})

	t.Run("computes hash from identity when hash is empty", func(t *testing.T) {
		podSpec := testPodSpec()
		identity := testIdentity()
		info := &CheckpointInfo{Enabled: true, Identity: &identity}
		require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, info, testPVCConfig()))
		assert.Len(t, info.Hash, 16)
	})

	t.Run("S3 and OCI storage set location", func(t *testing.T) {
		for _, tc := range []struct {
			storageType string
			config      controller_common.CheckpointStorageConfig
			wantLoc     string
		}{
			{"s3", controller_common.CheckpointStorageConfig{
				Type: controller_common.CheckpointStorageTypeS3,
				S3:   controller_common.CheckpointS3Config{URI: "s3://bucket/prefix"},
			}, "s3://bucket/prefix/" + testHash + ".tar"},
			{"oci", controller_common.CheckpointStorageConfig{
				Type: controller_common.CheckpointStorageTypeOCI,
				OCI:  controller_common.CheckpointOCIConfig{URI: "oci://registry/repo"},
			}, "oci://registry/repo:" + testHash},
		} {
			t.Run(tc.storageType, func(t *testing.T) {
				podSpec := testPodSpec()
				info := &CheckpointInfo{Enabled: true, Hash: testHash}
				require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, info, &controller_common.CheckpointConfig{Storage: tc.config}))
				assert.Equal(t, tc.wantLoc, info.Location)
			})
		}
	})

	t.Run("error cases", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			podSpec *corev1.PodSpec
			info    *CheckpointInfo
			config  *controller_common.CheckpointConfig
			errMsg  string
		}{
			{"hash empty and identity nil", testPodSpec(), &CheckpointInfo{Enabled: true}, testPVCConfig(), "identity is nil"},
			{"no containers", &corev1.PodSpec{}, testInfo(), testPVCConfig(), "no container found"},
			{"PVC name missing", testPodSpec(), testInfo(), &controller_common.CheckpointConfig{
				Storage: controller_common.CheckpointStorageConfig{Type: "pvc", PVC: controller_common.CheckpointPVCConfig{BasePath: "/checkpoints"}},
			}, "no PVC name"},
			{"PVC base path missing", testPodSpec(), testInfo(), &controller_common.CheckpointConfig{
				Storage: controller_common.CheckpointStorageConfig{Type: "pvc", PVC: controller_common.CheckpointPVCConfig{PVCName: "chrek-pvc"}},
			}, "no PVC base path"},
			{"S3 URI missing", testPodSpec(), testInfo(), &controller_common.CheckpointConfig{
				Storage: controller_common.CheckpointStorageConfig{Type: "s3"},
			}, "S3"},
			{"OCI URI missing", testPodSpec(), testInfo(), &controller_common.CheckpointConfig{
				Storage: controller_common.CheckpointStorageConfig{Type: "oci"},
			}, "OCI"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				err := InjectCheckpointIntoPodSpec(tc.podSpec, tc.info, tc.config)
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errMsg)
			})
		}
	})

	t.Run("falls back to first container when main not found", func(t *testing.T) {
		podSpec := &corev1.PodSpec{Containers: []corev1.Container{{Name: "sidecar", Image: "img", Command: []string{"python3"}}}}
		info := &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash}
		require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, info, testPVCConfig()))
		assert.Equal(t, []string{"sleep", "infinity"}, podSpec.Containers[0].Command)
	})
}

// --- ResolveCheckpointForService tests ---

func TestResolveCheckpointForService(t *testing.T) {
	ctx := context.Background()
	s := testScheme()

	t.Run("nil or disabled config returns disabled", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(s).Build()
		for _, cfg := range []*nvidiacomv1alpha1.ServiceCheckpointConfig{nil, {Enabled: false}} {
			info, err := ResolveCheckpointForService(ctx, c, testNamespace, cfg)
			require.NoError(t, err)
			assert.False(t, info.Enabled)
		}
	})

	t.Run("checkpointRef resolves ready CR", func(t *testing.T) {
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: "my-ckpt", Namespace: testNamespace},
			Spec:       nvidiacomv1alpha1.DynamoCheckpointSpec{Identity: testIdentity()},
			Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
				Phase: nvidiacomv1alpha1.DynamoCheckpointPhaseReady, IdentityHash: testHash,
				Location: "/checkpoints/" + testHash, StorageType: "pvc",
			},
		}
		c := fake.NewClientBuilder().WithScheme(s).WithObjects(ckpt).WithStatusSubresource(ckpt).Build()
		ref := "my-ckpt"

		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, CheckpointRef: &ref,
		})
		require.NoError(t, err)
		assert.True(t, info.Ready)
		assert.Equal(t, testHash, info.Hash)
		assert.Equal(t, "/checkpoints/"+testHash, info.Location)
	})

	t.Run("checkpointRef resolves not-ready CR", func(t *testing.T) {
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: "pending-ckpt", Namespace: testNamespace},
			Spec:       nvidiacomv1alpha1.DynamoCheckpointSpec{Identity: testIdentity()},
			Status:     nvidiacomv1alpha1.DynamoCheckpointStatus{Phase: nvidiacomv1alpha1.DynamoCheckpointPhaseCreating},
		}
		c := fake.NewClientBuilder().WithScheme(s).WithObjects(ckpt).WithStatusSubresource(ckpt).Build()
		ref := "pending-ckpt"

		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, CheckpointRef: &ref,
		})
		require.NoError(t, err)
		assert.False(t, info.Ready)
	})

	t.Run("checkpointRef errors when CR not found", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(s).Build()
		ref := "nonexistent"
		_, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, CheckpointRef: &ref,
		})
		assert.ErrorContains(t, err, "nonexistent")
	})

	t.Run("identity lookup finds existing checkpoint by label", func(t *testing.T) {
		identity := testIdentity()
		hash, err := ComputeIdentityHash(identity)
		require.NoError(t, err)

		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{
				Name: hash, Namespace: testNamespace,
				Labels: map[string]string{consts.KubeLabelCheckpointHash: hash},
			},
			Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{Identity: identity},
			Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
				Phase: nvidiacomv1alpha1.DynamoCheckpointPhaseReady, IdentityHash: hash,
				Location: "/checkpoints/" + hash, StorageType: "pvc",
			},
		}
		c := fake.NewClientBuilder().WithScheme(s).WithObjects(ckpt).WithStatusSubresource(ckpt).Build()

		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, Identity: &identity,
		})
		require.NoError(t, err)
		assert.True(t, info.Ready)
		assert.Equal(t, hash, info.Hash)
	})

	t.Run("identity lookup returns not-ready when no CR found", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(s).Build()
		identity := testIdentity()
		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, Identity: &identity,
		})
		require.NoError(t, err)
		assert.False(t, info.Ready)
		assert.Len(t, info.Hash, 16)
	})

	t.Run("errors when enabled but no ref and no identity", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(s).Build()
		_, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{Enabled: true})
		assert.ErrorContains(t, err, "no checkpointRef or identity")
	})
}
