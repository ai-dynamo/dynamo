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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testHash      = "abc123def4567890"
	testNamespace = "default"
)

func testPVCConfig() *configv1alpha1.CheckpointConfiguration {
	return &configv1alpha1.CheckpointConfiguration{
		Enabled: true,
		Storage: configv1alpha1.CheckpointStorageConfiguration{
			Type: configv1alpha1.CheckpointStorageTypePVC,
			PVC: configv1alpha1.CheckpointPVCConfig{
				PVCName:  "snapshot-pvc",
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

type createHookClient struct {
	client.Client
	onCreate func(ctx context.Context, obj client.Object) error
}

func (c *createHookClient) Create(ctx context.Context, obj client.Object, opts ...client.CreateOption) error {
	if c.onCreate != nil {
		if err := c.onCreate(ctx, obj); err != nil {
			return err
		}
		c.onCreate = nil
	}

	return c.Client.Create(ctx, obj, opts...)
}

// --- Resource helper tests ---

func TestHelpers(t *testing.T) {
	// checkpointInfoFromObject — ready
	hash, err := ComputeIdentityHash(testIdentity())
	require.NoError(t, err)
	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: hash},
		Spec:       nvidiacomv1alpha1.DynamoCheckpointSpec{Identity: testIdentity()},
		Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
			Phase:        nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
			IdentityHash: hash,
			Location:     "/checkpoints/" + hash,
			StorageType:  "pvc",
		},
	}
	info, err := checkpointInfoFromObject(ckpt)
	require.NoError(t, err)
	assert.True(t, info.Enabled)
	assert.True(t, info.Ready)
	assert.Equal(t, hash, info.Hash)
	assert.Equal(t, "/checkpoints/"+hash, info.Location)
	assert.Equal(t, ckpt.Name, info.CheckpointName)

	// checkpointInfoFromObject — not ready
	ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseCreating
	info, err = checkpointInfoFromObject(ckpt)
	require.NoError(t, err)
	assert.False(t, info.Ready)
}

func TestArtifactVersionHelpers(t *testing.T) {
	t.Run("new checkpoints default to version 1", func(t *testing.T) {
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{}
		assert.Nil(t, ckpt.Annotations)
		assert.Equal(t, "checkpoint-job-"+testHash+"-"+consts.DefaultCheckpointArtifactVersion, "checkpoint-job-"+testHash+"-"+consts.DefaultCheckpointArtifactVersion)
	})

	t.Run("annotation overrides desired version", func(t *testing.T) {
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					consts.KubeAnnotationCheckpointArtifactVersion: "3",
				},
			},
		}
		assert.Equal(t, "3", ckpt.Annotations[consts.KubeAnnotationCheckpointArtifactVersion])
		assert.Equal(t, "checkpoint-job-"+testHash+"-3", "checkpoint-job-"+testHash+"-"+ckpt.Annotations[consts.KubeAnnotationCheckpointArtifactVersion])
	})
}

func TestResolveCheckpointStorage(t *testing.T) {
	config := testPVCConfig()

	location, storageType, err := ResolveCheckpointStorage(testHash, "", config)
	require.NoError(t, err)
	assert.Equal(t, "/checkpoints/"+testHash+"/versions/"+consts.DefaultCheckpointArtifactVersion, location)
	assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointStorageType("pvc"), storageType)

	location, storageType, err = ResolveCheckpointStorage(testHash, "7", config)
	require.NoError(t, err)
	assert.Equal(t, "/checkpoints/"+testHash+"/versions/7", location)
	assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointStorageType("pvc"), storageType)
}

func TestCreateOrGetAutoCheckpointDeduplicatesConcurrentSameHashCheckpoint(t *testing.T) {
	ctx := context.Background()
	s := testScheme()

	identity := testIdentity()
	hash, err := ComputeIdentityHash(identity)
	require.NoError(t, err)

	friendly := &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "friendly-checkpoint",
			Namespace: testNamespace,
			Labels: map[string]string{
				consts.KubeLabelCheckpointHash: hash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Identity: identity,
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{},
			},
		},
		Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
			IdentityHash: hash,
			Phase:        nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
		},
	}

	baseClient := fake.NewClientBuilder().WithScheme(s).Build()
	c := &createHookClient{
		Client: baseClient,
		onCreate: func(ctx context.Context, obj client.Object) error {
			_, ok := obj.(*nvidiacomv1alpha1.DynamoCheckpoint)
			if !ok {
				return nil
			}
			return baseClient.Create(ctx, friendly.DeepCopy())
		},
	}

	ckpt, err := CreateOrGetAutoCheckpoint(ctx, c, testNamespace, identity, corev1.PodTemplateSpec{})
	require.NoError(t, err)
	assert.Equal(t, friendly.Name, ckpt.Name)

	list := &nvidiacomv1alpha1.DynamoCheckpointList{}
	require.NoError(t, baseClient.List(ctx, list))
	require.Len(t, list.Items, 1)
	assert.Equal(t, friendly.Name, list.Items[0].Name)
}

func TestCreateOrGetAutoCheckpointSetsDefaultArtifactVersion(t *testing.T) {
	ctx := context.Background()
	s := testScheme()
	c := fake.NewClientBuilder().WithScheme(s).Build()

	ckpt, err := CreateOrGetAutoCheckpoint(ctx, c, testNamespace, testIdentity(), corev1.PodTemplateSpec{})
	require.NoError(t, err)
	require.NotNil(t, ckpt.Annotations)
	assert.Equal(t, consts.DefaultCheckpointArtifactVersion, ckpt.Annotations[consts.KubeAnnotationCheckpointArtifactVersion])
}

// --- Injection idempotency tests ---

func TestInjectionIdempotency(t *testing.T) {
	// Volume injection is idempotent
	podSpec := &corev1.PodSpec{Volumes: []corev1.Volume{{Name: consts.CheckpointVolumeName}, {Name: consts.PodInfoVolumeName}}}
	InjectCheckpointVolume(podSpec, "snapshot-pvc")
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

func TestApplyCheckpointPodMetadata(t *testing.T) {
	t.Run("checkpoint source metadata uses annotations for location and storage", func(t *testing.T) {
		labels := map[string]string{}
		annotations := map[string]string{}

		ApplyCheckpointSourcePodMetadata(labels, annotations, testHash, "/checkpoints/"+testHash, "pvc")

		assert.Equal(t, consts.KubeLabelValueTrue, labels[consts.KubeLabelIsCheckpointSource])
		assert.Equal(t, testHash, labels[consts.KubeLabelCheckpointHash])
		assert.Equal(t, "/checkpoints/"+testHash, annotations[consts.KubeAnnotationCheckpointLocation])
		assert.Equal(t, "pvc", annotations[consts.KubeAnnotationCheckpointStorageType])
	})

	t.Run("restore metadata clears stale values when checkpoint is not ready", func(t *testing.T) {
		labels := map[string]string{
			consts.KubeLabelIsRestoreTarget: consts.KubeLabelValueTrue,
			consts.KubeLabelCheckpointHash:  "stale-hash",
		}
		annotations := map[string]string{
			consts.KubeAnnotationCheckpointLocation:    "/checkpoints/stale-hash",
			consts.KubeAnnotationCheckpointStorageType: "pvc",
		}

		ApplyRestorePodMetadata(labels, annotations, &CheckpointInfo{Enabled: true, Ready: false})

		_, hasRestoreTarget := labels[consts.KubeLabelIsRestoreTarget]
		_, hasCheckpointHash := labels[consts.KubeLabelCheckpointHash]
		_, hasLocation := annotations[consts.KubeAnnotationCheckpointLocation]
		_, hasStorageType := annotations[consts.KubeAnnotationCheckpointStorageType]
		assert.False(t, hasRestoreTarget)
		assert.False(t, hasCheckpointHash)
		assert.False(t, hasLocation)
		assert.False(t, hasStorageType)
	})
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

	t.Run("ready checkpoint preserves published versioned location", func(t *testing.T) {
		podSpec := testPodSpec()
		info := &CheckpointInfo{
			Enabled:     true,
			Ready:       true,
			Hash:        testHash,
			Location:    "/checkpoints/" + testHash + "/versions/2",
			StorageType: "pvc",
		}
		require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, info, testPVCConfig()))
		assert.Equal(t, "/checkpoints/"+testHash+"/versions/2", info.Location)
		assert.Equal(t, nvidiacomv1alpha1.DynamoCheckpointStorageType("pvc"), info.StorageType)
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

	t.Run("PVC storage injects volumes and mounts", func(t *testing.T) {
		podSpec := testPodSpec()
		require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, testInfo(), testPVCConfig()))

		// Volumes
		volNames := make(map[string]bool)
		for _, v := range podSpec.Volumes {
			volNames[v.Name] = true
			if v.Name == consts.CheckpointVolumeName {
				assert.Equal(t, "snapshot-pvc", v.PersistentVolumeClaim.ClaimName)
			}
			if v.Name == consts.PodInfoVolumeName {
				require.NotNil(t, v.DownwardAPI)
				fieldPaths := map[string]string{}
				for _, item := range v.DownwardAPI.Items {
					if item.FieldRef != nil {
						fieldPaths[item.Path] = item.FieldRef.FieldPath
					}
				}
				assert.Equal(t, "metadata.labels['"+consts.KubeLabelDynamoNamespace+"']", fieldPaths[consts.PodInfoFileDynNamespace])
				assert.Equal(t, "metadata.labels['"+consts.KubeLabelDynamoWorkerHash+"']", fieldPaths[consts.PodInfoFileDynNamespaceWorkerSuffix])
				assert.Equal(t, "metadata.labels['"+consts.KubeLabelDynamoComponentType+"']", fieldPaths[consts.PodInfoFileDynComponent])
				assert.Equal(t, "metadata.labels['"+consts.KubeLabelDynamoGraphDeploymentName+"']", fieldPaths[consts.PodInfoFileDynParentDGDName])
				assert.Equal(t, consts.PodInfoFieldPodNamespace, fieldPaths[consts.PodInfoFileDynParentDGDNamespace])
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
			config      configv1alpha1.CheckpointStorageConfiguration
			wantLoc     string
		}{
			{"s3", configv1alpha1.CheckpointStorageConfiguration{
				Type: configv1alpha1.CheckpointStorageTypeS3,
				S3:   configv1alpha1.CheckpointS3Config{URI: "s3://bucket/prefix"},
			}, "s3://bucket/prefix/" + testHash + ".tar"},
			{"oci", configv1alpha1.CheckpointStorageConfiguration{
				Type: configv1alpha1.CheckpointStorageTypeOCI,
				OCI:  configv1alpha1.CheckpointOCIConfig{URI: "oci://registry/repo"},
			}, "oci://registry/repo:" + testHash},
		} {
			t.Run(tc.storageType, func(t *testing.T) {
				podSpec := testPodSpec()
				info := &CheckpointInfo{Enabled: true, Hash: testHash}
				require.NoError(t, InjectCheckpointIntoPodSpec(podSpec, info, &configv1alpha1.CheckpointConfiguration{Storage: tc.config}))
				assert.Equal(t, tc.wantLoc, info.Location)
			})
		}
	})

	t.Run("error cases", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			podSpec *corev1.PodSpec
			info    *CheckpointInfo
			config  *configv1alpha1.CheckpointConfiguration
			errMsg  string
		}{
			{"hash empty and identity nil", testPodSpec(), &CheckpointInfo{Enabled: true}, testPVCConfig(), "identity is nil"},
			{"no containers", &corev1.PodSpec{}, testInfo(), testPVCConfig(), "no container found"},
			{"PVC name missing", testPodSpec(), testInfo(), &configv1alpha1.CheckpointConfiguration{
				Storage: configv1alpha1.CheckpointStorageConfiguration{Type: "pvc", PVC: configv1alpha1.CheckpointPVCConfig{BasePath: "/checkpoints"}},
			}, "no PVC name"},
			{"S3 URI missing", testPodSpec(), testInfo(), &configv1alpha1.CheckpointConfiguration{
				Storage: configv1alpha1.CheckpointStorageConfiguration{Type: "s3"},
			}, "S3"},
			{"OCI URI missing", testPodSpec(), testInfo(), &configv1alpha1.CheckpointConfiguration{
				Storage: configv1alpha1.CheckpointStorageConfiguration{Type: "oci"},
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
		hash, err := ComputeIdentityHash(testIdentity())
		require.NoError(t, err)
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: hash, Namespace: testNamespace},
			Spec:       nvidiacomv1alpha1.DynamoCheckpointSpec{Identity: testIdentity()},
			Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
				Phase:        nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
				IdentityHash: hash,
				Location:     "/checkpoints/" + hash,
				StorageType:  "pvc",
			},
		}
		c := fake.NewClientBuilder().WithScheme(s).WithObjects(ckpt).WithStatusSubresource(ckpt).Build()
		ref := hash

		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, CheckpointRef: &ref,
		})
		require.NoError(t, err)
		assert.True(t, info.Exists)
		assert.True(t, info.Ready)
		assert.Equal(t, hash, info.Hash)
		assert.Equal(t, "/checkpoints/"+hash, info.Location)
		assert.Equal(t, hash, info.CheckpointName)
	})

	t.Run("checkpointRef resolves not-ready CR", func(t *testing.T) {
		hash, err := ComputeIdentityHash(testIdentity())
		require.NoError(t, err)
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: hash, Namespace: testNamespace},
			Spec:       nvidiacomv1alpha1.DynamoCheckpointSpec{Identity: testIdentity()},
			Status:     nvidiacomv1alpha1.DynamoCheckpointStatus{Phase: nvidiacomv1alpha1.DynamoCheckpointPhaseCreating},
		}
		c := fake.NewClientBuilder().WithScheme(s).WithObjects(ckpt).WithStatusSubresource(ckpt).Build()
		ref := hash

		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, CheckpointRef: &ref,
		})
		require.NoError(t, err)
		assert.True(t, info.Exists)
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

	t.Run("checkpointRef resolves human-readable checkpoint names", func(t *testing.T) {
		hash, err := ComputeIdentityHash(testIdentity())
		require.NoError(t, err)
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: "not-the-hash", Namespace: testNamespace},
			Spec:       nvidiacomv1alpha1.DynamoCheckpointSpec{Identity: testIdentity()},
			Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
				IdentityHash: hash,
			},
		}
		c := fake.NewClientBuilder().WithScheme(s).WithObjects(ckpt).WithStatusSubresource(ckpt).Build()
		ref := "not-the-hash"

		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, CheckpointRef: &ref,
		})
		require.NoError(t, err)
		assert.Equal(t, "not-the-hash", info.CheckpointName)
		assert.Equal(t, hash, info.Hash)
	})

	t.Run("identity lookup finds existing checkpoint by identity hash", func(t *testing.T) {
		identity := testIdentity()
		hash, err := ComputeIdentityHash(identity)
		require.NoError(t, err)

		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: "friendly-name", Namespace: testNamespace},
			Spec:       nvidiacomv1alpha1.DynamoCheckpointSpec{Identity: identity},
			Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
				Phase:        nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
				IdentityHash: hash,
				Location:     "/checkpoints/" + hash,
				StorageType:  "pvc",
			},
		}
		c := fake.NewClientBuilder().WithScheme(s).WithObjects(ckpt).WithStatusSubresource(ckpt).Build()

		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, Identity: &identity,
		})
		require.NoError(t, err)
		assert.True(t, info.Exists)
		assert.True(t, info.Ready)
		assert.Equal(t, hash, info.Hash)
		assert.Equal(t, "friendly-name", info.CheckpointName)
	})

	t.Run("identity lookup returns existing not-ready checkpoint", func(t *testing.T) {
		identity := testIdentity()
		hash, err := ComputeIdentityHash(identity)
		require.NoError(t, err)

		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: "friendly-name", Namespace: testNamespace},
			Spec:       nvidiacomv1alpha1.DynamoCheckpointSpec{Identity: identity},
			Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
				Phase:        nvidiacomv1alpha1.DynamoCheckpointPhaseCreating,
				IdentityHash: hash,
			},
		}
		c := fake.NewClientBuilder().WithScheme(s).WithObjects(ckpt).WithStatusSubresource(ckpt).Build()

		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, Identity: &identity,
		})
		require.NoError(t, err)
		assert.True(t, info.Exists)
		assert.False(t, info.Ready)
		assert.Equal(t, hash, info.Hash)
	})

	t.Run("identity lookup returns not-ready when no CR found", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(s).Build()
		identity := testIdentity()
		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, Identity: &identity,
		})
		require.NoError(t, err)
		assert.False(t, info.Exists)
		assert.False(t, info.Ready)
		assert.Len(t, info.Hash, 16)
	})

	t.Run("errors when enabled but no ref and no identity", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(s).Build()
		_, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{Enabled: true})
		assert.ErrorContains(t, err, "no checkpointRef or identity")
	})
}
