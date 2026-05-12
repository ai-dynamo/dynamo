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
	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	testHash      = "abc123def4567890"
	testNamespace = "default"
)

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
	_ = appsv1.AddToScheme(s)
	return s
}

func testInfo() *CheckpointInfo {
	return &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash}
}

func testSnapshotAgentDaemonSet() *appsv1.DaemonSet {
	return &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "snapshot-agent",
			Namespace: testNamespace,
			Labels: map[string]string{
				snapshotprotocol.SnapshotAgentLabelKey: snapshotprotocol.SnapshotAgentLabelValue,
			},
		},
		Spec: appsv1.DaemonSetSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name: snapshotprotocol.SnapshotAgentContainerName,
						VolumeMounts: []corev1.VolumeMount{{
							Name:      "checkpoints",
							MountPath: "/checkpoints",
						}},
					}},
					Volumes: []corev1.Volume{{
						Name: "checkpoints",
						VolumeSource: corev1.VolumeSource{
							PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
								ClaimName: "snapshot-pvc",
							},
						},
					}},
				},
			},
		},
	}
}

func TestStorageFromConfig(t *testing.T) {
	t.Run("empty config uses daemonset discovery", func(t *testing.T) {
		_, ok, err := StorageFromConfig(configv1alpha1.CheckpointStorageConfiguration{})
		require.NoError(t, err)
		assert.False(t, ok)
	})

	t.Run("legacy s3 type is ignored", func(t *testing.T) {
		_, ok, err := StorageFromConfig(configv1alpha1.CheckpointStorageConfiguration{
			Type: configv1alpha1.CheckpointStorageTypeS3,
		})
		require.NoError(t, err)
		assert.False(t, ok)
	})

	t.Run("unknown storage type is rejected", func(t *testing.T) {
		_, _, err := StorageFromConfig(configv1alpha1.CheckpointStorageConfiguration{
			Type: "typo",
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "checkpoint.storage.type")
	})

	t.Run("pvc config resolves storage", func(t *testing.T) {
		storage, ok, err := StorageFromConfig(configv1alpha1.CheckpointStorageConfiguration{
			Type: snapshotprotocol.StorageTypePVC,
			PVC: configv1alpha1.CheckpointPVCConfig{
				PVCName:  "namespace-snapshots",
				BasePath: "/snapshots/",
			},
		})
		require.NoError(t, err)
		require.True(t, ok)
		assert.Equal(t, snapshotprotocol.StorageTypePVC, storage.Type)
		assert.Equal(t, "namespace-snapshots", storage.PVCName)
		assert.Equal(t, "/snapshots", storage.BasePath)
	})

	t.Run("pvc config normalizes clean base path", func(t *testing.T) {
		storage, ok, err := StorageFromConfig(configv1alpha1.CheckpointStorageConfiguration{
			Type: snapshotprotocol.StorageTypePVC,
			PVC: configv1alpha1.CheckpointPVCConfig{
				PVCName:  "namespace-snapshots",
				BasePath: "/snapshots//foo/../bar/",
			},
		})
		require.NoError(t, err)
		require.True(t, ok)
		assert.Equal(t, "/snapshots/bar", storage.BasePath)
	})

	t.Run("pvc config rejects relative base path", func(t *testing.T) {
		_, _, err := StorageFromConfig(configv1alpha1.CheckpointStorageConfiguration{
			Type: snapshotprotocol.StorageTypePVC,
			PVC: configv1alpha1.CheckpointPVCConfig{
				PVCName:  "namespace-snapshots",
				BasePath: "snapshots",
			},
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "must be absolute")
	})

	t.Run("pvc config rejects invalid access mode", func(t *testing.T) {
		_, _, err := StorageFromConfig(configv1alpha1.CheckpointStorageConfiguration{
			Type: snapshotprotocol.StorageTypePVC,
			PVC: configv1alpha1.CheckpointPVCConfig{
				PVCName:    "namespace-snapshots",
				BasePath:   "/snapshots",
				Create:     true,
				AccessMode: "RWX",
			},
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "checkpoint.storage.pvc.accessMode")
	})

	t.Run("pre-provisioned pvc config does not validate create-only access mode", func(t *testing.T) {
		storage, ok, err := StorageFromConfig(configv1alpha1.CheckpointStorageConfiguration{
			Type: snapshotprotocol.StorageTypePVC,
			PVC: configv1alpha1.CheckpointPVCConfig{
				PVCName:    "namespace-snapshots",
				BasePath:   "/snapshots",
				Create:     false,
				AccessMode: "RWX",
			},
		})
		require.NoError(t, err)
		require.True(t, ok)
		assert.Equal(t, "namespace-snapshots", storage.PVCName)
	})
}

func TestEnsureStoragePVC(t *testing.T) {
	ctx := context.Background()

	storageConfig := configv1alpha1.CheckpointStorageConfiguration{
		Type: snapshotprotocol.StorageTypePVC,
		PVC: configv1alpha1.CheckpointPVCConfig{
			PVCName:  "namespace-snapshots",
			BasePath: "/snapshots",
		},
	}

	t.Run("empty config is no-op without client", func(t *testing.T) {
		require.NoError(t, EnsureStoragePVC(ctx, nil, testNamespace, configv1alpha1.CheckpointStorageConfiguration{}))
	})

	t.Run("missing existing PVC returns clear error", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(testScheme()).Build()
		err := EnsureStoragePVC(ctx, c, testNamespace, storageConfig)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "checkpoint storage PVC default/namespace-snapshots does not exist")
		assert.Contains(t, err.Error(), "checkpoint.storage.pvc.create is false")
	})

	t.Run("existing PVC is reused", func(t *testing.T) {
		pvc := &corev1.PersistentVolumeClaim{
			ObjectMeta: metav1.ObjectMeta{Name: "namespace-snapshots", Namespace: testNamespace},
		}
		c := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(pvc).Build()
		require.NoError(t, EnsureStoragePVC(ctx, c, testNamespace, storageConfig))
	})

	t.Run("create true creates namespace PVC", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(testScheme()).Build()
		config := storageConfig
		config.PVC.Create = true
		config.PVC.Size = "10Gi"
		config.PVC.StorageClassName = "efs-sc"
		config.PVC.AccessMode = string(corev1.ReadWriteMany)

		require.NoError(t, EnsureStoragePVC(ctx, c, testNamespace, config))

		pvc := &corev1.PersistentVolumeClaim{}
		require.NoError(t, c.Get(ctx, types.NamespacedName{Name: "namespace-snapshots", Namespace: testNamespace}, pvc))
		assert.Equal(t, []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany}, pvc.Spec.AccessModes)
		storageRequest := pvc.Spec.Resources.Requests[corev1.ResourceStorage]
		assert.Equal(t, "10Gi", storageRequest.String())
		require.NotNil(t, pvc.Spec.StorageClassName)
		assert.Equal(t, "efs-sc", *pvc.Spec.StorageClassName)
		require.NotNil(t, pvc.Spec.VolumeMode)
		assert.Equal(t, corev1.PersistentVolumeFilesystem, *pvc.Spec.VolumeMode)
		assert.Equal(t, "checkpoint-storage", pvc.Labels["app.kubernetes.io/component"])
	})

	t.Run("create true defaults to ReadWriteMany and cluster default storage class", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(testScheme()).Build()
		config := storageConfig
		config.PVC.PVCName = "defaulted-snapshots"
		config.PVC.Create = true
		config.PVC.Size = "1Gi"

		require.NoError(t, EnsureStoragePVC(ctx, c, testNamespace, config))

		pvc := &corev1.PersistentVolumeClaim{}
		require.NoError(t, c.Get(ctx, types.NamespacedName{Name: "defaulted-snapshots", Namespace: testNamespace}, pvc))
		assert.Equal(t, []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany}, pvc.Spec.AccessModes)
		assert.Nil(t, pvc.Spec.StorageClassName)
	})

	t.Run("create true requires size", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(testScheme()).Build()
		config := storageConfig
		config.PVC.Create = true

		err := EnsureStoragePVC(ctx, c, testNamespace, config)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "checkpoint.storage.pvc.size is required")
	})

	t.Run("create true rejects non-positive size", func(t *testing.T) {
		for _, size := range []string{"0", "-1Gi"} {
			t.Run(size, func(t *testing.T) {
				c := fake.NewClientBuilder().WithScheme(testScheme()).Build()
				config := storageConfig
				config.PVC.Create = true
				config.PVC.Size = size

				err := EnsureStoragePVC(ctx, c, testNamespace, config)
				require.Error(t, err)
				assert.Contains(t, err.Error(), "must be greater than zero")
			})
		}
	})
}

func TestApplyRestorePodMetadataWithStorageConfig(t *testing.T) {
	labels := map[string]string{}
	annotations := map[string]string{
		snapshotprotocol.CheckpointStorageBasePathAnnotation: "/stale",
	}
	storageConfig := configv1alpha1.CheckpointStorageConfiguration{
		Type: snapshotprotocol.StorageTypePVC,
		PVC: configv1alpha1.CheckpointPVCConfig{
			PVCName:  "namespace-snapshots",
			BasePath: "/snapshots/",
		},
	}

	require.NoError(t, ApplyRestorePodMetadataWithStorageConfig(
		labels,
		annotations,
		&CheckpointInfo{Enabled: true, Ready: true, Hash: testHash},
		storageConfig,
	))

	assert.Equal(t, "true", labels[snapshotprotocol.RestoreTargetLabel])
	assert.Equal(t, testHash, labels[snapshotprotocol.CheckpointIDLabel])
	assert.Equal(t, snapshotprotocol.StorageTypePVC, annotations[snapshotprotocol.CheckpointStorageTypeAnnotation])
	assert.Equal(t, "/snapshots", annotations[snapshotprotocol.CheckpointStorageBasePathAnnotation])

	t.Run("enabled restore requires annotations map", func(t *testing.T) {
		err := ApplyRestorePodMetadataWithStorageConfig(
			map[string]string{},
			nil,
			&CheckpointInfo{Enabled: true, Ready: true, Hash: testHash},
			storageConfig,
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "annotations map is required")
	})

	t.Run("invalid storage config does not mutate metadata", func(t *testing.T) {
		labels := map[string]string{"existing": "label"}
		annotations := map[string]string{
			snapshotprotocol.CheckpointStorageBasePathAnnotation: "/stale",
		}

		err := ApplyRestorePodMetadataWithStorageConfig(
			labels,
			annotations,
			&CheckpointInfo{Enabled: true, Ready: true, Hash: testHash},
			configv1alpha1.CheckpointStorageConfiguration{
				Type: snapshotprotocol.StorageTypePVC,
				PVC: configv1alpha1.CheckpointPVCConfig{
					PVCName:  "namespace-snapshots",
					BasePath: "relative",
				},
			},
		)

		require.Error(t, err)
		assert.Equal(t, map[string]string{"existing": "label"}, labels)
		assert.Equal(t, map[string]string{
			snapshotprotocol.CheckpointStorageBasePathAnnotation: "/stale",
		}, annotations)
	})
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
				snapshotprotocol.CheckpointIDLabel: hash,
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

	ckpt, err := CreateOrGetAutoCheckpoint(ctx, c, testNamespace, identity, corev1.PodTemplateSpec{}, nil)
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

	ckpt, err := CreateOrGetAutoCheckpoint(ctx, c, testNamespace, testIdentity(), corev1.PodTemplateSpec{}, nil)
	require.NoError(t, err)
	require.NotNil(t, ckpt.Annotations)
	assert.Equal(t, snapshotprotocol.DefaultCheckpointArtifactVersion, ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation])
}

func TestCreateOrGetAutoCheckpointRejectsGMSSnapshotWhenGateDisabled(t *testing.T) {
	t.Setenv(consts.DynamoOperatorAllowGMSSnapshotEnvVar, "")
	ctx := context.Background()
	s := testScheme()
	c := fake.NewClientBuilder().WithScheme(s).Build()

	_, err := CreateOrGetAutoCheckpoint(
		ctx,
		c,
		testNamespace,
		testIdentity(),
		corev1.PodTemplateSpec{},
		&nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true},
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "GMS + Snapshot is temporarily disabled")
}

// --- InjectCheckpointIntoPodSpec tests ---

func TestInjectCheckpointIntoPodSpec(t *testing.T) {
	t.Run("not ready checkpoint leaves pod spec untouched", func(t *testing.T) {
		podSpec := testPodSpec()
		originalCmd := append([]string(nil), podSpec.Containers[0].Command...)
		originalArgs := append([]string(nil), podSpec.Containers[0].Args...)
		info := &CheckpointInfo{Enabled: true, Ready: false, Hash: testHash}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build()

		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))

		assert.Equal(t, originalCmd, podSpec.Containers[0].Command)
		assert.Equal(t, originalArgs, podSpec.Containers[0].Args)
		for _, volume := range podSpec.Volumes {
			assert.NotEqual(t, snapshotprotocol.SnapshotControlVolumeName, volume.Name)
			assert.NotEqual(t, snapshotprotocol.CheckpointVolumeName, volume.Name)
			assert.NotEqual(t, consts.PodInfoVolumeName, volume.Name)
		}
		for _, env := range podSpec.Containers[0].Env {
			assert.NotEqual(t, snapshotprotocol.SnapshotControlDirEnv, env.Name)
		}
	})

	t.Run("ready checkpoint injects podinfo and overrides command", func(t *testing.T) {
		podSpec := testPodSpec()
		info := &CheckpointInfo{Enabled: true, Ready: true, Identity: ptr.To(testIdentity())}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build()
		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		assert.Equal(t, []string{"sleep", "infinity"}, podSpec.Containers[0].Command)
		assert.Nil(t, podSpec.Containers[0].Args)
		assert.Len(t, info.Hash, 16)

		volumes := map[string]corev1.Volume{}
		for _, volume := range podSpec.Volumes {
			volumes[volume.Name] = volume
		}
		require.Contains(t, volumes, consts.PodInfoVolumeName)
		require.NotNil(t, volumes[consts.PodInfoVolumeName].DownwardAPI)

		fields := map[string]string{}
		for _, item := range volumes[consts.PodInfoVolumeName].DownwardAPI.Items {
			if item.FieldRef != nil {
				fields[item.Path] = item.FieldRef.FieldPath
			}
		}
		assert.Equal(t, "metadata.labels['"+consts.KubeLabelDynamoNamespace+"']", fields[consts.PodInfoFileDynNamespace])
		assert.Equal(t, "metadata.labels['"+consts.KubeLabelDynamoWorkerHash+"']", fields[consts.PodInfoFileDynNamespaceWorkerSuffix])
		assert.Equal(t, "metadata.labels['"+consts.KubeLabelDynamoComponentType+"']", fields[consts.PodInfoFileDynComponent])
		assert.Equal(t, "metadata.labels['"+consts.KubeLabelDynamoGraphDeploymentName+"']", fields[consts.PodInfoFileDynParentDGDName])
		assert.Equal(t, consts.PodInfoFieldPodNamespace, fields[consts.PodInfoFileDynParentDGDNamespace])

		mountPaths := map[string]string{}
		for _, mount := range podSpec.Containers[0].VolumeMounts {
			mountPaths[mount.Name] = mount.MountPath
		}
		assert.Equal(t, consts.PodInfoMountPath, mountPaths[consts.PodInfoVolumeName])
	})

	t.Run("ready checkpoint targets the container named main", func(t *testing.T) {
		podSpec := &corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "main", Image: "main:latest", Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"}},
				{Name: "sidecar", Image: "sidecar:latest", Command: []string{"sidecar"}, Args: []string{"run"}},
			},
		}
		info := &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build()

		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		assert.Equal(t, []string{"sleep", "infinity"}, podSpec.Containers[0].Command)
		assert.Nil(t, podSpec.Containers[0].Args)
		assert.Equal(t, []string{"sidecar"}, podSpec.Containers[1].Command)
		assert.Equal(t, []string{"run"}, podSpec.Containers[1].Args)
	})

	t.Run("failover targets shape every engine container", func(t *testing.T) {
		podSpec := &corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "engine-0", Image: "main:latest", Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"}},
				{Name: "engine-1", Image: "main:latest", Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"}},
				{Name: "sidecar", Image: "sidecar:latest", Command: []string{"sidecar"}, Args: []string{"run"}},
			},
		}
		info := &CheckpointInfo{
			Enabled:                 true,
			Ready:                   true,
			Hash:                    testHash,
			RestoreTargetContainers: []string{"engine-0", "engine-1"},
		}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build()

		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		for _, name := range []string{"engine-0", "engine-1"} {
			c := findContainer(podSpec, name)
			require.NotNil(t, c, "container %q not found", name)
			assert.Equal(t, []string{"sleep", "infinity"}, c.Command, "engine %s command", name)
			assert.Nil(t, c.Args, "engine %s args", name)
			gotSubPath := ""
			for _, m := range c.VolumeMounts {
				if m.Name == snapshotprotocol.SnapshotControlVolumeName {
					gotSubPath = m.SubPath
				}
			}
			assert.Equal(t, name, gotSubPath, "engine %s control-volume subPath", name)
		}
		sidecar := findContainer(podSpec, "sidecar")
		require.NotNil(t, sidecar)
		assert.Equal(t, []string{"sidecar"}, sidecar.Command, "sidecar must not be rewritten")
	})

	t.Run("ready checkpoint uses configured PVC storage without daemonset discovery", func(t *testing.T) {
		podSpec := testPodSpec()
		info := &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash}
		pvc := &corev1.PersistentVolumeClaim{
			ObjectMeta: metav1.ObjectMeta{Name: "namespace-snapshots", Namespace: testNamespace},
		}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(pvc).Build()
		storageConfig := configv1alpha1.CheckpointStorageConfiguration{
			Type: snapshotprotocol.StorageTypePVC,
			PVC: configv1alpha1.CheckpointPVCConfig{
				PVCName:  "namespace-snapshots",
				BasePath: "/snapshots",
			},
		}

		require.NoError(t, InjectCheckpointIntoPodSpecWithStorageConfig(
			context.Background(),
			reader,
			testNamespace,
			podSpec,
			info,
			storageConfig,
			snapshotprotocol.DefaultSeccompLocalhostProfile,
		))

		volumes := map[string]corev1.Volume{}
		for _, volume := range podSpec.Volumes {
			volumes[volume.Name] = volume
		}
		require.Contains(t, volumes, snapshotprotocol.CheckpointVolumeName)
		require.NotNil(t, volumes[snapshotprotocol.CheckpointVolumeName].PersistentVolumeClaim)
		assert.Equal(t, "namespace-snapshots", volumes[snapshotprotocol.CheckpointVolumeName].PersistentVolumeClaim.ClaimName)

		mounts := map[string]string{}
		for _, mount := range podSpec.Containers[0].VolumeMounts {
			mounts[mount.Name] = mount.MountPath
		}
		assert.Equal(t, "/snapshots", mounts[snapshotprotocol.CheckpointVolumeName])
	})

	t.Run("ready gms checkpoint injects restore sidecars and loader mount", func(t *testing.T) {
		podSpec := testPodSpec()
		podSpec.Containers[0].Resources.Claims = []corev1.ResourceClaim{{Name: "gpu"}}
		info := &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash, GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true}}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build()

		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		gmsServer := findContainer(podSpec, gms.ServerContainerName)
		require.NotNil(t, gmsServer)
		loader := findContainer(podSpec, GMSLoaderContainer)
		require.NotNil(t, loader)
		serverCount := 0
		loaderCount := 0
		for _, container := range podSpec.InitContainers {
			switch container.Name {
			case gms.ServerContainerName:
				serverCount++
			case GMSLoaderContainer:
				loaderCount++
			}
		}
		assert.Equal(t, 1, serverCount)
		assert.Equal(t, 1, loaderCount)

		// Restore: server and loader are init sidecars (restartPolicy=Always)
		assert.NotNil(t, gmsServer.RestartPolicy, "restore gms-server should have RestartPolicy")
		assert.Equal(t, corev1.ContainerRestartPolicyAlways, *gmsServer.RestartPolicy)
		assert.Nil(t, gmsServer.StartupProbe, "restore gms-server should not have StartupProbe")
		assert.NotNil(t, loader.RestartPolicy, "restore gms-loader should have RestartPolicy")
		assert.Equal(t, corev1.ContainerRestartPolicyAlways, *loader.RestartPolicy)

		mounts := map[string]string{}
		for _, mount := range loader.VolumeMounts {
			mounts[mount.Name] = mount.MountPath
		}
		assert.Equal(t, "/checkpoints", mounts[snapshotprotocol.CheckpointVolumeName])
		assert.Equal(t, gms.SharedMountPath, mounts[gms.SharedVolumeName])

		env := map[string]string{}
		for _, item := range loader.Env {
			env[item.Name] = item.Value
		}
		assert.Equal(t, "/checkpoints/gms/"+testHash+"/versions/1", env["GMS_CHECKPOINT_DIR"])
		assert.Equal(t, []string{"python3", "-m", "gpu_memory_service.cli.server"}, gmsServer.Command)
		assert.Equal(t, []string{"python3", "-m", "gpu_memory_service.cli.snapshot.loader"}, loader.Command)
	})

	t.Run("error cases", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			podSpec *corev1.PodSpec
			info    *CheckpointInfo
			reader  client.Reader
			errMsg  string
		}{
			{"hash empty and identity nil", testPodSpec(), &CheckpointInfo{Enabled: true, Ready: true}, fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build(), "identity is nil"},
			{"no containers", &corev1.PodSpec{}, testInfo(), fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build(), "restore target container"},
			{"snapshot daemonset missing", testPodSpec(), testInfo(), fake.NewClientBuilder().WithScheme(testScheme()).Build(), "no snapshot-agent daemonset found"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				err := InjectCheckpointIntoPodSpec(context.Background(), tc.reader, testNamespace, tc.podSpec, tc.info, snapshotprotocol.DefaultSeccompLocalhostProfile)
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errMsg)
			})
		}
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
		t.Setenv(consts.DynamoOperatorAllowGMSSnapshotEnvVar, "1")
		hash, err := ComputeIdentityHash(testIdentity())
		require.NoError(t, err)
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: hash, Namespace: testNamespace},
			Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
				Identity:         testIdentity(),
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true},
			},
			Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
				Phase:        nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
				IdentityHash: hash,
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
		assert.Equal(t, hash, info.CheckpointName)
		require.NotNil(t, info.GPUMemoryService)
		assert.True(t, info.GPUMemoryService.Enabled)
	})

	t.Run("checkpointRef rejects GMS checkpoint when gate is disabled", func(t *testing.T) {
		t.Setenv(consts.DynamoOperatorAllowGMSSnapshotEnvVar, "")
		hash, err := ComputeIdentityHash(testIdentity())
		require.NoError(t, err)
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: hash, Namespace: testNamespace},
			Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
				Identity:         testIdentity(),
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true},
			},
			Status: nvidiacomv1alpha1.DynamoCheckpointStatus{
				Phase:        nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
				IdentityHash: hash,
			},
		}
		c := fake.NewClientBuilder().WithScheme(s).WithObjects(ckpt).WithStatusSubresource(ckpt).Build()
		ref := hash

		_, err = ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true, CheckpointRef: &ref,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "GMS + Snapshot is temporarily disabled")
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

// --- ApplyRestorePodMetadata target-containers annotation ---

func TestApplyRestorePodMetadata_DefaultsToMainContainer(t *testing.T) {
	labels := map[string]string{}
	annotations := map[string]string{}
	ApplyRestorePodMetadata(labels, annotations, &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash})
	assert.Equal(t, consts.MainContainerName, annotations[snapshotprotocol.TargetContainersAnnotation])
}

func TestApplyRestorePodMetadata_FailoverTargets(t *testing.T) {
	labels := map[string]string{}
	annotations := map[string]string{}
	ApplyRestorePodMetadata(labels, annotations, &CheckpointInfo{
		Enabled:                 true,
		Ready:                   true,
		Hash:                    testHash,
		RestoreTargetContainers: []string{"engine-0", "engine-1"},
	})
	assert.Equal(t, "engine-0,engine-1", annotations[snapshotprotocol.TargetContainersAnnotation])
}

func TestApplyRestorePodMetadata_DisabledClearsAnnotation(t *testing.T) {
	labels := map[string]string{}
	annotations := map[string]string{
		snapshotprotocol.TargetContainersAnnotation: "stale",
	}
	ApplyRestorePodMetadata(labels, annotations, &CheckpointInfo{Enabled: false})
	_, ok := annotations[snapshotprotocol.TargetContainersAnnotation]
	assert.False(t, ok, "target-containers annotation must be cleared when checkpoint disabled")
}

// findContainer is a test helper that locates a container by name across both
// regular containers and init containers.
func findContainer(podSpec *corev1.PodSpec, name string) *corev1.Container {
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == name {
			return &podSpec.Containers[i]
		}
	}
	for i := range podSpec.InitContainers {
		if podSpec.InitContainers[i].Name == name {
			return &podSpec.InitContainers[i]
		}
	}
	return nil
}
