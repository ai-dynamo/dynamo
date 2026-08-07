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
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
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

func assertRestoreStandbyMode(
	t *testing.T,
	container *corev1.Container,
	command []string,
	args []string,
) {
	t.Helper()
	assert.Equal(t, command, container.Command)
	assert.Equal(t, args, container.Args)

	found := false
	for _, env := range container.Env {
		if env.Name == snapshotprotocol.RestoreStandbyModeEnv {
			found = true
			assert.Equal(t, "1", env.Value)
		}
	}
	assert.True(t, found, "restore standby mode env should be injected")
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

func mustExpectedAutoCheckpoint(
	t *testing.T,
	scheme *runtime.Scheme,
	identity nvidiacomv1alpha1.DynamoCheckpointIdentity,
	podTemplate corev1.PodTemplateSpec,
	targetContainerName string,
	deletionPolicy nvidiacomv1alpha1.CheckpointDeletionPolicy,
	gpuMemoryService *nvidiacomv1alpha1.GPUMemoryServiceSpec,
	owner client.Object,
) *nvidiacomv1alpha1.DynamoCheckpoint {
	t.Helper()
	expected, err := ExpectedAutoCheckpoint(
		scheme,
		testNamespace,
		testHash,
		identity,
		podTemplate,
		targetContainerName,
		deletionPolicy,
		gpuMemoryService,
		owner,
	)
	require.NoError(t, err)
	return expected
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

func TestCreateOrGetAutoCheckpointDoesNotReuseDifferentCheckpointWithSameLegacyHash(t *testing.T) {
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

	expected := mustExpectedAutoCheckpoint(t, s, identity, corev1.PodTemplateSpec{}, "", "", nil, nil)
	ckpt, err := CreateOrGetAutoCheckpoint(ctx, c, expected)
	require.NoError(t, err)
	assert.Equal(t, "checkpoint-"+testHash, ckpt.Name)

	list := &nvidiacomv1alpha1.DynamoCheckpointList{}
	require.NoError(t, baseClient.List(ctx, list))
	require.Len(t, list.Items, 2)
}

func TestCreateOrGetAutoCheckpointSetsDefaultArtifactVersion(t *testing.T) {
	ctx := context.Background()
	s := testScheme()
	c := fake.NewClientBuilder().WithScheme(s).Build()

	expected := mustExpectedAutoCheckpoint(t, s, testIdentity(), corev1.PodTemplateSpec{}, "", "", nil, nil)
	ckpt, err := CreateOrGetAutoCheckpoint(ctx, c, expected)
	require.NoError(t, err)
	require.NotNil(t, ckpt.Annotations)
	assert.Equal(t, snapshotprotocol.DefaultCheckpointArtifactVersion, ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation])
	assert.Equal(t, "true", ckpt.Annotations[consts.CheckpointAutoAnnotation])
	assert.Equal(t, string(nvidiacomv1alpha1.CheckpointDeletionPolicyDelete), ckpt.Annotations[consts.CheckpointDeletionPolicyAnnotation])
	assert.Equal(t, testHash, ckpt.Labels[snapshotprotocol.CheckpointIDLabel])
	assert.True(t, commonController.ContainsFinalizer(ckpt))

	stored := &nvidiacomv1alpha1.DynamoCheckpoint{}
	require.NoError(t, c.Get(ctx, types.NamespacedName{Name: ckpt.Name, Namespace: ckpt.Namespace}, stored))
	assert.True(t, commonController.ContainsFinalizer(stored))
}

func TestCreateOrGetAutoCheckpointAcceptsGMSCheckpoint(t *testing.T) {
	ctx := context.Background()
	s := testScheme()
	c := fake.NewClientBuilder().WithScheme(s).Build()

	expected := mustExpectedAutoCheckpoint(
		t, s, testIdentity(), corev1.PodTemplateSpec{}, "", "",
		&nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true}, nil,
	)
	ckpt, err := CreateOrGetAutoCheckpoint(ctx, c, expected)
	require.NoError(t, err)
	require.NotNil(t, ckpt.Spec.GPUMemoryService)
	assert.True(t, ckpt.Spec.GPUMemoryService.Enabled)
}

func TestCreateOrGetAutoCheckpointValidatesCaptureSpec(t *testing.T) {
	ctx := context.Background()
	s := testScheme()
	owner := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: testNamespace,
			UID:       types.UID("dgd-uid"),
		},
	}
	podTemplate := corev1.PodTemplateSpec{Spec: corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:    consts.MainContainerName,
			Image:   "worker:expected",
			Command: []string{"python3"},
			Args:    []string{"-m", "dynamo.vllm", "--load-format", "gms"},
			Ports:   []corev1.ContainerPort{{Name: "system", ContainerPort: 9090}},
			LivenessProbe: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					Exec: &corev1.ExecAction{Command: []string{"test", "-f", "/ready"}},
				},
			},
		}},
	}}
	gmsSpec := &nvidiacomv1alpha1.GPUMemoryServiceSpec{
		Enabled: true,
		Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
	}

	for _, tt := range []struct {
		name    string
		mutate  func(*nvidiacomv1alpha1.DynamoCheckpoint)
		wantErr string
	}{
		{
			name: "poisoned image and command",
			mutate: func(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) {
				main := &ckpt.Spec.Job.PodTemplateSpec.Spec.Containers[0]
				main.Image = "worker:poisoned"
				main.Command = []string{"steal-checkpoint"}
			},
			wantErr: "checkpoint capture spec differs",
		},
		{
			name: "poisoned executable probe",
			mutate: func(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) {
				ckpt.Spec.Job.PodTemplateSpec.Spec.Containers[0].
					LivenessProbe.Exec.Command = []string{"steal-checkpoint"}
			},
			wantErr: "checkpoint capture spec differs",
		},
		{
			name: "poisoned GMS spec",
			mutate: func(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) {
				ckpt.Spec.GPUMemoryService.DeviceClassName = "poisoned.example/gpu"
			},
			wantErr: "checkpoint capture spec differs",
		},
		{
			name: "status and omitted API defaults are harmless",
			mutate: func(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) {
				ckpt.Spec.Identity.TensorParallelSize = 0
				ckpt.Spec.Identity.PipelineParallelSize = 0
				ckpt.Spec.Job.TargetContainerName = ""
				ckpt.Spec.Job.ActiveDeadlineSeconds = nil
				ckpt.Spec.Job.PodTemplateSpec.Spec.Containers[0].Ports[0].Protocol = ""
				ckpt.Spec.GPUMemoryService.Mode = ""
				ckpt.Spec.GPUMemoryService.DeviceClassName = ""
				ckpt.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseReady
				ckpt.Status.Message = "observed status is not capture provenance"
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			expected := mustExpectedAutoCheckpoint(
				t, s, testIdentity(), podTemplate, consts.MainContainerName,
				nvidiacomv1alpha1.CheckpointDeletionPolicyDelete, gmsSpec, owner,
			)
			existing := expected.DeepCopy()
			tt.mutate(existing)
			c := fake.NewClientBuilder().WithScheme(s).WithObjects(existing).Build()

			_, err := CreateOrGetAutoCheckpoint(ctx, c, expected)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestCreateOrGetAutoCheckpointRetainStoresDeletionPolicy(t *testing.T) {
	ctx := context.Background()
	s := testScheme()
	owner := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: testNamespace,
			UID:       types.UID("dgd-uid"),
		},
	}
	c := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()

	expected := mustExpectedAutoCheckpoint(
		t, s, testIdentity(), corev1.PodTemplateSpec{}, "",
		nvidiacomv1alpha1.CheckpointDeletionPolicyRetain, nil, owner,
	)
	ckpt, err := CreateOrGetAutoCheckpoint(ctx, c, expected)
	require.NoError(t, err)

	assert.Empty(t, ckpt.OwnerReferences)
	assert.Equal(t, string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain), ckpt.Annotations[consts.CheckpointDeletionPolicyAnnotation])
}

func TestCreateOrGetAutoCheckpointUpdatesExistingDeletionPolicyAndFinalizer(t *testing.T) {
	ctx := context.Background()
	s := testScheme()
	owner := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: testNamespace,
			UID:       types.UID("dgd-uid"),
		},
	}
	expected := mustExpectedAutoCheckpoint(
		t, s, testIdentity(), corev1.PodTemplateSpec{}, "",
		nvidiacomv1alpha1.CheckpointDeletionPolicyDelete, nil, owner,
	)
	existing := expected.DeepCopy()
	existing.Finalizers = nil
	existing.OwnerReferences = nil
	delete(existing.Annotations, consts.CheckpointDeletionPolicyAnnotation)
	c := fake.NewClientBuilder().WithScheme(s).WithObjects(owner, existing).Build()

	ckpt, err := CreateOrGetAutoCheckpoint(ctx, c, expected)
	require.NoError(t, err)
	assert.Equal(t, string(nvidiacomv1alpha1.CheckpointDeletionPolicyDelete), ckpt.Annotations[consts.CheckpointDeletionPolicyAnnotation])
	assert.True(t, commonController.ContainsFinalizer(ckpt))
	require.Len(t, ckpt.OwnerReferences, 1)
	assert.Equal(t, owner.UID, ckpt.OwnerReferences[0].UID)
}

func TestCreateOrGetAutoCheckpointAdoptionUsesOptimisticLock(t *testing.T) {
	ctx := context.Background()
	s := testScheme()
	owner := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Name: "test-dgd", Namespace: testNamespace, UID: types.UID("dgd-uid"),
	}}
	expected := mustExpectedAutoCheckpoint(
		t, s, testIdentity(), corev1.PodTemplateSpec{}, "",
		nvidiacomv1alpha1.CheckpointDeletionPolicyDelete, nil, owner,
	)
	existing := expected.DeepCopy()
	existing.Finalizers = nil
	existing.OwnerReferences = nil
	delete(existing.Annotations, consts.CheckpointDeletionPolicyAnnotation)

	mutated := false
	c := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(owner, existing).
		WithInterceptorFuncs(interceptor.Funcs{
			Patch: func(
				ctx context.Context,
				c client.WithWatch,
				obj client.Object,
				patch client.Patch,
				opts ...client.PatchOption,
			) error {
				if !mutated {
					mutated = true
					current := &nvidiacomv1alpha1.DynamoCheckpoint{}
					require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(obj), current))
					current.Spec.Job.TargetContainerName = "changed-after-verification"
					require.NoError(t, c.Update(ctx, current))
				}
				return c.Patch(ctx, obj, patch, opts...)
			},
		}).
		Build()

	_, err := CreateOrGetAutoCheckpoint(ctx, c, expected)
	require.Error(t, err)
	assert.True(t, apierrors.IsConflict(err), "expected optimistic-lock conflict, got %v", err)

	stored := &nvidiacomv1alpha1.DynamoCheckpoint{}
	require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(expected), stored))
	assert.Equal(t, "changed-after-verification", stored.Spec.Job.TargetContainerName)
	assert.Empty(t, stored.Finalizers)
	assert.Empty(t, stored.OwnerReferences)
	assert.Empty(t, stored.Annotations[consts.CheckpointDeletionPolicyAnnotation])
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
		}
		for _, env := range podSpec.Containers[0].Env {
			assert.NotEqual(t, snapshotprotocol.SnapshotControlDirEnv, env.Name)
		}
	})

	t.Run("ready checkpoint enables restore standby mode", func(t *testing.T) {
		podSpec := testPodSpec()
		info := &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build()
		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		assertRestoreStandbyMode(t, &podSpec.Containers[0], []string{"python3"}, []string{"-m", "dynamo.vllm"})

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
		assertRestoreStandbyMode(t, &podSpec.Containers[0], []string{"python3"}, []string{"-m", "dynamo.vllm"})
		assert.Equal(t, []string{"sidecar"}, podSpec.Containers[1].Command)
		assert.Equal(t, []string{"run"}, podSpec.Containers[1].Args)
	})

	t.Run("paused restore shapes only named target containers", func(t *testing.T) {
		podSpec := &corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "engine-0", Image: "main:latest", Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"}},
				{Name: "engine-1", Image: "main:latest", Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"}},
				{Name: "engine-2", Image: "main:latest", Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"}},
				{Name: "gms-server", Image: "gms:latest", Command: []string{"gms-server"}, Args: []string{"--serve"}, Env: []corev1.EnvVar{{Name: "GMS", Value: "server"}}},
				{Name: "gms-loader", Image: "gms:latest", Command: []string{"gms-loader"}, Args: []string{"--load"}, Env: []corev1.EnvVar{{Name: "GMS", Value: "loader"}}},
				{Name: "user-helper", Image: "helper:latest", Command: []string{"helper"}, Args: []string{"run"}, Env: []corev1.EnvVar{{Name: "HELPER", Value: "true"}}},
			},
		}
		nonTargets := make(map[string]corev1.Container)
		for _, name := range []string{"gms-server", "gms-loader", "user-helper"} {
			nonTargets[name] = *findContainer(podSpec, name).DeepCopy()
		}
		info := &CheckpointInfo{
			Enabled: true,
			Ready:   true,
			Hash:    testHash,
			GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
				Enabled: true,
				Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
			},
			RestoreTargetContainers: []string{"engine-0", "engine-1", "engine-2"},
			RestorePaused:           true,
		}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build()

		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		for _, name := range info.RestoreTargetContainers {
			c := findContainer(podSpec, name)
			require.NotNil(t, c, "container %q not found", name)
			assertRestoreStandbyMode(t, c, []string{"python3"}, []string{"-m", "dynamo.vllm"})
			assert.Contains(t, c.Env, corev1.EnvVar{Name: snapshotRestorePausedEnv, Value: "1"})
			gotSubPath := ""
			for _, m := range c.VolumeMounts {
				if m.Name == snapshotprotocol.SnapshotControlVolumeName {
					gotSubPath = m.SubPath
				}
			}
			assert.Equal(t, name, gotSubPath, "engine %s control-volume subPath", name)
		}
		for name, before := range nonTargets {
			after := findContainer(podSpec, name)
			require.NotNil(t, after)
			assert.True(t, equality.Semantic.DeepEqual(before, *after), "%s must cold-start unchanged", name)
		}
		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		count := 0
		for i := range podSpec.InitContainers {
			if podSpec.InitContainers[i].Name == gms.ServerContainerName {
				count++
			}
		}
		assert.Equal(t, 1, count)
	})

	t.Run("paused restore rejects unsupported GMS mode", func(t *testing.T) {
		podSpec := testPodSpec()
		info := &CheckpointInfo{
			Enabled:       true,
			Ready:         true,
			Hash:          testHash,
			RestorePaused: true,
			GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
				Enabled: true,
				Mode:    nvidiacomv1alpha1.GMSModeInterPod,
			},
		}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build()

		require.ErrorContains(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile), `mode "interPod" is not implemented`)
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

		restore, err := ResolvePodSpecRestore(
			context.Background(),
			reader,
			testNamespace,
			info,
			storageConfig,
		)
		require.NoError(t, err)
		require.NotNil(t, restore)
		require.NoError(t, InjectResolvedCheckpointIntoPodSpec(
			podSpec,
			restore,
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

	t.Run("ready gms checkpoint wires declared restore client", func(t *testing.T) {
		podSpec := testPodSpec()
		podSpec.Containers[0].Resources.Claims = []corev1.ResourceClaim{{Name: "gpu"}}
		podSpec.Containers = append(podSpec.Containers, corev1.Container{Name: "gms-loader", Image: "loader:latest"})
		info := &CheckpointInfo{
			Enabled: true,
			Ready:   true,
			Hash:    testHash,
			GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
				Enabled:               true,
				ExtraClientContainers: []string{"gms-loader"},
			},
		}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build()

		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		gmsServer := findContainer(podSpec, gms.ServerContainerName)
		require.NotNil(t, gmsServer, "gms-server is a native sidecar (init+restartPolicy=Always)")
		loader := findContainer(podSpec, "gms-loader")
		require.NotNil(t, loader, "gms-loader is a regular container")
		serverInitCount := 0
		for _, container := range podSpec.InitContainers {
			if container.Name == gms.ServerContainerName {
				serverInitCount++
			}
		}
		loaderCount := 0
		for _, container := range podSpec.Containers {
			if container.Name == "gms-loader" {
				loaderCount++
			}
		}
		assert.Equal(t, 1, serverInitCount, "injection is idempotent for server")
		assert.Equal(t, 1, loaderCount, "injection is idempotent for loader")

		assert.Equal(t, corev1.ContainerRestartPolicyAlways, *gmsServer.RestartPolicy)
		assert.Nil(t, gmsServer.StartupProbe, "no StartupProbe — clients drive readiness via connect-retry")
		assert.Nil(t, loader.RestartPolicy, "loader is a regular container; pod RestartPolicy applies")

		mounts := map[string]string{}
		for _, mount := range loader.VolumeMounts {
			mounts[mount.Name] = mount.MountPath
		}
		assert.Empty(t, mounts[snapshotprotocol.CheckpointVolumeName])
		assert.Equal(t, gms.SharedMountPath, mounts[gms.SharedVolumeName])

		assert.Equal(t, []string{"python3", "-m", "gpu_memory_service.cli.server"}, gmsServer.Command)
		assert.Empty(t, loader.Command)
	})

	t.Run("error cases", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			podSpec *corev1.PodSpec
			info    *CheckpointInfo
			reader  client.Reader
			errMsg  string
		}{
			{"ready checkpoint without hash", testPodSpec(), &CheckpointInfo{Enabled: true, Ready: true}, fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build(), "checkpoint is ready but hash is not set"},
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

	t.Run("deprecated Manual value without checkpointRef is ignored", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(s).Build()
		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{
			Enabled: true,
			Mode:    nvidiacomv1alpha1.CheckpointModeManual,
		})
		require.NoError(t, err)
		assert.True(t, info.Enabled)
		assert.False(t, info.Exists)
	})

	t.Run("config without ref or identity resolves enabled without error", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(s).Build()
		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{Enabled: true})
		require.NoError(t, err)
		assert.True(t, info.Enabled)
		assert.False(t, info.Exists)
	})

	t.Run("checkpointRef resolves ready CR", func(t *testing.T) {
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

	t.Run("enabled without ref or identity waits for auto-created checkpoint", func(t *testing.T) {
		c := fake.NewClientBuilder().WithScheme(s).Build()
		info, err := ResolveCheckpointForService(ctx, c, testNamespace, &nvidiacomv1alpha1.ServiceCheckpointConfig{Enabled: true})
		require.NoError(t, err)
		assert.True(t, info.Enabled)
		assert.False(t, info.Exists)
		assert.False(t, info.Ready)
		assert.Equal(t, nvidiacomv1alpha1.CheckpointStartupPolicyImmediate, info.StartupPolicy)
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

func TestApplyRestoreCandidateMetadata(t *testing.T) {
	t.Run("ready checkpoint stamps candidate metadata without restore labels", func(t *testing.T) {
		labels := map[string]string{
			snapshotprotocol.CheckpointIDLabel: "stale",
		}
		annotations := map[string]string{
			snapshotprotocol.CheckpointArtifactVersionAnnotation: "stale",
		}

		err := ApplyRestoreCandidateMetadata(labels, annotations, &CheckpointInfo{
			Enabled:                 true,
			Exists:                  true,
			Ready:                   true,
			CheckpointName:          "worker-checkpoint",
			StartupPolicy:           nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint,
			RestoreTargetContainers: []string{"engine-0", "engine-1"},
		})
		require.NoError(t, err)

		assert.Empty(t, labels[snapshotprotocol.CheckpointIDLabel])
		assert.Empty(t, labels[snapshotprotocol.RestoreTargetLabel])
		assert.Empty(t, annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation])
		assert.Equal(t, consts.KubeLabelValueTrue, annotations[consts.CheckpointRestoreCandidateAnnotation])
		assert.Equal(t, "worker-checkpoint", annotations[consts.CheckpointNameAnnotation])
		assert.Equal(t, string(nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint), annotations[consts.CheckpointStartupPolicyAnnotation])
		assert.Equal(t, "engine-0,engine-1", annotations[snapshotprotocol.TargetContainersAnnotation])
	})

	t.Run("disabled clears stale candidate metadata", func(t *testing.T) {
		labels := map[string]string{
			snapshotprotocol.CheckpointIDLabel: "stale",
		}
		annotations := map[string]string{
			consts.CheckpointRestoreCandidateAnnotation: consts.KubeLabelValueTrue,
			consts.CheckpointNameAnnotation:             "stale",
			consts.CheckpointStartupPolicyAnnotation:    string(nvidiacomv1alpha1.CheckpointStartupPolicyImmediate),
			snapshotprotocol.TargetContainersAnnotation: consts.MainContainerName,
		}

		err := ApplyRestoreCandidateMetadata(labels, annotations, &CheckpointInfo{Enabled: false})
		require.NoError(t, err)

		assert.Empty(t, labels[snapshotprotocol.CheckpointIDLabel])
		assert.NotContains(t, annotations, consts.CheckpointRestoreCandidateAnnotation)
		assert.NotContains(t, annotations, consts.CheckpointNameAnnotation)
		assert.NotContains(t, annotations, consts.CheckpointStartupPolicyAnnotation)
		assert.NotContains(t, annotations, snapshotprotocol.TargetContainersAnnotation)
	})
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
