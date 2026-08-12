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
	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
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
	return s
}

func testInfo() *CheckpointInfo {
	return &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash}
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

	ckpt, err := CreateOrGetAutoCheckpoint(ctx, c, testNamespace, testHash, identity, corev1.PodTemplateSpec{}, "", "", nil, nil)
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

	ckpt, err := CreateOrGetAutoCheckpoint(ctx, c, testNamespace, testHash, testIdentity(), corev1.PodTemplateSpec{}, "", "", nil, nil)
	require.NoError(t, err)
	require.NotNil(t, ckpt.Annotations)
	assert.Equal(t, snapshotprotocol.DefaultCheckpointArtifactVersion, ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation])
	assert.Equal(t, "true", ckpt.Annotations[consts.CheckpointAutoAnnotation])
	assert.Equal(t, string(nvidiacomv1alpha1.CheckpointDeletionPolicyDelete), ckpt.Annotations[consts.CheckpointDeletionPolicyAnnotation])
	assert.Equal(t, testHash, ckpt.Labels[snapshotprotocol.CheckpointIDLabel])
	assert.Empty(t, ckpt.Finalizers)

	stored := &nvidiacomv1alpha1.DynamoCheckpoint{}
	require.NoError(t, c.Get(ctx, types.NamespacedName{Name: ckpt.Name, Namespace: ckpt.Namespace}, stored))
	assert.Empty(t, stored.Finalizers)
}

func TestCreateOrGetAutoCheckpointAcceptsGMSCheckpoint(t *testing.T) {
	ctx := context.Background()
	s := testScheme()
	c := fake.NewClientBuilder().WithScheme(s).Build()

	ckpt, err := CreateOrGetAutoCheckpoint(
		ctx,
		c,
		testNamespace,
		testHash,
		testIdentity(),
		corev1.PodTemplateSpec{},
		"",
		"",
		&nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true},
		nil,
	)
	require.NoError(t, err)
	require.NotNil(t, ckpt.Spec.GPUMemoryService)
	assert.True(t, ckpt.Spec.GPUMemoryService.Enabled)
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

	ckpt, err := CreateOrGetAutoCheckpoint(
		ctx,
		c,
		testNamespace,
		testHash,
		testIdentity(),
		corev1.PodTemplateSpec{},
		"",
		nvidiacomv1alpha1.CheckpointDeletionPolicyRetain,
		nil,
		owner,
	)
	require.NoError(t, err)

	assert.Empty(t, ckpt.OwnerReferences)
	assert.Equal(t, string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain), ckpt.Annotations[consts.CheckpointDeletionPolicyAnnotation])
}

func TestCreateOrGetAutoCheckpointUpdatesExistingDeletionPolicyAndOwner(t *testing.T) {
	ctx := context.Background()
	s := testScheme()
	owner := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: testNamespace,
			UID:       types.UID("dgd-uid"),
		},
	}
	existing := &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "checkpoint-" + testHash,
			Namespace: testNamespace,
			Labels: map[string]string{
				snapshotprotocol.CheckpointIDLabel: testHash,
			},
			Annotations: map[string]string{
				consts.CheckpointAutoAnnotation: consts.KubeLabelValueTrue,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Identity: testIdentity(),
		},
	}
	c := fake.NewClientBuilder().WithScheme(s).WithObjects(owner, existing).Build()

	ckpt, err := CreateOrGetAutoCheckpoint(
		ctx,
		c,
		testNamespace,
		testHash,
		testIdentity(),
		corev1.PodTemplateSpec{},
		"",
		nvidiacomv1alpha1.CheckpointDeletionPolicyDelete,
		nil,
		owner,
	)
	require.NoError(t, err)
	assert.Equal(t, string(nvidiacomv1alpha1.CheckpointDeletionPolicyDelete), ckpt.Annotations[consts.CheckpointDeletionPolicyAnnotation])
	assert.Empty(t, ckpt.Finalizers)
	require.Len(t, ckpt.OwnerReferences, 1)
	assert.Equal(t, owner.UID, ckpt.OwnerReferences[0].UID)
}

// --- InjectCheckpointIntoPodSpec tests ---

func TestInjectCheckpointIntoPodSpec(t *testing.T) {
	const removedCheckpointVolumeName = "checkpoint-storage"

	t.Run("not ready checkpoint leaves pod spec untouched", func(t *testing.T) {
		podSpec := testPodSpec()
		originalCmd := append([]string(nil), podSpec.Containers[0].Command...)
		originalArgs := append([]string(nil), podSpec.Containers[0].Args...)
		info := &CheckpointInfo{Enabled: true, Ready: false, Hash: testHash}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).Build()

		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))

		assert.Equal(t, originalCmd, podSpec.Containers[0].Command)
		assert.Equal(t, originalArgs, podSpec.Containers[0].Args)
		for _, volume := range podSpec.Volumes {
			assert.NotEqual(t, snapshotprotocol.SnapshotControlVolumeName, volume.Name)
			assert.NotEqual(t, removedCheckpointVolumeName, volume.Name)
		}
		for _, env := range podSpec.Containers[0].Env {
			assert.NotEqual(t, snapshotprotocol.SnapshotControlDirEnv, env.Name)
		}
	})

	t.Run("ready checkpoint enables restore standby mode", func(t *testing.T) {
		podSpec := testPodSpec()
		info := &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).Build()
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
		reader := fake.NewClientBuilder().WithScheme(testScheme()).Build()

		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		assertRestoreStandbyMode(t, &podSpec.Containers[0], []string{"python3"}, []string{"-m", "dynamo.vllm"})
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
		reader := fake.NewClientBuilder().WithScheme(testScheme()).Build()

		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info, snapshotprotocol.DefaultSeccompLocalhostProfile))
		for _, name := range []string{"engine-0", "engine-1"} {
			c := findContainer(podSpec, name)
			require.NotNil(t, c, "container %q not found", name)
			assertRestoreStandbyMode(t, c, []string{"python3"}, []string{"-m", "dynamo.vllm"})
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

	t.Run("ready checkpoint preserves unrelated storage without checkpoint PVC injection", func(t *testing.T) {
		podSpec := testPodSpec()
		podSpec.Volumes = []corev1.Volume{{
			Name: "model-cache",
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "model-cache"},
			},
		}}
		podSpec.Containers[0].VolumeMounts = []corev1.VolumeMount{{Name: "model-cache", MountPath: "/models"}}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).Build()

		t.Log("Inject restore control wiring without consulting snapshot-agent storage")
		require.NoError(t, InjectCheckpointIntoPodSpec(
			context.Background(),
			reader,
			testNamespace,
			podSpec,
			&CheckpointInfo{Enabled: true, Ready: true, Hash: testHash},
			snapshotprotocol.DefaultSeccompLocalhostProfile,
		))

		t.Log("Verify unrelated storage remains and the removed checkpoint storage is absent")
		require.Len(t, podSpec.Volumes, 2)
		assert.Equal(t, "model-cache", podSpec.Volumes[0].Name)
		assert.Equal(t, "model-cache", podSpec.Volumes[0].PersistentVolumeClaim.ClaimName)
		assert.Equal(t, "model-cache", podSpec.Containers[0].VolumeMounts[0].Name)
		assert.Equal(t, "/models", podSpec.Containers[0].VolumeMounts[0].MountPath)
		for _, volume := range podSpec.Volumes {
			assert.NotEqual(t, removedCheckpointVolumeName, volume.Name)
		}
		for _, mount := range podSpec.Containers[0].VolumeMounts {
			assert.NotEqual(t, removedCheckpointVolumeName, mount.Name)
		}
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
		reader := fake.NewClientBuilder().WithScheme(testScheme()).Build()

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
		assert.Empty(t, mounts[removedCheckpointVolumeName])
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
			{"ready checkpoint without hash", testPodSpec(), &CheckpointInfo{Enabled: true, Ready: true}, fake.NewClientBuilder().WithScheme(testScheme()).Build(), "checkpoint is ready but hash is not set"},
			{"no containers", &corev1.PodSpec{}, testInfo(), fake.NewClientBuilder().WithScheme(testScheme()).Build(), "restore target container"},
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

	t.Log("Apply restore identity and target-container metadata")
	ApplyRestorePodMetadata(labels, annotations, &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash})

	t.Log("Verify operator-owned restore metadata contains no storage location")
	assert.Equal(t, consts.MainContainerName, annotations[snapshotprotocol.TargetContainersAnnotation])
	assert.NotContains(t, annotations, "nvidia.com/snapshot-storage-type")
	assert.NotContains(t, annotations, "nvidia.com/snapshot-storage-base-path")
}

func TestApplyRestorePodMetadataIgnoresLegacyStorageAnnotations(t *testing.T) {
	labels := map[string]string{}
	annotations := map[string]string{
		"nvidia.com/snapshot-storage-type":      "pvc",
		"nvidia.com/snapshot-storage-base-path": "/untrusted",
	}

	t.Log("Apply restore metadata to a pod carrying legacy storage annotations")
	ApplyRestorePodMetadata(labels, annotations, &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash})

	t.Log("Verify legacy values are not interpreted or rewritten")
	assert.Equal(t, "pvc", annotations["nvidia.com/snapshot-storage-type"])
	assert.Equal(t, "/untrusted", annotations["nvidia.com/snapshot-storage-base-path"])
	assert.Equal(t, testHash, labels[snapshotprotocol.CheckpointIDLabel])
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
