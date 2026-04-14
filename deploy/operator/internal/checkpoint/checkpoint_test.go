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

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
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
	return &CheckpointInfo{Enabled: true, Hash: testHash}
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
	assert.Equal(t, snapshotprotocol.DefaultCheckpointArtifactVersion, ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation])
}

// --- InjectCheckpointIntoPodSpec tests ---

func TestInjectCheckpointIntoPodSpec(t *testing.T) {
	t.Run("ready checkpoint injects podinfo and overrides command", func(t *testing.T) {
		podSpec := testPodSpec()
		info := &CheckpointInfo{Enabled: true, Ready: true, Identity: ptr.To(testIdentity())}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build()
		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info))
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
				{Name: "sidecar", Image: "sidecar:latest", Command: []string{"sidecar"}, Args: []string{"run"}},
				{Name: consts.MainContainerName, Image: "main:latest", Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"}},
			},
		}
		info := &CheckpointInfo{Enabled: true, Ready: true, Hash: testHash}
		reader := fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build()

		require.NoError(t, InjectCheckpointIntoPodSpec(context.Background(), reader, testNamespace, podSpec, info))
		assert.Equal(t, []string{"sidecar"}, podSpec.Containers[0].Command)
		assert.Equal(t, []string{"run"}, podSpec.Containers[0].Args)
		assert.Equal(t, []string{"sleep", "infinity"}, podSpec.Containers[1].Command)
		assert.Nil(t, podSpec.Containers[1].Args)
	})

	t.Run("error cases", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			podSpec *corev1.PodSpec
			info    *CheckpointInfo
			reader  client.Reader
			errMsg  string
		}{
			{"hash empty and identity nil", testPodSpec(), &CheckpointInfo{Enabled: true}, fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build(), "identity is nil"},
			{"no containers", &corev1.PodSpec{}, testInfo(), fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build(), "no container found"},
			{"main container missing", &corev1.PodSpec{Containers: []corev1.Container{{Name: "sidecar", Image: "img", Command: []string{"python3"}}}}, testInfo(), fake.NewClientBuilder().WithScheme(testScheme()).WithObjects(testSnapshotAgentDaemonSet()).Build(), "main container not found"},
			{"snapshot daemonset missing", testPodSpec(), testInfo(), fake.NewClientBuilder().WithScheme(testScheme()).Build(), "no snapshot-agent daemonset found"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				err := InjectCheckpointIntoPodSpec(context.Background(), tc.reader, testNamespace, tc.podSpec, tc.info)
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
		hash, err := ComputeIdentityHash(testIdentity())
		require.NoError(t, err)
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: hash, Namespace: testNamespace},
			Spec:       nvidiacomv1alpha1.DynamoCheckpointSpec{Identity: testIdentity()},
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
