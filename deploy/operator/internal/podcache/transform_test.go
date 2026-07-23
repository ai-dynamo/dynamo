/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package podcache

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestProjectConsumerContract(t *testing.T) {
	now := metav1.Now()
	mode := int32(0440)
	pod := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name:              "worker-0",
			Namespace:         "inference",
			UID:               types.UID("pod-uid"),
			ResourceVersion:   "42",
			Generation:        3,
			Labels:            map[string]string{"dynamo": "worker", "dcd": "old-worker"},
			Annotations:       map[string]string{"topology-label": "topology.kubernetes.io/zone"},
			OwnerReferences:   []metav1.OwnerReference{{Name: "worker", UID: types.UID("owner-uid"), Controller: ptr.To(true)}},
			Finalizers:        []string{"test.example/finalizer"},
			DeletionTimestamp: &now,
			ManagedFields:     []metav1.ManagedFieldsEntry{{Manager: "large-manager"}},
		},
		Spec: corev1.PodSpec{
			NodeName: "node-a",
			Containers: []corev1.Container{{
				Name:    "main",
				Image:   "large-image",
				Command: []string{"python"},
				Args:    []string{"-m", "dynamo"},
				Env:     []corev1.EnvVar{{Name: "LARGE", Value: "discard-me"}},
				Resources: corev1.ResourceRequirements{
					Limits: corev1.ResourceList{corev1.ResourceMemory: resourceQuantity("1Gi")},
				},
				ReadinessProbe: &corev1.Probe{InitialDelaySeconds: 10},
			}},
			InitContainers: []corev1.Container{{Name: "discarded-init", Image: "large-init"}},
			Volumes: []corev1.Volume{
				{
					Name: "pod-labels",
					VolumeSource: corev1.VolumeSource{DownwardAPI: &corev1.DownwardAPIVolumeSource{
						DefaultMode: ptr.To(int32(0644)),
						Items: []corev1.DownwardAPIVolumeFile{
							{
								Path: "zone",
								FieldRef: &corev1.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.labels['topology.kubernetes.io/zone']",
								},
								Mode: &mode,
							},
							{
								Path:             "resources",
								ResourceFieldRef: &corev1.ResourceFieldSelector{Resource: "limits.memory"},
							},
						},
					}},
				},
				{Name: "discarded-secret", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "large"}}},
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			PodIP: "10.0.0.1",
			Conditions: []corev1.PodCondition{
				{Type: corev1.PodScheduled, Status: corev1.ConditionTrue, Message: "discard-me"},
				{Type: corev1.PodReady, Status: corev1.ConditionTrue, Reason: "discard-me"},
			},
			ContainerStatuses: []corev1.ContainerStatus{
				{
					Name:         "profiler",
					Ready:        true,
					RestartCount: 7,
					Image:        "discard-me",
					State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{
						Reason: "ImagePullBackOff", Message: "pull failed",
					}},
				},
				{
					Name: "worker",
					State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
						ExitCode: 17, Signal: 9, Reason: "Error", Message: "worker failed", ContainerID: "discard-me",
					}},
				},
			},
			InitContainerStatuses: []corev1.ContainerStatus{{
				Name: "init",
				State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{
					Reason: "ErrImagePull", Message: "init pull failed",
				}},
			}},
		},
	}
	wantMetadata := *pod.ObjectMeta.DeepCopy()
	wantMetadata.ManagedFields = nil

	got := Project(pod)

	t.Run("metadata supports topology checkpoint snapshot failover and Recreate", func(t *testing.T) {
		assert.Equal(t, wantMetadata, got.ObjectMeta)
	})
	t.Run("topology and snapshot retain node assignment and label field paths", func(t *testing.T) {
		assert.Equal(t, "node-a", got.Spec.NodeName)
		require.Len(t, got.Spec.Volumes, 1)
		require.Len(t, got.Spec.Volumes[0].DownwardAPI.Items, 1)
		assert.Equal(t, "metadata.labels['topology.kubernetes.io/zone']", got.Spec.Volumes[0].DownwardAPI.Items[0].FieldRef.FieldPath)
	})
	t.Run("model retains Ready identity command and arguments", func(t *testing.T) {
		require.Len(t, got.Spec.Containers, 1)
		assert.Equal(t, corev1.Container{Name: "main", Command: []string{"python"}, Args: []string{"-m", "dynamo"}}, got.Spec.Containers[0])
		assert.Equal(t, []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}, got.Status.Conditions)
	})
	t.Run("failover DGDR and Recreate retain terminal and diagnostic state", func(t *testing.T) {
		assert.Equal(t, corev1.PodRunning, got.Status.Phase)
		require.Len(t, got.Status.ContainerStatuses, 2)
		assert.Equal(t, "ImagePullBackOff", got.Status.ContainerStatuses[0].State.Waiting.Reason)
		assert.Equal(t, int32(17), got.Status.ContainerStatuses[1].State.Terminated.ExitCode)
		assert.Equal(t, "worker failed", got.Status.ContainerStatuses[1].State.Terminated.Message)
		require.Len(t, got.Status.InitContainerStatuses, 1)
		assert.Equal(t, "ErrImagePull", got.Status.InitContainerStatuses[0].State.Waiting.Reason)
	})
	t.Run("unused heavyweight fields are removed", func(t *testing.T) {
		assert.Nil(t, got.ManagedFields)
		assert.Empty(t, got.Spec.InitContainers)
		assert.Empty(t, got.Spec.Containers[0].Image)
		assert.Empty(t, got.Spec.Containers[0].Env)
		assert.Empty(t, got.Spec.Containers[0].Resources)
		assert.Empty(t, got.Status.PodIP)
		assert.Empty(t, got.Status.ContainerStatuses[0].Image)
		assert.Zero(t, got.Status.ContainerStatuses[0].RestartCount)
		assert.Zero(t, got.Status.ContainerStatuses[1].State.Terminated.Signal)
		assert.Empty(t, got.Status.ContainerStatuses[1].State.Terminated.ContainerID)
	})

	first := got.DeepCopy()
	assert.Equal(t, first, Project(got), "projection must be idempotent")
}

func TestTransformLeavesOtherObjectsUnchanged(t *testing.T) {
	configMap := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "config"}}
	got, err := Transform(configMap)
	require.NoError(t, err)
	assert.Same(t, configMap, got)
}

func TestProjectedPodSupportsNarrowMetadataPatch(t *testing.T) {
	before := Project(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-0",
			Namespace: "inference",
			Labels:    map[string]string{"existing": "value"},
		},
		Spec: corev1.PodSpec{
			NodeName: "node-a",
			Containers: []corev1.Container{{
				Name: "main",
			}},
		},
	})
	after := before.DeepCopy()
	after.Labels["topology.kubernetes.io/zone"] = "us-west-2a"

	patch, err := client.MergeFrom(before).Data(after)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"metadata": {
			"labels": {
				"topology.kubernetes.io/zone": "us-west-2a"
			}
		}
	}`, string(patch))
}

func TestConfigureAddsPodEntry(t *testing.T) {
	options := cache.Options{}
	require.NoError(t, Configure(&options))

	podEntries := 0
	for obj, byObject := range options.ByObject {
		if _, ok := obj.(*corev1.Pod); !ok {
			continue
		}
		podEntries++
		require.NotNil(t, byObject.Transform)
	}
	assert.Equal(t, 1, podEntries)
}

func TestConfigure(t *testing.T) {
	podKey := &corev1.Pod{}
	configMapKey := &corev1.ConfigMap{}
	namespaces := map[string]cache.Config{"inference": {}}
	options := cache.Options{
		DefaultNamespaces: namespaces,
		ByObject: map[client.Object]cache.ByObject{
			podKey:       {Namespaces: namespaces},
			configMapKey: {},
		},
	}

	require.NoError(t, Configure(&options))
	require.NoError(t, Configure(&options), "configuration must be idempotent")

	podEntries := 0
	for obj, byObject := range options.ByObject {
		if _, ok := obj.(*corev1.Pod); !ok {
			continue
		}
		podEntries++
		assert.Equal(t, namespaces, byObject.Namespaces)
		require.NotNil(t, byObject.Transform)
	}
	assert.Equal(t, 1, podEntries)
	_, configMapStillPresent := options.ByObject[configMapKey]
	assert.True(t, configMapStillPresent)
}

func TestConfigureRejectsDuplicatePodEntries(t *testing.T) {
	options := cache.Options{ByObject: map[client.Object]cache.ByObject{
		&corev1.Pod{}: {},
		&corev1.Pod{}: {},
	}}
	assert.ErrorContains(t, Configure(&options), "multiple Pod cache configurations")
}

func resourceQuantity(value string) resource.Quantity {
	return resource.MustParse(value)
}
