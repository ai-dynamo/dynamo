/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func TestValidateSSHVolumePreservesAuthoredStorage(t *testing.T) {
	for _, test := range []struct {
		name    string
		source  corev1.VolumeSource
		wantErr bool
	}{
		{"matching Secret", corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "mpi.ssh"}}, false},
		{"persistent data", corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "data"}}, true},
		{"different Secret", corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "application"}}, true},
		{"restricted keys", corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{
			SecretName: "mpi.ssh", Items: []corev1.KeyToPath{{Key: "public.key", Path: "public.key"}},
		}}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Bind the authored storage to the runtime SSH volume name")
			pod := corev1.PodSpec{Volumes: []corev1.Volume{{Name: "ssh-secret", VolumeSource: test.source}}}
			before := pod.DeepCopy()

			t.Log("Require a compatible MPI key binding without changing the authored source")
			err := validateSSHVolume(&pod, "mpi.ssh")
			if test.wantErr {
				require.ErrorContains(t, err, `volume "ssh-secret" is reserved for MPI SSH Secret "mpi.ssh"`)
				require.Equal(t, *before, pod)
				return
			}
			require.NoError(t, err)
			require.Equal(t, *before, pod)
		})
	}
}

func TestAddConductorSSHKeyValidatesAuthoredStorage(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name   string
		source corev1.VolumeSource
		mount  corev1.VolumeMount
		err    string
	}{
		{name: "Secret", source: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "application"}}, err: "requires emptyDir"},
		{name: "PVC", source: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "data"}}, err: "requires emptyDir"},
		{name: "missing source", err: "requires emptyDir"},
		{name: "missing mount", source: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}, err: "without subPath"},
		{name: "subPath", source: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}, mount: corev1.VolumeMount{Name: conductorSSHKeyVolumeName, MountPath: conductorSSHKeyDir, SubPath: "keys"}, err: "without subPath"},
		{name: "EmptyDir", source: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}, mount: corev1.VolumeMount{Name: conductorSSHKeyVolumeName, MountPath: conductorSSHKeyDir}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Author the conductor key source, destination and runtime mount")
			pod := corev1.PodSpec{
				Containers: []corev1.Container{{
					Name: commonconsts.MainContainerName, Image: "lpu-runtime",
					VolumeMounts: []corev1.VolumeMount{test.mount},
				}},
				Volumes: []corev1.Volume{
					{Name: runtimeSSHVolumeName, VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{
						SecretName: "mpi.ssh", DefaultMode: ptr.To(int32(0644)),
					}}},
					{Name: conductorSSHKeyVolumeName, VolumeSource: test.source},
				},
			}
			before := pod.DeepCopy()

			t.Log("Require a shared Pod-local directory without changing authored storage")
			err := addConductorSSHKey(&pod, &pod.Containers[0], "mpi.ssh")
			if test.err != "" {
				require.ErrorContains(t, err, test.err)
				require.Equal(t, *before, pod)
				return
			}
			require.NoError(t, err)
			require.Equal(t, before.Volumes, pod.Volumes)
			require.Equal(t, before.Containers, pod.Containers)
			require.Len(t, pod.InitContainers, 1)
			require.Equal(t, []string{"/bin/install"}, pod.InitContainers[0].Command)
			require.Equal(t, []string{"-m", "0600", "/ssh-pk/private.key", "/tmp/dynamo-lpu-ssh/private.key"}, pod.InitContainers[0].Args)
		})
	}
}

func TestXTRuntimesRejectSSHVolumeCollision(t *testing.T) {
	t.Log("Start each XT role with an authored volume using the fixed SSH name")
	base := corev1.PodSpec{
		Containers: []corev1.Container{{Name: commonconsts.MainContainerName, Image: "lpu-runtime"}},
		Volumes:    []corev1.Volume{{Name: "ssh-secret", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}}},
	}
	base.Containers[0].VolumeMounts = []corev1.VolumeMount{{Name: lpuConfigVolumeName, MountPath: lpuConfigMountPath}}
	configureAgentScheduling(&base, BuildFamilyXT)
	direct, agent, conductor := base.DeepCopy(), base.DeepCopy(), base.DeepCopy()

	t.Log("Propagate the collision through every XT runtime configuration path")
	const wantError = `volume "ssh-secret" is reserved for MPI SSH Secret "mpi.ssh"`
	require.ErrorContains(t, configureDirectHybridAgentRuntime(direct, "graph-lpu", "mpi.ssh", false), wantError)
	require.ErrorContains(t, configureNodeLocalAgentRuntime(agent, BuildFamilyXT, false, "mpi.ssh"), wantError)
	require.ErrorContains(t, configureNodeLocalConductorRuntime(conductor, BuildFamilyXT, "lpu-wkr-m-0", "mpi.ssh"), wantError)

	t.Log("Preserve the authored volume in every rejected runtime")
	for _, pod := range []*corev1.PodSpec{direct, agent, conductor} {
		require.Equal(t, base.Volumes[0].VolumeSource, testVolumeByName(t, pod.Volumes, "ssh-secret"))
	}
}
