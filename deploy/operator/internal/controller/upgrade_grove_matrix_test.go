//go:build clustertest && upgradecontract

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

const upgradeMPISSHSecret = "upgrade-mpi-ssh"

// Canonical matrix paths must exercise the named backend, topology, and provider.
func checkUpgradeGroveMatrix(t *testing.T, relative string, dgd *api.DynamoGraphDeployment, objects []unstructured.Unstructured) {
	t.Helper()
	parts := strings.Split(filepath.ToSlash(relative), "/")
	if len(parts) != 5 || parts[0] != "latest" || parts[4] != "grove" {
		return
	}
	t.Log("Check the canonical Grove topology independently of the saved golden")
	require.Equal(t, parts[1], dgd.Spec.BackendFramework)
	require.Equal(t, "grove", dgd.Annotations["nvidia.com/workload-provider"])
	expected := map[string]int{"frontend": 1}
	expectedRoles := map[string]int{"frontend/frontend": 1}
	workers := []string{"worker"}
	if parts[2] == "disaggregated" {
		workers = []string{"prefill", "decode"}
	}
	for _, role := range workers {
		expected[role] = 1
		if parts[3] == "multi" {
			expected[role] = 2
			expectedRoles[role+"/"+role+"-ldr"] = 1
			expectedRoles[role+"/"+role+"-wkr"] = 1
		} else {
			expectedRoles[role+"/"+role] = 1
		}
	}
	require.Len(t, dgd.Spec.Components, len(expected))
	for _, component := range dgd.Spec.Components {
		count, found := expected[component.ComponentName]
		require.True(t, found, "unexpected component %s", component.ComponentName)
		require.Equal(t, component.ComponentName, string(component.ComponentType))
		if count == 2 {
			require.NotNil(t, component.Multinode)
			require.EqualValues(t, 2, component.Multinode.NodeCount)
		} else {
			require.Nil(t, component.Multinode)
		}
	}
	pcs := 0
	for _, object := range objects {
		require.NotContains(t, []string{"DynamoComponentDeployment", "Deployment", "LeaderWorkerSet"}, object.GetKind(), "Grove scenario selected another workload program")
		if object.GetKind() != upgradePodCliqueSetKind {
			continue
		}
		pcs++
		cliques, found, err := unstructured.NestedSlice(object.Object, "spec", "template", "cliques")
		require.NoError(t, err)
		require.True(t, found)
		require.NoError(t, upgradeGroveRoleError(expectedRoles, cliques))
		if parts[1] == "trtllm" && parts[3] == "multi" {
			t.Log("Check that every TRT-LLM worker mounts the configured MPI SSH Secret")
			for _, value := range cliques {
				clique := value.(map[string]any)
				labels := clique["labels"].(map[string]any)
				if labels["nvidia.com/dynamo-component"] == "frontend" {
					continue
				}
				pod, found, err := unstructured.NestedMap(clique, "spec", "podSpec")
				require.NoError(t, err)
				require.True(t, found)
				var spec corev1.PodSpec
				require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(pod, &spec))
				secretVolumes, sshMounts := 0, 0
				for _, volume := range spec.Volumes {
					if volume.Name != upgradeMPISSHSecret {
						continue
					}
					require.NotNil(t, volume.Secret)
					require.Equal(t, upgradeMPISSHSecret, volume.Secret.SecretName)
					secretVolumes++
				}
				for _, container := range spec.Containers {
					for _, mount := range container.VolumeMounts {
						if mount.MountPath != "/ssh-pk" {
							continue
						}
						require.Equal(t, upgradeMPISSHSecret, mount.Name)
						sshMounts++
					}
				}
				require.Equal(t, 1, secretVolumes)
				require.Positive(t, sshMounts)
			}
		}
	}
	require.Equal(t, 1, pcs)
}

// Count component/role pairs so duplicate leaders cannot stand in for workers.
func upgradeGroveRoleError(expected map[string]int, cliques []any) error {
	observed := map[string]int{}
	for _, value := range cliques {
		clique := value.(map[string]any)
		labels := clique["labels"].(map[string]any)
		role, found, err := unstructured.NestedString(clique, "spec", "roleName")
		if err != nil {
			return err
		}
		if !found || clique["name"] != role {
			return fmt.Errorf("clique name and roleName disagree")
		}
		component := labels["nvidia.com/dynamo-component"].(string)
		observed[component+"/"+role]++
	}
	if diff := cmp.Diff(expected, observed); diff != "" {
		return fmt.Errorf("Grove component roles differ: %s", diff)
	}
	return nil
}

func TestUpgradeGroveMultinodeRoles(t *testing.T) {
	for _, roles := range [][]string{{"worker-ldr", "worker-wkr"}, {"worker-ldr", "worker-ldr"}, {"worker-ldr"}} {
		t.Run(strings.Join(roles, "+"), func(t *testing.T) {
			t.Log("Require a distinct leader and worker even when a golden update accepts the output")
			expected := map[string]int{"worker/worker-ldr": 1, "worker/worker-wkr": 1}
			var cliques []any
			for _, role := range roles {
				cliques = append(cliques, map[string]any{
					"name": role, "labels": map[string]any{"nvidia.com/dynamo-component": "worker"},
					"spec": map[string]any{"roleName": role},
				})
			}
			err := upgradeGroveRoleError(expected, cliques)
			if len(roles) == 2 && roles[1] == "worker-wkr" {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
			}
		})
	}
}
