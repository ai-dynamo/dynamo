/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

const (
	selectedCyborgConfigSuffix       = "-decode"
	selectedCyborgInfiniBandName     = "infiniband"
	selectedCyborgInfiniBandPath     = "/dev/infiniband"
	selectedCyborgServerHostsFileEnv = "SERVER_HOSTS_FILE"
	selectedCyborgTokenizerDirEnv    = "TOKENIZER_DIR"
	selectedCyborgTotalReplicasEnv   = "TOTAL_REPLICAS"
)

func selectedCyborgConfigMapName(dgdName string) string {
	return boundedAuxiliaryName(dgdName, selectedCyborgConfigSuffix)
}

// ApplySelectedCyborgContainerDefaults installs the operator-owned V2 Cyborg
// bindings before the user's main-container override is merged. container and
// workload must be non-nil, and workload must contain a projection. The
// function mutates container and reads workload and lpxPodSpec without mutating them.
func ApplySelectedCyborgContainerDefaults(
	container *corev1.Container,
	workload *SelectedWorkload,
	dgdName string,
	totalReplicas int32,
	lpxPodSpec corev1.PodSpec,
) error {
	projection := workload.modelProjections[0]

	configMapName := selectedCyborgConfigMapName(dgdName)
	container.VolumeMounts = append(
		container.VolumeMounts,
		corev1.VolumeMount{Name: selectedCyborgInfiniBandName, MountPath: selectedCyborgInfiniBandPath},
		corev1.VolumeMount{Name: lpuConfigVolumeName, MountPath: lpuConfigMountPath},
	)
	container.Env = append(
		container.Env,
		corev1.EnvVar{
			Name: selectedCyborgTokenizerDirEnv,
			ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{
				LocalObjectReference: corev1.LocalObjectReference{Name: configMapName},
				Key:                  "tokenizer_dir",
			}},
		},
		corev1.EnvVar{Name: selectedCyborgTotalReplicasEnv, Value: strconv.FormatInt(int64(totalReplicas), 10)},
	)
	cyborgBatchSize, ioFPGACount, err := cyborgRuntimeIO(&projection.configuredBuild, totalReplicas)
	if err != nil {
		return err
	}
	// SWA cache IDs depend on the final, user-overridable batch size and are
	// bound only after the main-container override is merged.
	applyCyborgRuntimeIO(container, cyborgBatchSize, ioFPGACount)
	lpxContainer := findMainContainer(lpxPodSpec.Containers)
	mountIndex := slices.IndexFunc(lpxContainer.VolumeMounts, func(mount corev1.VolumeMount) bool {
		return mount.Name == commonconsts.ModelStorageVolumeName
	})
	if mountIndex < 0 {
		return fmt.Errorf("selected LPX main container requires model storage volume mount %q", commonconsts.ModelStorageVolumeName)
	}
	modelStoragePath := lpxContainer.VolumeMounts[mountIndex].MountPath
	if strings.TrimSpace(modelStoragePath) == "" {
		return fmt.Errorf("model storage volume %q has no mount path", commonconsts.ModelStorageVolumeName)
	}
	if err := applyCyborgManifestPath(container, projection, modelStoragePath); err != nil {
		return err
	}
	hostsFile := lpuConfigMountPath + "/lpu_servers"
	if workload.scalingGroupReplicas > 1 {
		// Resolve the replica through the downward API before Kubernetes expands
		// SERVER_HOSTS_FILE. The image entrypoint remains untouched.
		container.Env = append(container.Env, corev1.EnvVar{
			Name: "LPX_ENGINE_REPLICA",
			ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{
				FieldPath: "metadata.labels['" + grovecommon.LabelPodCliqueScalingGroupReplicaIndex + "']",
			}},
		})
		hostsFile += "-$(LPX_ENGINE_REPLICA)"
	}
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  selectedCyborgServerHostsFileEnv,
		Value: hostsFile,
	})
	return nil
}

// ApplySelectedCyborgPodDefaults installs the operator-owned V2 Cyborg
// volumes before the user's PodSpec override is merged. podSpec must be non-nil
// and is mutated in place.
func ApplySelectedCyborgPodDefaults(podSpec *corev1.PodSpec, dgdName string) {
	podSpec.Volumes = append(
		podSpec.Volumes,
		corev1.Volume{
			Name: lpuConfigVolumeName,
			VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: selectedCyborgConfigMapName(dgdName)},
			}},
		},
		corev1.Volume{
			Name: selectedCyborgInfiniBandName,
			VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{
				Path: selectedCyborgInfiniBandPath,
				Type: ptr.To(corev1.HostPathDirectory),
			}},
		},
	)
}

// renderSelectedCyborgConfigMap requires a nonnil build and at least one Agent template name.
func renderSelectedCyborgConfigMap(
	namespace string,
	dgdName string,
	plan *MaterializationPlan,
	modelStoragePath string,
	build *Build,
) (*corev1.ConfigMap, error) {
	tokenizerDir := ""
	if build.Path != "" {
		tokenizerPath := build.RuntimeTokenizerPath
		if strings.TrimSpace(tokenizerPath) == "" {
			return nil, fmt.Errorf("capnp manifest build is missing model.tokenizer.path")
		}
		runtimePath, runtimeErr := buildRuntimePath(build.Path, modelStoragePath)
		if runtimeErr != nil {
			return nil, fmt.Errorf("failed to determine tokenizer_dir: %w", runtimeErr)
		}
		tokenizerDir = filepath.Join(runtimePath, tokenizerPath)
	}

	data := map[string]string{"tokenizer_dir": tokenizerDir}
	for replica := int32(0); replica < plan.Replicas; replica++ {
		// Cyborg supplies the PCS prefix, so only materialize the relative Agent name.
		serverPrefix := materializedCliqueNameForReplica(plan.LPXScalingGroupTemplate, plan.Agents[0].TemplateName, replica) + "-"
		var servers strings.Builder
		offset := 0
		for index, partition := range build.Partitions {
			if index != 0 {
				servers.WriteByte('\n')
			}
			servers.WriteString(serverPrefix)
			servers.WriteString(strconv.Itoa(offset))
			offset += partition.effectiveNodeCount()
		}
		key := "lpu_servers"
		if plan.Replicas > 1 {
			key += "-" + strconv.Itoa(int(replica))
		}
		data[key] = servers.String()
	}

	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      selectedCyborgConfigMapName(dgdName),
			Namespace: namespace,
		},
		Data: data,
	}, nil
}
