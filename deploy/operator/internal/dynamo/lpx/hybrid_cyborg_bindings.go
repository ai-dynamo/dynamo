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

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

const (
	selectedCyborgServerHostsFileEnv = "SERVER_HOSTS_FILE"
	selectedCyborgTokenizerDirEnv    = "TOKENIZER_DIR"
	selectedCyborgTotalReplicasEnv   = "TOTAL_REPLICAS"
)

// ApplySelectedCyborgContainerDefaults installs the operator-owned V2 Cyborg
// bindings before the user's main-container override is merged. container and
// workload must be non-nil, and workload must contain a projection. The
// function mutates container and reads workload and lpxPodSpec without mutating them.
func ApplySelectedCyborgContainerDefaults(
	container *corev1.Container,
	workload *SelectedWorkload,
	configMapName string,
	totalReplicas int32,
	lpxPodSpec corev1.PodSpec,
) error {
	projection := workload.modelProjections[0]

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
	lpxContainer := common.FindContainerByName(lpxPodSpec.Containers, commonconsts.MainContainerName)
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
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  selectedCyborgServerHostsFileEnv,
		Value: runtimeTemporaryStorageMountPath + "/lpu_servers",
	})
	return nil
}

// RenderCyborgConfigMap renders the XT hybrid configuration before Cyborg defaults are merged.
// The workload must contain the selected hybrid model; agentPodSpec supplies its merged storage.
func (w *SelectedWorkload) RenderCyborgConfigMap(
	namespace string,
	root string,
	agentPodSpec corev1.PodSpec,
) (*corev1.ConfigMap, error) {
	storage, err := lpuModelStorageBinding(agentPodSpec)
	if err != nil {
		return nil, err
	}
	plan, err := w.PlanNodeLocalMaterialization(root)
	if err != nil {
		return nil, err
	}
	build := &w.modelProjections[0].configuredBuild
	tokenizerDir := ""
	if build.Path != "" {
		tokenizerPath := build.RuntimeTokenizerPath
		if strings.TrimSpace(tokenizerPath) == "" {
			return nil, fmt.Errorf("capnp manifest build is missing model.tokenizer.path")
		}
		runtimePath, runtimeErr := buildRuntimePath(build.Path, storage.mount.MountPath)
		if runtimeErr != nil {
			return nil, fmt.Errorf("failed to determine tokenizer_dir: %w", runtimeErr)
		}
		tokenizerDir = filepath.Join(runtimePath, tokenizerPath)
	}

	// Cyborg supplies the PCS prefix; startup supplies this engine's Grove index.
	serverPrefix := plan.LPXScalingGroupTemplate + "-${GROVE_PCSG_INDEX}-" + plan.Agents[0].TemplateName + "-"
	servers := make([]string, len(build.Partitions))
	offset := 0
	for index, partition := range build.Partitions {
		servers[index] = serverPrefix + strconv.Itoa(offset)
		offset += partition.effectiveNodeCount()
	}

	return renderRuntimeConfigMap(namespace, root+"-decode", map[string]string{
		"tokenizer_dir": tokenizerDir,
		"lpu_servers":   strings.Join(servers, "\n"),
	})
}
