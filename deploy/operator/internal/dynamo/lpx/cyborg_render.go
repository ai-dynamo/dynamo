/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"slices"
	"strconv"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
)

// configureHybridCyborg consumes a fresh hybrid clique from an admitted source
// whose conductor resources were validated during workload selection.
func configureHybridCyborg(
	cyborg *grovev1alpha1.PodCliqueTemplateSpec,
	projection *ModelProjection,
	workloadDigest string,
	modelStorage lpuModelStorage,
	agentTemplateNames []string,
	plan *MaterializationPlan,
	cyborgConfigMap *corev1.ConfigMap,
) error {
	container := common.FindContainerByName(cyborg.Spec.PodSpec.Containers, commonconsts.MainContainerName)

	// Apply manifest-aware runtime bindings to the selected Cyborg container.
	cyborgStorage, err := lpuModelStorageBinding(cyborg.Spec.PodSpec)
	if err != nil {
		return err
	}
	if !apiequality.Semantic.DeepEqual(cyborgStorage.volume, modelStorage.volume) || !apiequality.Semantic.DeepEqual(cyborgStorage.mount, modelStorage.mount) {
		return fmt.Errorf("selected Cyborg podTemplate conflicts with model storage mount %q", modelStorage.mount.MountPath)
	}
	cyborgBatchSize, ioFPGACount, err := cyborgRuntimeIO(&projection.configuredBuild, cyborg.Spec.Replicas)
	if err != nil {
		return err
	}
	cyborgBatchSize, err = configuredCyborgBatchSize(container, cyborgBatchSize)
	if err != nil {
		return err
	}
	if projection.configuredBuild.Family == BuildFamilyXT {
		setContainerEnv(container, false, corev1.EnvVar{Name: "RDMA_PORT", Value: strconv.Itoa(lpuRDMAPort)})
	}
	applyCyborgRuntimeIO(container, cyborgBatchSize, ioFPGACount)
	applyCyborgSWACacheIDs(container, cyborgBatchSize)
	if err := applyCyborgManifestPath(container, projection, modelStorage.mount.MountPath); err != nil {
		return err
	}
	configFile := ""
	if cyborgConfigMap != nil && slices.ContainsFunc(container.Env, func(variable corev1.EnvVar) bool {
		return variable.Name == selectedCyborgServerHostsFileEnv && variable.Value == runtimeTemporaryStorageMountPath+"/lpu_servers" && variable.ValueFrom == nil
	}) {
		configFile = "lpu_servers"
		if err := validateRuntimeConfigStorage(&cyborg.Spec.PodSpec, container, configFile); err != nil {
			return err
		}
	}
	wrapCyborgStartup(container, cyborgBatchSize, configFile)

	cyborg.Spec.PodSpec.SchedulerName = "default-scheduler"
	delete(cyborg.Labels, commonconsts.KubeLabelKaiSchedulerQueue)
	cyborg.Annotations = roleAnnotations(
		cyborg.Annotations,
		lpxv1alpha1.PodRoleCyborgWorker,
		workloadDigest,
	)
	if cyborgConfigMap != nil {
		if err := withLPUConfigVolume(&cyborg.Spec.PodSpec, cyborgConfigMap.Name, true); err != nil {
			return err
		}
		cyborg.Annotations[commonconsts.AnnotationExtraResourcesHash] = LPUConfigMapHash(cyborgConfigMap)
	}
	if projection.configuredBuild.Family == BuildFamilyXT {
		cyborg.Spec.StartsAfter = appendUnique(cyborg.Spec.StartsAfter, agentTemplateNames...)
	}
	if plan.ConductorTemplate != "" {
		cyborg.Spec.StartsAfter = appendUnique(cyborg.Spec.StartsAfter, plan.ConductorTemplate)
	}
	if projection.configuredBuild.Family != BuildFamilyXT {
		cyborg.Spec.StartsAfter = appendUnique(cyborg.Spec.StartsAfter, agentTemplateNames...)
	}
	return nil
}
