/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"strconv"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
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
	container := findMainContainer(cyborg.Spec.PodSpec.Containers)

	// Apply manifest-aware runtime bindings to the selected Cyborg container.
	if err := withLPUModelStorage(&cyborg.Spec.PodSpec, container, modelStorage); err != nil {
		return err
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
	if err := wrapCyborgDecodeForSwaBatch(container, cyborgBatchSize); err != nil {
		return err
	}

	cyborg.Spec.PodSpec.SchedulerName = "default-scheduler"
	delete(cyborg.Labels, commonconsts.KubeLabelKaiSchedulerQueue)
	cyborg.Annotations = roleAnnotations(
		cyborg.Annotations,
		lpxv1alpha1.PodRoleCyborgWorker,
		workloadDigest,
	)
	if cyborgConfigMap != nil {
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
