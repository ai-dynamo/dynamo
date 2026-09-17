/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
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
	cyborgConfigMap *corev1.ConfigMap,
) error {
	container := common.FindContainerByName(cyborg.Spec.PodSpec.Containers, commonconsts.MainContainerName)

	// Apply manifest-aware runtime bindings to the selected Cyborg container.
	cyborgStorage, err := lpuModelStorageBinding(cyborg.Spec.PodSpec)
	if err != nil {
		return err
	}
	if cyborgStorage.mount.MountPath != modelStorage.mount.MountPath {
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
	applyCyborgRuntimeIO(container, cyborgBatchSize, ioFPGACount)
	applyCyborgSWACacheIDs(container, cyborgBatchSize)
	if err := applyCyborgManifestPath(container, projection, modelStorage.mount.MountPath); err != nil {
		return err
	}

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
	cyborg.Spec.StartsAfter = appendUnique(cyborg.Spec.StartsAfter, agentTemplateNames...)
	return nil
}
