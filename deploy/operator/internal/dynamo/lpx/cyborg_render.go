/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
)

// configureHybridCyborg consumes a fresh hybrid clique from an admitted source
// whose conductor resources were validated during workload selection.
func configureHybridCyborg(
	cyborg *grovev1alpha1.PodCliqueTemplateSpec,
	projection *Model,
	workloadDigest string,
	modelStoragePath string,
	cyborgConfigMap *corev1.ConfigMap,
	cyborgConfigHash string,
) error {
	container := common.FindContainerByName(cyborg.Spec.PodSpec.Containers, commonconsts.MainContainerName)

	// Apply manifest-aware runtime bindings to the selected Cyborg container.
	cyborgStoragePath, err := lpuModelStoragePath(cyborg.Spec.PodSpec)
	if err != nil {
		return err
	}
	if cyborgStoragePath != modelStoragePath {
		return fmt.Errorf("selected Cyborg podTemplate conflicts with model storage mount %q", modelStoragePath)
	}
	if err := validateCyborgReplicas(&projection.component.configuredBuild, cyborg.Spec.Replicas); err != nil {
		return err
	}
	if err := applyCyborgManifestPath(container, projection, modelStoragePath); err != nil {
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
		cyborg.Annotations[v1alpha1.AnnotationExtraResourcesHash] = cyborgConfigHash
	}
	return nil
}
