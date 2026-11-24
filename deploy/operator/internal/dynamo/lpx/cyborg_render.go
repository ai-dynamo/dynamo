/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
)

const (
	gbuildManifestPathEnv = "GBUILD_MANIFEST_PATH"
	cyborgWeightsPathEnv  = "CYBORG_WEIGHTS_PATH"
)

// configureHybridCyborg consumes a fresh hybrid clique from an admitted source
// whose conductor resources were validated during workload selection.
func configureHybridCyborg(
	cyborg *grovev1alpha1.PodCliqueTemplateSpec,
	projection *ModelProjection,
	workloadDigest string,
	modelStoragePath string,
	agentTemplateNames []string,
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
	if err := validateCyborgReplicas(projection.configuredBuild, cyborg.Spec.Replicas); err != nil {
		return err
	}

	// Publish the manifest, the local selection and the checkpoint ahead of authored bindings.
	buildRoot, err := projection.buildRuntimePath(modelStoragePath)
	if err != nil {
		return fmt.Errorf("resolve GBuild manifest path: %w", err)
	}
	prependEnv(container, corev1.EnvVar{Name: gbuildManifestPathEnv, Value: filepath.Join(buildRoot, gbuildDeploymentV1CapnpFile)})
	applyLocalPartitionIDs(container, projection)
	if projection.checkpoint != nil {
		prependEnv(container, corev1.EnvVar{Name: cyborgWeightsPathEnv, Value: projection.checkpoint.SnapshotPath(modelStoragePath)})
	}

	cyborg.Annotations = roleAnnotations(
		cyborg.Annotations,
		lpxv1alpha1.PodRoleCyborgWorker,
		workloadDigest,
	)
	configureNodeLocalConductorRuntime(&cyborg.Spec.PodSpec, strings.Join(agentTemplateNames, ":"))
	cyborg.Spec.StartsAfter = slices.Clone(agentTemplateNames)
	return nil
}

// validateCyborgReplicas requires a nonnil normalized build and validates its Cyborg replica domain.
func validateCyborgReplicas(build Build, replicas int32) error {
	// Require every physical endpoint and client-owned transaction in this replica domain.
	ioFPGACount := build.ioFPGACount
	ioFanoutFactor := build.ioFanoutFactor
	if replicas%ioFPGACount != 0 {
		return fmt.Errorf("decode service Cyborg replicas %d must be divisible by ioFpgaCount %d", replicas, ioFPGACount)
	}
	if replicas/ioFPGACount%ioFanoutFactor != 0 {
		return fmt.Errorf(
			"decode service Cyborg replicas %d must provide fanoutFactor %d clients for each of %d I/O FPGA endpoints",
			replicas,
			ioFanoutFactor,
			ioFPGACount,
		)
	}

	return nil
}
