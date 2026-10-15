/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
)

const gbuildManifestPathEnv = "GBUILD_MANIFEST_PATH"

// MinimumCyborgReplicas returns one complete client group for a resolved hybrid workload.
func (w *Workload) MinimumCyborgReplicas() (int32, error) {
	build := &w.modelProjections[0].configuredBuild
	replicas := int64(build.ioFPGACount) * int64(build.ioFanoutFactor)
	if replicas > math.MaxInt32 {
		return 0, fmt.Errorf("minimum Cyborg replicas %d exceeds the PodClique replica limit %d", replicas, math.MaxInt32)
	}
	return int32(replicas), nil
}

// ValidateCyborgReplicas ensures that the specified Cyborg replicas are in a valid configuration.
func (w *Workload) ValidateCyborgReplicas(replicas int32) error {
	build := &w.modelProjections[0].configuredBuild
	return validateCyborgReplicas(build, replicas)
}

// applyCyborgManifestPath projects an authoritative manifest location into one Cyborg container.
func applyCyborgManifestPath(container *corev1.Container, projection *ModelProjection, modelStoragePath string) error {
	buildRoot, err := buildRuntimePath(projection.configuredBuild.path, projection.runtimeBuildRef, modelStoragePath)
	if err != nil {
		return fmt.Errorf("resolve GBuild manifest path: %w", err)
	}

	manifestPath, err := projection.contractFormat.filename()
	if err != nil {
		return err
	}

	// Kubernetes expands environment references in order; publish the manifest before authored bindings.
	env := make([]corev1.EnvVar, 0, len(container.Env)+1)
	env = append(env, corev1.EnvVar{
		Name:  gbuildManifestPathEnv,
		Value: filepath.Join(buildRoot, manifestPath),
	})
	for _, variable := range container.Env {
		if variable.Name != gbuildManifestPathEnv {
			env = append(env, variable)
		}
	}
	container.Env = env
	return nil
}

// validateCyborgReplicas requires a nonnil normalized build and validates its Cyborg replica domain.
func validateCyborgReplicas(build *Build, replicas int32) error {
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

// configureHybridCyborg consumes a fresh hybrid clique from an admitted source
// whose conductor resources were validated during workload selection.
// cyborgConfigMap is the nonnil Agent-address ConfigMap rendered for that workload.
func configureHybridCyborg(
	cyborg *grovev1alpha1.PodCliqueTemplateSpec,
	projection *ModelProjection,
	workloadDigest string,
	modelStoragePath string,
	agentTemplateNames []string,
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
	if err := validateCyborgReplicas(&projection.configuredBuild, cyborg.Spec.Replicas); err != nil {
		return err
	}
	if err := applyCyborgManifestPath(container, projection, modelStoragePath); err != nil {
		return err
	}
	applyLocalPartitionIDs(container, projection)

	cyborg.Annotations = roleAnnotations(
		cyborg.Annotations,
		lpxv1alpha1.PodRoleCyborgWorker,
		workloadDigest,
	)
	ensureLPUConfigVolume(&cyborg.Spec.PodSpec, cyborgConfigMap.Name)
	cyborg.Annotations[v1alpha1.AnnotationExtraResourcesHash] = cyborgConfigHash
	cyborg.Spec.StartsAfter = slices.Clone(agentTemplateNames)
	return nil
}

// renderCyborgConfigMap renders the hybrid Agent endpoints of either LPU family.
// It lists one Agent per remote runtime partition in compiler order; an all-local
// selection renders an empty list. The workload and plan must be non-nil,
// validated, and describe the selected hybrid model.
// The workload and plan are not mutated. The caller assigns the returned ConfigMap's namespace.
func (w *Workload) renderCyborgConfigMap(plan *MaterializationPlan) (*corev1.ConfigMap, string, error) {
	projection := w.modelProjections[0]
	build := &projection.configuredBuild

	// Cyborg runs a selected prop-sync chain as one runtime partition and
	// addresses it through its first member. XT builds have already collapsed
	// these chains; HX builds keep each member as a physical partition.
	chained := make([]bool, len(build.partitions))
	for _, position := range projection.propSyncEdges {
		chained[position+1] = true
	}

	// Cyborg supplies the PCS prefix; startup supplies this workload's Grove index.
	serverPrefix := plan.ScalingGroupTemplate + "-${GROVE_PCSG_INDEX}-" + plan.Agents[0].TemplateName + "-"
	servers := make([]string, 0, len(build.partitions))
	offset := 0
	for index, partition := range build.partitions {
		if !chained[index] {
			servers = append(servers, serverPrefix+strconv.Itoa(offset))
		}
		offset += partition.effectiveNodeCount()
	}

	return renderRuntimeConfigMap(plan.ResourcePrefix+"-decode", map[string]string{
		"lpu_servers": strings.Join(servers, "\n"),
	})
}
