/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"fmt"
	"strings"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// groveWorkloadRenderer owns rendering of the desired Grove PodCliqueSet using read-only
// Kubernetes access. It does not reconcile resources, register watches, own
// finalizers, or write status.
type groveWorkloadRenderer struct {
	reader                client.Reader
	config                *configv1alpha1.OperatorConfiguration
	runtimeConfig         *commoncontroller.RuntimeConfig
	dockerSecretRetriever DockerSecretRetriever
}

// grovePodCliqueSetRender couples the desired PCS to the exact observation used
// to decide compatibility and the worker hash suffix.
type grovePodCliqueSetRender struct {
	existing          *grovev1alpha1.PodCliqueSet
	desired           *grovev1alpha1.PodCliqueSet
	gpuShapes         map[string]dynamo.GPUShape
	runtimeStatuses   map[string]dynamo.ComponentRuntimeStatus
	powerCapPreserved bool
}

func newGroveWorkloadRenderer(
	reader client.Reader,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
	dockerSecretRetriever DockerSecretRetriever,
) *groveWorkloadRenderer {
	return &groveWorkloadRenderer{
		reader:                reader,
		config:                config,
		runtimeConfig:         runtimeConfig,
		dockerSecretRetriever: dockerSecretRetriever,
	}
}

func (r *groveWorkloadRenderer) Render(
	ctx context.Context,
	req groveReconcileRequest,
	restartState *dynamo.RestartState,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
	workerGenerationChanged bool,
) (*grovePodCliqueSetRender, error) {
	if req.DGD == nil {
		return nil, fmt.Errorf("cannot render Grove PodCliqueSet without a DynamoGraphDeployment")
	}
	if r.reader == nil {
		return nil, fmt.Errorf("cannot render Grove PodCliqueSet without a Kubernetes reader")
	}

	existingPodCliqueSet := &grovev1alpha1.PodCliqueSet{}
	key := types.NamespacedName{
		Name:      dynamo.PCSNameForDGD(req.DGD, req.IsDelegated),
		Namespace: req.DGD.Namespace,
	}
	if err := r.reader.Get(ctx, key, existingPodCliqueSet); err != nil {
		if !apierrors.IsNotFound(err) {
			return nil, fmt.Errorf("get PodCliqueSet %s: %w", key, err)
		}
		existingPodCliqueSet = nil
	}

	// Render ordinary workloads and retain the observed server-owned fields.
	workerHashSuffix := shouldRenderGroveWorkerHashSuffix(req, existingPodCliqueSet, workerGenerationChanged)
	desired, err := dynamo.GenerateGrovePodCliqueSet(
		ctx, req.DGD, req.IsDelegated, r.config, r.runtimeConfig, r.reader,
		r.dockerSecretRetriever, restartState, existingPodCliqueSet, workerHashSuffix, checkpointInfos, nil,
	)
	if err != nil {
		return nil, err
	}
	prepareGroveTopologyConstraintUpgrade(desired, existingPodCliqueSet)
	preserveGrovePodCliqueSetOrder(desired, existingPodCliqueSet)
	preserveGrovePodCliqueSetReplicas(
		desired,
		existingPodCliqueSet,
		checkpointInfos,
	)

	// Prove cap-only divergence against the observed PCS before choosing its generation identity.
	preserved := false
	if len(dynamo.DesiredWorkerPowerCaps(req.DGD)) > 0 && !powerCapRolloutBlocked(req.DGD.Status.RollingUpdate) && existingPodCliqueSet != nil && metav1.IsControlledBy(existingPodCliqueSet, req.DGD) {
		retained, applicable := retainedGrovePowerCaps(req, existingPodCliqueSet)
		if applicable {
			candidate, renderErr := dynamo.GenerateGrovePodCliqueSet(ctx, req.DGD, req.IsDelegated, r.config, r.runtimeConfig, r.reader, r.dockerSecretRetriever, restartState, existingPodCliqueSet, false, checkpointInfos, &retained)
			if renderErr != nil {
				return nil, renderErr
			}
			prepareGroveTopologyConstraintUpgrade(candidate, existingPodCliqueSet)
			preserveGrovePodCliqueSetOrder(candidate, existingPodCliqueSet)
			preserveGrovePodCliqueSetReplicas(candidate, existingPodCliqueSet, checkpointInfos)
			composed, composeErr := provideroverride.ComposeGroveOverrides(req.DGD, candidate)
			if composeErr != nil {
				return nil, composeErr
			}
			comparisonCandidate := &grovev1alpha1.PodCliqueSet{}
			if err := runtime.DefaultUnstructuredConverter.FromUnstructured(composed.Object, comparisonCandidate); err != nil {
				return nil, err
			}
			if apiequality.Semantic.DeepEqual(grovePowerCapWorkerSpec(comparisonCandidate), grovePowerCapWorkerSpec(existingPodCliqueSet)) {
				desired, preserved = candidate, true
			}
		}
	}

	gpuShapes, err := dynamo.ResolveGroveGPUShapes(ctx, r.reader, req.DGD, req.IsDelegated, desired)
	if err != nil {
		return nil, err
	}
	return &grovePodCliqueSetRender{
		existing:          existingPodCliqueSet,
		desired:           desired,
		gpuShapes:         gpuShapes,
		runtimeStatuses:   dynamo.ResolveGroveComponentRuntimeStatuses(req.DGD, desired),
		powerCapPreserved: preserved,
	}, nil
}

// grovePowerCapWorkerSpec compares worker intent using the pinned Grove alpha.15 admission defaults.
// It retains common PCS settings but excludes independently reconciled nonworker cliques. Rollout policy
// is excluded like the worker hash excludes it; only this comparison copy drops it, never the written PCS.
func grovePowerCapWorkerSpec(pcs *grovev1alpha1.PodCliqueSet) grovev1alpha1.PodCliqueSetSpec {
	spec := *pcs.Spec.DeepCopy()
	spec.UpdateStrategy = nil
	if spec.Template.TerminationDelay == nil {
		spec.Template.TerminationDelay = &metav1.Duration{Duration: 4 * time.Hour}
	}
	if spec.Template.HeadlessServiceConfig == nil {
		spec.Template.HeadlessServiceConfig = &grovev1alpha1.HeadlessServiceConfig{PublishNotReadyAddresses: true}
	}

	// Keep only scaling groups with worker members.
	workers := map[string]bool{}
	for _, clique := range spec.Template.Cliques {
		workers[clique.Name] = dynamo.IsWorkerComponent(clique.Labels[commonconsts.KubeLabelDynamoComponentType])
	}
	groups := spec.Template.PodCliqueScalingGroupConfigs[:0]
	for _, group := range spec.Template.PodCliqueScalingGroupConfigs {
		hasWorker := false
		for _, name := range group.CliqueNames {
			hasWorker = hasWorker || workers[name]
		}
		if !hasWorker {
			continue
		}
		if group.Replicas == nil {
			group.Replicas = new(int32(1))
		}
		if group.ScaleConfig != nil && group.ScaleConfig.MinReplicas == nil {
			group.ScaleConfig.MinReplicas = new(*group.Replicas)
		}
		group.RollingUpdate = nil
		groups = append(groups, group)
	}
	spec.Template.PodCliqueScalingGroupConfigs = groups

	// Normalize provider defaults without discarding any authored worker drift.
	cliques := spec.Template.Cliques[:0]
	for _, clique := range spec.Template.Cliques {
		if !workers[clique.Name] {
			continue
		}
		if clique.Spec.Replicas == 0 {
			clique.Spec.Replicas = 1
		}
		if clique.Spec.ScaleConfig != nil && clique.Spec.ScaleConfig.MinReplicas == nil {
			clique.Spec.ScaleConfig.MinReplicas = new(clique.Spec.Replicas)
		}
		clique.RollingUpdate = nil
		template := &corev1.PodTemplateSpec{Spec: clique.Spec.PodSpec}
		defaultPowerCapTemplate(template)
		clique.Spec.PodSpec = template.Spec
		cliques = append(cliques, clique)
	}
	spec.Template.Cliques = cliques
	return spec
}

// retainedGrovePowerCaps requires one coherent active worker identity and at least one cap difference.
func retainedGrovePowerCaps(req groveReconcileRequest, pcs *grovev1alpha1.PodCliqueSet) (dynamo.GroveWorkerIdentity, bool) {
	retained := dynamo.GroveWorkerIdentity{PowerCaps: map[string]string{}}
	changed, found := false, false
	for _, clique := range pcs.Spec.Template.Cliques {
		component := clique.Labels[commonconsts.KubeLabelDynamoComponent]
		if !dynamo.IsWorkerComponent(clique.Labels[commonconsts.KubeLabelDynamoComponentType]) {
			continue
		}
		hash := clique.Labels[commonconsts.KubeLabelDynamoWorkerHash]
		if found && hash != retained.Hash {
			return retained, false
		}
		retained.Hash, found = hash, true
		for i := range req.DGD.Spec.Components {
			source := &req.DGD.Spec.Components[i]
			if source.ComponentName != component {
				continue
			}
			// Role-template workers have no component template and therefore no explicit cap.
			var want string
			var wantPresent bool
			if source.PodTemplate != nil {
				want, wantPresent = source.PodTemplate.Annotations[commonconsts.KubeAnnotationGPUPowerLimit]
			}
			old, oldPresent := clique.Annotations[commonconsts.KubeAnnotationGPUPowerLimit]
			if wantPresent != oldPresent {
				return retained, false
			}
			if oldPresent {
				if previous, exists := retained.PowerCaps[component]; exists && previous != old {
					return retained, false
				}
				retained.PowerCaps[component] = old
				changed = changed || old != want
			}
		}
	}
	current := currentWorkerHashes(req.DGD)
	if retained.Hash != "" && !current.contains(retained.Hash) {
		return retained, false
	}
	return retained, changed && found
}

func shouldRenderGroveWorkerHashSuffix(
	req groveReconcileRequest,
	existing *grovev1alpha1.PodCliqueSet,
	hashChanged bool,
) bool {
	managedComponents := req.ManagedComponents()
	if !hasWorkerComponents(managedComponents) {
		return false
	}
	if existing == nil || podCliqueSetUsesGroveWorkerHashSuffix(req, existing) {
		return true
	}

	return hashChanged
}

func hasWorkerComponents(components []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
	for i := range components {
		if dynamo.IsWorkerComponent(string(components[i].ComponentType)) {
			return true
		}
	}
	return false
}

func podCliqueSetUsesGroveWorkerHashSuffix(
	req groveReconcileRequest,
	pcs *grovev1alpha1.PodCliqueSet,
) bool {
	managedComponents := req.ManagedComponents()
	for i := range managedComponents {
		component := &managedComponents[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		clique := podCliqueSetCliqueForComponent(pcs, component.ComponentName)
		if clique == nil || clique.Labels[commonconsts.KubeLabelDynamoWorkerHash] == "" {
			return false
		}
	}
	return true
}

// podCliqueSetObservesWorkerHash reports whether every worker clique in pcs carries the
// computed DGD worker hash, or, for a pre-existing legacy PCS, that no clique is stamped.
func podCliqueSetObservesWorkerHash(
	req groveReconcileRequest,
	pcs *grovev1alpha1.PodCliqueSet,
	acceptAllUnstamped bool,
) (bool, error) {
	managedComponents := req.ManagedComponents()
	if !hasWorkerComponents(managedComponents) {
		return true, nil
	}
	want, err := dynamo.ComputeDGDWorkersSpecHash(req.DGD)
	if err != nil {
		return false, fmt.Errorf("compute desired Grove worker hash: %w", err)
	}
	allCanonical := true
	allUnstamped := acceptAllUnstamped
	for i := range managedComponents {
		component := &managedComponents[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		clique := podCliqueSetCliqueForComponent(pcs, component.ComponentName)
		if clique == nil {
			return false, nil
		}
		hash := clique.Labels[commonconsts.KubeLabelDynamoWorkerHash]
		allCanonical = allCanonical && hash == want
		allUnstamped = allUnstamped && hash == ""
	}
	return allCanonical || allUnstamped, nil
}

func podCliqueSetCliqueForComponent(
	pcs *grovev1alpha1.PodCliqueSet,
	componentName string,
) *grovev1alpha1.PodCliqueTemplateSpec {
	if pcs == nil {
		return nil
	}
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique != nil && clique.Labels[commonconsts.KubeLabelDynamoComponent] == componentName {
			return clique
		}
	}
	return nil
}

func preserveGrovePodCliqueSetOrder(
	desired *grovev1alpha1.PodCliqueSet,
	existing *grovev1alpha1.PodCliqueSet,
) {
	if desired == nil || existing == nil {
		return
	}
	desired.Spec.Template.Cliques = orderLikeExisting(
		existing.Spec.Template.Cliques,
		desired.Spec.Template.Cliques,
		podCliqueTemplateName,
	)
	desired.Spec.Template.PodCliqueScalingGroupConfigs = orderLikeExisting(
		existing.Spec.Template.PodCliqueScalingGroupConfigs,
		desired.Spec.Template.PodCliqueScalingGroupConfigs,
		podCliqueScalingGroupConfigName,
	)
	desired.Spec.Template.ResourceClaimTemplates = orderLikeExisting(
		existing.Spec.Template.ResourceClaimTemplates,
		desired.Spec.Template.ResourceClaimTemplates,
		resourceClaimTemplateConfigName,
	)
}

// prepareGroveTopologyConstraintUpgrade performs the first half of Grove's
// supported legacy topology migration. A pre-alpha.9 constraint has
// packDomain but no topologyName; Grove requires that object to be repaired by
// adding topologyName before packDomain can be migrated to pack.required.
// Keeping the legacy packing shape for this reconciliation lets the next
// reconciliation apply the generated modern shape without recreating the PCS.
func prepareGroveTopologyConstraintUpgrade(
	desired *grovev1alpha1.PodCliqueSet,
	existing *grovev1alpha1.PodCliqueSet,
) {
	if desired == nil || existing == nil {
		return
	}

	prepareLegacyGroveTopologyConstraintRepair(
		desired.Spec.Template.TopologyConstraint,
		existing.Spec.Template.TopologyConstraint,
	)

	existingCliqueConstraints := make(
		map[string]*grovev1alpha1.TopologyConstraint,
		len(existing.Spec.Template.Cliques),
	)
	for _, clique := range existing.Spec.Template.Cliques {
		if clique != nil {
			existingCliqueConstraints[clique.Name] = clique.TopologyConstraint
		}
	}
	for _, clique := range desired.Spec.Template.Cliques {
		if clique != nil {
			prepareLegacyGroveTopologyConstraintRepair(
				clique.TopologyConstraint,
				existingCliqueConstraints[clique.Name],
			)
		}
	}

	existingScalingGroupConstraints := make(
		map[string]*grovev1alpha1.TopologyConstraint,
		len(existing.Spec.Template.PodCliqueScalingGroupConfigs),
	)
	for i := range existing.Spec.Template.PodCliqueScalingGroupConfigs {
		config := &existing.Spec.Template.PodCliqueScalingGroupConfigs[i]
		existingScalingGroupConstraints[config.Name] = config.TopologyConstraint
	}
	for i := range desired.Spec.Template.PodCliqueScalingGroupConfigs {
		config := &desired.Spec.Template.PodCliqueScalingGroupConfigs[i]
		prepareLegacyGroveTopologyConstraintRepair(
			config.TopologyConstraint,
			existingScalingGroupConstraints[config.Name],
		)
	}
}

func prepareLegacyGroveTopologyConstraintRepair(
	desired *grovev1alpha1.TopologyConstraint,
	existing *grovev1alpha1.TopologyConstraint,
) {
	if desired == nil || existing == nil {
		return
	}
	// A constraint without an explicit desired name inherits from its repaired
	// parent and can migrate packDomain directly in the same update.
	if desired.TopologyName == "" || existing.TopologyName != "" || existing.PackDomain == "" {
		return
	}

	desired.PackDomain = existing.PackDomain
	if existing.Pack == nil {
		desired.Pack = nil
		return
	}
	pack := *existing.Pack
	desired.Pack = &pack
}

// Keep Grove-owned scaling state stable across parent PCS reconciliation.
// Horizontal replicas are driven through scale subresources.
//
//nolint:gocyclo // Grove scale fields share one ordered snapshot traversal.
func preserveGrovePodCliqueSetReplicas(
	desired *grovev1alpha1.PodCliqueSet,
	existing *grovev1alpha1.PodCliqueSet,
	checkpointInfoByComponent map[string]*checkpoint.CheckpointInfo,
) {
	if desired == nil || existing == nil {
		return
	}

	// Let checkpoint-gated components retain their generated zero replicas.
	replicaPreserveSkips := map[string]struct{}{}
	for componentName, info := range checkpointInfoByComponent {
		if info != nil &&
			info.Enabled &&
			info.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint &&
			!info.Ready {
			replicaPreserveSkips[strings.ToLower(componentName)] = struct{}{}
		}
	}

	cliquesInScalingGroups := make(map[string]struct{})
	for _, config := range desired.Spec.Template.PodCliqueScalingGroupConfigs {
		for _, cliqueName := range config.CliqueNames {
			cliquesInScalingGroups[cliqueName] = struct{}{}
		}
	}

	cliqueReplicasByName := make(map[string]int32, len(existing.Spec.Template.Cliques))
	for _, clique := range existing.Spec.Template.Cliques {
		if clique == nil || clique.Name == "" {
			continue
		}
		cliqueReplicasByName[clique.Name] = clique.Spec.Replicas
	}
	for _, clique := range desired.Spec.Template.Cliques {
		if clique == nil {
			continue
		}
		if _, inScalingGroup := cliquesInScalingGroups[clique.Name]; inScalingGroup {
			continue
		}
		if componentName := clique.Labels[commonconsts.KubeLabelDynamoComponent]; componentName != "" {
			if _, skip := replicaPreserveSkips[strings.ToLower(componentName)]; skip {
				continue
			}
		}
		if replicas, ok := cliqueReplicasByName[clique.Name]; ok {
			clique.Spec.Replicas = replicas
		}
	}

	// Index existing scaling-group replicas by name.
	scalingGroupReplicasByName := make(map[string]*int32, len(existing.Spec.Template.PodCliqueScalingGroupConfigs))
	for _, config := range existing.Spec.Template.PodCliqueScalingGroupConfigs {
		if config.Name != "" {
			scalingGroupReplicasByName[config.Name] = config.Replicas
		}
	}

	// Preserve scale-owned fields in generated configuration order.
	for i := range desired.Spec.Template.PodCliqueScalingGroupConfigs {
		config := &desired.Spec.Template.PodCliqueScalingGroupConfigs[i]
		if _, skip := replicaPreserveSkips[strings.ToLower(config.Name)]; skip {
			continue
		}
		if replicas, ok := scalingGroupReplicasByName[config.Name]; ok {
			config.Replicas = replicas
		}
	}
}

func orderLikeExisting[T any](existing []T, desired []T, nameOf func(T) string) []T {
	if len(existing) == 0 || len(desired) < 2 {
		return desired
	}
	desiredByName := make(map[string]T, len(desired))
	for _, item := range desired {
		if name := nameOf(item); name != "" {
			desiredByName[name] = item
		}
	}
	ordered := make([]T, 0, len(desired))
	used := make(map[string]struct{}, len(desired))
	for _, existingItem := range existing {
		name := nameOf(existingItem)
		if desiredItem, ok := desiredByName[name]; ok {
			ordered = append(ordered, desiredItem)
			used[name] = struct{}{}
		}
	}
	for _, item := range desired {
		name := nameOf(item)
		if name == "" {
			ordered = append(ordered, item)
			continue
		}
		if _, ok := used[name]; !ok {
			ordered = append(ordered, item)
		}
	}
	return ordered
}

func podCliqueTemplateName(clique *grovev1alpha1.PodCliqueTemplateSpec) string {
	if clique == nil {
		return ""
	}
	return clique.Name
}

func podCliqueScalingGroupConfigName(config grovev1alpha1.PodCliqueScalingGroupConfig) string {
	return config.Name
}

func resourceClaimTemplateConfigName(config grovev1alpha1.ResourceClaimTemplateConfig) string {
	return config.Name
}
