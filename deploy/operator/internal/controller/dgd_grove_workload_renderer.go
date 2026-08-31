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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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

// grovePodCliqueSetRender couples the desired PCS and rendered DGD to the
// exact observation used to decide compatibility and the worker hash suffix.
type grovePodCliqueSetRender struct {
	existing         *grovev1alpha1.PodCliqueSet
	desired          *grovev1alpha1.PodCliqueSet
	renderDeployment *nvidiacomv1beta1.DynamoGraphDeployment
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
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	restartState *dynamo.RestartState,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
	workerGenerationChanged bool,
) (*grovePodCliqueSetRender, error) {
	if dgd == nil {
		return nil, fmt.Errorf("cannot render Grove PodCliqueSet without a DynamoGraphDeployment")
	}
	if r.reader == nil {
		return nil, fmt.Errorf("cannot render Grove PodCliqueSet without a Kubernetes reader")
	}
	existingPodCliqueSet := &grovev1alpha1.PodCliqueSet{}
	key := types.NamespacedName{
		Name:      dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components),
		Namespace: dgd.Namespace,
	}
	if err := r.reader.Get(ctx, key, existingPodCliqueSet); err != nil {
		if !apierrors.IsNotFound(err) {
			return nil, fmt.Errorf("get PodCliqueSet %s: %w", key, err)
		}
		existingPodCliqueSet = nil
	}

	workerHashSuffixNeeded := shouldRenderGroveWorkerHashSuffix(dgd, existingPodCliqueSet, workerGenerationChanged)
	renderDeployment, err := groveRenderDeployment(dgd, existingPodCliqueSet, workerHashSuffixNeeded)
	if err != nil {
		return nil, err
	}
	desired, err := r.renderPodCliqueSet(
		ctx, renderDeployment, existingPodCliqueSet, restartState, checkpointInfos,
	)
	if err != nil {
		return nil, err
	}
	return &grovePodCliqueSetRender{
		existing:         existingPodCliqueSet,
		desired:          desired,
		renderDeployment: renderDeployment,
	}, nil
}

func (r *groveWorkloadRenderer) renderPodCliqueSet(
	ctx context.Context,
	renderDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	existing *grovev1alpha1.PodCliqueSet,
	restartState *dynamo.RestartState,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) (*grovev1alpha1.PodCliqueSet, error) {
	existingRestartAnnotations := restartAnnotationsFromPodCliqueSet(existing)
	desired, err := dynamo.GenerateGrovePodCliqueSet(
		ctx,
		renderDeployment,
		r.config,
		r.runtimeConfig,
		r.reader,
		r.dockerSecretRetriever,
		restartState,
		existingRestartAnnotations,
		checkpointInfos,
	)
	if err != nil {
		return nil, err
	}

	prepareGroveTopologyConstraintUpgrade(desired, existing)
	preserveGrovePodCliqueSetOrder(desired, existing)
	preserveGrovePodCliqueSetReplicas(desired, existing, checkpointInfos)
	return desired, nil
}

func groveRenderDeployment(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	workerHashSuffix bool,
) (*nvidiacomv1beta1.DynamoGraphDeployment, error) {
	renderDeployment := dgd.DeepCopy()
	applyGroveCompatibility(renderDeployment, pcs)
	if !workerHashSuffix {
		return renderDeployment, nil
	}
	if err := applyGroveWorkerHashSuffix(renderDeployment, dgd); err != nil {
		return nil, err
	}
	return renderDeployment, nil
}

func applyGroveWorkerHashSuffix(
	renderDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	hashSource *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	workerHash, err := dynamo.ComputeDGDWorkersSpecHash(hashSource)
	if err != nil {
		return fmt.Errorf("compute Grove worker hash suffix: %w", err)
	}
	for i := range renderDeployment.Spec.Components {
		component := &renderDeployment.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		if component.PodTemplate == nil {
			component.PodTemplate = &corev1.PodTemplateSpec{}
		}
		if component.PodTemplate.Labels == nil {
			component.PodTemplate.Labels = make(map[string]string)
		}
		component.PodTemplate.Labels[commonconsts.KubeLabelDynamoWorkerHash] = workerHash
	}
	return nil
}

func shouldRenderGroveWorkerHashSuffix(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	existing *grovev1alpha1.PodCliqueSet,
	workerGenerationChanged bool,
) bool {
	if !dgdHasWorkerComponents(dgd) {
		return false
	}
	if existing == nil || podCliqueSetUsesGroveWorkerHashSuffix(dgd, existing) {
		return true
	}

	return workerGenerationChanged
}

func dgdHasWorkerComponents(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	for i := range dgd.Spec.Components {
		if dynamo.IsWorkerComponent(string(dgd.Spec.Components[i].ComponentType)) {
			return true
		}
	}
	return false
}

func podCliqueSetUsesGroveWorkerHashSuffix(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
) bool {
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
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

func applyGroveCompatibility(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
) {
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		componentType := string(component.ComponentType)
		if !groveComponentTypeCanUseLegacyWorkerSelector(componentType) {
			continue
		}
		if podCliqueSetHasLegacyWorkerSelector(pcs, component.ComponentName, componentType) {
			applyLegacyGroveWorkerComponentType(component, componentType)
		}
	}
}

func restartAnnotationsFromPodCliqueSet(pcs *grovev1alpha1.PodCliqueSet) map[string]string {
	restartAnnotations := make(map[string]string)
	if pcs == nil {
		return restartAnnotations
	}
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique.Annotations != nil {
			if timestamp, ok := clique.Annotations[commonconsts.RestartAnnotation]; ok {
				if componentName, ok := clique.Labels[commonconsts.KubeLabelDynamoComponent]; ok {
					restartAnnotations[componentName] = timestamp
				}
			}
		}
	}
	return restartAnnotations
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

// Grove horizontal replicas are driven through scale subresources after
// creation; keep existing template values so DGD replica changes do not update
// the PodCliqueSet spec.
func preserveGrovePodCliqueSetReplicas(
	desired *grovev1alpha1.PodCliqueSet,
	existing *grovev1alpha1.PodCliqueSet,
	checkpointInfoByComponent ...map[string]*checkpoint.CheckpointInfo,
) {
	if desired == nil || existing == nil {
		return
	}
	replicaPreserveSkips := map[string]struct{}{}
	if len(checkpointInfoByComponent) > 0 {
		for componentName, info := range checkpointInfoByComponent[0] {
			if info != nil &&
				info.Enabled &&
				info.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint &&
				!info.Ready {
				replicaPreserveSkips[strings.ToLower(componentName)] = struct{}{}
			}
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

	scalingGroupReplicasByName := make(
		map[string]*int32,
		len(existing.Spec.Template.PodCliqueScalingGroupConfigs),
	)
	for _, config := range existing.Spec.Template.PodCliqueScalingGroupConfigs {
		if config.Name == "" {
			// Defensive only; generated PCSG configs always have names.
			continue
		}
		scalingGroupReplicasByName[config.Name] = config.Replicas
	}
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

func groveComponentTypeCanUseLegacyWorkerSelector(componentType string) bool {
	return componentType == commonconsts.ComponentTypePrefill ||
		componentType == commonconsts.ComponentTypeDecode
}

func podCliqueSetHasLegacyWorkerSelector(
	pcs *grovev1alpha1.PodCliqueSet,
	componentName string,
	componentType string,
) bool {
	if pcs == nil {
		return false
	}
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique == nil || clique.Labels[commonconsts.KubeLabelDynamoComponent] != componentName {
			continue
		}
		if hasLegacyWorkerSelector(clique.Labels, componentType) {
			return true
		}
	}
	return false
}

func applyLegacyGroveWorkerComponentType(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	subComponentType string,
) {
	component.ComponentType = nvidiacomv1beta1.ComponentTypeWorker
	if component.PodTemplate == nil {
		component.PodTemplate = &corev1.PodTemplateSpec{}
	}
	if component.PodTemplate.Labels == nil {
		component.PodTemplate.Labels = map[string]string{}
	}
	if _, ok := component.PodTemplate.Labels[commonconsts.KubeLabelDynamoSubComponentType]; !ok {
		component.PodTemplate.Labels[commonconsts.KubeLabelDynamoSubComponentType] = subComponentType
	}
}
