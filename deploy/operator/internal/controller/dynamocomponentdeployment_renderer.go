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
	"maps"
	"strings"
	"sync"

	"emperror.dev/errors"
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

const dcdWorkloadRoleLabel = "role"

// dcdWorkloadRenderer contains the dependencies required to render the
// workload-facing resources of a DynamoComponentDeployment. It deliberately
// does not own reconciliation, watches, finalizers, or status.
//
// Keeping this concrete and package-private establishes a reusable rendering
// boundary without committing to a public provider framework. Composite
// workload programs can reuse this unit without constructing a
// DynamoComponentDeploymentReconciler.
type dcdWorkloadRenderer struct {
	reader                client.Reader
	config                *configv1alpha1.OperatorConfiguration
	runtimeConfig         *commonController.RuntimeConfig
	dockerSecretRetriever DockerSecretRetriever
}

func newDCDWorkloadRenderer(
	reader client.Reader,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commonController.RuntimeConfig,
	dockerSecretRetriever DockerSecretRetriever,
) *dcdWorkloadRenderer {
	return &dcdWorkloadRenderer{
		reader:                reader,
		config:                config,
		runtimeConfig:         runtimeConfig,
		dockerSecretRetriever: dockerSecretRetriever,
	}
}

func (r *DynamoComponentDeploymentReconciler) workloadRenderer() *dcdWorkloadRenderer {
	return newDCDWorkloadRenderer(r.Client, r.Config, r.RuntimeConfig, r.DockerSecretRetriever)
}

// renderMultinodePodTemplateSpecs renders the leader and worker pod templates
// shared by LWS and composite multinode workload resources. The caller remains
// responsible for composing those templates into its provider-native object.
func (r *dcdWorkloadRenderer) renderMultinodePodTemplateSpecs(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (*corev1.PodTemplateSpec, *corev1.PodTemplateSpec, error) {
	podLabels, err := r.getDCDWorkloadPodLabels(ctx, dcd)
	if err != nil {
		return nil, nil, err
	}
	containerGPUs := r.containerGPUCount(ctx, dcd)

	leaderLabels := make(map[string]string, len(podLabels))
	maps.Copy(leaderLabels, podLabels)
	leaderPodTemplateSpec, err := r.generateLeaderPodTemplateSpec(ctx, dcd, leaderLabels, containerGPUs)
	if err != nil {
		return nil, nil, err
	}

	workerLabels := make(map[string]string, len(podLabels))
	maps.Copy(workerLabels, podLabels)
	workerPodTemplateSpec, err := r.generateWorkerPodTemplateSpec(ctx, dcd, workerLabels, containerGPUs)
	if err != nil {
		return nil, nil, err
	}

	return leaderPodTemplateSpec, workerPodTemplateSpec, nil
}

func (r *dcdWorkloadRenderer) containerGPUCount(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) dynamo.ContainerGPUCount {
	return sync.OnceValues(func() (int64, error) {
		return dynamo.ResolveContainerGPUs(ctx, r.reader, dcd.Namespace, &dcd.Spec.DynamoComponentDeploymentSharedSpec)
	})
}

func (r *dcdWorkloadRenderer) generateLeaderPodTemplateSpec(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	labels map[string]string,
	containerGPUs dynamo.ContainerGPUCount,
) (*corev1.PodTemplateSpec, error) {
	leaderPodTemplateSpec, err := r.generatePodTemplateSpec(ctx, dcd, dynamo.RoleLeader, containerGPUs)
	if err != nil {
		return nil, errors.Wrap(err, "failed to generate leader pod template")
	}

	maps.Copy(leaderPodTemplateSpec.ObjectMeta.Labels, labels)
	leaderPodTemplateSpec.ObjectMeta.Labels[dcdWorkloadRoleLabel] = string(dynamo.RoleLeader)
	delete(leaderPodTemplateSpec.ObjectMeta.Labels, commonconsts.KubeLabelDynamoSelector)

	if err := checkMainContainer(&leaderPodTemplateSpec.Spec); err != nil {
		return nil, errors.Wrap(err, "generateLeaderPodTemplateSpec: failed to check main container")
	}

	return leaderPodTemplateSpec, nil
}

func (r *dcdWorkloadRenderer) generateWorkerPodTemplateSpec(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	labels map[string]string,
	containerGPUs dynamo.ContainerGPUCount,
) (*corev1.PodTemplateSpec, error) {
	workerPodTemplateSpec, err := r.generatePodTemplateSpec(ctx, dcd, dynamo.RoleWorker, containerGPUs)
	if err != nil {
		return nil, errors.Wrap(err, "failed to generate worker pod template")
	}

	maps.Copy(workerPodTemplateSpec.ObjectMeta.Labels, labels)
	workerPodTemplateSpec.ObjectMeta.Labels[dcdWorkloadRoleLabel] = string(dynamo.RoleWorker)
	delete(workerPodTemplateSpec.ObjectMeta.Labels, commonconsts.KubeLabelDynamoSelector)

	if err := checkMainContainer(&workerPodTemplateSpec.Spec); err != nil {
		return nil, errors.Wrap(err, "generateWorkerPodTemplateSpec: failed to check LWS worker main container")
	}

	return workerPodTemplateSpec, nil
}

func (r *dcdWorkloadRenderer) generatePodTemplateSpec(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	role dynamo.Role,
	containerGPUs dynamo.ContainerGPUCount,
) (*corev1.PodTemplateSpec, error) {
	component := &dcd.Spec.DynamoComponentDeploymentSharedSpec
	componentType, err := r.getDCDWorkloadComponentType(ctx, dcd)
	if err != nil {
		return nil, err
	}
	podLabels := dynamo.GetDCDKubeLabels(dcd)
	podAnnotations := dynamo.GetDCDKubeAnnotations(dcd)
	kubeName := dcd.Name

	// Convert user-provided metrics annotation into controller-managed label.
	// By default (no annotation), metrics are enabled.
	if podAnnotations[commonconsts.KubeAnnotationEnableMetrics] != commonconsts.KubeLabelValueFalse {
		podLabels[commonconsts.KubeLabelMetricsEnabled] = commonconsts.KubeLabelValueTrue
	}

	if parentName := dcd.GetLabels()[commonconsts.KubeLabelDynamoGraphDeploymentName]; parentName != "" {
		podLabels[commonconsts.KubeLabelDynamoGraphDeploymentName] = parentName
	} else if parentName := dcd.GetParentGraphDeploymentName(); parentName != "" {
		podLabels[commonconsts.KubeLabelDynamoGraphDeploymentName] = parentName
	}
	if componentType != "" {
		podLabels[commonconsts.KubeLabelDynamoComponentType] = componentType
	}
	if componentName := dynamo.GetDCDComponentName(dcd); componentName != "" {
		podLabels[commonconsts.KubeLabelDynamoComponent] = componentName
	}
	if dynamoNamespace := dynamo.GetDCDDynamoNamespace(dcd); dynamoNamespace != "" {
		podLabels[commonconsts.KubeLabelDynamoNamespace] = dynamoNamespace
	}
	if workerHash := dcd.GetLabels()[commonconsts.KubeLabelDynamoWorkerHash]; workerHash != "" {
		podLabels[commonconsts.KubeLabelDynamoWorkerHash] = workerHash
	}

	var checkpointInfo *checkpoint.CheckpointInfo
	if checkpointConfig := dynamo.GetCheckpoint(component); r.runtimeConfig.Gate.Enabled(features.Checkpoint) && checkpointConfig != nil {
		info, err := checkpoint.ResolveCheckpointForService(
			ctx,
			r.reader,
			dcd.Namespace,
			dynamo.ToAlphaCheckpointConfig(checkpointConfig),
		)
		if err != nil {
			return nil, errors.Wrap(err, "failed to resolve checkpoint")
		}
		if dynamo.IsIntraPodFailoverEnabled(&dcd.Spec.DynamoComponentDeploymentSharedSpec) {
			info.RestoreTargetContainers = dynamo.IntraPodFailoverEngineContainerNames()
		}
		if err := gms.OverlayClients(
			&info.GPUMemoryService,
			info.CheckpointName,
			info.Exists,
			dynamo.GetGPUMemoryService(component),
		); err != nil {
			return nil, errors.Wrap(err, "failed to apply checkpoint gpuMemoryService config")
		}
		checkpointInfo = info
	}

	podSpec, err := dynamo.GenerateBasePodSpecForController(
		dcd,
		r.dockerSecretRetriever,
		r.config,
		role,
		commonconsts.MultinodeDeploymentTypeLWS,
		checkpointInfo,
		containerGPUs,
		dynamo.GenerateBasePodSpecForControllerOptions{
			WorkloadComponentType: nvidiacomv1beta1.ComponentType(componentType),
		},
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to generate base pod spec")
	}
	if r.runtimeConfig.Gate.Enabled(features.Checkpoint) {
		if checkpointInfo == nil ||
			string(checkpointInfo.StartupPolicy) == string(nvidiacomv1beta1.CheckpointStartupPolicyWaitForCheckpoint) {
			if err := checkpoint.InjectCheckpointIntoPodSpecWithStorageConfig(
				ctx,
				r.reader,
				dcd.Namespace,
				podSpec,
				checkpointInfo,
				r.config.Checkpoint.Storage,
				r.config.Checkpoint.EffectiveSeccompProfile(),
			); err != nil {
				return nil, errors.Wrap(err, "failed to inject checkpoint config")
			}
		}
	}

	// Apply the transactional power gate only after every command-rendering step.
	powerGateInputs, err := r.powerGateInputs(ctx, dcd, componentType, role)
	if err != nil {
		return nil, errors.Wrap(err, "failed to resolve transactional power gate")
	}
	if powerGateInputs != nil {
		if err := dynamo.ApplyPowerGate(podSpec, *powerGateInputs); err != nil {
			return nil, errors.Wrap(err, "failed to apply transactional power gate")
		}
	}

	if len(podSpec.Containers) == 0 {
		return nil, errors.New("no containers found in base pod spec")
	}

	podLabels[commonconsts.KubeLabelDynamoSelector] = kubeName

	if commonController.IsK8sDiscoveryEnabled(r.config.Discovery.Backend, podAnnotations) {
		podLabels[commonconsts.KubeLabelDynamoDiscoveryBackend] = "kubernetes"
		podLabels[commonconsts.KubeLabelDynamoDiscoveryEnabled] = commonconsts.KubeLabelValueTrue
	}

	if checkpointInfo != nil &&
		(checkpointInfo.StartupPolicy == "" ||
			string(checkpointInfo.StartupPolicy) == string(nvidiacomv1beta1.CheckpointStartupPolicyImmediate)) {
		if err := checkpoint.ApplyRestoreCandidateMetadata(podLabels, podAnnotations, checkpointInfo); err != nil {
			return nil, errors.Wrap(err, "failed to apply checkpoint candidate metadata")
		}
	} else if err := checkpoint.ApplyRestorePodMetadataWithStorageConfig(
		podLabels,
		podAnnotations,
		checkpointInfo,
		r.config.Checkpoint.Storage,
	); err != nil {
		return nil, errors.Wrap(err, "failed to apply checkpoint metadata")
	}

	if podSpec.ServiceAccountName == "" {
		serviceAccounts := &corev1.ServiceAccountList{}
		err = r.reader.List(ctx, serviceAccounts, client.InNamespace(dcd.Namespace), client.MatchingLabels{
			commonconsts.KubeLabelDynamoComponentPod: commonconsts.KubeLabelValueTrue,
		})
		if err != nil {
			return nil, errors.Wrapf(err, "failed to list service accounts in namespace %s", dcd.Namespace)
		}
		if len(serviceAccounts.Items) > 0 {
			podSpec.ServiceAccountName = serviceAccounts.Items[0].Name
		} else {
			podSpec.ServiceAccountName = DefaultServiceAccountName
		}
	}

	return &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      podLabels,
			Annotations: podAnnotations,
		},
		Spec: *podSpec,
	}, nil
}

// powerGateInputs resolves immutable gate identity and bounds from the
// owning DGD and its DGPB. Unenrolled DGDs retain the static deployment path.
func (r *dcdWorkloadRenderer) powerGateInputs(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	componentType string,
	role dynamo.Role,
) (*dynamo.PowerGateInputs, error) {
	// Require the direct DGD controller identity before consulting its policy.
	dgdOwner := metav1.GetControllerOf(dcd)
	if !isDGDControllerReference(dgdOwner) {
		return nil, nil
	}

	// Use the immutable parent enrollment annotation as the sole mode authority.
	key := types.NamespacedName{Namespace: dcd.Namespace, Name: dgdOwner.Name}
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err := r.reader.Get(ctx, key, dgd); err != nil {
		return nil, fmt.Errorf("read owning DynamoGraphDeployment %s: %w", key, err)
	}
	if dgd.UID != dgdOwner.UID {
		return nil, fmt.Errorf("DynamoGraphDeployment %s UID %q does not match owner UID %q", key, dgd.UID, dgdOwner.UID)
	}
	if dgd.Annotations[nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation] !=
		nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence {
		return nil, nil
	}

	// In transactional mode the immutable parent, not the mutable child DCD,
	// decides whether this workload is a power-managed worker. Bind the DCD's
	// immutable resource name to exactly one parent component before trusting
	// any child classification fields.
	parentComponent, err := parentComponentForDCD(dgd, dcd)
	if err != nil {
		return nil, err
	}
	componentName := parentComponent.ComponentName
	if dcd.Spec.ComponentType != parentComponent.ComponentType ||
		dynamo.GetDCDComponentName(dcd) != componentName {
		return nil, fmt.Errorf(
			"transactional DynamoComponentDeployment %s/%s does not match parent component %q",
			dcd.Namespace,
			dcd.Name,
			componentName,
		)
	}
	if !dynamo.IsWorkerComponent(string(parentComponent.ComponentType)) {
		if componentType != string(parentComponent.ComponentType) {
			return nil, fmt.Errorf(
				"transactional DynamoComponentDeployment %s/%s workload type %q does not match nonworker parent component %q",
				dcd.Namespace,
				dcd.Name,
				componentType,
				componentName,
			)
		}
		return nil, nil
	}
	if !dynamo.IsWorkerComponent(componentType) {
		return nil, fmt.Errorf(
			"transactional DynamoComponentDeployment %s/%s does not match power-managed parent component %q",
			dcd.Namespace,
			dcd.Name,
			componentName,
		)
	}

	// The one DGPB is named after its owning DGD and is mandatory once enrolled.
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := r.reader.Get(ctx, key, dgpb); err != nil {
		if k8serrors.IsNotFound(err) {
			return nil, fmt.Errorf("transactional DynamoGraphPowerBudget %s is not available", key)
		}
		return nil, fmt.Errorf("read DynamoGraphPowerBudget %s: %w", key, err)
	}

	// Bind the rendered workload to the same exact DGD generation on both objects.
	dgpbOwner := metav1.GetControllerOf(dgpb)
	if !isDGDControllerReference(dgpbOwner) ||
		dgpbOwner.Name != dgdOwner.Name ||
		dgpbOwner.UID != dgdOwner.UID ||
		dgpb.Status.DGDUID != string(dgdOwner.UID) {
		return nil, fmt.Errorf("DynamoGraphPowerBudget %s does not match DGD UID %q", key, dgdOwner.UID)
	}
	if role != dynamo.RoleMain {
		return nil, fmt.Errorf("DynamoGraphPowerBudget %s cannot gate unsupported role %q", key, role)
	}

	// Resolve exactly one bounded component row for the main-container allocation.
	var matched *nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus
	for index := range dgpb.Status.Components {
		if dgpb.Status.Components[index].Name != componentName {
			continue
		}
		if matched != nil {
			return nil, fmt.Errorf("DynamoGraphPowerBudget %s has duplicate component %q", key, componentName)
		}
		matched = &dgpb.Status.Components[index]
	}
	if matched == nil || matched.PhysicalGPUsPerReplica < 1 || matched.InGateBoundWattsPerGPU < 1 {
		return nil, fmt.Errorf("DynamoGraphPowerBudget %s lacks complete bounds for component %q", key, componentName)
	}

	return &dynamo.PowerGateInputs{
		DGDUID:                   string(dgdOwner.UID),
		Component:                componentName,
		ExpectedPhysicalGPUCount: matched.PhysicalGPUsPerReplica,
		InGateBoundWattsPerGPU:   matched.InGateBoundWattsPerGPU,
	}, nil
}

func parentComponentForDCD(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, error) {
	var matched *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
	for index := range dgd.Spec.Components {
		component := &dgd.Spec.Components[index]
		baseName := dynamo.GetDCDResourceName(dgd, component.ComponentName, "")
		if dcd.Name != baseName && !hasDCDWorkerHashSuffix(dcd.Name, baseName) {
			continue
		}
		if matched != nil {
			return nil, fmt.Errorf("DynamoComponentDeployment %s/%s matches multiple parent components", dcd.Namespace, dcd.Name)
		}
		matched = component
	}
	if matched == nil {
		return nil, fmt.Errorf("DynamoComponentDeployment %s/%s does not match an owning DGD component", dcd.Namespace, dcd.Name)
	}
	return matched, nil
}

func hasDCDWorkerHashSuffix(name, baseName string) bool {
	if len(name) != len(baseName)+9 || !strings.HasPrefix(name, baseName+"-") {
		return false
	}
	for _, character := range name[len(baseName)+1:] {
		if !strings.ContainsRune("0123456789abcdef", character) {
			return false
		}
	}
	return true
}

func isDGDControllerReference(owner *metav1.OwnerReference) bool {
	return owner != nil &&
		owner.Kind == nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind &&
		strings.HasPrefix(owner.APIVersion, nvidiacomv1beta1.GroupVersion.Group+"/")
}

func (r *dcdWorkloadRenderer) generateService(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (*corev1.Service, bool, error) {
	deleteStub := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dynamo.NormalizeKubeResourceName(dcd.Name),
			Namespace: dcd.Namespace,
		},
	}

	annotations := dynamo.GetDCDKubeAnnotations(dcd)
	isK8sDiscovery := commonController.IsK8sDiscoveryEnabled(r.config.Discovery.Backend, annotations)

	if !(isK8sDiscovery || dcd.IsFrontendComponent()) {
		return deleteStub, true, nil
	}

	dynamoNamespace := dynamo.GetDCDDynamoNamespace(dcd)
	if dynamoNamespace == "" {
		return nil, false, fmt.Errorf("expected DynamoComponentDeployment %s to have a dynamoNamespace", dcd.Name)
	}

	componentType, err := r.getDCDWorkloadComponentType(ctx, dcd)
	if err != nil {
		return nil, false, err
	}

	svc, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
		ServiceName:     dcd.Name,
		Namespace:       dcd.Namespace,
		ComponentType:   componentType,
		DynamoNamespace: dynamoNamespace,
		ComponentName:   dynamo.GetDCDComponentName(dcd),
		Labels:          dynamo.GetDCDKubeLabels(dcd),
		Annotations:     annotations,
		IsK8sDiscovery:  isK8sDiscovery,
	})
	if err != nil {
		return nil, false, err
	}
	if dcd.IsMultinode() {
		svc.Spec.Selector[dcdWorkloadRoleLabel] = string(dynamo.RoleLeader)
	}
	return svc, false, nil
}

func (r *dcdWorkloadRenderer) getDCDWorkloadPodLabels(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (map[string]string, error) {
	labels := dynamo.GetDCDKubeLabels(dcd)
	componentType, err := r.getDCDWorkloadComponentType(ctx, dcd)
	if err != nil {
		return nil, err
	}
	if componentType == "" {
		return labels, nil
	}
	labels[commonconsts.KubeLabelDynamoComponentType] = componentType
	specType := string(dcd.Spec.ComponentType)
	if componentType == commonconsts.ComponentTypeWorker &&
		(specType == commonconsts.ComponentTypePrefill || specType == commonconsts.ComponentTypeDecode) &&
		labels[commonconsts.KubeLabelDynamoSubComponentType] == "" {
		labels[commonconsts.KubeLabelDynamoSubComponentType] = specType
	}
	return labels, nil
}

// getDCDWorkloadComponentType returns the component type that should be
// rendered into pod metadata, env, and service selectors for this DCD. It keeps
// legacy-compatible worker generations as "worker" even when the v1beta1 DCD
// spec is represented as a more specific prefill/decode worker component.
func (r *dcdWorkloadRenderer) getDCDWorkloadComponentType(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (string, error) {
	if dcd == nil {
		return "", nil
	}

	componentType := dynamo.GetDCDWorkloadComponentType(dcd)
	if componentType == commonconsts.ComponentTypeWorker || !dynamo.IsWorkerComponent(componentType) {
		return componentType, nil
	}

	if hasLegacyWorkerSelector(dcd.GetLabels(), componentType) {
		return commonconsts.ComponentTypeWorker, nil
	}

	legacy, err := r.hasExistingLegacyWorkerSelector(ctx, dcd, componentType)
	if err != nil {
		return "", err
	}
	if legacy {
		return commonconsts.ComponentTypeWorker, nil
	}

	return componentType, nil
}

func (r *dcdWorkloadRenderer) hasExistingLegacyWorkerSelector(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	componentType string,
) (bool, error) {
	if dcd == nil || r == nil || r.reader == nil {
		return false, nil
	}

	deployment := &appsv1.Deployment{}
	if err := r.reader.Get(ctx, types.NamespacedName{Name: dcd.Name, Namespace: dcd.Namespace}, deployment); err == nil {
		if hasLegacyWorkerSelector(deployment.Spec.Template.Labels, componentType) {
			return true, nil
		}
	} else if !k8serrors.IsNotFound(err) {
		return false, fmt.Errorf("failed to get deployment %s/%s: %w", dcd.Namespace, dcd.Name, err)
	}

	if r.runtimeConfig.Gate.Enabled(features.LWS) {
		lwsName := leaderWorkerSetName(dcd)
		leaderWorkerSet := &leaderworkersetv1.LeaderWorkerSet{}
		if err := r.reader.Get(ctx, types.NamespacedName{Name: lwsName, Namespace: dcd.Namespace}, leaderWorkerSet); err == nil {
			template := leaderWorkerSet.Spec.LeaderWorkerTemplate
			if template.LeaderTemplate != nil && hasLegacyWorkerSelector(template.LeaderTemplate.Labels, componentType) {
				return true, nil
			}
			if hasLegacyWorkerSelector(template.WorkerTemplate.Labels, componentType) {
				return true, nil
			}
		} else if !k8serrors.IsNotFound(err) {
			return false, fmt.Errorf("failed to get leaderworkerset %s/%s: %w", dcd.Namespace, lwsName, err)
		}
	}

	serviceName := dynamo.NormalizeKubeResourceName(dcd.Name)
	service := &corev1.Service{}
	if err := r.reader.Get(ctx, types.NamespacedName{Name: serviceName, Namespace: dcd.Namespace}, service); err == nil {
		return hasLegacyWorkerSelector(service.Spec.Selector, componentType), nil
	} else if !k8serrors.IsNotFound(err) {
		return false, fmt.Errorf("failed to get service %s/%s: %w", dcd.Namespace, serviceName, err)
	}

	return false, nil
}
