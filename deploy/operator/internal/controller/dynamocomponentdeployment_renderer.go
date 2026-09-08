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
	"sync"

	"emperror.dev/errors"
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
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
	"k8s.io/apimachinery/pkg/runtime/schema"
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

	checkpointInfo, err := r.resolveCheckpointInfo(ctx, dcd, component)
	if err != nil {
		return nil, err
	}

	podSpec, err := dynamo.GenerateBasePodSpecForController(
		dcd,
		r.dockerSecretRetriever,
		r.config,
		role,
		commonconsts.MultinodeDeploymentTypeLWS,
		containerGPUs,
		dynamo.GenerateBasePodSpecForControllerOptions{
			WorkloadComponentType: nvidiacomv1beta1.ComponentType(componentType),
		},
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to generate base pod spec")
	}
	if len(podSpec.Containers) == 0 {
		return nil, errors.New("no containers found in base pod spec")
	}

	podLabels[commonconsts.KubeLabelDynamoSelector] = kubeName

	if commonController.IsK8sDiscoveryEnabled(r.config.Discovery.Backend, podAnnotations) {
		podLabels[commonconsts.KubeLabelDynamoDiscoveryBackend] = "kubernetes"
		podLabels[commonconsts.KubeLabelDynamoDiscoveryEnabled] = commonconsts.KubeLabelValueTrue
	}

	if r.runtimeConfig.Gate.Enabled(features.Checkpoint) {
		if err := checkpoint.ApplyRestoreCandidateMetadata(podAnnotations, checkpointInfo); err != nil {
			return nil, errors.Wrap(err, "failed to apply checkpoint candidate metadata")
		}
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

func (r *dcdWorkloadRenderer) resolveCheckpointInfo(
	ctx context.Context,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) (*checkpoint.CheckpointInfo, error) {
	checkpointConfig := dynamo.GetCheckpoint(component)
	if !r.runtimeConfig.Gate.Enabled(features.Checkpoint) || checkpointConfig == nil {
		return nil, nil
	}

	alphaCheckpointConfig := dynamo.ToAlphaCheckpointConfig(checkpointConfig)
	automaticSnapshotJob, err := automaticSnapshotJobReferenceForDCD(dcd)
	if err != nil {
		return nil, err
	}
	var info *checkpoint.CheckpointInfo
	if checkpointConfig.CheckpointRef == nil || *checkpointConfig.CheckpointRef == "" {
		// A DGD-generated DCD temporarily has no reference while its automatic
		// SnapshotJob is pending.
		startupPolicy := alphaCheckpointConfig.StartupPolicy
		if startupPolicy == "" {
			startupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
		}
		info = &checkpoint.CheckpointInfo{
			Enabled:              true,
			AutomaticCapture:     automaticSnapshotJob != nil,
			StartupPolicy:        startupPolicy,
			AutomaticSnapshotJob: automaticSnapshotJob,
		}
		// Preserve an explicit capture target across the pending-to-Ready handoff.
		if alphaCheckpointConfig.TargetContainerName != "" {
			info.RestoreTargetContainers = []string{alphaCheckpointConfig.TargetContainerName}
		}
	} else {
		workerHash := dynamo.GetDCDEffectiveWorkerHash(dcd)
		var expectedWorkerHash *string
		if dynamo.IsWorkerComponent(string(component.ComponentType)) {
			expectedWorkerHash = &workerHash
		}
		info, err = checkpoint.ResolvePodSnapshotForService(
			ctx,
			r.reader,
			dcd.Namespace,
			alphaCheckpointConfig,
			expectedWorkerHash,
			podSnapshotUseForDCD(dcd, automaticSnapshotJob),
		)
		if err != nil {
			return nil, errors.Wrap(err, "failed to resolve checkpoint")
		}
		if automaticSnapshotJob != nil {
			info.AutomaticCapture = true
			info.AutomaticSnapshotJob = automaticSnapshotJob
		}
	}
	if dynamo.IsIntraPodFailoverEnabled(&dcd.Spec.DynamoComponentDeploymentSharedSpec) {
		info.RestoreTargetContainers = dynamo.IntraPodFailoverEngineContainerNames()
	}

	serviceGMS := dynamo.GetGPUMemoryService(component)
	if info.NativeSnapshot != nil {
		err = gms.OverlayCompatibleSnapshotClients(&info.GPUMemoryService, info.CheckpointName, serviceGMS)
	} else {
		err = gms.OverlayClients(&info.GPUMemoryService, info.CheckpointName, info.Exists, serviceGMS)
	}
	if err != nil {
		return nil, errors.Wrap(err, "failed to apply checkpoint gpuMemoryService config")
	}
	return info, nil
}

func podSnapshotUseForDCD(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	automaticSnapshotJob *checkpoint.SnapshotJobReference,
) checkpoint.PodSnapshotUse {
	if automaticSnapshotJob == nil {
		return checkpoint.ExplicitPodSnapshotUse()
	}
	ownerUID, managed := managedDGDUIDForDCD(dcd)
	if !managed {
		return checkpoint.ExplicitPodSnapshotUse()
	}
	return checkpoint.ManagedPodSnapshotUse(ownerUID)
}

func automaticSnapshotJobReferenceForDCD(dcd *nvidiacomv1beta1.DynamoComponentDeployment) (*checkpoint.SnapshotJobReference, error) {
	if _, managed := managedDGDUIDForDCD(dcd); !managed {
		return nil, nil
	}
	reference, found, err := checkpoint.AutomaticSnapshotJobReferenceFromAnnotations(
		dynamo.GetPodTemplateAnnotations(&dcd.Spec.DynamoComponentDeploymentSharedSpec),
	)
	if err != nil {
		return nil, errors.Wrap(err, "invalid automatic SnapshotJob restore candidate")
	}
	if !found {
		return nil, nil
	}
	return reference, nil
}

func managedDGDUIDForDCD(dcd *nvidiacomv1beta1.DynamoComponentDeployment) (types.UID, bool) {
	// A concrete DGD controller reference selects the supported managed path;
	// Kubernetes authorization remains the security boundary for Snapshot access.
	controller := metav1.GetControllerOf(dcd)
	if controller == nil ||
		controller.Kind != nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind ||
		controller.UID == "" {
		return "", false
	}

	// Accept any served DGD API version from Dynamo's API group.
	groupVersion, err := schema.ParseGroupVersion(controller.APIVersion)
	if err != nil || groupVersion.Group != nvidiacomv1beta1.GroupVersion.Group {
		return "", false
	}

	return controller.UID, true
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
