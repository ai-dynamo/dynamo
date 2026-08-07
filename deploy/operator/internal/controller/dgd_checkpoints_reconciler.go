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

	"github.com/imdario/mergo"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/discovery"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
)

// dgdCheckpointsResult is the cohesive output consumed by later workload
// rendering and status projection.
type dgdCheckpointsResult struct {
	Infos    map[string]*checkpoint.CheckpointInfo
	Statuses map[string]nvidiacomv1beta1.ComponentCheckpointStatus
}

// dgdCheckpointsReconciler owns checkpoint discovery, automatic checkpoint
// resources, checkpoint job rendering, and their resolved program inputs.
type dgdCheckpointsReconciler struct {
	dgdResourceSyncer
	config                *configv1alpha1.OperatorConfiguration
	runtimeConfig         *commoncontroller.RuntimeConfig
	dockerSecretRetriever dockerSecretRetriever
}

func newDGDCheckpointsReconciler(
	syncer dgdResourceSyncer,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
	dockerSecretRetriever dockerSecretRetriever,
) *dgdCheckpointsReconciler {
	return &dgdCheckpointsReconciler{
		dgdResourceSyncer:     syncer,
		config:                config,
		runtimeConfig:         runtimeConfig,
		dockerSecretRetriever: dockerSecretRetriever,
	}
}

func (r *dgdCheckpointsReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (dgdCheckpointsResult, error) {
	result := dgdCheckpointsResult{
		Statuses: make(map[string]nvidiacomv1beta1.ComponentCheckpointStatus),
		Infos:    make(map[string]*checkpoint.CheckpointInfo),
	}
	logger := log.FromContext(ctx)
	storageEnsured := false

	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		componentName := component.ComponentName
		checkpointConfig := dynamo.GetCheckpoint(component)
		if checkpointConfig == nil {
			continue
		}
		if !r.runtimeConfig.Gate.Enabled(features.Checkpoint) {
			return dgdCheckpointsResult{}, fmt.Errorf("component %s: checkpoint functionality is disabled in the operator configuration", componentName)
		}

		logger.Info("Reconciling checkpoint for component", "component", componentName)
		if !storageEnsured {
			if err := checkpoint.EnsureStoragePVC(ctx, r.Client, dgd.Namespace, r.config.Checkpoint.Storage); err != nil {
				logger.Error(err, "Failed to ensure checkpoint storage PVC", "component", componentName)
				return dgdCheckpointsResult{}, fmt.Errorf("failed to ensure checkpoint storage PVC for component %s: %w", componentName, err)
			}
			storageEnsured = true
		}

		alphaCheckpointConfig := dynamo.ToAlphaCheckpointConfig(checkpointConfig)
		startupPolicy := alphaCheckpointConfig.StartupPolicy
		if startupPolicy == "" {
			startupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
		}

		var info *checkpoint.CheckpointInfo
		var err error
		hasCheckpointRef := checkpointConfig.CheckpointRef != nil && *checkpointConfig.CheckpointRef != ""
		if !hasCheckpointRef {
			workerHash, hashErr := checkpointWorkerHashForComponent(dgd, componentName)
			if hashErr != nil {
				return dgdCheckpointsResult{}, fmt.Errorf("failed to compute checkpoint worker hash for component %s: %w", componentName, hashErr)
			}
			checkpointID := checkpoint.DGDCheckpointID(
				dgd.Namespace,
				dgd.Name,
				string(dgd.UID),
				componentName,
				workerHash,
			)
			checkpointName := fmt.Sprintf("checkpoint-%s", checkpointID)
			refConfig := *alphaCheckpointConfig.DeepCopy()
			refConfig.CheckpointRef = &checkpointName
			info, err = checkpoint.ResolveCheckpointForService(ctx, r.Client, dgd.Namespace, &refConfig)
			if apierrors.IsNotFound(err) {
				info = nil
				err = nil
			}
			if err == nil && info == nil {
				info = &checkpoint.CheckpointInfo{
					Enabled:       true,
					StartupPolicy: startupPolicy,
				}
			}
		} else {
			info, err = checkpoint.ResolveCheckpointForService(ctx, r.Client, dgd.Namespace, alphaCheckpointConfig)
		}
		if err != nil {
			logger.Error(err, "Failed to resolve checkpoint for component", "component", componentName)
			return dgdCheckpointsResult{}, fmt.Errorf("failed to resolve checkpoint for component %s: %w", componentName, err)
		}

		info.StartupPolicy = startupPolicy
		if len(info.RestoreTargetContainers) == 0 && alphaCheckpointConfig.TargetContainerName != "" {
			info.RestoreTargetContainers = []string{alphaCheckpointConfig.TargetContainerName}
		}
		if dynamo.IsIntraPodFailoverEnabled(component) {
			info.RestoreTargetContainers = dynamo.IntraPodFailoverEngineContainerNames(component)
		}
		if err := gms.OverlayClients(&info.GPUMemoryService, info.CheckpointName, info.Exists, dynamo.GetGPUMemoryService(component)); err != nil {
			return dgdCheckpointsResult{}, fmt.Errorf("failed to apply checkpoint gpuMemoryService config for component %s: %w", componentName, err)
		}
		result.Infos[componentName] = info

		if !hasCheckpointRef {
			if !info.Exists {
				logger.Info("Creating DGD-managed DynamoCheckpoint CR", "component", componentName)
			}
			ckpt, err := r.createCheckpointCR(ctx, dgd, componentName, component)
			if err != nil {
				logger.Error(err, "Failed to create DynamoCheckpoint CR", "component", componentName)
				return dgdCheckpointsResult{}, fmt.Errorf("failed to create checkpoint for component %s: %w", componentName, err)
			}
			info.Exists = true
			info.CheckpointName = ckpt.Name
			info.Hash, err = checkpoint.CheckpointID(ckpt)
			if err != nil {
				return dgdCheckpointsResult{}, fmt.Errorf("failed to resolve checkpoint ID for component %s: %w", componentName, err)
			}
			if info.GPUMemoryService == nil {
				info.GPUMemoryService = ckpt.Spec.GPUMemoryService
			}
			info.Ready = ckpt.Status.Phase == nvidiacomv1alpha1.DynamoCheckpointPhaseReady
		}

		result.Statuses[componentName] = nvidiacomv1beta1.ComponentCheckpointStatus{
			CheckpointName: info.CheckpointName,
			CheckpointID:   info.Hash,
			IdentityHash:   info.Hash,
			Ready:          info.Ready,
		}
	}

	return result, nil
}

// createCheckpointCR creates a DGD-managed DynamoCheckpoint CR for a component.
//
//nolint:gocyclo
func (r *dgdCheckpointsReconciler) createCheckpointCR(
	ctx context.Context,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	expected, claimTemplateName, gpuCount, deviceClassName, err := r.expectedCheckpointCR(
		r.Scheme(),
		dynamoDeployment,
		componentName,
		component,
	)
	if err != nil {
		return nil, err
	}
	if expected.Spec.GPUMemoryService != nil && expected.Spec.GPUMemoryService.Enabled {
		if err := r.syncCheckpointGMSResourceClaimTemplate(
			ctx,
			dynamoDeployment,
			claimTemplateName,
			gpuCount,
			deviceClassName,
		); err != nil {
			return nil, err
		}
	}

	ckpt, err := checkpoint.CreateOrGetAutoCheckpoint(ctx, r.Client, expected)
	if err != nil {
		return nil, err
	}
	if expected.Spec.GPUMemoryService != nil && expected.Spec.GPUMemoryService.Enabled {
		if err := r.adoptCheckpointGMSResourceClaimTemplate(ctx, ckpt, claimTemplateName); err != nil {
			return nil, err
		}
	}
	return ckpt, nil
}

// expectedCheckpointCR constructs the operator-owned checkpoint object without
// reading or writing Kubernetes resources.
//
//nolint:gocyclo
func (r *dgdCheckpointsReconciler) expectedCheckpointCR(
	scheme *runtime.Scheme,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) (*nvidiacomv1alpha1.DynamoCheckpoint, string, int, string, error) {
	workerHash, err := checkpointWorkerHashForComponent(dynamoDeployment, componentName)
	if err != nil {
		return nil, "", 0, "", fmt.Errorf("failed to compute checkpoint worker hash for component %s: %w", componentName, err)
	}
	return r.expectedCheckpointCRForWorkerHash(
		scheme,
		dynamoDeployment,
		componentName,
		component,
		workerHash,
	)
}

func (r *dgdCheckpointsReconciler) expectedCheckpointCRForWorkerHash(
	scheme *runtime.Scheme,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	workerHash string,
) (*nvidiacomv1alpha1.DynamoCheckpoint, string, int, string, error) {
	checkpointConfig := dynamo.GetCheckpoint(component)
	if checkpointConfig == nil {
		return nil, "", 0, "", fmt.Errorf("checkpoint config is required")
	}

	checkpointID := checkpoint.DGDCheckpointID(
		dynamoDeployment.Namespace,
		dynamoDeployment.Name,
		string(dynamoDeployment.UID),
		componentName,
		workerHash,
	)

	backendFramework, err := dynamo.BackendFrameworkForComponent(component, dynamoDeployment)
	if err != nil {
		return nil, "", 0, "", fmt.Errorf("failed to determine backend framework for component %s: %w", componentName, err)
	}
	if (backendFramework == "" || backendFramework == dynamo.BackendFrameworkNoop) &&
		checkpointConfig.Identity != nil &&
		checkpointConfig.Identity.BackendFramework != "" {
		backendFramework, err = dynamo.ParseBackendFramework(checkpointConfig.Identity.BackendFramework)
		if err != nil {
			return nil, "", 0, "", fmt.Errorf("invalid legacy checkpoint identity backend framework for component %s: %w", componentName, err)
		}
	}
	if backendFramework == "" || backendFramework == dynamo.BackendFrameworkNoop {
		return nil, "", 0, "", fmt.Errorf("checkpoint backend framework for component %s could not be determined; set spec.backendFramework or use a recognizable worker command", componentName)
	}

	podTemplate, err := r.buildCheckpointJobPodTemplate(
		dynamoDeployment,
		component,
		componentName,
		backendFramework,
	)
	if err != nil {
		return nil, "", 0, "", fmt.Errorf("failed to build checkpoint job pod template: %w", err)
	}
	if commoncontroller.IsK8sDiscoveryEnabled(r.config.Discovery.Backend, dynamoDeployment.Annotations) &&
		podTemplate.Spec.ServiceAccountName == "" {
		podTemplate.Spec.ServiceAccountName = discovery.GetK8sDiscoveryServiceAccountName(dynamoDeployment.Name)
	}
	if podTemplate.Labels == nil {
		podTemplate.Labels = map[string]string{}
	}
	podTemplate.Labels[consts.KubeLabelDynamoGraphDeploymentName] = dynamoDeployment.Name
	podTemplate.Labels[consts.KubeLabelDynamoComponent] = componentName
	if workerHash != "" {
		podTemplate.Labels[consts.KubeLabelDynamoWorkerHash] = workerHash
	}

	targetContainerName := consts.MainContainerName
	if checkpointConfig.TargetContainerName != "" {
		targetContainerName = checkpointConfig.TargetContainerName
	}
	targetContainer, err := findPodTemplateContainer(&podTemplate, targetContainerName)
	if err != nil {
		return nil, "", 0, "", err
	}
	var gmsSpec *nvidiacomv1alpha1.GPUMemoryServiceSpec
	if converted := gms.ToAlphaSpec(dynamo.GetGPUMemoryService(component)); converted != nil {
		gmsSpec = converted.DeepCopy()
		gmsSpec.ExtraClientContainers = nil
		if checkpointConfig.Job != nil {
			gmsSpec.ExtraClientContainers = append([]string(nil), checkpointConfig.Job.GMSClientContainers...)
		}
	}
	var checkpointGMSClaimTemplateName string
	var checkpointGMSGPUCount int
	var checkpointGMSDeviceClassName string
	if gmsSpec != nil && gmsSpec.Enabled {
		checkpointGMSClaimTemplateName = checkpointGMSResourceClaimTemplateName(checkpointID)
		checkpointGMSGPUCount, err = dra.ExtractGPUCountFromResourceRequirements(targetContainer.Resources)
		if err != nil {
			return nil, "", 0, "", fmt.Errorf("invalid GPU resource requirements for GMS checkpoint %s/%s: %w", dynamoDeployment.Name, componentName, err)
		}
		checkpointGMSDeviceClassName = gmsSpec.DeviceClassName
		if checkpointGMSDeviceClassName == "" {
			checkpointGMSDeviceClassName = dra.DefaultDeviceClassName
		}
		if err := prepareCheckpointGMSPodTemplate(&podTemplate, targetContainerName, checkpointID, gmsSpec); err != nil {
			return nil, "", 0, "", err
		}
	}
	deletionPolicy := nvidiacomv1alpha1.CheckpointDeletionPolicy(checkpointConfig.DeletionPolicy)
	if deletionPolicy == "" {
		deletionPolicy = nvidiacomv1alpha1.CheckpointDeletionPolicyDelete
	}

	ckpt, err := checkpoint.ExpectedAutoCheckpoint(
		scheme,
		dynamoDeployment.Namespace,
		checkpointID,
		nvidiacomv1alpha1.DynamoCheckpointIdentity{
			Model:            fmt.Sprintf("%s/%s", dynamoDeployment.Namespace, dynamoDeployment.Name),
			BackendFramework: string(backendFramework),
			ExtraParameters: map[string]string{
				"dgdUID":       string(dynamoDeployment.UID),
				"component":    componentName,
				"checkpointID": checkpointID,
			},
		},
		podTemplate,
		targetContainerName,
		deletionPolicy,
		gmsSpec,
		dynamoDeployment,
	)
	if err != nil {
		return nil, "", 0, "", err
	}
	return ckpt, checkpointGMSClaimTemplateName, checkpointGMSGPUCount, checkpointGMSDeviceClassName, nil
}

func checkpointGMSResourceClaimTemplateName(checkpointID string) string {
	return dra.ResourceClaimTemplateName("checkpoint-"+checkpointID, "worker")
}

func findPodTemplateContainer(podTemplate *corev1.PodTemplateSpec, containerName string) (*corev1.Container, error) {
	for i := range podTemplate.Spec.Containers {
		if podTemplate.Spec.Containers[i].Name == containerName {
			return &podTemplate.Spec.Containers[i], nil
		}
	}
	return nil, fmt.Errorf("checkpoint job pod template: pod spec has no container named %q", containerName)
}

func (r *dgdCheckpointsReconciler) syncCheckpointGMSResourceClaimTemplate(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	claimTemplateName string,
	gpuCount int,
	deviceClassName string,
) error {
	_, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*resourcev1.ResourceClaimTemplate, bool, error) {
		return dra.GenerateResourceClaimTemplate(ctx, r.Client, claimTemplateName, dgd.Namespace, gpuCount, deviceClassName)
	})
	if err != nil {
		return fmt.Errorf("failed to sync checkpoint GMS ResourceClaimTemplate %s/%s: %w", dgd.Namespace, claimTemplateName, err)
	}
	return nil
}

func (r *dgdCheckpointsReconciler) adoptCheckpointGMSResourceClaimTemplate(
	ctx context.Context,
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
	claimTemplateName string,
) error {
	template := &resourcev1.ResourceClaimTemplate{}
	key := types.NamespacedName{Name: claimTemplateName, Namespace: ckpt.Namespace}
	if err := r.Get(ctx, key, template); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("failed to get checkpoint GMS ResourceClaimTemplate %s/%s: %w", ckpt.Namespace, claimTemplateName, err)
	}
	if metav1.IsControlledBy(template, ckpt) {
		return nil
	}

	ownerReferences := template.GetOwnerReferences()
	filtered := make([]metav1.OwnerReference, 0, len(ownerReferences))
	for _, ref := range ownerReferences {
		if ref.Controller != nil && *ref.Controller {
			continue
		}
		filtered = append(filtered, ref)
	}
	template.SetOwnerReferences(filtered)
	if err := controllerutil.SetControllerReference(ckpt, template, r.Scheme()); err != nil {
		return fmt.Errorf("failed to set DynamoCheckpoint owner on GMS ResourceClaimTemplate %s/%s: %w", ckpt.Namespace, claimTemplateName, err)
	}
	if err := r.Update(ctx, template); err != nil {
		return fmt.Errorf("failed to update checkpoint GMS ResourceClaimTemplate owner %s/%s: %w", ckpt.Namespace, claimTemplateName, err)
	}
	return nil
}

func prepareCheckpointGMSPodTemplate(
	podTemplate *corev1.PodTemplateSpec,
	targetContainerName string,
	checkpointID string,
	gmsSpec *nvidiacomv1alpha1.GPUMemoryServiceSpec,
) error {
	switch gmsSpec.Mode {
	case "", nvidiacomv1alpha1.GMSModeIntraPod:
	case nvidiacomv1alpha1.GMSModeInterPod:
		return fmt.Errorf("gpuMemoryService checkpoint jobs for mode %q are not implemented", gmsSpec.Mode)
	default:
		return fmt.Errorf("gpuMemoryService checkpoint job has unsupported mode %q", gmsSpec.Mode)
	}

	targetContainer, err := findPodTemplateContainer(podTemplate, targetContainerName)
	if err != nil {
		return err
	}
	ensureCheckpointGMSPodClaim(&podTemplate.Spec, checkpointGMSResourceClaimTemplateName(checkpointID))
	checkpoint.EnsureIntraPodGPUMemoryService(
		&podTemplate.Spec,
		[]*corev1.Container{targetContainer},
		gmsSpec.ExtraClientContainers,
	)
	return nil
}

func ensureCheckpointGMSPodClaim(podSpec *corev1.PodSpec, claimTemplateName string) {
	foundToleration := false
	for i := range podSpec.Tolerations {
		toleration := podSpec.Tolerations[i]
		if toleration.Key == consts.KubeResourceGPUNvidia && toleration.Effect == corev1.TaintEffectNoSchedule {
			foundToleration = true
			break
		}
	}
	if !foundToleration {
		podSpec.Tolerations = append(podSpec.Tolerations, corev1.Toleration{
			Key:      consts.KubeResourceGPUNvidia,
			Operator: corev1.TolerationOpExists,
			Effect:   corev1.TaintEffectNoSchedule,
		})
	}

	podClaim := corev1.PodResourceClaim{
		Name:                      dra.ClaimName,
		ResourceClaimTemplateName: &claimTemplateName,
	}
	for i := range podSpec.ResourceClaims {
		if podSpec.ResourceClaims[i].Name == dra.ClaimName {
			podSpec.ResourceClaims[i] = podClaim
			return
		}
	}
	podSpec.ResourceClaims = append(podSpec.ResourceClaims, podClaim)
}

func (r *dgdCheckpointsReconciler) deleteAutoCheckpointsForDGD(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	checkpoints := &nvidiacomv1alpha1.DynamoCheckpointList{}
	if err := r.List(
		ctx,
		checkpoints,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
	); err != nil {
		return err
	}
	for i := range checkpoints.Items {
		ckpt := &checkpoints.Items[i]
		if ckpt.Annotations == nil || ckpt.Annotations[consts.CheckpointAutoAnnotation] != consts.KubeLabelValueTrue {
			continue
		}
		if ckpt.Annotations[consts.CheckpointDeletionPolicyAnnotation] == string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain) {
			if err := r.detachRetainedAutoCheckpoint(ctx, ckpt); err != nil {
				return err
			}
			continue
		}
		if err := r.Delete(ctx, ckpt); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("delete auto checkpoint %s/%s: %w", ckpt.Namespace, ckpt.Name, err)
		}
	}
	return nil
}

func (r *dgdCheckpointsReconciler) detachRetainedAutoCheckpoint(
	ctx context.Context,
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
) error {
	updated := ckpt.DeepCopy()
	updated.OwnerReferences = nil
	if updated.Labels != nil {
		delete(updated.Labels, consts.KubeLabelDynamoGraphDeploymentName)
	}
	if equality.Semantic.DeepEqual(ckpt.OwnerReferences, updated.OwnerReferences) &&
		equality.Semantic.DeepEqual(ckpt.Labels, updated.Labels) {
		return nil
	}
	if err := r.Patch(ctx, updated, client.MergeFrom(ckpt)); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("detach retained auto checkpoint %s/%s: %w", ckpt.Namespace, ckpt.Name, err)
	}
	return nil
}

func checkpointWorkerHashForComponent(dgd *nvidiacomv1beta1.DynamoGraphDeployment, componentName string) (string, error) {
	if dgd == nil {
		return "", nil
	}
	component := dgd.GetComponentByName(componentName)
	if component == nil || !dynamo.IsWorkerComponent(string(component.ComponentType)) {
		return "", nil
	}
	desired, err := desiredWorkerHashes(dgd)
	if err != nil {
		return "", err
	}
	return activeWorkerHashForDCDGeneration(dgd, desired), nil
}

// buildCheckpointJobPodTemplate builds a checkpoint job template from the same
// component defaults used for regular DGD pods, then keeps only the target
// container plus any checkpoint-job sidecars supplied by the user.
//
//nolint:gocyclo
func (r *dgdCheckpointsReconciler) buildCheckpointJobPodTemplate(
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	componentName string,
	backendFramework dynamo.BackendFramework,
) (corev1.PodTemplateSpec, error) {
	targetContainerName := consts.MainContainerName
	if checkpointConfig := dynamo.GetCheckpoint(component); checkpointConfig != nil && checkpointConfig.TargetContainerName != "" {
		targetContainerName = checkpointConfig.TargetContainerName
	}

	// Create a copy of the component spec stripped of features that buildCheckpointJob
	// or the checkpoint controller handle independently. GenerateBasePodSpec would
	// otherwise apply DGD-specific transforms (DRA claims, GMS server sidecar,
	// frontend sidecar, failover transforms) that conflict with the checkpoint path's
	// own setup.
	componentForJob := component.DeepCopy()
	componentForJob.Experimental = nil
	componentForJob.FrontendSidecar = nil

	// Use the normal DGD path so graph-level defaults such as spec.env,
	// annotations, labels, and pod-template metadata are applied consistently.
	podSpec, err := dynamo.GeneratePodSpecForComponent(
		componentForJob,
		backendFramework,
		r.dockerSecretRetriever,
		dynamoDeployment,
		dynamo.RoleCheckpoint, // Use checkpoint role
		1,                     // Single node for checkpoint job
		r.config,
		consts.MultinodeDeploymentTypeGrove, // Use Grove (single-node backends return early)
		componentName,
		nil, // No checkpoint info for checkpoint creation jobs
		nil, // Use default deployer
	)
	if err != nil {
		return corev1.PodTemplateSpec{}, fmt.Errorf("failed to generate base pod spec: %w", err)
	}

	if podSpec == nil {
		return corev1.PodTemplateSpec{}, fmt.Errorf("checkpoint job pod spec is nil")
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == targetContainerName {
			podSpec.Containers = []corev1.Container{*podSpec.Containers[i].DeepCopy()}
			break
		}
	}
	if len(podSpec.Containers) != 1 || podSpec.Containers[0].Name != targetContainerName {
		return corev1.PodTemplateSpec{}, fmt.Errorf("checkpoint target container %q not found", targetContainerName)
	}

	// Override RestartPolicy for job (must be Never or OnFailure)
	podSpec.RestartPolicy = corev1.RestartPolicyNever

	// Seed the checkpoint job pod-template metadata from the component's own
	// PodTemplate.ObjectMeta so workload-level labels/annotations (e.g. Istio
	// sidecar opt-out or policy annotations) are not silently dropped on the
	// auto-created checkpoint job. GeneratePodSpecForComponent only returns the
	// PodSpec, so the template metadata must be carried over explicitly here.
	// Precedence: component pod-template metadata < controller-managed labels <
	// explicit checkpoint.job.podTemplate overrides (applied below).
	podLabels := map[string]string{}
	podAnnotations := map[string]string{}
	if component.PodTemplate != nil {
		for k, v := range component.PodTemplate.Labels {
			podLabels[k] = v
		}
		for k, v := range component.PodTemplate.Annotations {
			podAnnotations[k] = v
		}
	}
	podLabels[consts.KubeLabelDynamoComponent] = componentName

	podTemplate := corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      podLabels,
			Annotations: podAnnotations,
		},
		Spec: *podSpec,
	}
	if checkpointConfig := dynamo.GetCheckpoint(component); checkpointConfig != nil && checkpointConfig.Job != nil {
		if overrides := checkpointConfig.Job.PodTemplate; overrides != nil {
			if len(overrides.Labels) > 0 {
				if podTemplate.Labels == nil {
					podTemplate.Labels = make(map[string]string, len(overrides.Labels))
				}
				for k, v := range overrides.Labels {
					podTemplate.Labels[k] = v
				}
			}
			if len(overrides.Annotations) > 0 {
				if podTemplate.Annotations == nil {
					podTemplate.Annotations = make(map[string]string, len(overrides.Annotations))
				}
				for k, v := range overrides.Annotations {
					podTemplate.Annotations[k] = v
				}
			}

			overlay := overrides.Spec.DeepCopy()
			containers := overlay.Containers
			initContainers := overlay.InitContainers
			volumes := overlay.Volumes
			overlay.Containers = nil
			overlay.InitContainers = nil
			overlay.Volumes = nil
			if err := mergo.Merge(&podTemplate.Spec, *overlay, mergo.WithOverride); err != nil {
				return corev1.PodTemplateSpec{}, fmt.Errorf("failed to merge checkpoint job pod spec: %w", err)
			}

			podTemplate.Spec.Volumes = mergeNamedSlice(podTemplate.Spec.Volumes, volumes, func(v corev1.Volume) string { return v.Name })
			podTemplate.Spec.InitContainers = mergeNamedSlice(podTemplate.Spec.InitContainers, initContainers, func(c corev1.Container) string { return c.Name })
			for _, override := range containers {
				if override.Name == "" {
					podTemplate.Spec.Containers = append(podTemplate.Spec.Containers, override)
					continue
				}
				var existing *corev1.Container
				for i := range podTemplate.Spec.Containers {
					if podTemplate.Spec.Containers[i].Name == override.Name {
						existing = &podTemplate.Spec.Containers[i]
						break
					}
				}
				if existing == nil {
					podTemplate.Spec.Containers = append(podTemplate.Spec.Containers, override)
					continue
				}

				baseEnv := existing.Env
				user := override.DeepCopy()
				if err := mergo.Merge(existing, *user, mergo.WithOverride); err != nil {
					return corev1.PodTemplateSpec{}, fmt.Errorf("failed to merge checkpoint job container %q: %w", override.Name, err)
				}
				existing.Env = dynamo.MergeEnvs(baseEnv, user.Env)
				if user.LivenessProbe != nil {
					existing.LivenessProbe = user.LivenessProbe.DeepCopy()
				}
				if user.ReadinessProbe != nil {
					existing.ReadinessProbe = user.ReadinessProbe.DeepCopy()
				}
				if user.StartupProbe != nil {
					existing.StartupProbe = user.StartupProbe.DeepCopy()
				}
			}
		}
	}
	return podTemplate, nil
}
