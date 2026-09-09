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
	"errors"
	"fmt"
	"strings"

	snapshotv1alpha1 "github.com/ai-dynamo/snapshot/api/v1alpha1"
	"github.com/imdario/mergo"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
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

// errAutomaticSnapshotCleanupPending keeps the DGD finalizer in place while
// Snapshot finishes the asynchronous deletion of managed SnapshotJobs.
var errAutomaticSnapshotCleanupPending = errors.New("automatic snapshot cleanup pending")

// dgdCheckpointsReconciler owns checkpoint discovery, automatic SnapshotJobs,
// capture Pod rendering, and their resolved program inputs.
type dgdCheckpointsReconciler struct {
	dgdResourceSyncer
	config                *configv1alpha1.OperatorConfiguration
	runtimeConfig         *commoncontroller.RuntimeConfig
	dockerSecretRetriever DockerSecretRetriever
}

func newDGDCheckpointsReconciler(
	syncer dgdResourceSyncer,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
	dockerSecretRetriever DockerSecretRetriever,
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
		checkpointName := strings.TrimSpace(ptr.Deref(checkpointConfig.CheckpointRef, ""))
		hasCheckpointRef := checkpointName != ""

		alphaCheckpointConfig := dynamo.ToAlphaCheckpointConfig(checkpointConfig)
		startupPolicy := alphaCheckpointConfig.StartupPolicy
		if startupPolicy == "" {
			startupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
		}

		// Derive the compatibility identity expected by captured and restored workers.
		workerHash, err := checkpointWorkerHashForComponent(dgd, componentName)
		if err != nil {
			return dgdCheckpointsResult{}, fmt.Errorf("failed to compute checkpoint worker hash for component %s: %w", componentName, err)
		}
		workerComponent := dynamo.IsWorkerComponent(string(component.ComponentType))
		var expectedWorkerHash *string
		if workerComponent {
			expectedWorkerHash = &workerHash
		}

		var info *checkpoint.CheckpointInfo
		if workerComponent && workerHash == "" {
			// Grove records the active worker generation after synchronizing its
			// first PodCliqueSet. Do not capture or resolve a generation-less
			// worker while that durable identity is still being initialized.
			logger.Info("Waiting for active worker hash before checkpoint reconciliation", "component", componentName)
			info = &checkpoint.CheckpointInfo{
				Enabled:        true,
				CheckpointName: checkpointName,
				StartupPolicy:  startupPolicy,
			}
			if hasCheckpointRef {
				// Explicit restore must remain fail-closed while compatibility
				// identity is unavailable, even when Immediate was requested.
				info.StartupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint
			} else {
				info.AutomaticCapture = true
			}
		} else if !hasCheckpointRef {
			info, err = r.reconcileAutomaticSnapshotJob(
				ctx,
				dgd,
				componentName,
				component,
				workerHash,
				startupPolicy,
			)
		} else {
			// Resolve explicit references against the standalone PodSnapshot API.
			info, err = checkpoint.ResolvePodSnapshotForService(
				ctx,
				r.Client,
				dgd.Namespace,
				alphaCheckpointConfig,
				expectedWorkerHash,
				checkpoint.ExplicitPodSnapshotUse(),
			)
		}
		if err != nil {
			logger.Error(err, "Failed to resolve checkpoint for component", "component", componentName)
			return dgdCheckpointsResult{}, fmt.Errorf("failed to resolve checkpoint for component %s: %w", componentName, err)
		}

		if info.StartupPolicy == "" {
			info.StartupPolicy = startupPolicy
		}
		if len(info.RestoreTargetContainers) == 0 && alphaCheckpointConfig.TargetContainerName != "" {
			info.RestoreTargetContainers = []string{alphaCheckpointConfig.TargetContainerName}
		}
		if dynamo.IsIntraPodFailoverEnabled(component) {
			info.RestoreTargetContainers = dynamo.IntraPodFailoverEngineContainerNames()
		}

		// Apply client settings from the resolved artifact, or from the service
		// while an automatic capture is still pending.
		serviceGMS := dynamo.GetGPUMemoryService(component)
		if info.NativeSnapshot != nil {
			err = gms.OverlayCompatibleSnapshotClients(&info.GPUMemoryService, info.CheckpointName, serviceGMS)
		} else {
			err = gms.OverlayClients(&info.GPUMemoryService, info.CheckpointName, info.Exists, serviceGMS)
		}
		if err != nil {
			return dgdCheckpointsResult{}, fmt.Errorf("failed to apply checkpoint gpuMemoryService config for component %s: %w", componentName, err)
		}
		result.Infos[componentName] = info

		result.Statuses[componentName] = nvidiacomv1beta1.ComponentCheckpointStatus{
			CheckpointName: info.CheckpointName,
			Ready:          info.Ready,
		}
	}

	return result, nil
}

// reconcileAutomaticSnapshotJob converges one DGD-managed automatic capture
// and returns the restore observation consumed by workload rendering.
//
//nolint:gocyclo
func (r *dgdCheckpointsReconciler) reconcileAutomaticSnapshotJob(
	ctx context.Context,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	workerHash string,
	startupPolicy nvidiacomv1alpha1.CheckpointStartupPolicy,
) (*checkpoint.CheckpointInfo, error) {
	checkpointConfig := dynamo.GetCheckpoint(component)
	if checkpointConfig == nil {
		return nil, fmt.Errorf("checkpoint config is required")
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
		return nil, fmt.Errorf("failed to determine backend framework for component %s: %w", componentName, err)
	}
	if backendFramework == "" || backendFramework == dynamo.BackendFrameworkNoop {
		return nil, fmt.Errorf("checkpoint backend framework for component %s could not be determined; set spec.backendFramework or use a recognizable worker command", componentName)
	}

	podTemplate, err := r.buildCheckpointJobPodTemplate(
		dynamoDeployment,
		component,
		componentName,
		backendFramework,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to build SnapshotJob pod template: %w", err)
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
		return nil, err
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
	if gmsSpec != nil && gmsSpec.Enabled {
		checkpointGMSClaimTemplateName = checkpointGMSResourceClaimTemplateName(checkpointID)
		checkpointGMSGPUCount, err := dra.ExtractGPUCountFromResourceRequirements(targetContainer.Resources)
		if err != nil {
			return nil, fmt.Errorf("invalid GPU resource requirements for GMS checkpoint %s/%s: %w", dynamoDeployment.Name, componentName, err)
		}
		checkpointGMSDeviceClassName := gmsSpec.DeviceClassName
		if checkpointGMSDeviceClassName == "" {
			checkpointGMSDeviceClassName = dra.DefaultDeviceClassName
		}
		if err := r.syncCheckpointGMSResourceClaimTemplate(
			ctx,
			dynamoDeployment,
			checkpointGMSClaimTemplateName,
			checkpointGMSGPUCount,
			checkpointGMSDeviceClassName,
		); err != nil {
			return nil, err
		}
		if err := prepareCheckpointGMSPodTemplate(
			&podTemplate,
			targetContainerName,
			checkpointID,
			gmsSpec,
		); err != nil {
			return nil, err
		}
	}
	deletionPolicy := nvidiacomv1alpha1.CheckpointDeletionPolicy(checkpointConfig.DeletionPolicy)
	if deletionPolicy == "" {
		deletionPolicy = nvidiacomv1alpha1.CheckpointDeletionPolicyDelete
	}

	gmsMode, err := automaticSnapshotGMSMode(gmsSpec)
	if err != nil {
		return nil, err
	}

	// SnapshotJob owns the one-shot capture state machine while Dynamo supplies
	// the rendered workload and compatibility metadata.
	desired := buildAutomaticSnapshotJob(
		dynamoDeployment,
		componentName,
		checkpointID,
		workerHash,
		podTemplate,
		targetContainerName,
		deletionPolicy,
		gmsMode,
	)
	snapshotJob, err := r.syncAutomaticSnapshotJob(
		ctx,
		dynamoDeployment,
		desired,
		deletionPolicy,
	)
	if err != nil {
		return nil, err
	}
	if gmsSpec != nil && gmsSpec.Enabled {
		if err := r.adoptCheckpointGMSResourceClaimTemplate(ctx, snapshotJob, checkpointGMSClaimTemplateName); err != nil {
			return nil, err
		}
	}
	if err := r.syncAutomaticPodSnapshotLifecycle(ctx, snapshotJob, deletionPolicy); err != nil {
		return nil, err
	}

	var expectedWorkerHash *string
	if dynamo.IsWorkerComponent(string(component.ComponentType)) {
		expectedWorkerHash = &workerHash
	}
	return r.resolveAutomaticSnapshotJob(
		ctx,
		snapshotJob,
		dynamo.ToAlphaCheckpointConfig(checkpointConfig),
		expectedWorkerHash,
		startupPolicy,
	)
}

const automaticSnapshotActiveDeadlineSeconds = int64(3600)

func buildAutomaticSnapshotJob(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	checkpointID string,
	workerHash string,
	podTemplate corev1.PodTemplateSpec,
	targetContainerName string,
	deletionPolicy nvidiacomv1alpha1.CheckpointDeletionPolicy,
	gmsMode string,
) *snapshotv1alpha1.SnapshotJob {
	name := fmt.Sprintf("checkpoint-%s", checkpointID)
	labels := map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		consts.KubeLabelDynamoComponent:           componentName,
	}
	if workerHash != "" {
		labels[consts.KubeLabelDynamoWorkerHash] = workerHash
	}
	annotations := map[string]string{
		consts.CheckpointAutoAnnotation:           consts.KubeLabelValueTrue,
		consts.CheckpointDeletionPolicyAnnotation: string(deletionPolicy),
		consts.CheckpointOwnerUIDAnnotation:       string(dgd.UID),
	}

	// The generated PodSnapshot carries both lifecycle identity and immutable
	// restore compatibility metadata; Snapshot owns its runtime fields.
	snapshotLabels := make(map[string]string, len(labels))
	for key, value := range labels {
		snapshotLabels[key] = value
	}
	snapshotAnnotations := map[string]string{
		consts.CheckpointAutoAnnotation:               consts.KubeLabelValueTrue,
		consts.CheckpointDeletionPolicyAnnotation:     string(deletionPolicy),
		consts.CheckpointOwnerUIDAnnotation:           string(dgd.UID),
		consts.SnapshotCompatibilityVersionAnnotation: consts.SnapshotCompatibilityVersion,
		consts.SnapshotWorkerHashAnnotation:           workerHash,
		consts.SnapshotGMSModeAnnotation:              gmsMode,
	}

	return &snapshotv1alpha1.SnapshotJob{
		TypeMeta: metav1.TypeMeta{
			APIVersion: snapshotv1alpha1.GroupVersion.String(),
			Kind:       "SnapshotJob",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   dgd.Namespace,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: snapshotv1alpha1.SnapshotJobSpec{
			PodTemplate:           podTemplate,
			ActiveDeadlineSeconds: ptr.To(automaticSnapshotActiveDeadlineSeconds),
			PodSnapshotTemplate: snapshotv1alpha1.PodSnapshotTemplate{
				Metadata: &snapshotv1alpha1.PodSnapshotTemplateMetadata{
					Labels:      snapshotLabels,
					Annotations: snapshotAnnotations,
				},
				TargetContainers: []string{targetContainerName},
			},
		},
	}
}

// syncAutomaticSnapshotJob creates the immutable SnapshotJob or reconciles
// only the lifecycle metadata that remains mutable. DGD and desired must be
// non-nil.
func (r *dgdCheckpointsReconciler) syncAutomaticSnapshotJob(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired *snapshotv1alpha1.SnapshotJob,
	deletionPolicy nvidiacomv1alpha1.CheckpointDeletionPolicy,
) (*snapshotv1alpha1.SnapshotJob, error) {
	key := client.ObjectKeyFromObject(desired)
	existing := &snapshotv1alpha1.SnapshotJob{}
	if err := r.Get(ctx, key, existing); err != nil {
		if !apierrors.IsNotFound(err) {
			return nil, fmt.Errorf("get automatic SnapshotJob %s: %w", key, err)
		}

		// Delete-policy jobs use Kubernetes ownership; retained jobs remain
		// independent so they can finish after the DGD is removed.
		created := desired.DeepCopy()
		if deletionPolicy == nvidiacomv1alpha1.CheckpointDeletionPolicyDelete {
			if err := controllerutil.SetControllerReference(dgd, created, r.Scheme()); err != nil {
				return nil, fmt.Errorf("set DGD owner on automatic SnapshotJob %s: %w", key, err)
			}
		}
		if err := r.Create(ctx, created); err != nil {
			return nil, fmt.Errorf("create automatic SnapshotJob %s: %w", key, err)
		}
		return created, nil
	}

	// A deterministic name may be reused only by this graph incarnation.
	// Capture inputs that participate in the worker hash select a new name;
	// inputs outside that hash intentionally preserve the existing one-shot job,
	// matching the previous automatic-capture invalidation contract.
	if existing.Annotations[consts.CheckpointAutoAnnotation] != consts.KubeLabelValueTrue ||
		existing.Annotations[consts.CheckpointOwnerUIDAnnotation] != string(dgd.UID) {
		return nil, fmt.Errorf("SnapshotJob %s already exists and is not managed by DGD uid %q", key, dgd.UID)
	}
	if controller := metav1.GetControllerOf(existing); controller != nil && controller.UID != dgd.UID {
		return nil, fmt.Errorf("SnapshotJob %s is controlled by unexpected uid %q", key, controller.UID)
	}
	// Reconcile Dynamo-owned metadata and the existing deletion policy without
	// modifying SnapshotJob.spec.
	updated := existing.DeepCopy()
	if updated.Labels == nil {
		updated.Labels = map[string]string{}
	}
	for key, value := range desired.Labels {
		updated.Labels[key] = value
	}
	if updated.Annotations == nil {
		updated.Annotations = map[string]string{}
	}
	for key, value := range desired.Annotations {
		updated.Annotations[key] = value
	}
	if deletionPolicy == nvidiacomv1alpha1.CheckpointDeletionPolicyDelete {
		if err := controllerutil.SetControllerReference(dgd, updated, r.Scheme()); err != nil {
			return nil, fmt.Errorf("set DGD owner on automatic SnapshotJob %s: %w", key, err)
		}
	} else {
		updated.OwnerReferences = removeControllerReferenceByUID(updated.OwnerReferences, dgd.UID)
	}
	if equality.Semantic.DeepEqual(existing.Labels, updated.Labels) &&
		equality.Semantic.DeepEqual(existing.Annotations, updated.Annotations) &&
		equality.Semantic.DeepEqual(existing.OwnerReferences, updated.OwnerReferences) {
		return existing, nil
	}
	if err := r.Patch(ctx, updated, client.MergeFrom(existing)); err != nil {
		return nil, fmt.Errorf("patch automatic SnapshotJob %s lifecycle metadata: %w", key, err)
	}
	return updated, nil
}

// syncAutomaticPodSnapshotLifecycle keeps mutable Dynamo policy outside the
// immutable SnapshotJob spec while preserving it on the generated artifact.
func (r *dgdCheckpointsReconciler) syncAutomaticPodSnapshotLifecycle(
	ctx context.Context,
	job *snapshotv1alpha1.SnapshotJob,
	deletionPolicy nvidiacomv1alpha1.CheckpointDeletionPolicy,
) error {
	if job.Status.PodSnapshotName == "" {
		return nil
	}
	snapshot := &snapshotv1alpha1.PodSnapshot{}
	key := client.ObjectKey{Namespace: job.Namespace, Name: job.Status.PodSnapshotName}
	if err := r.Get(ctx, key, snapshot); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("get automatic PodSnapshot %s: %w", key, err)
	}
	if snapshot.Labels[snapshotv1alpha1.SnapshotJobOwnerLabel] != job.Name ||
		(job.UID != "" && snapshot.Labels[snapshotv1alpha1.SnapshotJobOwnerUIDLabel] != string(job.UID)) {
		return nil
	}
	if !automaticSnapshotResourceMatchesOwnerUID(snapshot, job.Annotations[consts.CheckpointOwnerUIDAnnotation]) {
		return nil
	}
	if snapshot.Annotations[consts.CheckpointDeletionPolicyAnnotation] == string(deletionPolicy) {
		return nil
	}

	updated := snapshot.DeepCopy()
	if updated.Annotations == nil {
		updated.Annotations = map[string]string{}
	}
	updated.Annotations[consts.CheckpointDeletionPolicyAnnotation] = string(deletionPolicy)
	if err := r.Patch(ctx, updated, client.MergeFrom(snapshot)); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("patch automatic PodSnapshot %s lifecycle metadata: %w", key, err)
	}
	return nil
}

func removeControllerReferenceByUID(refs []metav1.OwnerReference, uid types.UID) []metav1.OwnerReference {
	filtered := make([]metav1.OwnerReference, 0, len(refs))
	for _, ref := range refs {
		if ref.UID == uid && ref.Controller != nil && *ref.Controller {
			continue
		}
		filtered = append(filtered, ref)
	}
	return filtered
}

func (r *dgdCheckpointsReconciler) resolveAutomaticSnapshotJob(
	ctx context.Context,
	snapshotJob *snapshotv1alpha1.SnapshotJob,
	config *nvidiacomv1alpha1.ServiceCheckpointConfig,
	expectedWorkerHash *string,
	startupPolicy nvidiacomv1alpha1.CheckpointStartupPolicy,
) (*checkpoint.CheckpointInfo, error) {
	info := &checkpoint.CheckpointInfo{
		Enabled:          true,
		AutomaticCapture: true,
		CheckpointName:   snapshotJob.Status.PodSnapshotName,
		StartupPolicy:    startupPolicy,
	}
	if snapshotv1alpha1.IsSnapshotJobFailed(snapshotJob) {
		failed := meta.FindStatusCondition(snapshotJob.Status.Conditions, snapshotv1alpha1.SnapshotJobConditionFailed)
		failure := failed.Reason
		if failed.Message != "" {
			failure += ": " + failed.Message
		}
		return nil, fmt.Errorf("automatic SnapshotJob %s/%s failed: %s", snapshotJob.Namespace, snapshotJob.Name, failure)
	}
	if snapshotJob.UID != "" {
		info.AutomaticSnapshotJob = &checkpoint.SnapshotJobReference{
			Name: snapshotJob.Name,
			UID:  snapshotJob.UID,
		}
	}
	if !snapshotv1alpha1.IsSnapshotJobCompleted(snapshotJob) {
		return info, nil
	}
	if info.AutomaticSnapshotJob == nil {
		return nil, fmt.Errorf("completed automatic SnapshotJob %s/%s has no UID", snapshotJob.Namespace, snapshotJob.Name)
	}
	if snapshotJob.Status.PodSnapshotName == "" || snapshotJob.Status.PodSnapshotUID == "" {
		return nil, fmt.Errorf("completed SnapshotJob %s/%s has no PodSnapshot identity", snapshotJob.Namespace, snapshotJob.Name)
	}

	// Completion, not capture alone, is the restore boundary because helper
	// containers such as the GMS saver may still be persisting state.
	refConfig := config.DeepCopy()
	refConfig.CheckpointRef = ptr.To(snapshotJob.Status.PodSnapshotName)
	resolved, err := checkpoint.ResolvePodSnapshotForService(
		ctx,
		r.Client,
		snapshotJob.Namespace,
		refConfig,
		expectedWorkerHash,
		checkpoint.ManagedPodSnapshotUse(types.UID(snapshotJob.Annotations[consts.CheckpointOwnerUIDAnnotation])),
	)
	if apierrors.IsNotFound(err) {
		return info, nil
	}
	if err != nil {
		return nil, err
	}
	if resolved.NativeSnapshot.UID != snapshotJob.Status.PodSnapshotUID {
		return nil, fmt.Errorf(
			"SnapshotJob %s/%s produced PodSnapshot uid %q, found uid %q",
			snapshotJob.Namespace,
			snapshotJob.Name,
			snapshotJob.Status.PodSnapshotUID,
			resolved.NativeSnapshot.UID,
		)
	}
	if !resolved.Ready {
		return info, nil
	}
	resolved.AutomaticCapture = true
	resolved.StartupPolicy = startupPolicy
	resolved.AutomaticSnapshotJob = info.AutomaticSnapshotJob
	return resolved, nil
}

func automaticSnapshotGMSMode(spec *nvidiacomv1alpha1.GPUMemoryServiceSpec) (string, error) {
	if spec == nil || !spec.Enabled {
		return consts.SnapshotGMSModeDisabled, nil
	}
	switch spec.Mode {
	case "", nvidiacomv1alpha1.GMSModeIntraPod:
		return string(nvidiacomv1alpha1.GMSModeIntraPod), nil
	default:
		return "", fmt.Errorf("automatic SnapshotJob has unsupported gpuMemoryService mode %q", spec.Mode)
	}
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
	return nil, fmt.Errorf("SnapshotJob pod template: pod spec has no container named %q", containerName)
}

func (r *dgdCheckpointsReconciler) syncCheckpointGMSResourceClaimTemplate(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	claimTemplateName string,
	gpuCount int,
	deviceClassName string,
) error {
	// DynamoCheckpoint takes controller ownership after the DGD creates this template.
	_, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*resourcev1.ResourceClaimTemplate, bool, error) {
		return dra.GenerateResourceClaimTemplate(ctx, r.Client, claimTemplateName, dgd.Namespace, gpuCount, deviceClassName)
	}, commoncontroller.WithSharedOwnership())
	if err != nil {
		return fmt.Errorf("failed to sync checkpoint GMS ResourceClaimTemplate %s/%s: %w", dgd.Namespace, claimTemplateName, err)
	}
	return nil
}

func (r *dgdCheckpointsReconciler) adoptCheckpointGMSResourceClaimTemplate(
	ctx context.Context,
	owner client.Object,
	claimTemplateName string,
) error {
	template := &resourcev1.ResourceClaimTemplate{}
	key := types.NamespacedName{Name: claimTemplateName, Namespace: owner.GetNamespace()}
	if err := r.Get(ctx, key, template); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("failed to get checkpoint GMS ResourceClaimTemplate %s/%s: %w", owner.GetNamespace(), claimTemplateName, err)
	}
	if metav1.IsControlledBy(template, owner) {
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
	if err := controllerutil.SetControllerReference(owner, template, r.Scheme()); err != nil {
		return fmt.Errorf("failed to set capture owner on GMS ResourceClaimTemplate %s/%s: %w", owner.GetNamespace(), claimTemplateName, err)
	}
	if err := r.Update(ctx, template); err != nil {
		return fmt.Errorf("failed to update checkpoint GMS ResourceClaimTemplate owner %s/%s: %w", owner.GetNamespace(), claimTemplateName, err)
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
		return fmt.Errorf("gpuMemoryService SnapshotJobs for mode %q are not implemented", gmsSpec.Mode)
	default:
		return fmt.Errorf("gpuMemoryService SnapshotJob has unsupported mode %q", gmsSpec.Mode)
	}

	targetContainer, err := findPodTemplateContainer(podTemplate, targetContainerName)
	if err != nil {
		return err
	}
	for _, clientContainerName := range gmsSpec.ExtraClientContainers {
		if _, err := findPodTemplateContainer(podTemplate, clientContainerName); err != nil {
			return fmt.Errorf("gpuMemoryService client container %q: %w", clientContainerName, err)
		}
	}
	ensureCheckpointGMSPodClaim(&podTemplate.Spec, checkpointGMSResourceClaimTemplateName(checkpointID))
	checkpoint.EnsureIntraPodGPUMemoryService(
		&podTemplate.Spec,
		[]*corev1.Container{targetContainer},
		gmsSpec.ExtraClientContainers,
		true,
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
	retainedSnapshotJobs, cleanupPending, err := r.deleteAutomaticSnapshotJobsForDGD(ctx, dgd)
	if err != nil {
		return err
	}
	if cleanupPending {
		return errAutomaticSnapshotCleanupPending
	}
	if err := r.deleteAutomaticPodSnapshotsForDGD(ctx, dgd, retainedSnapshotJobs); err != nil {
		return err
	}

	return nil
}

func (r *dgdCheckpointsReconciler) deleteAutomaticSnapshotJobsForDGD(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (map[types.UID]string, bool, error) {
	jobs := &snapshotv1alpha1.SnapshotJobList{}
	if err := r.List(
		ctx,
		jobs,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
	); err != nil {
		if snapshotAPIUnavailable(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("list automatic SnapshotJobs for DGD %s/%s: %w", dgd.Namespace, dgd.Name, err)
	}

	retainedSnapshotJobs := make(map[types.UID]string)
	cleanupPending := false
	for i := range jobs.Items {
		job := &jobs.Items[i]
		if !automaticSnapshotResourceBelongsToDGD(job, dgd) {
			continue
		}
		deletionPolicy := nvidiacomv1alpha1.CheckpointDeletionPolicy(job.Annotations[consts.CheckpointDeletionPolicyAnnotation])
		if deletionPolicy == "" {
			deletionPolicy = nvidiacomv1alpha1.CheckpointDeletionPolicyDelete
		}
		// Once deletion has been accepted by the API server, Snapshot owns any
		// finalizer delay and Dynamo can continue cleaning up its artifacts.
		if job.DeletionTimestamp != nil {
			if deletionPolicy == nvidiacomv1alpha1.CheckpointDeletionPolicyRetain && job.UID != "" {
				retainedSnapshotJobs[job.UID] = job.Name
			}
			continue
		}
		if err := r.syncAutomaticPodSnapshotLifecycle(ctx, job, deletionPolicy); err != nil {
			return nil, false, err
		}
		if deletionPolicy == nvidiacomv1alpha1.CheckpointDeletionPolicyRetain {
			if job.UID == "" {
				return nil, false, fmt.Errorf("retained automatic SnapshotJob %s/%s has no UID", job.Namespace, job.Name)
			}
			retainedSnapshotJobs[job.UID] = job.Name
			if err := r.detachRetainedAutomaticSnapshotJob(ctx, job, dgd.UID); err != nil {
				return nil, false, err
			}
			continue
		}
		cleanupPending = true
		if err := r.Delete(ctx, job); err != nil && !apierrors.IsNotFound(err) {
			return nil, false, fmt.Errorf("delete automatic SnapshotJob %s/%s: %w", job.Namespace, job.Name, err)
		}
	}
	return retainedSnapshotJobs, cleanupPending, nil
}

func (r *dgdCheckpointsReconciler) deleteAutomaticPodSnapshotsForDGD(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	retainedSnapshotJobs map[types.UID]string,
) error {
	snapshots := &snapshotv1alpha1.PodSnapshotList{}
	if err := r.List(
		ctx,
		snapshots,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
	); err != nil {
		if snapshotAPIUnavailable(err) {
			return nil
		}
		return fmt.Errorf("list automatic PodSnapshots for DGD %s/%s: %w", dgd.Namespace, dgd.Name, err)
	}

	for i := range snapshots.Items {
		snapshot := &snapshots.Items[i]
		if !automaticSnapshotResourceBelongsToDGD(snapshot, dgd) {
			continue
		}
		jobUID := types.UID(snapshot.Labels[snapshotv1alpha1.SnapshotJobOwnerUIDLabel])
		if jobUID != "" && retainedSnapshotJobs[jobUID] == snapshot.Labels[snapshotv1alpha1.SnapshotJobOwnerLabel] {
			continue
		}
		if snapshot.Annotations[consts.CheckpointDeletionPolicyAnnotation] == string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain) {
			continue
		}
		if err := r.Delete(ctx, snapshot); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("delete automatic PodSnapshot %s/%s: %w", snapshot.Namespace, snapshot.Name, err)
		}
	}
	return nil
}

func snapshotAPIUnavailable(err error) bool {
	return runtime.IsNotRegisteredError(err) || meta.IsNoMatchError(err) || apierrors.IsNotFound(err)
}

func automaticSnapshotResourceBelongsToDGD(resource client.Object, dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	return automaticSnapshotResourceMatchesOwnerUID(resource, string(dgd.UID))
}

func automaticSnapshotResourceMatchesOwnerUID(resource client.Object, ownerUID string) bool {
	return resource.GetAnnotations()[consts.CheckpointAutoAnnotation] == consts.KubeLabelValueTrue &&
		resource.GetAnnotations()[consts.CheckpointOwnerUIDAnnotation] == ownerUID
}

func (r *dgdCheckpointsReconciler) detachRetainedAutomaticSnapshotJob(
	ctx context.Context,
	job *snapshotv1alpha1.SnapshotJob,
	dgdUID types.UID,
) error {
	updated := job.DeepCopy()
	updated.SetOwnerReferences(removeControllerReferenceByUID(updated.GetOwnerReferences(), dgdUID))
	if equality.Semantic.DeepEqual(job.OwnerReferences, updated.OwnerReferences) {
		return nil
	}
	if err := r.Patch(ctx, updated, client.MergeFrom(job)); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("detach retained automatic SnapshotJob %s/%s: %w", job.Namespace, job.Name, err)
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

// buildCheckpointJobPodTemplate builds a SnapshotJob capture Pod template from the same
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
		1,                     // Single node for SnapshotJob capture
		r.config,
		consts.MultinodeDeploymentTypeGrove, // Use Grove (single-node backends return early)
		componentName,
		nil,                                     // Use default deployer
		func() (int64, error) { return 0, nil }, // Checkpoint jobs are single-node
	)
	if err != nil {
		return corev1.PodTemplateSpec{}, fmt.Errorf("failed to generate base pod spec: %w", err)
	}

	if podSpec == nil {
		return corev1.PodTemplateSpec{}, fmt.Errorf("SnapshotJob pod spec is nil")
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

	// Seed the SnapshotJob pod-template metadata from the component's own
	// PodTemplate.ObjectMeta so workload-level labels/annotations (e.g. Istio
	// sidecar opt-out or policy annotations) are not silently dropped on the
	// auto-created SnapshotJob. GeneratePodSpecForComponent only returns the
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
				return corev1.PodTemplateSpec{}, fmt.Errorf("failed to merge SnapshotJob pod spec: %w", err)
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
					return corev1.PodTemplateSpec{}, fmt.Errorf("failed to merge SnapshotJob container %q: %w", override.Name, err)
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
