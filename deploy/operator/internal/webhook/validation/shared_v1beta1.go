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

package validation

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controllercommon "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	k8svalidation "k8s.io/apimachinery/pkg/util/validation"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// SharedSpecValidatorV1Beta1 validates v1beta1 DynamoComponentDeploymentSharedSpec fields.
// DGD components use it directly; DCD can move to it when its admission path is ported to v1beta1.
type SharedSpecValidatorV1Beta1 struct {
	spec         *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
	fieldPath    string
	grovePathway bool
	mgr          ctrl.Manager
}

func NewSharedSpecValidatorV1Beta1(
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fieldPath string,
	grovePathway bool,
	mgr ctrl.Manager,
) *SharedSpecValidatorV1Beta1 {
	return &SharedSpecValidatorV1Beta1{
		spec:         spec,
		fieldPath:    fieldPath,
		grovePathway: grovePathway,
		mgr:          mgr,
	}
}

func (v *SharedSpecValidatorV1Beta1) Validate(ctx context.Context) (admission.Warnings, error) {
	var errs []error
	if v.spec.Replicas != nil && *v.spec.Replicas < 0 {
		errs = append(errs, fmt.Errorf("%s.replicas must be non-negative", v.fieldPath))
	}
	if err := v.validateMinAvailable(); err != nil {
		errs = append(errs, err)
	}
	if err := v.validatePodTemplate(); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateCompilationCache(); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateSharedMemory(); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateCheckpointConfig(); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateGMSClientContainerNames(); err != nil {
		errs = append(errs, err)
	}
	if err := v.validatePodTemplateAnnotations(); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateEPPConfig(ctx); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateGPUMemoryService(); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateFailover(); err != nil {
		errs = append(errs, err)
	}
	if err := v.validateSnapshotWithGPUMemoryService(); err != nil {
		errs = append(errs, err)
	}
	return nil, errors.Join(errs...)
}

func (v *SharedSpecValidatorV1Beta1) validateMinAvailable() error {
	if v.spec.MinAvailable == nil {
		return nil
	}
	if !v.grovePathway {
		return fmt.Errorf("%s.minAvailable is currently supported only for Grove-backed DynamoGraphDeployment components", v.fieldPath)
	}
	if *v.spec.MinAvailable <= 0 {
		return fmt.Errorf("%s.minAvailable must be greater than 0", v.fieldPath)
	}
	replicas := int32(1)
	if v.spec.Replicas != nil {
		replicas = *v.spec.Replicas
	}
	if replicas > 0 && replicas < *v.spec.MinAvailable {
		return fmt.Errorf("%s.replicas must be 0 or greater than or equal to minAvailable", v.fieldPath)
	}
	return nil
}

func (v *SharedSpecValidatorV1Beta1) validatePodTemplate() error {
	if v.spec.PodTemplate == nil {
		if v.spec.FrontendSidecar != nil {
			return fmt.Errorf("%s.frontendSidecar requires podTemplate.spec.containers", v.fieldPath)
		}
		return nil
	}

	containers := v.spec.PodTemplate.Spec.Containers
	if v.spec.FrontendSidecar != nil {
		target := *v.spec.FrontendSidecar
		if target == "" {
			return fmt.Errorf("%s.frontendSidecar must not be empty", v.fieldPath)
		}
		found := false
		for i := range containers {
			if containers[i].Name == target {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("%s.frontendSidecar %q does not match any podTemplate.spec.containers name",
				v.fieldPath, target)
		}
	}

	for i := range containers {
		container := containers[i]
		if container.Name != consts.MainContainerName && container.Image == "" {
			return fmt.Errorf("%s.podTemplate.spec.containers[%d].image is required for sidecar container %q",
				v.fieldPath, i, container.Name)
		}
	}
	for i := range v.spec.PodTemplate.Spec.InitContainers {
		container := v.spec.PodTemplate.Spec.InitContainers[i]
		if container.Image == "" {
			return fmt.Errorf("%s.podTemplate.spec.initContainers[%d].image is required for init container %q",
				v.fieldPath, i, container.Name)
		}
	}
	return nil
}

func (v *SharedSpecValidatorV1Beta1) validateCompilationCache() error {
	if v.spec.CompilationCache == nil {
		return nil
	}
	if v.spec.CompilationCache.PVCName == "" {
		return fmt.Errorf("%s.compilationCache.pvcName is required", v.fieldPath)
	}
	return nil
}

func (v *SharedSpecValidatorV1Beta1) validateSharedMemory() error {
	if v.spec.SharedMemorySize == nil {
		return nil
	}
	if v.spec.SharedMemorySize.Sign() < 0 {
		return fmt.Errorf("%s.sharedMemorySize must be non-negative", v.fieldPath)
	}
	return nil
}

func (v *SharedSpecValidatorV1Beta1) validateCheckpointConfig() error {
	checkpoint := betaCheckpoint(v.spec)
	if checkpoint == nil {
		return nil
	}
	if checkpoint.Job != nil && checkpoint.CheckpointRef != nil && *checkpoint.CheckpointRef != "" {
		return fmt.Errorf("%s.experimental.checkpoint.job cannot be set when checkpointRef is specified", v.fieldPath)
	}
	if checkpoint.TargetContainerName != "" {
		if errs := k8svalidation.IsDNS1123Label(checkpoint.TargetContainerName); len(errs) > 0 {
			return fmt.Errorf(
				"%s.experimental.checkpoint.targetContainerName %q is not a valid Kubernetes container name: %s",
				v.fieldPath,
				checkpoint.TargetContainerName,
				strings.Join(errs, "; "),
			)
		}
	}
	if checkpoint.Job == nil {
		return nil
	}
	if len(checkpoint.Job.GMSClientContainers) > 0 {
		gms := betaGPUMemoryService(v.spec)
		if gms == nil {
			return fmt.Errorf("%s.experimental.checkpoint.job.gmsClientContainers requires gpuMemoryService to be set", v.fieldPath)
		}
		if betaGMSMode(gms.Mode) == nvidiacomv1beta1.GMSModeInterPod {
			return fmt.Errorf("%s.experimental.checkpoint.job.gmsClientContainers is only supported with gpuMemoryService.mode=IntraPod", v.fieldPath)
		}
	}
	for i, name := range checkpoint.Job.GMSClientContainers {
		if errs := k8svalidation.IsDNS1123Label(name); len(errs) > 0 {
			return fmt.Errorf(
				"%s.experimental.checkpoint.job.gmsClientContainers[%d] %q is not a valid Kubernetes container name: %s",
				v.fieldPath,
				i,
				name,
				strings.Join(errs, "; "),
			)
		}
	}
	return nil
}

func (v *SharedSpecValidatorV1Beta1) validateGMSClientContainerNames() error {
	gms := betaGPUMemoryService(v.spec)
	if gms == nil {
		return nil
	}
	for i, name := range gms.ExtraClientContainers {
		if errs := k8svalidation.IsDNS1123Label(name); len(errs) > 0 {
			return fmt.Errorf(
				"%s.experimental.gpuMemoryService.extraClientContainers[%d] %q is not a valid Kubernetes container name: %s",
				v.fieldPath,
				i,
				name,
				strings.Join(errs, "; "),
			)
		}
	}
	if len(gms.ExtraClientContainers) > 0 && betaGMSMode(gms.Mode) == nvidiacomv1beta1.GMSModeInterPod {
		return fmt.Errorf("%s.experimental.gpuMemoryService.extraClientContainers is only supported with mode=IntraPod", v.fieldPath)
	}
	if len(gms.ExtraClientPods) > 0 {
		return fmt.Errorf("%s.experimental.gpuMemoryService.extraClientPods is reserved for inter-pod GMS and is not implemented yet", v.fieldPath)
	}
	return nil
}

func (v *SharedSpecValidatorV1Beta1) validatePodTemplateAnnotations() error {
	annotations := dynamo.GetPodTemplateAnnotations(v.spec)
	return validateVLLMDistributedExecutorBackendAnnotation(v.fieldPath+".podTemplate.metadata.annotations", annotations)
}

func (v *SharedSpecValidatorV1Beta1) validateEPPConfig(ctx context.Context) error {
	if v.spec.ComponentType != nvidiacomv1beta1.ComponentTypeEPP {
		return nil
	}
	if err := v.checkInferencePoolAPIAvailability(ctx); err != nil {
		return fmt.Errorf("%s: cannot deploy EPP component: %w", v.fieldPath, err)
	}
	if v.spec.IsMultinode() {
		return fmt.Errorf("%s: EPP component cannot be multinode (multinode field must be nil or nodeCount must be 1)", v.fieldPath)
	}
	if v.spec.Replicas != nil && *v.spec.Replicas != 1 {
		return fmt.Errorf("%s: EPP component must have exactly 1 replica (found %d replicas)", v.fieldPath, *v.spec.Replicas)
	}
	if v.spec.EPPConfig == nil {
		return fmt.Errorf("%s.eppConfig is required for EPP components", v.fieldPath)
	}
	if v.spec.EPPConfig.ConfigMapRef == nil && v.spec.EPPConfig.Config == nil {
		return fmt.Errorf("%s.eppConfig: either configMapRef or config must be specified (no default configuration provided)", v.fieldPath)
	}
	if v.spec.EPPConfig.ConfigMapRef != nil && v.spec.EPPConfig.Config != nil {
		return fmt.Errorf("%s.eppConfig: configMapRef and config are mutually exclusive, only one can be specified", v.fieldPath)
	}
	if v.spec.EPPConfig.ConfigMapRef != nil && v.spec.EPPConfig.ConfigMapRef.Name == "" {
		return fmt.Errorf("%s.eppConfig.configMapRef.name is required", v.fieldPath)
	}
	return nil
}

func (v *SharedSpecValidatorV1Beta1) checkInferencePoolAPIAvailability(ctx context.Context) error {
	if v.mgr == nil {
		return fmt.Errorf("manager is required to detect InferencePool API availability")
	}
	if !controllercommon.DetectInferencePoolAvailability(ctx, v.mgr) {
		return fmt.Errorf(
			"InferencePool API group (%s) is not available in the cluster. "+
				"EPP requires the Gateway API Inference Extension to be installed. "+
				"Please install the Gateway API Inference Extension before deploying EPP components",
			epp.InferencePoolGroup)
	}
	return nil
}

func (v *SharedSpecValidatorV1Beta1) validateGPUMemoryService() error {
	gms := betaGPUMemoryService(v.spec)
	if gms == nil {
		return nil
	}

	isWorker := v.spec.ComponentType == nvidiacomv1beta1.ComponentTypeWorker ||
		v.spec.ComponentType == nvidiacomv1beta1.ComponentTypePrefill ||
		v.spec.ComponentType == nvidiacomv1beta1.ComponentTypeDecode
	if !isWorker {
		return fmt.Errorf(
			"%s.experimental.gpuMemoryService: GPU memory service is only supported for worker components (type must be worker, prefill, or decode)",
			v.fieldPath)
	}

	gpuCount, err := dra.ExtractGPUCountFromResourceRequirements(dynamo.GetMainContainerResources(v.spec))
	if err != nil || gpuCount < 1 {
		return fmt.Errorf(
			"%s.experimental.gpuMemoryService: GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1",
			v.fieldPath)
	}

	return nil
}

func (v *SharedSpecValidatorV1Beta1) validateFailover() error {
	failover := betaFailover(v.spec)
	if failover == nil {
		return nil
	}

	var errs []error
	gms := betaGPUMemoryService(v.spec)
	failoverMode := betaGMSMode(failover.Mode)

	switch failoverMode {
	case nvidiacomv1beta1.GMSModeIntraPod:
		if gms == nil {
			errs = append(errs, fmt.Errorf(
				"%s.experimental.failover: intraPod failover requires gpuMemoryService to be set",
				v.fieldPath))
		} else if betaGMSMode(gms.Mode) != nvidiacomv1beta1.GMSModeIntraPod {
			errs = append(errs, fmt.Errorf(
				"%s.experimental.failover: failover.mode %q must match gpuMemoryService.mode %q",
				v.fieldPath, failoverMode, gms.Mode))
		}
		if failover.NumShadows != 0 && failover.NumShadows != 1 {
			errs = append(errs, fmt.Errorf(
				"%s.experimental.failover.numShadows=%d is invalid for mode=%q: intraPod uses a fixed 1 primary + 1 shadow sidecar; "+
					"use failover.mode=%q to configure numShadows",
				v.fieldPath, failover.NumShadows, nvidiacomv1beta1.GMSModeIntraPod, nvidiacomv1beta1.GMSModeInterPod))
		}

	case nvidiacomv1beta1.GMSModeInterPod:
		if gms == nil {
			errs = append(errs, fmt.Errorf(
				"%s.experimental.failover: interPod failover requires gpuMemoryService.mode=%q",
				v.fieldPath, nvidiacomv1beta1.GMSModeInterPod))
		} else if betaGMSMode(gms.Mode) != nvidiacomv1beta1.GMSModeInterPod {
			detected := string(gms.Mode)
			if detected == "" {
				detected = unsetValue
			}
			errs = append(errs, fmt.Errorf(
				"%s.experimental.failover: interPod failover requires gpuMemoryService.mode=%q (got %q)",
				v.fieldPath, nvidiacomv1beta1.GMSModeInterPod, detected))
		}
		if failover.NumShadows < 1 {
			errs = append(errs, fmt.Errorf("%s.experimental.failover.numShadows must be >= 1", v.fieldPath))
		}

		gpuCount, err := dra.ExtractGPUCountFromResourceRequirements(dynamo.GetMainContainerResources(v.spec))
		if err != nil {
			errs = append(errs, fmt.Errorf("%s.podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu: %w", v.fieldPath, err))
		} else if gpuCount < 1 {
			errs = append(errs, fmt.Errorf("%s: GMS failover requires at least 1 GPU in podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu", v.fieldPath))
		}

		switch v.spec.ComponentType {
		case nvidiacomv1beta1.ComponentTypeEPP, nvidiacomv1beta1.ComponentTypeFrontend, nvidiacomv1beta1.ComponentTypePlanner:
			errs = append(errs, fmt.Errorf("%s: GMS failover is not supported for type %q", v.fieldPath, v.spec.ComponentType))
		}

	default:
		errs = append(errs, fmt.Errorf("%s.experimental.failover.mode %q is invalid", v.fieldPath, failover.Mode))
	}

	return errors.Join(errs...)
}

func (v *SharedSpecValidatorV1Beta1) validateSnapshotWithGPUMemoryService() error {
	checkpoint := betaCheckpoint(v.spec)
	gms := betaGPUMemoryService(v.spec)
	if checkpoint == nil || !checkpoint.Enabled || gms == nil {
		return nil
	}
	if os.Getenv(consts.DynamoOperatorAllowGMSSnapshotEnvVar) == "1" {
		return nil
	}
	return fmt.Errorf(
		"%s.experimental.checkpoint: GMS + Snapshot is temporarily disabled; disable gpuMemoryService or enable the internal GMS + Snapshot gate",
		v.fieldPath)
}
