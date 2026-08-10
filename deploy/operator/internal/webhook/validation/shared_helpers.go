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
	"fmt"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	unsetValue = "<unset>"

	vllmDistributedExecutorBackendMP  = "mp"
	vllmDistributedExecutorBackendRay = "ray"

	runtimeVersionOverrideRequiredMessage = "is required when the specified main container image has no parseable semantic-version tag"
)

// runtimeVersionValidationSource identifies the API representation whose field
// paths must be used for runtime-version validation errors.
type runtimeVersionValidationSource uint8

const (
	runtimeVersionSourceV1Beta1 runtimeVersionValidationSource = iota
	runtimeVersionSourceV1Alpha1
	runtimeVersionSourceDisabled
)

// runtimeVersionValidationSourceForRequest uses RequestKind because it preserves
// the GVK the client submitted when the API server converts the object for
// an equivalent-version webhook. For unconverted requests, RequestKind is nil;
// the handler endpoint GVK is then the source representation.
func runtimeVersionValidationSourceForRequest(ctx context.Context, fallbackGVK schema.GroupVersionKind) runtimeVersionValidationSource {
	request, err := admission.RequestFromContext(ctx)
	if err == nil && request.RequestKind != nil {
		return runtimeVersionValidationSourceForGVK(schema.GroupVersionKind{
			Group:   request.RequestKind.Group,
			Version: request.RequestKind.Version,
			Kind:    request.RequestKind.Kind,
		})
	}
	return runtimeVersionValidationSourceForGVK(fallbackGVK)
}

func runtimeVersionValidationSourceForGVK(gvk schema.GroupVersionKind) runtimeVersionValidationSource {
	if gvk.GroupVersion() == nvidiacomv1alpha1.GroupVersion {
		return runtimeVersionSourceV1Alpha1
	}
	return runtimeVersionSourceV1Beta1
}

func (v *sharedValidation) validatesRuntimeVersionFor(source runtimeVersionValidationSource) bool {
	return v.runtimeVersionSource == source
}

func (v *sharedValidation) validatesRequestSourceFor(source runtimeVersionValidationSource) bool {
	return v.requestSource == source
}

// runtimeVersionImageAndPath returns the main image and its v1beta1 field path.
// spec and fldPath must not be nil.
func runtimeVersionImageAndPath(
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) (string, *field.Path) {
	imagePath := fldPath.Child("podTemplate", "spec", "containers")

	// Resolve the exact container path when the named main container exists.
	if spec.PodTemplate != nil {
		if index := containerIndexByName(spec.PodTemplate.Spec.Containers, consts.MainContainerName); index >= 0 {
			imagePath = imagePath.Index(index).Child("image")
			return spec.PodTemplate.Spec.Containers[index].Image, imagePath
		}
	}
	return "", imagePath
}

// runtimeVersionImageAndPathV1Alpha1 returns the main image and its v1alpha1 field path.
// spec and fldPath must not be nil.
func runtimeVersionImageAndPathV1Alpha1(
	spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) (string, *field.Path) {
	imagePath := fldPath.Child("extraPodSpec", "mainContainer", "image")
	if spec.ExtraPodSpec != nil && spec.ExtraPodSpec.MainContainer != nil {
		return spec.ExtraPodSpec.MainContainer.Image, imagePath
	}
	return "", imagePath
}

// runtimeVersionOverrideRequired reports whether image cannot provide a version and override is absent.
func runtimeVersionOverrideRequired(image, override string) bool {
	if override != "" {
		return false
	}
	_, err := runtimeversion.ParseImageVersion(image)
	return err != nil
}

func hasContainerNamed(containers []corev1.Container, name string) bool {
	for i := range containers {
		if containers[i].Name == name {
			return true
		}
	}
	return false
}

func podTemplateContainers(podTemplate *corev1.PodTemplateSpec) []corev1.Container {
	if podTemplate == nil {
		return nil
	}
	return podTemplate.Spec.Containers
}

func invalidVLLMDistributedExecutorBackendAnnotation(annotations map[string]string) (string, bool) {
	value, exists := annotations[consts.KubeAnnotationVLLMDistributedExecutorBackend]
	if !exists {
		return "", false
	}

	switch strings.ToLower(value) {
	case vllmDistributedExecutorBackendMP, vllmDistributedExecutorBackendRay:
		return "", false
	default:
		return value, true
	}
}

// inferencePoolAvailabilityError checks the InferencePool API.
// ctx and mgr must not be nil.
func inferencePoolAvailabilityError(ctx context.Context, mgr ctrl.Manager) error {
	available, err := features.DetectInferencePoolAvailability(ctx, mgr)
	if err != nil {
		return fmt.Errorf("detect InferencePool API availability: %w", err)
	}
	if available {
		return nil
	}
	return fmt.Errorf(
		"InferencePool API group (%s) is not available in the cluster; install the Gateway API Inference Extension before deploying EPP components",
		epp.InferencePoolGroup,
	)
}

// validateElasticEPRequiresCommand rejects a vLLM component that requests the
// elastic-EP Ray topology (--enable-elastic-ep with --data-parallel-backend ray,
// including the -dpb alias and flag=value spellings) but omits the main
// container command. The operator starts the single-pod Ray head by rewriting
// the container to run "ray start ... && <command>", which needs an explicit
// executable; with only the image ENTRYPOINT (empty command) it cannot build
// that command, so elastic EP would silently never start. Fail closed here with
// an actionable error rather than admit a request the operator cannot fulfill.
func validateElasticEPRequiresCommand(
	backendFramework string,
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) field.ErrorList {
	var allErrs field.ErrorList
	if backendFramework != string(dynamo.BackendFrameworkVLLM) || spec.PodTemplate == nil {
		return allErrs
	}
	containers := spec.PodTemplate.Spec.Containers
	index := containerIndexByName(containers, consts.MainContainerName)
	if index < 0 {
		return allErrs
	}
	mainContainer := &containers[index]
	if len(mainContainer.Command) > 0 || !dynamo.IsElasticEPRayLaunch(mainContainer) {
		return allErrs
	}
	commandPath := fldPath.Child("podTemplate", "spec", "containers").Index(index).Child("command")
	allErrs = append(allErrs, field.Required(
		commandPath,
		"elastic expert parallelism (--enable-elastic-ep with --data-parallel-backend ray) requires an explicit "+
			"container command; the operator starts the single-pod Ray head by wrapping that command and cannot "+
			"start it from the image ENTRYPOINT",
	))
	return allErrs
}

func gpuMemoryServiceFor(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) *nvidiacomv1beta1.GPUMemoryServiceSpec {
	return gpuMemoryServiceForExperimental(component.Experimental)
}

func gpuMemoryServiceForExperimental(experimental *nvidiacomv1beta1.ExperimentalSpec) *nvidiacomv1beta1.GPUMemoryServiceSpec {
	if experimental == nil {
		return nil
	}
	return experimental.GPUMemoryService
}

func failoverFor(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) *nvidiacomv1beta1.FailoverSpec {
	return failoverForExperimental(component.Experimental)
}

func failoverForExperimental(experimental *nvidiacomv1beta1.ExperimentalSpec) *nvidiacomv1beta1.FailoverSpec {
	if experimental == nil {
		return nil
	}
	return experimental.Failover
}

func groveForExperimental(experimental *nvidiacomv1beta1.ExperimentalSpec) *nvidiacomv1beta1.GroveSpec {
	if experimental == nil {
		return nil
	}
	return experimental.Grove
}

func forceScalingGroupFor(experimental *nvidiacomv1beta1.ExperimentalSpec) bool {
	grove := groveForExperimental(experimental)
	return grove != nil && grove.ForceScalingGroup
}

func effectiveGMSMode(mode nvidiacomv1beta1.GPUMemoryServiceMode) nvidiacomv1beta1.GPUMemoryServiceMode {
	if mode == "" {
		return nvidiacomv1beta1.GMSModeIntraPod
	}
	return mode
}

func isInterPodGMS(gms *nvidiacomv1beta1.GPUMemoryServiceSpec) bool {
	return gms != nil && effectiveGMSMode(gms.Mode) == nvidiacomv1beta1.GMSModeInterPod
}

func isInterPodFailover(failover *nvidiacomv1beta1.FailoverSpec) bool {
	return failover != nil && effectiveGMSMode(failover.Mode) == nvidiacomv1beta1.GMSModeInterPod
}

func effectiveNumShadows(failover *nvidiacomv1beta1.FailoverSpec) int32 {
	if failover.NumShadows < 1 {
		return 1
	}
	return failover.NumShadows
}
