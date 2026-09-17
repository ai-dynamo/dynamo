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
	runtimefeatures "github.com/ai-dynamo/dynamo/deploy/operator/internal/features/runtime"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
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
	return !v.ratchetRuntimeVersion && v.runtimeVersionSource == source
}

func (v *sharedValidation) hasRuntimeVersionSource(source runtimeVersionValidationSource) bool {
	return v.runtimeVersionSource == source
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

// toleratesMissingRuntimeVersionOverride reports whether componentType may omit
// runtimeVersionOverride when its image tag carries no resolvable version.
//
// EPP is never exempt. Whether eppConfig is required (legacy Go EPP) or
// forbidden (native Rust EPP, 1.5.0+) is decided entirely by the resolved
// runtime version, so an unresolvable version enforces neither half of the
// rule: eppRuntimeCompatibilityError returns nil and the component is admitted
// with no contract checked at all. Standalone DynamoComponentDeployments set
// allowMissingRuntimeVersionOverride, so without this carve-out an EPP DCD on a
// non-semver tag (a CI sha tag, :latest, a digest ref) would silently skip the
// check that the DynamoGraphDeployment path still performs.
func (v *sharedValidation) toleratesMissingRuntimeVersionOverride(componentType string) bool {
	return v.allowMissingRuntimeVersionOverride && componentType != consts.ComponentTypeEPP
}

// runtimeVersionOverrideRequired reports whether image cannot provide a version and override is absent.
func runtimeVersionOverrideRequired(image, override string) bool {
	if override != "" {
		return false
	}
	_, err := runtimeversion.ParseImageVersion(image)
	return err != nil
}

type eppRuntimeContract struct {
	componentType          string
	image                  string
	runtimeVersionOverride string
	hasEPPConfig           bool
}

// eppRuntimeContractV1Beta1 returns the complete runtime-contract inputs represented by spec.
// spec must not be nil. image is the source-version main container image.
func eppRuntimeContractV1Beta1(spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, image string) eppRuntimeContract {
	return eppRuntimeContract{
		componentType:          string(spec.ComponentType),
		image:                  image,
		runtimeVersionOverride: spec.RuntimeVersionOverride,
		hasEPPConfig:           spec.EPPConfig != nil,
	}
}

// eppRuntimeContractV1Alpha1 returns the complete runtime-contract inputs represented by spec.
// spec must not be nil. image is the source-version main container image.
func eppRuntimeContractV1Alpha1(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec, image string) eppRuntimeContract {
	return eppRuntimeContract{
		componentType:          spec.ComponentType,
		image:                  image,
		runtimeVersionOverride: spec.RuntimeVersionOverride,
		hasEPPConfig:           spec.EPPConfig != nil,
	}
}

// eppRuntimeCompatibilityError returns the cross-field error for an EPP runtime-contract mismatch.
// Invalid or unavailable runtime versions are reported by the runtime-version
// validator instead, which for an EPP component always demands a resolvable
// version (see toleratesMissingRuntimeVersionOverride) -- so returning nil here
// defers the report rather than admitting an unchecked contract.
func eppRuntimeCompatibilityError(contract eppRuntimeContract, eppConfigPath *field.Path) *field.Error {
	if contract.componentType != consts.ComponentTypeEPP {
		return nil
	}

	version, err := runtimeversion.Resolve(contract.image, contract.runtimeVersionOverride)
	if err != nil {
		return nil
	}

	if runtimefeatures.NativeRustEPP.Enabled(&version) {
		if contract.hasEPPConfig {
			return field.Forbidden(
				eppConfigPath,
				"must be omitted for native Rust EPP images with runtime version 1.5.0 or later",
			)
		}
		return nil
	}

	if !contract.hasEPPConfig {
		return field.Required(
			eppConfigPath,
			"is required for legacy Go EPP images with runtime version earlier than 1.5.0",
		)
	}
	return nil
}

// eppRuntimeCompatibilityUpdateError ratchets only an identical pre-existing contract violation.
// eppConfigPath must not be nil. sameEPPConfig reports full source-version value equality.
func eppRuntimeCompatibilityUpdateError(
	newContract eppRuntimeContract,
	oldContract eppRuntimeContract,
	sameEPPConfig bool,
	eppConfigPath *field.Path,
) *field.Error {
	newErr := eppRuntimeCompatibilityError(newContract, eppConfigPath)
	if newErr == nil {
		return nil
	}
	if newContract == oldContract && sameEPPConfig {
		return nil
	}
	return newErr
}

func hasContainerNamed(containers []corev1.Container, name string) bool {
	for i := range containers {
		if containers[i].Name == name {
			return true
		}
	}
	return false
}

func containerIndexByName(containers []corev1.Container, name string) int {
	// Return the first exact match so callers can update that container in place.
	for i := range containers {
		if containers[i].Name == name {
			return i
		}
	}
	return -1
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

// validateElasticEPRequiresCommand requires an explicit main container command on a vLLM
// component that requests the elastic-EP Ray topology (--enable-elastic-ep with
// --data-parallel-backend ray, including the -dpb alias and flag=value spellings).
//
// The operator starts the single-pod Ray head by rewriting the container to run
// "ray start ... && <command>". With only the image ENTRYPOINT (empty command) there is
// nothing to wrap, so elastic EP would silently never start: fail closed instead.
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

// validateElasticEPSingleReplica rejects a vLLM component that requests the elastic-EP
// Ray topology with more than one replica.
//
// The leader is the Ray head; elastic EP grows by adding followers to it, not leaders. Two
// replicas mean two independent Ray clusters, but the operator renders one identity for
// them: one "<component>-ray" headless Service selecting every pod with the component
// labels, and one follower per component, so a follower would join whichever head DNS
// resolved with no way to name its leader. Per-replica identity (a Ray Service and
// follower set each) is not rendered yet, and without this check the operator silently
// emits neither the Ray Service nor the follower, leaving leaders that can never grow and
// no indication why.
//
// It must therefore fire on EXACTLY the shapes that derive a follower, and no others. The
// component type and node-count guards below are what keep it from rejecting shapes the
// feature never touches:
//
//   - a non-worker component (a frontend carrying the flags) never reaches synthesis --
//     IsWorkerComponent excludes it -- so it has no Ray Service or follower to collide over
//   - a multinode component takes the LWS path, which never renders RoleFollower, so its
//     replicas are LWS groups rather than competing Ray heads
//
// Both were accepted before this rule existed. Rejecting them now would be a regression for
// configurations that have nothing to do with elastic EP's single-leader constraint, and the
// error message -- "capacity is added by scaling followers" -- would name a follower those
// shapes will never get.
func validateElasticEPSingleReplica(
	backendFramework string,
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) field.ErrorList {
	var allErrs field.ErrorList
	if backendFramework != string(dynamo.BackendFrameworkVLLM) || spec.PodTemplate == nil {
		return allErrs
	}
	if spec.Replicas == nil || *spec.Replicas <= 1 {
		return allErrs
	}
	if !dynamo.IsWorkerComponent(string(spec.ComponentType)) {
		return allErrs
	}
	if spec.GetNumberOfNodes() > 1 {
		return allErrs
	}
	containers := spec.PodTemplate.Spec.Containers
	index := containerIndexByName(containers, consts.MainContainerName)
	if index < 0 || !dynamo.IsElasticEPRayLaunch(&containers[index]) {
		return allErrs
	}
	allErrs = append(allErrs, field.Invalid(
		fldPath.Child("replicas"),
		*spec.Replicas,
		"elastic expert parallelism (--enable-elastic-ep with --data-parallel-backend ray) supports a single "+
			"leader replica: the leader is the Ray head, and capacity is added by scaling followers rather than "+
			"leaders. Multiple independent elastic-EP clusters are not rendered yet, because each replica would "+
			"need its own Ray Service and follower set; use replicas: 1, or declare separate components",
	))
	return allErrs
}

// validateElasticEPSingleReplicaRatcheted applies the single-replica rule but lets an
// unchanged pre-existing violation through, so existing objects stay editable.
//
// The rule is gated and admission never re-runs when a cluster-wide gate flips on, so a
// component with replicas > 1 can already exist once the gate is enabled. The rule is
// stateless, so it would then fire on every UPDATE and block any edit to that component,
// not just its replica count. oldSpec is nil on create: nothing to ratchet against.
func validateElasticEPSingleReplicaRatcheted(
	backendFramework string,
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	oldSpec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) field.ErrorList {
	newErrs := validateElasticEPSingleReplica(backendFramework, spec, fldPath)
	if len(newErrs) == 0 || oldSpec == nil {
		return newErrs
	}
	// Tolerate only an identical violation: any change to the replica count, in either
	// direction and including making it worse, re-asserts the rule.
	if len(validateElasticEPSingleReplica(backendFramework, oldSpec, fldPath)) == 0 {
		return newErrs
	}
	if ptr.Deref(spec.Replicas, 0) != ptr.Deref(oldSpec.Replicas, 0) {
		return newErrs
	}
	return nil
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
	return grove != nil && ptr.Deref(grove.ForceScalingGroup, false)
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
