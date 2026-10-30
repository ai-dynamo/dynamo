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
	"slices"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
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

	runtimeVersionOverrideRequiredMessage = "is required when the specified Dynamo runtime container image has no parseable semantic-version tag"
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

// runtimeVersionImageAndPath returns the Dynamo runtime image and its v1beta1 field path.
// spec and fldPath must not be nil.
func runtimeVersionImageAndPath(
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) (string, *field.Path) {
	// In sidecar mode the engine image says nothing about Dynamo compatibility.
	if spec.PodTemplate != nil {
		if index := slices.IndexFunc(spec.PodTemplate.Spec.InitContainers, func(c corev1.Container) bool { return c.Name == consts.RuntimeContainerName }); index >= 0 {
			return spec.PodTemplate.Spec.InitContainers[index].Image, fldPath.Child("podTemplate", "spec", "initContainers").Index(index).Child("image")
		}
	}

	imagePath := fldPath.Child("podTemplate", "spec", "containers")

	// Resolve the exact container path when the named main container exists.
	if spec.PodTemplate != nil {
		if index := mainContainerIndex(spec.PodTemplate.Spec.Containers); index >= 0 {
			imagePath = imagePath.Index(index).Child("image")
			return spec.PodTemplate.Spec.Containers[index].Image, imagePath
		}
	}
	return "", imagePath
}

// runtimeVersionImageAndPathV1Alpha1 returns the runtime image and its alpha field path.
// spec and fldPath must not be nil.
func runtimeVersionImageAndPathV1Alpha1(
	spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) (string, *field.Path) {
	// Alpha represents init containers directly, so live fields determine the mode.
	if spec.ExtraPodSpec != nil && spec.ExtraPodSpec.PodSpec != nil {
		if index := slices.IndexFunc(spec.ExtraPodSpec.PodSpec.InitContainers, func(c corev1.Container) bool { return c.Name == consts.RuntimeContainerName }); index >= 0 {
			return spec.ExtraPodSpec.PodSpec.InitContainers[index].Image, fldPath.Child("extraPodSpec", "initContainers").Index(index).Child("image")
		}
	}

	imagePath := fldPath.Child("extraPodSpec", "mainContainer", "image")
	if spec.ExtraPodSpec != nil && spec.ExtraPodSpec.MainContainer != nil {
		return spec.ExtraPodSpec.MainContainer.Image, imagePath
	}
	return "", imagePath
}

// toleratesMissingRuntimeVersionOverride reports whether componentType may omit
// runtimeVersionOverride when its image tag carries no resolvable version.
//
// Components without a Dynamo role do not carry a Dynamo runtime contract.
// EPP is never exempt. Whether eppConfig is required (legacy Go EPP) or
// forbidden (native Rust EPP, 1.5.0+) is decided entirely by the resolved
// runtime version, so an unresolvable version enforces neither half of the
// rule: eppRuntimeCompatibilityError returns nil and the component is admitted
// with no contract checked at all. Standalone DynamoComponentDeployments set
// allowMissingRuntimeVersionOverride, so without this carve-out an EPP DCD on a
// non-semver tag (a CI sha tag, :latest, a digest ref) would silently skip the
// check that the DynamoGraphDeployment path still performs.
func (v *sharedValidation) toleratesMissingRuntimeVersionOverride(componentType string) bool {
	switch componentType {
	case consts.ComponentTypeEPP:
		return false
	case consts.ComponentTypeFrontend, consts.ComponentTypeWorker, consts.ComponentTypePrefill,
		consts.ComponentTypeDecode, consts.ComponentTypePlanner:
		return v.allowMissingRuntimeVersionOverride
	default:
		return true
	}
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

func mainContainerIndex(containers []corev1.Container) int {
	// Return the first exact match so callers can update that container in place.
	for i := range containers {
		if containers[i].Name == consts.MainContainerName {
			return i
		}
	}
	return -1
}

// restoreEntrypointInputs is the complete input of the checkpoint restore
// entrypoint rule for one restore destination: the Pod template that renders it,
// the restore target and the launch it receives.
type restoreEntrypointInputs struct {
	// roleName names the role that renders the destination. Before rendering it is
	// the role whose complete PodTemplate authored the input, or empty for the
	// component-level PodTemplate, which serves every role.
	roleName        string
	targetContainer string
	// roleIndex and containerIndex locate the target in the authored lists for error
	// paths only; the destination keeps its identity when siblings are reordered or
	// inserted. roleIndex is -1 for the component-level PodTemplate and
	// containerIndex is -1 when the target container is missing.
	roleIndex      int
	containerIndex int
	// command and args are the launch restore admission sees: the rendered launch when
	// the operator can derive it, otherwise the authored one.
	command []string
	args    []string
	// authoredSupported reports whether the authored launch already satisfies the
	// contract, so a rejection comes from the operator rewriting it.
	authoredSupported bool
}

// supported reports whether the launch satisfies the restore entrypoint contract.
func (i restoreEntrypointInputs) supported() bool {
	return checkpoint.HasSupportedRestoreEntrypoint(&corev1.Container{Command: i.command, Args: i.args})
}

// equal compares the destination identity and launch, treating nil and empty launch slices alike.
func (i restoreEntrypointInputs) equal(other restoreEntrypointInputs) bool {
	return i.roleName == other.roleName &&
		i.targetContainer == other.targetContainer &&
		slices.Equal(i.command, other.command) &&
		slices.Equal(i.args, other.args)
}

// restoreEntrypointInputsForTemplate resolves the target container launch of one Pod template.
// roleIndex is -1 for the component-level template; podTemplate may be nil.
func restoreEntrypointInputsForTemplate(
	podTemplate *corev1.PodTemplateSpec,
	roleName string,
	roleIndex int,
	targetContainer string,
) restoreEntrypointInputs {
	inputs := restoreEntrypointInputs{
		roleName:        roleName,
		targetContainer: targetContainer,
		roleIndex:       roleIndex,
		containerIndex:  -1,
	}

	// An authored command replaces the operator's /bin/sh -c default, so a missing one leaves the shell.
	containers := podTemplateContainers(podTemplate)
	for i := range containers {
		if containers[i].Name == targetContainer {
			inputs.containerIndex = i
			inputs.command = containers[i].Command
			inputs.args = containers[i].Args
			break
		}
	}
	inputs.authoredSupported = inputs.supported()
	return inputs
}

// checkpointRestoreEntrypointOptions carries what rendering needs besides the component:
// the graph's explicit backend, the graph annotations propagated to components, and
// whether the graph renders through Grove or LeaderWorkerSet.
type checkpointRestoreEntrypointOptions struct {
	backendFramework string
	annotations      map[string]string
	grovePathway     bool
}

// restoreEntrypointInputsFor resolves the entrypoint of every restore destination
// that restore admission will see for the checkpoint target container. The
// component-level PodTemplate is one destination; in role-template mode each role
// with a complete PodTemplate is one, mirroring how rendering selects the template.
// The operator rewrites the launch of the main container while rendering, so its
// destinations carry the rendered launch whenever it can be derived. It reports
// false when the component does not enable checkpointing for a component type that
// supports it, or when the launch is not authored by the user (native sidecars).
// spec must not be nil.
func restoreEntrypointInputsFor(
	ctx context.Context,
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	options checkpointRestoreEntrypointOptions,
) ([]restoreEntrypointInputs, bool) {
	if spec.Experimental == nil || spec.Experimental.Checkpoint == nil || !spec.Experimental.Checkpoint.Enabled {
		return nil, false
	}
	switch spec.ComponentType {
	case nvidiacomv1beta1.ComponentTypeWorker, nvidiacomv1beta1.ComponentTypePrefill, nvidiacomv1beta1.ComponentTypeDecode:
	default:
		return nil, false
	}
	if dynamo.GetDynamoSidecar(spec) != nil {
		return nil, false
	}

	targetContainer := spec.Experimental.Checkpoint.TargetContainerName
	if targetContainer == "" {
		targetContainer = consts.MainContainerName
	}

	// A component-level PodTemplate is the complete source for every role.
	var inputs []restoreEntrypointInputs
	if spec.PodTemplate != nil || !dynamo.HasRolePodTemplates(spec) {
		inputs = []restoreEntrypointInputs{restoreEntrypointInputsForTemplate(spec.PodTemplate, "", -1, targetContainer)}
	} else {
		// Role-template mode renders each role from its own complete PodTemplate.
		for i := range spec.Roles {
			if spec.Roles[i].PodTemplate != nil {
				inputs = append(inputs, restoreEntrypointInputsForTemplate(spec.Roles[i].PodTemplate, spec.Roles[i].Name, i, targetContainer))
			}
		}
	}

	// Only the main container's launch is rewritten while rendering.
	if targetContainer != consts.MainContainerName {
		return inputs, true
	}
	return renderRestoreEntrypointInputs(ctx, spec, options, inputs), true
}

// renderRestoreEntrypointInputs replaces the authored launch of each main-container
// destination with the launch the operator renders for the roles it serves. It keeps
// the authored inputs when the launch cannot be derived, which includes a template
// without the main container, so admission is never stricter than restore Pod
// admission on what it cannot model.
func renderRestoreEntrypointInputs(
	ctx context.Context,
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	options checkpointRestoreEntrypointOptions,
	authored []restoreEntrypointInputs,
) []restoreEntrypointInputs {
	deploymentType := consts.MultinodeDeploymentTypeLWS
	if options.grovePathway {
		deploymentType = consts.MultinodeDeploymentTypeGrove
	}
	launches, err := dynamo.RenderMainContainerLaunches(ctx, spec, options.backendFramework, options.annotations, deploymentType)
	if err != nil {
		return authored
	}

	// Each rendered role becomes one destination located at the authored container that produced it.
	rendered := make([]restoreEntrypointInputs, 0, len(launches))
	for _, launch := range launches {
		for _, input := range authored {
			// The component-level template (no role name) serves every role; a role template only its own.
			if input.roleName != "" && input.roleName != string(launch.Role) {
				continue
			}
			input.roleName = string(launch.Role)
			input.command = launch.Command
			input.args = launch.Args
			rendered = append(rendered, input)
		}
	}
	if len(rendered) == 0 {
		return authored
	}
	return rendered
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

// validateElasticEPRequiresCommand rejects an operator-managed vLLM leader
// that requests the elastic-EP Ray topology but omits the main container
// command. Complete role PodTemplates own their launch commands and skip this
// operator rewrite.
func validateElasticEPRequiresCommand(
	backendFramework string,
	spec *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
) field.ErrorList {
	if backendFramework != string(dynamo.BackendFrameworkVLLM) || dynamo.HasRolePodTemplates(spec) {
		return nil
	}

	// The global template is also the leader source in legacy template mode.
	return validateElasticEPTemplateRequiresCommand(spec.PodTemplate, fldPath.Child("podTemplate"))
}

func validateElasticEPTemplateRequiresCommand(
	podTemplate *corev1.PodTemplateSpec,
	fldPath *field.Path,
) field.ErrorList {
	if podTemplate == nil {
		return nil
	}
	containers := podTemplate.Spec.Containers
	index := mainContainerIndex(containers)
	if index < 0 {
		return nil
	}
	mainContainer := &containers[index]
	if len(mainContainer.Command) > 0 || !dynamo.IsElasticEPRayLaunch(mainContainer) {
		return nil
	}
	commandPath := fldPath.Child("spec", "containers").Index(index).Child("command")
	return field.ErrorList{field.Required(
		commandPath,
		"elastic expert parallelism (--enable-elastic-ep with --data-parallel-backend ray) requires an explicit "+
			"container command; the operator starts the Ray head by wrapping that command and cannot "+
			"start it from the image ENTRYPOINT",
	)}
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

// groveNativeMinimumPath locates the owning component's native availability field.
// component and fldPath are non-nil; an omitted override uses the same path.
func groveNativeMinimumPath(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, fldPath *field.Path) *field.Path {
	if component.UsesPCSG() || component.IsLPX() {
		return fldPath.Child("providerOverride", "value", "minAvailable")
	}
	return fldPath.Child("providerOverride", "value", "spec", "minAvailable")
}

// hasRoleSidecarV1Alpha1 resolves component-wide mode from the authored leader.
// spec must not be nil.
func hasRoleSidecarV1Alpha1(spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) bool {
	for _, role := range spec.Roles {
		if role.Name != nvidiacomv1alpha1.ComponentRoleLeader || role.PodTemplate == nil {
			continue
		}
		for _, container := range role.PodTemplate.Spec.InitContainers {
			if container.Name == consts.RuntimeContainerName {
				return true
			}
		}
	}
	return false
}
