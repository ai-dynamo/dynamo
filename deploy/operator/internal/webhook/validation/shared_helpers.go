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
	"cmp"
	"context"
	"errors"
	"fmt"
	"maps"
	"net"
	"reflect"
	"slices"
	"strings"
	"syscall"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	runtimefeatures "github.com/ai-dynamo/dynamo/deploy/operator/internal/features/runtime"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	"github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/santhosh-tekuri/jsonschema/v6/kind"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
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

// traceClaimPathHasOnlyNamedSegments reports whether subPath is relative to the claim root and
// every segment names an entry: none is empty, "." or "..".
func traceClaimPathHasOnlyNamedSegments(subPath string) bool {
	if subPath == "" {
		return false
	}
	for segment := range strings.SplitSeq(subPath, "/") {
		if segment == "" || segment == "." || segment == ".." {
			return false
		}
	}
	return true
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

// inputSchemaErrors reports native JSON pointers at the shared spec owner. Lowering
// combines typed fields and overrides, so a native path need not name an override.
// Do not include input values in errors: arbitrary override strings may contain secrets.
func inputSchemaErrors(err *jsonschema.ValidationError, fldPath *field.Path) field.ErrorList {
	switch err.ErrorKind.(type) {
	case *kind.Schema, *kind.Group, *kind.Reference:
		allErrs := make(field.ErrorList, 0, len(err.Causes))
		for _, cause := range err.Causes {
			allErrs = append(allErrs, inputSchemaErrors(cause, fldPath)...)
		}
		slices.SortFunc(allErrs, func(a, b *field.Error) int { return cmp.Compare(a.Detail, b.Detail) })
		return allErrs
	}

	// Keep unions as one error, but retain the paths needed to repair nested input.
	path := inputJSONPointer(err.InstanceLocation)
	keyword := strings.Join(err.ErrorKind.KeywordPath(), "/")
	detail := fmt.Sprintf("generated AISimulate input at %s violates JSON Schema keyword %s", path, keyword)
	switch constraint := err.ErrorKind.(type) {
	case *kind.Required:
		names := slices.Sorted(slices.Values(constraint.Missing))
		detail += ": missing " + strings.Join(names, ", ")
	case *kind.AdditionalProperties:
		names := slices.Sorted(slices.Values(constraint.Properties))
		detail += ": unknown " + strings.Join(names, ", ")
	case *kind.AnyOf, *kind.OneOf:
		locations := map[string]bool{}
		pending := append([]*jsonschema.ValidationError(nil), err.Causes...)
		for len(pending) > 0 {
			cause := pending[0]
			pending = append(pending[1:], cause.Causes...)
			if len(cause.InstanceLocation) > len(err.InstanceLocation) {
				locations[inputJSONPointer(cause.InstanceLocation)] = true
			}
		}
		paths := make([]string, 0, len(locations))
		for location := range locations {
			paths = append(paths, location)
		}
		slices.Sort(paths)
		if len(paths) > 0 {
			detail += "; alternatives reject " + strings.Join(paths, ", ")
		}
	}
	return field.ErrorList{field.Invalid(fldPath, "<redacted>", detail)}
}

func inputJSONPointer(location []string) string {
	escape := strings.NewReplacer("~", "~0", "/", "~1")
	tokens := make([]string, len(location))
	for i, token := range location {
		tokens[i] = escape.Replace(token)
	}
	return "/" + strings.Join(tokens, "/")
}

// restoreLaunch is one restore destination's rendered launch, keyed by role and container name
// rather than list position, so reordering containers keeps its identity.
type restoreLaunch struct {
	role      dynamo.Role
	container corev1.Container
	authored  *corev1.Container
}

// renderRestoreLaunches returns each restore destination's launch, in role order, as the operator renders it.
// A role whose destination cannot be rendered is reported in the error, and the other roles' launches are still returned.
// ctx, reader, operatorConfig, dgd, and component must be non-nil.
func renderRestoreLaunches(
	ctx context.Context,
	reader client.Reader,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	grovePathway bool,
) ([]restoreLaunch, error) {
	// Restore uses the same destination rule as the operator's candidate metadata.
	target := consts.MainContainerName
	if name := dynamo.GetCheckpoint(component).TargetContainerName; name != "" && !dynamo.IsIntraPodFailoverEnabled(component) {
		target = name
	}

	deploymentType := consts.MultinodeDeploymentTypeLWS
	if grovePathway {
		deploymentType = consts.MultinodeDeploymentTypeGrove
	}
	mains, err := dynamo.PrepareDGDComponentMainContainers(ctx, reader, dgd, component, operatorConfig, deploymentType)

	launches := make([]restoreLaunch, 0, len(mains))
	var errs []error
	if err != nil {
		errs = append(errs, err)
	}
	for _, main := range mains {
		effective, err := dynamo.EffectiveComponentForRole(component, main.Role)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		var authored *corev1.Container
		if effective.PodTemplate != nil {
			if i := slices.IndexFunc(effective.PodTemplate.Spec.Containers, func(c corev1.Container) bool { return c.Name == target }); i >= 0 {
				authored = &effective.PodTemplate.Spec.Containers[i]
			}
		}

		// Only main is rewritten by the operator; another target must exist as authored.
		launch := main.Container
		if target != consts.MainContainerName {
			if authored == nil {
				errs = append(errs, fmt.Errorf("restore destination container %q not found in role %s", target, main.Role))
				continue
			}
			launch = *authored
		}
		launches = append(launches, restoreLaunch{
			role:      main.Role,
			container: corev1.Container{Name: target, Command: launch.Command, Args: launch.Args},
			authored:  authored,
		})
	}
	return launches, errors.Join(errs...)
}

func hasUnchangedRestoreLaunch(old []restoreLaunch, launch restoreLaunch) bool {
	return slices.ContainsFunc(old, func(o restoreLaunch) bool {
		return o.role == launch.role &&
			o.container.Name == launch.container.Name &&
			slices.Equal(o.container.Command, launch.container.Command) &&
			slices.Equal(o.container.Args, launch.container.Args)
	})
}

// restoreLaunchPath returns the target container's command in the role's PodTemplate, or the
// PodTemplate itself when the container is not authored. fldPath is the component path.
func restoreLaunchPath(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	launch restoreLaunch,
	fldPath *field.Path,
) *field.Path {
	templatePath := fldPath.Child("podTemplate")
	template := component.PodTemplate
	if dynamo.HasRolePodTemplates(component) {
		if i := slices.IndexFunc(component.Roles, func(r nvidiacomv1beta1.ComponentRoleSpec) bool { return r.Name == string(launch.role) }); i >= 0 {
			templatePath = fldPath.Child("roles").Index(i).Child("podTemplate")
			template = component.Roles[i].PodTemplate
		}
	}
	if template == nil {
		return templatePath
	}
	i := slices.IndexFunc(template.Spec.Containers, func(c corev1.Container) bool { return c.Name == launch.container.Name })
	if i < 0 {
		return templatePath
	}
	return templatePath.Child("spec", "containers").Index(i).Child("command")
}

func unsupportedRestoreLaunchDetail(launch restoreLaunch) string {
	detail := fmt.Sprintf(
		"checkpoint restore destination container %q (role %s) must directly invoke python -m <module>, "+
			"and the module must call maybe_run_restore_standby_mode(); rendered command=%q",
		launch.container.Name,
		launch.role,
		launch.container.Command,
	)
	switch {
	case launch.authored == nil || len(launch.authored.Command) == 0:
		detail += "; no command is authored, so this is the operator's default launch"
	case !slices.Equal(launch.authored.Command, launch.container.Command):
		detail += "; the operator rewrote the authored command into this form"
	}
	return detail
}

func (v *sharedValidation) restoreLaunchReader() *memoReader {
	if v.restoreReader == nil {
		v.restoreReader = newMemoReader(v.mgr.GetAPIReader())
	}
	return v.restoreReader
}

func renderErrorExplained(allErrs field.ErrorList, renderErr error) bool {
	return slices.ContainsFunc(allErrs, func(e *field.Error) bool {
		return e.Detail != "" && strings.Contains(renderErr.Error(), e.Detail)
	})
}

// restoreRenderInputsUnchanged reports whether an update leaves everything that renders the
// component's restore launch unchanged: the component apart from replicas, and its DGD's
// annotations and graph-wide spec. All arguments must be non-nil.
func restoreRenderInputsUnchanged(
	oldDGD, dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	oldComponent, component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) bool {
	oldSpec, spec := oldDGD.Spec, dgd.Spec
	oldSpec.Components, spec.Components = nil, nil
	oldShared, shared := *oldComponent, *component
	oldShared.Replicas, shared.Replicas = nil, nil
	return apiequality.Semantic.DeepEqual(oldShared, shared) &&
		apiequality.Semantic.DeepEqual(oldDGD.Annotations, dgd.Annotations) &&
		apiequality.Semantic.DeepEqual(oldSpec, spec)
}

// memoReader serves repeated Gets of the same object from the first read, so
// the old and new restore launches of one request see the same dependencies.
type memoReader struct {
	client.Reader
	gets map[memoKey]memoResult
}

type memoKey struct {
	kind string
	key  client.ObjectKey
}

type memoResult struct {
	obj client.Object
	err error
}

func newMemoReader(reader client.Reader) *memoReader {
	return &memoReader{Reader: reader, gets: map[memoKey]memoResult{}}
}

func restoreReadInfrastructureError(reader *memoReader) error {
	if reader == nil {
		return nil
	}
	keys := slices.SortedFunc(maps.Keys(reader.gets), func(a, b memoKey) int {
		return cmp.Or(cmp.Compare(a.kind, b.kind), cmp.Compare(a.key.Namespace, b.key.Namespace), cmp.Compare(a.key.Name, b.key.Name))
	})
	for _, key := range keys {
		err := reader.gets[key].err
		if isRestoreReadInfrastructureError(err) {
			return k8serrors.NewInternalError(fmt.Errorf("cannot read a restore launch dependency: %w", err))
		}
	}
	return nil
}

func isRestoreReadInfrastructureError(err error) bool {
	var networkError net.Error
	return errors.As(err, &networkError) ||
		errors.Is(err, syscall.EMFILE) || errors.Is(err, syscall.ENFILE) || errors.Is(err, syscall.ENOMEM) ||
		errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) ||
		k8serrors.IsInternalError(err) || k8serrors.IsServiceUnavailable(err) ||
		k8serrors.IsTimeout(err) || k8serrors.IsServerTimeout(err) || k8serrors.IsTooManyRequests(err)
}

func (r *memoReader) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	k := memoKey{kind: fmt.Sprintf("%T", obj), key: key}
	if cached, ok := r.gets[k]; ok {
		if cached.err == nil {
			reflect.ValueOf(obj).Elem().Set(reflect.ValueOf(cached.obj.DeepCopyObject()).Elem())
		}
		return cached.err
	}
	err := r.Reader.Get(ctx, key, obj, opts...)
	result := memoResult{err: err}
	if err == nil {
		result.obj = obj.DeepCopyObject().(client.Object)
	}
	r.gets[k] = result
	return err
}
