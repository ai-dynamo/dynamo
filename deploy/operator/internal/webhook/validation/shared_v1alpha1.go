/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateDynamoComponentDeploymentSharedSpecV1alpha1 supplements storage-version validation
// for source-only fields, lossy conversion, and source-schema gaps. It must not repeat
// rules whose inputs survive conversion. spec and fldPath must not be nil.
// oldEnv is nil on create, otherwise it is the previous source-version envs list.
func (v *sharedValidation) validateDynamoComponentDeploymentSharedSpecV1alpha1(
	spec *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec,
	fldPath *field.Path,
	dynamoNamespace string,
	oldEnv []corev1.EnvVar,
	orderedEnvironment bool,
) field.ErrorList {
	allErrs := field.ErrorList{}
	if value, invalid := invalidVLLMDistributedExecutorBackendAnnotation(spec.Annotations); invalid {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("annotations").Key(consts.KubeAnnotationVLLMDistributedExecutorBackend),
			value,
			`must be "mp" or "ray"`,
		))
	}

	if spec.DynamoNamespace != nil && *spec.DynamoNamespace != "" {
		v.warnf(
			"%s.dynamoNamespace is deprecated and ignored. Value %q will be replaced with %q. Remove this field from your configuration",
			fldPath,
			*spec.DynamoNamespace,
			dynamoNamespace,
		)
	}
	//nolint:staticcheck // SA1019: Intentionally warning about a deprecated preserved field.
	if spec.Autoscaling != nil {
		v.warnf(
			"%s.autoscaling is deprecated and ignored. Use DynamoGraphDeploymentScalingAdapter with HPA, KEDA, or Planner for autoscaling instead. See docs/kubernetes/autoscaling.md",
			fldPath,
		)
	}

	volumeMountsPath := fldPath.Child("volumeMounts")
	for i := range spec.VolumeMounts {
		allErrs = append(allErrs, v.validateVolumeMountV1alpha1(&spec.VolumeMounts[i], volumeMountsPath.Index(i))...)
	}

	// Alpha envs is not a map-list in the source schema, unlike the converted container env.
	if v.runtimeVersionSource == runtimeVersionSourceV1Alpha1 {
		allErrs = append(allErrs, v.validateEnvVars(spec.Envs, oldEnv, fldPath.Child("envs"), orderedEnvironment)...)
	}
	if spec.Ingress != nil {
		allErrs = append(allErrs, v.validateIngressSpecV1alpha1(spec.Ingress, fldPath.Child("ingress"))...)
	}
	if spec.EPPConfig != nil && v.runtimeVersionSource == runtimeVersionSourceV1Alpha1 {
		allErrs = append(allErrs, v.validateEPPConfigV1alpha1(spec.EPPConfig, fldPath.Child("eppConfig"))...)
	}
	if spec.FrontendSidecar != nil && spec.ExtraPodSpec != nil && spec.ExtraPodSpec.PodSpec != nil &&
		hasContainerNamed(spec.ExtraPodSpec.PodSpec.Containers, consts.FrontendSidecarContainerName) {
		allErrs = append(allErrs, field.Forbidden(
			fldPath.Child("frontendSidecar"),
			fmt.Sprintf("cannot inject frontend sidecar: a container named %q already exists in extraPodSpec.containers", consts.FrontendSidecarContainerName),
		))
	}
	if spec.Failover != nil {
		allErrs = append(allErrs, v.validateFailoverSpecV1alpha1(spec.Failover, fldPath.Child("failover"))...)
	}

	return allErrs
}

// validateVolumeMountV1alpha1 checks the source-only compilation-cache flag and mount-point relation.
// Conversion loses that flag and the original flat-list distinctions.
// volumeMount and fldPath must not be nil.
func (v *sharedValidation) validateVolumeMountV1alpha1(
	volumeMount *nvidiacomv1alpha1.VolumeMount,
	fldPath *field.Path,
) field.ErrorList {
	if volumeMount.UseAsCompilationCache || volumeMount.MountPoint != "" {
		return nil
	}
	return field.ErrorList{field.Required(
		fldPath.Child("mountPoint"),
		"is required when useAsCompilationCache is false",
	)}
}

// validateIngressSpecV1alpha1 validates ingress, which has no storage-version API counterpart.
// ingress and fldPath must not be nil.
func (v *sharedValidation) validateIngressSpecV1alpha1(
	ingress *nvidiacomv1alpha1.IngressSpec,
	fldPath *field.Path,
) field.ErrorList {
	if !ingress.Enabled || ingress.Host != "" {
		return nil
	}
	return field.ErrorList{field.Required(fldPath.Child("host"), "is required when ingress is enabled")}
}

// validateEPPConfigV1alpha1 covers the exactly-one rule missing from the Alpha source schema.
// Config-map reference values survive conversion and are validated by validateEPPConfig.
// config and fldPath must not be nil.
func (v *sharedValidation) validateEPPConfigV1alpha1(
	config *nvidiacomv1alpha1.EPPConfig,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}
	if (config.ConfigMapRef == nil) == (config.Config == nil) {
		allErrs = append(allErrs, field.Forbidden(
			fldPath,
			"exactly one of configMapRef or config is required",
		))
	}
	return allErrs
}

// validateFailoverSpecV1alpha1 covers the intraPod shadow limit absent from the Alpha source schema.
// Disabled source settings are not represented after conversion; configurable interPod shadow counts
// remain an intentional Alpha API difference. failover and fldPath must not be nil.
func (v *sharedValidation) validateFailoverSpecV1alpha1(
	failover *nvidiacomv1alpha1.FailoverSpec,
	fldPath *field.Path,
) field.ErrorList {
	if !failover.Enabled || failover.Mode != nvidiacomv1alpha1.GMSModeIntraPod ||
		failover.NumShadows == 0 || failover.NumShadows == 1 {
		return nil
	}
	return field.ErrorList{field.Invalid(
		fldPath.Child("numShadows"),
		failover.NumShadows,
		fmt.Sprintf("is invalid for mode=%q: intraPod uses a fixed 1 primary + 1 shadow sidecar; use failover.mode=%q to configure numShadows", nvidiacomv1alpha1.GMSModeIntraPod, nvidiacomv1alpha1.GMSModeInterPod),
	)}
}
