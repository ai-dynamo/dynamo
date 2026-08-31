/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// Image refs the project actually deploys, alongside the semver refs the
// migration rule was written for. The non-semver ones are the interesting
// cases: CI rewrites the EPP image to a sha tag, the shipped GAIE examples use
// a placeholder tag, and digest pinning strips the tag entirely. None of them
// resolve to a runtime version, so the eppConfig contract cannot be decided
// from the image alone.
const (
	imageCISHATag    = "registry.example/dynamo:abc1234-vllm"
	imagePlaceholder = "nvcr.io/nvidia/ai-dynamo/epp-image:my-tag"
	imageDigestRef   = "registry.example/dynamo-frontend@sha256:" +
		"0000000000000000000000000000000000000000000000000000000000000000"
	imageFloatingTag = "registry.example/dynamo-frontend:latest"
	imageRustEPP     = "registry.example/dynamo-frontend:1.5.0"
	imageRustEPPRC   = "registry.example/dynamo-frontend:1.5.0-rc.2"
	imageLegacyGoEPP = "registry.example/epp-image:1.4.1"
)

// TestEPPRuntimeCompatibilityErrorOverRealImageRefs pins which refs the
// eppConfig migration rule can actually decide. A ref that resolves to no
// version yields no verdict here by design; the runtime-version validator
// rejects it instead (see TestEPPRequiresResolvableRuntimeVersion).
func TestEPPRuntimeCompatibilityErrorOverRealImageRefs(t *testing.T) {
	tests := []struct {
		name         string
		image        string
		hasEPPConfig bool
		wantType     field.ErrorType
		wantNoError  bool
	}{
		{name: "sha tag is undecidable", image: imageCISHATag, wantNoError: true},
		{name: "sha tag with eppConfig is undecidable", image: imageCISHATag, hasEPPConfig: true, wantNoError: true},
		{name: "placeholder tag is undecidable", image: imagePlaceholder, wantNoError: true},
		{name: "digest ref is undecidable", image: imageDigestRef, wantNoError: true},
		{name: "floating tag is undecidable", image: imageFloatingTag, wantNoError: true},

		{name: "rust EPP without eppConfig is valid", image: imageRustEPP, wantNoError: true},
		{name: "rust EPP with eppConfig is forbidden", image: imageRustEPP, hasEPPConfig: true, wantType: field.ErrorTypeForbidden},
		{name: "rust EPP prerelease with eppConfig is forbidden", image: imageRustEPPRC, hasEPPConfig: true, wantType: field.ErrorTypeForbidden},

		{name: "legacy Go EPP without eppConfig is required", image: imageLegacyGoEPP, wantType: field.ErrorTypeRequired},
		{name: "legacy Go EPP with eppConfig is valid", image: imageLegacyGoEPP, hasEPPConfig: true, wantNoError: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			contract := eppRuntimeContract{
				componentType: consts.ComponentTypeEPP,
				image:         tt.image,
				hasEPPConfig:  tt.hasEPPConfig,
			}
			err := eppRuntimeCompatibilityError(contract, field.NewPath("spec", "eppConfig"))
			switch {
			case tt.wantNoError && err != nil:
				t.Fatalf("expected no error, got %v", err)
			case !tt.wantNoError && err == nil:
				t.Fatalf("expected a %s error, got none", tt.wantType)
			case !tt.wantNoError && err.Type != tt.wantType:
				t.Fatalf("expected %s, got %s (%s)", tt.wantType, err.Type, err.Detail)
			}
		})
	}
}

// TestEPPRuntimeCompatibilityErrorIgnoresNonEPPComponents keeps the rule scoped
// to EPP: a frontend on a legacy tag must not be told to add an eppConfig.
func TestEPPRuntimeCompatibilityErrorIgnoresNonEPPComponents(t *testing.T) {
	contract := eppRuntimeContract{componentType: consts.ComponentTypeFrontend, image: imageLegacyGoEPP}
	if err := eppRuntimeCompatibilityError(contract, field.NewPath("spec", "eppConfig")); err != nil {
		t.Fatalf("expected no error for a non-EPP component, got %v", err)
	}
}

func eppSharedSpec(componentType nvidiacomv1beta1.ComponentType, image string) *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	replicas := int32(1)
	return &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentType: componentType,
		Replicas:      &replicas,
		PodTemplate: &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: consts.MainContainerName, Image: image}},
			},
		},
	}
}

// runtimeVersionOverrideError returns the runtimeVersionOverride error from
// errs, or nil. That field is the one the EPP contract falls back to when the
// image carries no resolvable version.
func runtimeVersionOverrideError(errs field.ErrorList) *field.Error {
	for _, err := range errs {
		if strings.HasSuffix(err.Field, "runtimeVersionOverride") {
			return err
		}
	}
	return nil
}

// TestEPPRequiresResolvableRuntimeVersion is the end-to-end half: an EPP
// component whose image carries no resolvable version must be rejected rather
// than admitted with its eppConfig contract unchecked.
//
// allowMissingRuntimeVersionOverride is the standalone
// DynamoComponentDeployment path; without the EPP carve-out that path admitted
// every ref in this table, while the DynamoGraphDeployment path rejected them.
func TestEPPRequiresResolvableRuntimeVersion(t *testing.T) {
	for _, image := range []string{imageCISHATag, imagePlaceholder, imageDigestRef, imageFloatingTag} {
		for _, allowMissing := range []bool{false, true} {
			t.Run(image, func(t *testing.T) {
				v := &sharedValidation{
					runtimeVersionSource:               runtimeVersionSourceV1Beta1,
					allowMissingRuntimeVersionOverride: allowMissing,
				}
				errs := v.validateDynamoComponentDeploymentSharedSpec(
					eppSharedSpec(consts.ComponentTypeEPP, image),
					field.NewPath("spec"),
					dynamoComponentDeploymentSharedSpecValidationOptions{},
				)
				if runtimeVersionOverrideError(errs) == nil {
					t.Fatalf("EPP on unresolvable image %q was admitted with no runtime-version error "+
						"(allowMissingRuntimeVersionOverride=%v); its eppConfig contract went unchecked",
						image, allowMissing)
				}
			})
		}
	}
}

// TestNonEPPKeepsMissingRuntimeVersionOverrideExemption guards the other side of
// the carve-out: it must not silently tighten every other component type.
func TestNonEPPKeepsMissingRuntimeVersionOverrideExemption(t *testing.T) {
	v := &sharedValidation{
		runtimeVersionSource:               runtimeVersionSourceV1Beta1,
		allowMissingRuntimeVersionOverride: true,
	}
	errs := v.validateDynamoComponentDeploymentSharedSpec(
		eppSharedSpec(consts.ComponentTypeFrontend, imageCISHATag),
		field.NewPath("spec"),
		dynamoComponentDeploymentSharedSpecValidationOptions{},
	)
	if err := runtimeVersionOverrideError(errs); err != nil {
		t.Fatalf("non-EPP component lost its exemption: %v", err)
	}
}

// TestEPPRequiresResolvableRuntimeVersionOnUpdate covers the update path, where
// the create-path runtime-version block is skipped entirely
// (validatesRuntimeVersionFor is false once ratchetRuntimeVersion is set) and
// the update variants run instead. Changing an EPP to an image with no
// resolvable version must still be rejected rather than admitted uncontracted.
func TestEPPRequiresResolvableRuntimeVersionOnUpdate(t *testing.T) {
	for _, allowMissing := range []bool{false, true} {
		v := &sharedValidation{
			runtimeVersionSource:               runtimeVersionSourceV1Beta1,
			ratchetRuntimeVersion:              true,
			allowMissingRuntimeVersionOverride: allowMissing,
		}
		errs := v.validateDynamoComponentDeploymentSharedSpecUpdate(
			eppSharedSpec(consts.ComponentTypeEPP, imageCISHATag),
			eppSharedSpec(consts.ComponentTypeEPP, imageRustEPP),
			field.NewPath("spec"),
			true,
			schema.GroupKind{Group: "nvidia.com", Kind: "DynamoGraphDeployment"},
			false,
		)
		if runtimeVersionOverrideError(errs) == nil {
			t.Fatalf("EPP updated onto unresolvable image %q was admitted "+
				"(allowMissingRuntimeVersionOverride=%v)", imageCISHATag, allowMissing)
		}
	}
}

// TestEPPUnchangedUnresolvableImageIsRatcheted keeps the update ratchet intact:
// an unrelated edit to a component that already had an unresolvable image must
// not start failing, or every existing EPP object becomes unpatchable.
func TestEPPUnchangedUnresolvableImageIsRatcheted(t *testing.T) {
	v := &sharedValidation{
		runtimeVersionSource:  runtimeVersionSourceV1Beta1,
		ratchetRuntimeVersion: true,
	}
	errs := v.validateDynamoComponentDeploymentSharedSpecUpdate(
		eppSharedSpec(consts.ComponentTypeEPP, imageCISHATag),
		eppSharedSpec(consts.ComponentTypeEPP, imageCISHATag),
		field.NewPath("spec"),
		true,
		schema.GroupKind{Group: "nvidia.com", Kind: "DynamoGraphDeployment"},
		false,
	)
	if err := runtimeVersionOverrideError(errs); err != nil {
		t.Fatalf("unchanged pre-existing image lost its ratchet: %v", err)
	}
}
