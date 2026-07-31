/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package grove

import (
	"reflect"
	"testing"

	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
)

func TestHandlerIsTransparent(t *testing.T) {
	h := &handler{}
	pcs := &grovev1alpha1.PodCliqueSet{}
	want := pcs.DeepCopy()

	t.Log("Pass a PodCliqueSet through the temporary defaulting handler")
	if err := h.Default(t.Context(), pcs); err != nil {
		t.Fatalf("default PodCliqueSet: %v", err)
	}
	if !reflect.DeepEqual(pcs, want) {
		t.Fatalf("defaulted PodCliqueSet = %#v, want unchanged %#v", pcs, want)
	}

	t.Log("Pass the same PodCliqueSet through create validation")
	warnings, err := h.ValidateCreate(t.Context(), pcs)
	if err != nil {
		t.Fatalf("validate PodCliqueSet: %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("validation warnings = %v, want none", warnings)
	}
}

func TestConfigurationsMatchPodCliqueSetAdmission(t *testing.T) {
	t.Log("Build the temporary registrations at the production Grove paths")
	configurations := Configurations()
	if len(configurations.Mutating) != 1 || len(configurations.Mutating[0].Webhooks) != 1 {
		t.Fatalf("mutating registrations = %#v, want one configuration with one webhook", configurations.Mutating)
	}
	if len(configurations.Validating) != 1 || len(configurations.Validating[0].Webhooks) != 1 {
		t.Fatalf("validating registrations = %#v, want one configuration with one webhook", configurations.Validating)
	}

	t.Log("Verify both registrations target only PodCliqueSets")
	for _, rule := range []struct {
		name      string
		path      *string
		resources []string
	}{
		{
			name:      "mutating",
			path:      configurations.Mutating[0].Webhooks[0].ClientConfig.Service.Path,
			resources: configurations.Mutating[0].Webhooks[0].Rules[0].Resources,
		},
		{
			name:      "validating",
			path:      configurations.Validating[0].Webhooks[0].ClientConfig.Service.Path,
			resources: configurations.Validating[0].Webhooks[0].Rules[0].Resources,
		},
	} {
		if rule.path == nil || *rule.path == "" {
			t.Fatalf("%s webhook path is empty", rule.name)
		}
		if len(rule.resources) != 1 || rule.resources[0] != groveResource {
			t.Fatalf("%s webhook resources = %v, want [%s]", rule.name, rule.resources, groveResource)
		}
	}
}
