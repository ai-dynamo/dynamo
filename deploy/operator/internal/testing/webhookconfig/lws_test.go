/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package webhookconfig

import "testing"

func TestLeaderWorkerSetConfigurationsContainOnlyLeaderWorkerSetWebhooks(t *testing.T) {
	t.Log("Build the focused registrations from the upstream LeaderWorkerSet webhook markers")
	configurations := LeaderWorkerSetConfigurations()
	if len(configurations.Mutating) != 1 || len(configurations.Mutating[0].Webhooks) != 1 {
		t.Fatalf("mutating registrations = %#v, want one configuration with one webhook", configurations.Mutating)
	}
	if len(configurations.Validating) != 1 || len(configurations.Validating[0].Webhooks) != 1 {
		t.Fatalf("validating registrations = %#v, want one configuration with one webhook", configurations.Validating)
	}

	t.Log("Verify both registrations retain their service paths and target only LeaderWorkerSets")
	for name, path := range map[string]*string{
		"mutating":   configurations.Mutating[0].Webhooks[0].ClientConfig.Service.Path,
		"validating": configurations.Validating[0].Webhooks[0].ClientConfig.Service.Path,
	} {
		if path == nil || *path == "" {
			t.Fatalf("%s webhook path is empty", name)
		}
	}
	for _, rule := range []struct {
		name      string
		resources []string
	}{
		{name: "mutating", resources: configurations.Mutating[0].Webhooks[0].Rules[0].Resources},
		{name: "validating", resources: configurations.Validating[0].Webhooks[0].Rules[0].Resources},
	} {
		if len(rule.resources) != 1 || rule.resources[0] != lwsResource {
			t.Fatalf("%s webhook resources = %v, want [%s]", rule.name, rule.resources, lwsResource)
		}
	}
}
