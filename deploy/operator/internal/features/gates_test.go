/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package features

import (
	"context"
	"encoding/json"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDefaults(t *testing.T) {
	gates := Defaults()
	for _, name := range allNames {
		if got, want := gates.Enabled(name), name == GPUDiscovery; got != want {
			t.Errorf("Defaults().Enabled(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestGatesEnabled(t *testing.T) {
	gates := allEnabledGates()
	for _, name := range allNames {
		if !gates.Enabled(name) {
			t.Errorf("Gates.Enabled(%q) = false, want true", name)
		}
	}
}

func TestGateRegistryIsComplete(t *testing.T) {
	encoded, err := json.Marshal(allEnabledGates())
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	values := make(map[string]bool)
	if err := json.Unmarshal(encoded, &values); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if got, want := len(values), len(allNames); got != want {
		t.Fatalf("encoded gate count = %d, want %d", got, want)
	}
	for _, name := range allNames {
		if enabled, exists := values[string(name)]; !exists || !enabled {
			t.Errorf("encoded gate %q = %v, exists = %v; want true", name, enabled, exists)
		}
	}
}

func allEnabledGates() Gates {
	return Gates{
		Checkpoint:       true,
		Grove:            true,
		LWS:              true,
		KaiScheduler:     true,
		VolcanoScheduler: true,
		DRA:              true,
		Istio:            true,
		GPUDiscovery:     true,
	}
}

func TestAPIGroupServesVersion(t *testing.T) {
	apiGroups := &metav1.APIGroupList{
		Groups: []metav1.APIGroup{
			{
				Name: "resource.k8s.io",
				Versions: []metav1.GroupVersionForDiscovery{
					{GroupVersion: "resource.k8s.io/v1beta1", Version: "v1beta1"},
					{GroupVersion: "resource.k8s.io/v1beta2", Version: "v1beta2"},
				},
			},
			{
				Name:     "apps",
				Versions: []metav1.GroupVersionForDiscovery{{GroupVersion: "apps/v1", Version: "v1"}},
			},
		},
	}

	tests := []struct {
		name      string
		groupName string
		version   string
		want      bool
	}{
		{name: "group exists when version omitted", groupName: "resource.k8s.io", want: true},
		{name: "served beta version exists", groupName: "resource.k8s.io", version: "v1beta2", want: true},
		{name: "unserved v1 version is unavailable", groupName: "resource.k8s.io", version: "v1", want: false},
		{name: "different group with v1 exists", groupName: "apps", version: "v1", want: true},
		{name: "missing group is unavailable", groupName: "missing.example.com", version: "v1", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := apiGroupServesVersion(apiGroups, tt.groupName, tt.version); got != tt.want {
				t.Fatalf("apiGroupServesVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestResolveCheckpoint(t *testing.T) {
	tests := []struct {
		name              string
		enabled           bool
		snapshotAvailable bool
		want              bool
		wantErr           bool
	}{
		{name: "disabled does not require snapshot API", enabled: false, snapshotAvailable: false, want: false},
		{name: "enabled with snapshot API is enabled", enabled: true, snapshotAvailable: true, want: true},
		{name: "enabled without snapshot API fails", enabled: true, snapshotAvailable: false, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resolveCheckpoint(tt.enabled, tt.snapshotAvailable)
			if (err != nil) != tt.wantErr {
				t.Fatalf("resolveCheckpoint() error = %v, wantErr %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("resolveCheckpoint() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGateContext(t *testing.T) {
	want := Gates{Grove: true}
	got, ok := GateFrom(WithGate(context.Background(), want))
	if !ok {
		t.Fatal("GateFrom() did not find gate")
	}
	if !got.Enabled(Grove) {
		t.Error("GateFrom().Enabled(Grove) = false, want true")
	}
}

func TestMustGateFromPanicsWithoutGate(t *testing.T) {
	if _, ok := GateFrom(context.Background()); ok {
		t.Fatal("GateFrom() found unexpected gate")
	}
	defer func() {
		if recover() == nil {
			t.Fatal("MustGateFrom() did not panic")
		}
	}()
	MustGateFrom(context.Background())
}
