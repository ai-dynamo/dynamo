/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"os"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"sigs.k8s.io/yaml"
)

func TestParseFeatureGates(t *testing.T) {
	tests := []struct {
		name    string
		value   string
		enabled bool
		wantErr bool
	}{
		{name: "default disabled"},
		{name: "explicitly enabled", value: "DGDRV1Beta2=true", enabled: true},
		{name: "explicitly disabled", value: "DGDRV1Beta2=false"},
		{name: "unknown gate", value: "Unknown=true", wantErr: true},
		{name: "invalid value", value: "DGDRV1Beta2=yes", wantErr: true},
		{name: "missing value", value: "DGDRV1Beta2", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gates, err := parseFeatureGates(tt.value)
			if tt.wantErr {
				if err == nil {
					t.Fatal("parseFeatureGates() succeeded, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("parseFeatureGates() error = %v", err)
			}
			if got := gates[dgdrV1Beta2Gate]; got != tt.enabled {
				t.Fatalf("%s = %v, want %v", dgdrV1Beta2Gate, got, tt.enabled)
			}
		})
	}
}

func TestConfigureGeneratedDGDRV1Beta2CRDs(t *testing.T) {
	tests := []struct {
		file      string
		wantApply bool
	}{
		{file: "../../config/crd/bases/nvidia.com_dynamographdeploymentrequests.yaml", wantApply: true},
		{file: "../../config/crd/bases/nvidia.com_dynamographdeploymentruns.yaml"},
		{file: "../../config/crd/bases/nvidia.com_dynamographdeploymentcandidates.yaml"},
	}

	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			data, err := os.ReadFile(tt.file)
			if err != nil {
				t.Fatalf("read generated CRD: %v", err)
			}
			crd := &apiextensionsv1.CustomResourceDefinition{}
			if err := yaml.Unmarshal(data, crd); err != nil {
				t.Fatalf("decode generated CRD: %v", err)
			}

			if got := configureDGDRV1Beta2CRD(crd); got != tt.wantApply {
				t.Fatalf("configureDGDRV1Beta2CRD() = %v, want %v", got, tt.wantApply)
			}
			if crd.Name == dgdrCRDName {
				for _, version := range crd.Spec.Versions {
					if version.Name == "v1beta2" {
						t.Fatal("generated request CRD still contains v1beta2")
					}
				}
			}
		})
	}
}

func TestConfigureDGDRV1Beta2CRD(t *testing.T) {
	tests := []struct {
		name         string
		crdName      string
		wantApply    bool
		wantVersions []string
	}{
		{
			name:         "request retains stable versions",
			crdName:      dgdrCRDName,
			wantApply:    true,
			wantVersions: []string{"v1alpha1", "v1beta1"},
		},
		{
			name:         "run is skipped",
			crdName:      dgdrRunCRDName,
			wantVersions: []string{"v1alpha1", "v1beta1", "v1beta2"},
		},
		{
			name:         "candidate is skipped",
			crdName:      dgdrCandidateCRD,
			wantVersions: []string{"v1alpha1", "v1beta1", "v1beta2"},
		},
		{
			name:         "unrelated CRD is unchanged",
			crdName:      "dynamographdeployments.nvidia.com",
			wantApply:    true,
			wantVersions: []string{"v1alpha1", "v1beta1", "v1beta2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			crd := &apiextensionsv1.CustomResourceDefinition{}
			crd.Name = tt.crdName
			for _, version := range []string{"v1alpha1", "v1beta1", "v1beta2"} {
				crd.Spec.Versions = append(crd.Spec.Versions, apiextensionsv1.CustomResourceDefinitionVersion{Name: version})
			}

			if got := configureDGDRV1Beta2CRD(crd); got != tt.wantApply {
				t.Fatalf("configureDGDRV1Beta2CRD() = %v, want %v", got, tt.wantApply)
			}
			var versions []string
			for _, version := range crd.Spec.Versions {
				versions = append(versions, version.Name)
			}
			if len(versions) != len(tt.wantVersions) {
				t.Fatalf("versions = %v, want %v", versions, tt.wantVersions)
			}
			for i := range versions {
				if versions[i] != tt.wantVersions[i] {
					t.Fatalf("versions = %v, want %v", versions, tt.wantVersions)
				}
			}
		})
	}
}
