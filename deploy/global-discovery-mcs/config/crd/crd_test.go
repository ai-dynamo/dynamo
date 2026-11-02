// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package crd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	mcscrd "sigs.k8s.io/mcs-api/config/crd"
	"sigs.k8s.io/yaml"
)

func TestFiles(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, WriteDir(dir))
	want := map[string]string{
		"multicluster.x-k8s.io_serviceexports.yaml": "serviceexports.multicluster.x-k8s.io",
		"multicluster.x-k8s.io_serviceimports.yaml": "serviceimports.multicluster.x-k8s.io",
		"about.k8s.io_clusterproperties.yaml":       "clusterproperties.about.k8s.io",
	}
	for file, name := range want {
		data, err := os.ReadFile(filepath.Join(dir, file))
		require.NoError(t, err)
		var crd struct {
			Kind     string `json:"kind"`
			Metadata struct {
				Name        string            `json:"name"`
				Annotations map[string]string `json:"annotations"`
			} `json:"metadata"`
			Spec struct {
				Scope string `json:"scope"`
			} `json:"spec"`
		}
		require.NoError(t, yaml.Unmarshal(data, &crd), file)
		assert.Equal(t, "CustomResourceDefinition", crd.Kind)
		assert.Equal(t, name, crd.Metadata.Name)
		if file == "about.k8s.io_clusterproperties.yaml" {
			assert.Equal(t, "Cluster", crd.Spec.Scope)
			assert.NotEmpty(t, crd.Metadata.Annotations["api-approved.kubernetes.io"], "about.k8s.io is a protected group")
		} else {
			assert.Equal(t, "Namespaced", crd.Spec.Scope)
		}
	}
	require.Error(t, WriteDir(filepath.Join(dir, "missing")))
	assert.Len(t, Files(), len(want))
}

// The copies in bases/ must be the CRDs of the sigs.k8s.io/mcs-api version
// in go.mod below the header make sync-crds adds; run make sync-crds after
// changing that version.
func TestMCSAPICopiesMatchTheModule(t *testing.T) {
	files := Files()
	for name, upstream := range map[string][]byte{
		"multicluster.x-k8s.io_serviceexports.yaml": mcscrd.ServiceExportCRD,
		"multicluster.x-k8s.io_serviceimports.yaml": mcscrd.ServiceImportCRD,
	} {
		copied := string(files[name])
		require.Contains(t, copied, "by make sync-crds; the upstream header follows.\n#\n", name)
		body := copied[strings.Index(copied, "the upstream header follows.\n#\n")+len("the upstream header follows.\n#\n"):]
		assert.Equal(t, string(upstream), body, "%s differs from sigs.k8s.io/mcs-api; run make sync-crds", name)
	}
}
