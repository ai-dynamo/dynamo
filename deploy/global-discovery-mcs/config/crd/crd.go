// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package crd embeds the CustomResourceDefinitions mcs-controller needs, as
// kept in bases/: ServiceExport and ServiceImport, copied from
// sigs.k8s.io/mcs-api config/crd by make sync-crds, and the KEP-2149
// ClusterProperty, kept by hand. Tests install them into envtest; the image
// ships them in /opt/global-discovery-mcs/crds and charts copy the YAML files.
package crd

import (
	"embed"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

//go:embed bases/*.yaml
var bases embed.FS

// Files returns every CRD by file name.
func Files() map[string][]byte {
	out := map[string][]byte{}
	entries, err := fs.ReadDir(bases, "bases")
	if err != nil {
		panic(fmt.Sprintf("read embedded CRDs: %v", err))
	}
	for _, e := range entries {
		data, err := fs.ReadFile(bases, "bases/"+e.Name())
		if err != nil {
			panic(fmt.Sprintf("read embedded CRD %s: %v", e.Name(), err))
		}
		out[e.Name()] = data
	}
	return out
}

// WriteDir writes every CRD into dir, for envtest's CRDDirectoryPaths.
func WriteDir(dir string) error {
	for name, data := range Files() {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			return fmt.Errorf("write CRD %s: %w", name, err)
		}
	}
	return nil
}
