// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nsmount

import (
	"fmt"
	"path/filepath"
	"strings"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

// ResolveArtifactPath returns the existing checkpoint artifact layout rooted
// at the agent-owned base path. All variable components must be single clean
// path elements.
func ResolveArtifactPath(basePath, artifactID, version string) (string, error) {
	basePath = strings.TrimSpace(basePath)
	if !filepath.IsAbs(basePath) || filepath.Clean(basePath) != basePath {
		return "", fmt.Errorf("base path must be an absolute, clean path: %q", basePath)
	}
	artifactID = strings.TrimSpace(artifactID)
	if err := validatePathElement("artifact ID", artifactID); err != nil {
		return "", err
	}
	version = snapshotprotocol.ArtifactVersion(version)
	if err := validatePathElement("artifact version", version); err != nil {
		return "", err
	}
	return filepath.Join(basePath, artifactID, "versions", version), nil
}

func validatePathElement(label, value string) error {
	if value == "" || value == "." || value == ".." || filepath.Base(value) != value || strings.ContainsAny(value, `/\`) {
		return fmt.Errorf("%s must be a single clean path element: %q", label, value)
	}
	return nil
}
