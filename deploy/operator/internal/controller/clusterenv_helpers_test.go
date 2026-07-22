//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"fmt"
	"os/exec"
	"strings"

	semver "github.com/Masterminds/semver/v3"
)

func clusterTestOperatorVersion() (string, error) {
	output, err := exec.Command(
		"git", "describe", "--tags", "--match", "v[0-9]*", "--abbrev=12", "--long", "HEAD",
	).Output()
	if err != nil {
		return "", fmt.Errorf("describe current commit: %w", err)
	}
	version := strings.TrimPrefix(strings.TrimSpace(string(output)), "v")
	if _, err := semver.NewVersion(version); err != nil {
		return "", fmt.Errorf("git describe returned invalid semantic version %q: %w", version, err)
	}
	return version, nil
}
