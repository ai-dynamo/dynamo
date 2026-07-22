/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtimeversion

import (
	"fmt"
	"regexp"
	"strings"

	semver "github.com/Masterminds/semver/v3"
)

var imageTagPattern = regexp.MustCompile(`^[vV]?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-((?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$`)

// Version identifies a runtime compatibility version by semver core.
type Version struct {
	Major uint64
	Minor uint64
	Patch uint64
}

func (v Version) String() string {
	return fmt.Sprintf("%d.%d.%d", v.Major, v.Minor, v.Patch)
}

// Parse returns the compatibility version represented by an explicit override.
func Parse(value string) (Version, error) {
	version, err := semver.StrictNewVersion(value)
	if err != nil {
		return Version{}, fmt.Errorf("must be a semantic version such as \"1.1.0\"")
	}
	return fromSemver(version), nil
}

// ParseImageVersion returns a normalized compatibility version from an image tag.
func ParseImageVersion(image string) (Version, error) {
	tag := imageTag(image)
	if tag == "" {
		return Version{}, fmt.Errorf("image %q does not contain a tag", image)
	}
	trimmed := strings.TrimSpace(tag)
	if !imageTagPattern.MatchString(trimmed) {
		return Version{}, fmt.Errorf("image tag %q must contain a semantic version such as \"1.1.0\"", tag)
	}
	version, err := semver.StrictNewVersion(strings.TrimPrefix(strings.TrimPrefix(trimmed, "v"), "V"))
	if err != nil {
		return Version{}, fmt.Errorf("image tag %q must contain a semantic version such as \"1.1.0\"", tag)
	}
	return fromSemver(version), nil
}

func fromSemver(version *semver.Version) Version {
	return Version{Major: version.Major(), Minor: version.Minor(), Patch: version.Patch()}
}

func imageTag(image string) string {
	ref := strings.TrimSpace(image)
	if digest := strings.Index(ref, "@"); digest >= 0 {
		ref = ref[:digest]
	}
	lastSlash := strings.LastIndex(ref, "/")
	lastColon := strings.LastIndex(ref, ":")
	if lastColon <= lastSlash {
		return ""
	}
	return ref[lastColon+1:]
}
