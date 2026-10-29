// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package version holds build metadata. The Makefile sets the variables
// with -ldflags -X; other builds keep the defaults.
package version

var (
	// Service is the service name reported in logs and GetServerInfo.
	Service = "global-discovery-mcs"
	// Version is the release version.
	Version = "dev"
	// GitHash is the commit the binary was built from.
	GitHash = "unknown"
	// ReleaseTag is the image tag of a release build.
	ReleaseTag = ""
)

// KeysAndValues returns the build metadata as alternating key-value pairs for
// structured loggers.
func KeysAndValues() []any {
	return []any{"service", Service, "version", Version, "gitHash", GitHash, "releaseTag", ReleaseTag}
}

// String returns the version and commit in one line, for example
// "dev (unknown)".
func String() string {
	return Version + " (" + GitHash + ")"
}
