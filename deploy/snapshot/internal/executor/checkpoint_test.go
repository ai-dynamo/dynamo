// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package executor

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func TestRequiredSourceMounts(t *testing.T) {
	candidates := []types.SourceMountManifest{
		{Path: "/model-cache", Volume: "model-cache", ProvidedBy: "PersistentVolumeClaim/model-cache"},
		{Path: "/scratch", Volume: "scratch", ProvidedBy: "EmptyDir"},
		{Path: "/not-externalized", Volume: "ignored", ProvidedBy: "ConfigMap/ignored"},
	}

	mounts, runtimeManaged := requiredSourceMounts(candidates, map[string]string{
		"/model-cache":         "/model-cache",
		"/scratch":             "/scratch",
		"/etc/hosts":           "/etc/hosts",
		"/var/run/secrets/k8s": "/var/run/secrets/k8s",
	})

	require.NotNil(t, runtimeManaged)
	assert.Equal(t, 2, *runtimeManaged)
	assert.Equal(t, candidates[:2], mounts)
}

func TestRequiredSourceMountsRecordsKnownZero(t *testing.T) {
	mounts, runtimeManaged := requiredSourceMounts(nil, nil)

	require.NotNil(t, runtimeManaged)
	assert.Zero(t, *runtimeManaged)
	assert.Empty(t, mounts)
}

func TestRequiredSourceMountsMatchesRunAliases(t *testing.T) {
	candidates := []types.SourceMountManifest{
		{Path: "/var/run/model-cache", Volume: "model-cache", ProvidedBy: "PersistentVolumeClaim/model-cache"},
		{Path: "/run/credentials", Volume: "credentials", ProvidedBy: "Secret/credentials"},
	}

	mounts, runtimeManaged := requiredSourceMounts(candidates, map[string]string{
		"/run/model-cache":     "/run/model-cache",
		"/var/run/credentials": "/var/run/credentials",
		"/etc/hosts":           "/etc/hosts",
	})

	require.NotNil(t, runtimeManaged)
	assert.Equal(t, 1, *runtimeManaged)
	assert.Equal(t, candidates, mounts)
}

func TestRequiredSourceMountsCountsAliasedExternalMountOnce(t *testing.T) {
	candidates := []types.SourceMountManifest{
		{Path: "/run/data", Volume: "first", ProvidedBy: "EmptyDir"},
		{Path: "/var/run/data", Volume: "second", ProvidedBy: "EmptyDir"},
	}

	mounts, runtimeManaged := requiredSourceMounts(candidates, map[string]string{"/run/data": "/run/data"})

	require.NotNil(t, runtimeManaged)
	assert.Zero(t, *runtimeManaged)
	assert.Equal(t, candidates, mounts)
}
