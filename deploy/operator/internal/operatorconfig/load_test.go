/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package operatorconfig

import (
	"os"
	"path/filepath"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/stretchr/testify/require"
)

const validConfig = `apiVersion: operator.config.dynamo.nvidia.com/v1alpha1
kind: OperatorConfiguration
mpi:
  sshSecretName: mpi-ssh
  sshSecretNamespace: dynamo-system
rbac:
  plannerClusterRoleName: planner
  dgdrProfilingClusterRoleName: dgdr-profiling
  eppClusterRoleName: epp
lpx:
  enabled: true
`

func TestLoad(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte(validConfig), 0o600))

	config, err := Load(configPath)
	require.NoError(t, err)
	require.True(t, config.LPX.Enabled)
	require.Equal(t, 8080, config.Server.Metrics.Port)
	require.Equal(t, configv1alpha1.DiscoveryBackendKubernetes, config.Discovery.Backend)
}

func TestLoadRejectsInvalidConfig(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte(validConfig+`leaderElection:
  enabled: true
`), 0o600))

	_, err := Load(configPath)
	require.ErrorContains(t, err, "leaderElection.id")
}
