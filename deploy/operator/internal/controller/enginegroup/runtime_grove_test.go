//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"net/http"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGroveRuntimeProviderRejectsUnknownBackend(t *testing.T) {
	t.Log("resolve an unrecognized runtime without contacting an engine")
	provider := &groveRuntimeProvider{Client: fake.NewClientBuilder().Build(), HTTP: &http.Client{}}
	group := &api.DynamoGraphDeploymentEngineGroup{
		Labels: map[string]string{consts.KubeLabelDynamoEngineGroupRuntime: "unknown"}}
	_, err := provider.Resolve(t.Context(), group)
	require.ErrorIs(t, err, ErrRuntimeUnavailable)
}
