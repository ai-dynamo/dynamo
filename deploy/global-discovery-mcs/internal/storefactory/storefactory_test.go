// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package storefactory

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/memory"
)

func TestOpen(t *testing.T) {
	cfg := config.Default()
	st, err := Open(context.Background(), cfg)
	require.NoError(t, err)
	assert.IsType(t, &memory.Store{}, st)

	// The DynamoDB opener builds a client without touching the network, but
	// it needs a region.
	cfg.Store = config.StoreDynamoDB
	cfg.Region = ""
	t.Setenv("AWS_REGION", "")
	_, err = Open(context.Background(), cfg)
	assert.ErrorContains(t, err, "no AWS region")
	cfg.Region = "us-east-1"
	cfg.DynamoDBEndpoint = "http://127.0.0.1:1"
	st, err = Open(context.Background(), cfg)
	require.NoError(t, err)
	require.NoError(t, st.Close())

	cfg.Store = "etcd"
	_, err = Open(context.Background(), cfg)
	assert.ErrorContains(t, err, "unknown store")

	Register("test", func(context.Context, *config.Config) (store.Store, error) { return memory.New(), nil })
	t.Cleanup(func() {
		mu.Lock()
		delete(openers, "test")
		mu.Unlock()
	})
	assert.Equal(t, []string{"dynamodb", "memory", "test"}, Kinds())
	cfg.Store = "test"
	_, err = Open(context.Background(), cfg)
	require.NoError(t, err)
}
