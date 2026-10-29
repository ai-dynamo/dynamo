// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package storefactory

import (
	"context"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/dynamodb"
)

func init() {
	Register(config.StoreDynamoDB, openDynamoDB)
}

// openDynamoDB builds the DynamoDB store from the server configuration. The
// AWS client reads credentials and the region from the environment, with
// --region and --dynamodb-endpoint as overrides.
func openDynamoDB(ctx context.Context, cfg *config.Config) (store.Store, error) {
	client, err := dynamodb.NewClient(ctx, dynamodb.ClientOptions{Region: cfg.Region, Endpoint: cfg.DynamoDBEndpoint})
	if err != nil {
		return nil, err
	}
	return dynamodb.New(dynamodb.Options{Client: client, TableName: cfg.DynamoDBTable})
}
