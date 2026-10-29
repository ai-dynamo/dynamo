// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamodb

import (
	"context"
	"errors"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	ddb "github.com/aws/aws-sdk-go-v2/service/dynamodb"
)

// ClientOptions configures NewClient.
type ClientOptions struct {
	// Region overrides the region of the shared AWS configuration
	// (AWS_REGION, AWS_DEFAULT_REGION or the profile).
	Region string
	// Endpoint overrides the DynamoDB endpoint, for dynamodb-local.
	Endpoint string
}

// NewClient returns a DynamoDB client configured from the default AWS
// credential chain: environment, shared files, IRSA web identity, then the
// instance or task role. It fails when no region is configured.
func NewClient(ctx context.Context, opts ClientOptions) (*ddb.Client, error) {
	var load []func(*config.LoadOptions) error
	if opts.Region != "" {
		load = append(load, config.WithRegion(opts.Region))
	}
	cfg, err := config.LoadDefaultConfig(ctx, load...)
	if err != nil {
		return nil, fmt.Errorf("dynamodb: load AWS configuration: %w", err)
	}
	if cfg.Region == "" {
		return nil, errors.New("dynamodb: no AWS region; set one explicitly or through AWS_REGION")
	}
	return ddb.NewFromConfig(cfg, func(o *ddb.Options) {
		if opts.Endpoint != "" {
			o.BaseEndpoint = aws.String(opts.Endpoint)
		}
	}), nil
}
