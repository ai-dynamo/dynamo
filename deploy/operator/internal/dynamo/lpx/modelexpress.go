/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/tls"
	"fmt"
	"net/url"

	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

const modelExpressMaxMessageSize = 100 * 1024 * 1024

// NewModelExpressClient validates modelExpressURL and constructs a gRPC-backed client.
func NewModelExpressClient(modelExpressURL string) (modelpb.ModelServiceClient, error) {
	parsedModelExpressURL, err := url.Parse(modelExpressURL)
	if err != nil {
		return nil, fmt.Errorf("parse ModelExpress URL %q: %w", modelExpressURL, err)
	}

	target := modelExpressURL
	transportCredentials := insecure.NewCredentials()
	switch parsedModelExpressURL.Scheme {
	case "http", "https":
		if parsedModelExpressURL.Host == "" {
			return nil, fmt.Errorf("ModelExpress URL %q must include a host", modelExpressURL)
		}
		if parsedModelExpressURL.Path != "" && parsedModelExpressURL.Path != "/" {
			return nil, fmt.Errorf("ModelExpress URL %q must not include a path", modelExpressURL)
		}
		if parsedModelExpressURL.ForceQuery || parsedModelExpressURL.RawQuery != "" || parsedModelExpressURL.Fragment != "" {
			return nil, fmt.Errorf("ModelExpress URL %q must not include query or fragment", modelExpressURL)
		}

		target = parsedModelExpressURL.Host
		if parsedModelExpressURL.Scheme == "https" {
			transportCredentials = credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})
		}
	}

	conn, err := grpc.NewClient(
		target,
		grpc.WithTransportCredentials(transportCredentials),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(modelExpressMaxMessageSize),
			grpc.MaxCallSendMsgSize(modelExpressMaxMessageSize),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("create ModelExpress gRPC client for %q: %w", modelExpressURL, err)
	}

	return modelpb.NewModelServiceClient(conn), nil
}
