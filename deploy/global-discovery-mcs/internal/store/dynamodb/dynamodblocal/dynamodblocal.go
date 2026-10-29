// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package dynamodblocal runs amazon/dynamodb-local for tests of the
// DynamoDB store and of mcs-admin. One container serves every test of a
// test binary; tests isolate themselves with their own table or
// clusterset names.
//
// A test binary calls Endpoint (or Client) from its tests and Terminate
// from TestMain after m.Run. Tests are skipped when no Docker daemon is
// reachable. Setting MCS_TEST_DYNAMODB_ENDPOINT to the URL of a running
// DynamoDB-compatible endpoint uses it instead of a container.
package dynamodblocal

import (
	"context"
	"fmt"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	ddb "github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
)

const (
	// Image is the dynamodb-local image the tests run.
	Image = "amazon/dynamodb-local:2.6.1"
	// EndpointEnv names the variable that selects an existing endpoint.
	EndpointEnv = "MCS_TEST_DYNAMODB_ENDPOINT"
	// Region is the region the test clients sign requests for.
	Region = "us-east-1"
)

var (
	mu        sync.Mutex
	endpoint  string
	container testcontainers.Container
)

// Endpoint returns the URL of the shared dynamodb-local, starting it on the
// first call. It skips t when Docker is unavailable and fails t when the
// container does not start.
func Endpoint(t *testing.T) string {
	t.Helper()
	if ep := os.Getenv(EndpointEnv); ep != "" {
		return ep
	}
	mu.Lock()
	defer mu.Unlock()
	if endpoint != "" {
		return endpoint
	}
	testcontainers.SkipIfProviderIsNotHealthy(t)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	c, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		Image:        Image,
		ExposedPorts: []string{"8000/tcp"},
		Cmd:          []string{"-jar", "DynamoDBLocal.jar", "-inMemory"},
		WaitingFor:   wait.ForListeningPort("8000/tcp").WithStartupTimeout(2 * time.Minute),
		Started:      true,
	})
	if err != nil {
		if c != nil {
			_ = testcontainers.TerminateContainer(c)
		}
		t.Fatalf("start %s: %v", Image, err)
	}
	host, err := c.Host(ctx)
	if err != nil {
		_ = testcontainers.TerminateContainer(c)
		t.Fatalf("container host: %v", err)
	}
	port, err := c.MappedPort(ctx, "8000/tcp")
	if err != nil {
		_ = testcontainers.TerminateContainer(c)
		t.Fatalf("container port: %v", err)
	}
	container = c
	endpoint = "http://" + net.JoinHostPort(host, port.Port())
	return endpoint
}

// Client returns a client for the shared dynamodb-local.
func Client(t *testing.T) *ddb.Client {
	t.Helper()
	return NewClient(Endpoint(t))
}

// NewClient returns a client for a DynamoDB-compatible endpoint with static
// credentials, which dynamodb-local accepts.
func NewClient(endpoint string) *ddb.Client {
	return ddb.New(ddb.Options{
		Region:       Region,
		BaseEndpoint: aws.String(endpoint),
		Credentials:  credentials.NewStaticCredentialsProvider("local", "local", ""),
	})
}

// Terminate stops the container, if one was started. Call it from TestMain
// after m.Run.
func Terminate() error {
	mu.Lock()
	defer mu.Unlock()
	if container == nil {
		return nil
	}
	err := testcontainers.TerminateContainer(container)
	container, endpoint = nil, ""
	if err != nil {
		return fmt.Errorf("terminate %s: %w", Image, err)
	}
	return nil
}
