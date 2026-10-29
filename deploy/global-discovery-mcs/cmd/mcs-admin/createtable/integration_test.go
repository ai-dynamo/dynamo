//go:build integration

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package createtable

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	ddb "github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/dynamodb"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/dynamodb/dynamodblocal"
)

func TestMain(m *testing.M) {
	code := m.Run()
	if err := dynamodblocal.Terminate(); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
	os.Exit(code)
}

// TestRunAgainstDynamoDBLocal runs the command as an operator would, with
// credentials and region from the environment.
func TestRunAgainstDynamoDBLocal(t *testing.T) {
	endpoint := dynamodblocal.Endpoint(t)
	dir := t.TempDir()
	t.Setenv("AWS_CONFIG_FILE", filepath.Join(dir, "config"))
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(dir, "credentials"))
	t.Setenv("AWS_PROFILE", "")
	t.Setenv("AWS_REGION", dynamodblocal.Region)
	t.Setenv("AWS_ACCESS_KEY_ID", "local")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "local")

	args := []string{"--endpoint", endpoint, "--table", "mcs-admin-test", "--pitr=false", "--deletion-protection=false"}
	var stdout bytes.Buffer
	require.NoError(t, Run(t.Context(), Defaults{}, args, &stdout, os.Stderr))
	assert.Contains(t, stdout.String(), "created table mcs-admin-test")

	stdout.Reset()
	require.NoError(t, Run(t.Context(), Defaults{}, args, &stdout, os.Stderr), "running it again is safe")
	assert.Contains(t, stdout.String(), "table mcs-admin-test exists")

	client := dynamodblocal.NewClient(endpoint)
	out, err := client.DescribeTimeToLive(t.Context(),
		&ddb.DescribeTimeToLiveInput{TableName: aws.String("mcs-admin-test")})
	require.NoError(t, err)
	assert.Equal(t, dynamodb.AttributeExpiresAt, aws.ToString(out.TimeToLiveDescription.AttributeName))
	_, err = client.DeleteTable(t.Context(), &ddb.DeleteTableInput{TableName: aws.String("mcs-admin-test")})
	require.NoError(t, err)
}
