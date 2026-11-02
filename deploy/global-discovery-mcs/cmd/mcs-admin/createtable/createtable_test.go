// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package createtable

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	ddb "github.com/aws/aws-sdk-go-v2/service/dynamodb"
	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/dynamodb"
)

// fakeClient is a SchemaAPI whose table becomes ACTIVE at once.
type fakeClient struct {
	exists  bool
	created *ddb.CreateTableInput
	ttl     bool
	pitr    bool
}

func (f *fakeClient) CreateTable(
	_ context.Context, in *ddb.CreateTableInput, _ ...func(*ddb.Options),
) (*ddb.CreateTableOutput, error) {
	if f.exists {
		return nil, &ddbtypes.ResourceInUseException{Message: aws.String("exists")}
	}
	f.created = in
	return &ddb.CreateTableOutput{}, nil
}

func (f *fakeClient) DescribeTable(
	_ context.Context, in *ddb.DescribeTableInput, _ ...func(*ddb.Options),
) (*ddb.DescribeTableOutput, error) {
	def := dynamodb.CreateTableInput(dynamodb.TableOptions{TableName: aws.ToString(in.TableName)})
	return &ddb.DescribeTableOutput{Table: &ddbtypes.TableDescription{
		TableName:            in.TableName,
		TableStatus:          ddbtypes.TableStatusActive,
		KeySchema:            def.KeySchema,
		AttributeDefinitions: def.AttributeDefinitions,
	}}, nil
}

func (f *fakeClient) DescribeTimeToLive(
	context.Context, *ddb.DescribeTimeToLiveInput, ...func(*ddb.Options),
) (*ddb.DescribeTimeToLiveOutput, error) {
	return &ddb.DescribeTimeToLiveOutput{}, nil
}

func (f *fakeClient) UpdateTimeToLive(
	context.Context, *ddb.UpdateTimeToLiveInput, ...func(*ddb.Options),
) (*ddb.UpdateTimeToLiveOutput, error) {
	f.ttl = true
	return &ddb.UpdateTimeToLiveOutput{}, nil
}

func (f *fakeClient) DescribeContinuousBackups(
	context.Context, *ddb.DescribeContinuousBackupsInput, ...func(*ddb.Options),
) (*ddb.DescribeContinuousBackupsOutput, error) {
	return &ddb.DescribeContinuousBackupsOutput{}, nil
}

func (f *fakeClient) UpdateContinuousBackups(
	context.Context, *ddb.UpdateContinuousBackupsInput, ...func(*ddb.Options),
) (*ddb.UpdateContinuousBackupsOutput, error) {
	f.pitr = true
	return &ddb.UpdateContinuousBackupsOutput{}, nil
}

func runFake(t *testing.T, f *fakeClient, args ...string) (dynamodb.ClientOptions, string, string, error) {
	t.Helper()
	return runFakeWith(t, Defaults{}, f, args...)
}

func runFakeWith(
	t *testing.T, defaults Defaults, f *fakeClient, args ...string,
) (dynamodb.ClientOptions, string, string, error) {
	t.Helper()
	var got dynamodb.ClientOptions
	var stdout, stderr bytes.Buffer
	open := func(_ context.Context, opts dynamodb.ClientOptions) (dynamodb.SchemaAPI, error) {
		got = opts
		return f, nil
	}
	err := RunWith(t.Context(), defaults, args, &stdout, &stderr, open)
	return got, stdout.String(), stderr.String(), err
}

func TestCreate(t *testing.T) {
	f := &fakeClient{}
	opts, stdout, _, err := runFake(t, f,
		"--table", "mcs-test", "--region", "us-west-2", "--endpoint", "http://localhost:8000")
	require.NoError(t, err)
	assert.Equal(t, dynamodb.ClientOptions{Region: "us-west-2", Endpoint: "http://localhost:8000"}, opts)
	assert.Equal(t, "mcs-test", aws.ToString(f.created.TableName))
	assert.True(t, aws.ToBool(f.created.DeletionProtectionEnabled))
	assert.True(t, f.ttl)
	assert.True(t, f.pitr)
	assert.Equal(t, "created table mcs-test\nTTL on expires_at is enabled\npoint-in-time recovery is enabled\n", stdout)
}

func TestDefaultsFromStoreFlags(t *testing.T) {
	defaults := Defaults{Table: "from-store", Region: "eu-west-1", Endpoint: "http://store:8000"}

	f := &fakeClient{}
	opts, stdout, _, err := runFakeWith(t, defaults, f)
	require.NoError(t, err)
	assert.Equal(t, dynamodb.ClientOptions{Region: "eu-west-1", Endpoint: "http://store:8000"}, opts)
	assert.Equal(t, "from-store", aws.ToString(f.created.TableName))
	assert.Contains(t, stdout, "created table from-store\n")

	// A subcommand flag wins over the default.
	f = &fakeClient{}
	opts, _, _, err = runFakeWith(t, defaults, f,
		"--table", "own", "--region", "us-west-2", "--endpoint", "http://own:8000")
	require.NoError(t, err)
	assert.Equal(t, dynamodb.ClientOptions{Region: "us-west-2", Endpoint: "http://own:8000"}, opts)
	assert.Equal(t, "own", aws.ToString(f.created.TableName))

	// Empty defaults leave the built-in table name.
	f = &fakeClient{}
	_, _, _, err = runFakeWith(t, Defaults{}, f)
	require.NoError(t, err)
	assert.Equal(t, dynamodb.DefaultTableName, aws.ToString(f.created.TableName))
}

func TestExistingTableWithoutExtras(t *testing.T) {
	f := &fakeClient{exists: true}
	_, stdout, _, err := runFake(t, f,
		"--pitr=false", "--ttl=false", "--deletion-protection=false", "--wait-timeout", time.Second.String())
	require.NoError(t, err)
	assert.False(t, f.ttl)
	assert.False(t, f.pitr)
	assert.Equal(t, "table global-discovery-mcs exists; key schema checked\n", stdout)
}

func TestUsageErrors(t *testing.T) {
	for _, args := range [][]string{{"--nope"}, {"extra"}, {"--table", ""}} {
		_, _, _, err := runFake(t, &fakeClient{}, args...)
		assert.ErrorAs(t, err, new(UsageError), "%v", args)
	}
	_, _, stderr, err := runFake(t, &fakeClient{}, "-h")
	assert.ErrorIs(t, err, flag.ErrHelp)
	assert.Contains(t, stderr, "-endpoint string")
}

func TestClientError(t *testing.T) {
	boom := errors.New("no credentials")
	failing := func(context.Context, dynamodb.ClientOptions) (dynamodb.SchemaAPI, error) {
		return nil, boom
	}
	err := RunWith(t.Context(), Defaults{}, nil, &bytes.Buffer{}, &bytes.Buffer{}, failing)
	assert.ErrorIs(t, err, boom)
}
