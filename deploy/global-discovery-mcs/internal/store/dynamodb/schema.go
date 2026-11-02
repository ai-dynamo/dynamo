// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamodb

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	ddb "github.com/aws/aws-sdk-go-v2/service/dynamodb"
	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
)

// The table layout. docs/storage.md describes it, and
// testdata/create-table.json is the CreateTable request of
// DefaultTableOptions(DefaultTableName) in the JSON form that
// `aws dynamodb create-table --cli-input-json` accepts.
const (
	// DefaultTableName is the table name mcs-registry and mcs-admin default
	// to.
	DefaultTableName = "global-discovery-mcs"
	// AttributePK and AttributeSK are the partition and sort keys.
	AttributePK = attrPK
	AttributeSK = attrSK
	// AttributeUpdatedAt is the sort key of IndexByUpdatedAt, in Unix
	// nanoseconds.
	AttributeUpdatedAt = attrUpdatedAt
	// AttributeExpiresAt is the TTL attribute, in Unix seconds.
	AttributeExpiresAt = attrExpiresAt
	// IndexByUpdatedAt is a KEYS_ONLY local secondary index on
	// (pk, updated_at). v1 does not query it; it exists because a local
	// secondary index can only be created with the table, and it lets a
	// later incremental refresh read only the keys changed since a time.
	IndexByUpdatedAt = "by_updated_at"

	defaultWaitTimeout  = 5 * time.Minute
	defaultPollInterval = 2 * time.Second
)

// TableOptions configures CreateTable.
type TableOptions struct {
	// TableName is the table to create. Required.
	TableName string
	// DeletionProtection creates the table with deletion protection.
	DeletionProtection bool
	// PointInTimeRecovery enables continuous backups.
	PointInTimeRecovery bool
	// TimeToLive enables TTL on AttributeExpiresAt.
	TimeToLive bool
	// Tags are applied to the new table.
	Tags map[string]string
	// WaitTimeout bounds the wait for the table to become ACTIVE. Zero
	// means 5 minutes.
	WaitTimeout time.Duration
	// PollInterval is the DescribeTable interval while waiting. Zero means
	// 2 seconds.
	PollInterval time.Duration
}

// DefaultTableOptions returns the production settings: deletion
// protection, point-in-time recovery and TTL.
func DefaultTableOptions(tableName string) TableOptions {
	return TableOptions{
		TableName:           tableName,
		DeletionProtection:  true,
		PointInTimeRecovery: true,
		TimeToLive:          true,
	}
}

// CreateTableInput returns the CreateTable request of the table: string
// keys pk (HASH) and sk (RANGE), on-demand capacity, and the KEYS_ONLY
// local secondary index by_updated_at on (pk, updated_at). TTL and
// point-in-time recovery are separate calls; see CreateTable.
func CreateTableInput(opts TableOptions) *ddb.CreateTableInput {
	in := &ddb.CreateTableInput{
		TableName: aws.String(opts.TableName),
		AttributeDefinitions: []ddbtypes.AttributeDefinition{
			{AttributeName: aws.String(AttributePK), AttributeType: ddbtypes.ScalarAttributeTypeS},
			{AttributeName: aws.String(AttributeSK), AttributeType: ddbtypes.ScalarAttributeTypeS},
			{AttributeName: aws.String(AttributeUpdatedAt), AttributeType: ddbtypes.ScalarAttributeTypeN},
		},
		KeySchema: []ddbtypes.KeySchemaElement{
			{AttributeName: aws.String(AttributePK), KeyType: ddbtypes.KeyTypeHash},
			{AttributeName: aws.String(AttributeSK), KeyType: ddbtypes.KeyTypeRange},
		},
		LocalSecondaryIndexes: []ddbtypes.LocalSecondaryIndex{{
			IndexName: aws.String(IndexByUpdatedAt),
			KeySchema: []ddbtypes.KeySchemaElement{
				{AttributeName: aws.String(AttributePK), KeyType: ddbtypes.KeyTypeHash},
				{AttributeName: aws.String(AttributeUpdatedAt), KeyType: ddbtypes.KeyTypeRange},
			},
			Projection: &ddbtypes.Projection{ProjectionType: ddbtypes.ProjectionTypeKeysOnly},
		}},
		BillingMode:               ddbtypes.BillingModePayPerRequest,
		DeletionProtectionEnabled: aws.Bool(opts.DeletionProtection),
	}
	for _, k := range slices.Sorted(maps.Keys(opts.Tags)) {
		in.Tags = append(in.Tags, ddbtypes.Tag{Key: aws.String(k), Value: aws.String(opts.Tags[k])})
	}
	return in
}

// SchemaAPI is the subset of the DynamoDB client CreateTable uses.
// *dynamodb.Client implements it.
type SchemaAPI interface {
	CreateTable(ctx context.Context, in *ddb.CreateTableInput, optFns ...func(*ddb.Options)) (*ddb.CreateTableOutput, error)
	DescribeTable(ctx context.Context, in *ddb.DescribeTableInput, optFns ...func(*ddb.Options)) (*ddb.DescribeTableOutput, error)
	DescribeTimeToLive(ctx context.Context, in *ddb.DescribeTimeToLiveInput, optFns ...func(*ddb.Options)) (*ddb.DescribeTimeToLiveOutput, error)
	UpdateTimeToLive(ctx context.Context, in *ddb.UpdateTimeToLiveInput, optFns ...func(*ddb.Options)) (*ddb.UpdateTimeToLiveOutput, error)
	DescribeContinuousBackups(ctx context.Context, in *ddb.DescribeContinuousBackupsInput, optFns ...func(*ddb.Options)) (*ddb.DescribeContinuousBackupsOutput, error)
	UpdateContinuousBackups(ctx context.Context, in *ddb.UpdateContinuousBackupsInput, optFns ...func(*ddb.Options)) (*ddb.UpdateContinuousBackupsOutput, error)
}

var _ SchemaAPI = (*ddb.Client)(nil)

// CreateTable creates the table unless it exists, waits until it is
// ACTIVE, then enables TTL and point-in-time recovery as opts asks. It is
// idempotent: for an existing table it checks the key schema and only
// enables what is missing. It reports whether it created the table.
//
// It creates a single-region table. A Global Table is formed by adding
// replicas afterwards; see docs/storage.md.
func CreateTable(ctx context.Context, client SchemaAPI, opts TableOptions) (bool, error) {
	if opts.TableName == "" {
		return false, errors.New("dynamodb: TableOptions.TableName is required")
	}
	created := true
	if _, err := client.CreateTable(ctx, CreateTableInput(opts)); err != nil {
		if _, ok := errors.AsType[*ddbtypes.ResourceInUseException](err); !ok {
			return false, fmt.Errorf("dynamodb: create table %s: %w", opts.TableName, err)
		}
		created = false
	}
	desc, err := waitActive(ctx, client, opts)
	if err != nil {
		return created, err
	}
	if err := checkKeySchema(desc); err != nil {
		return created, err
	}
	if opts.TimeToLive {
		if err := EnsureTTL(ctx, client, opts.TableName); err != nil {
			return created, err
		}
	}
	if opts.PointInTimeRecovery {
		if err := EnsurePointInTimeRecovery(ctx, client, opts.TableName); err != nil {
			return created, err
		}
	}
	return created, nil
}

// waitActive polls DescribeTable until the table is ACTIVE.
func waitActive(ctx context.Context, client SchemaAPI, opts TableOptions) (*ddbtypes.TableDescription, error) {
	ctx, cancel := context.WithTimeout(ctx, cmp.Or(opts.WaitTimeout, defaultWaitTimeout))
	defer cancel()
	poll := cmp.Or(opts.PollInterval, defaultPollInterval)
	for {
		out, err := client.DescribeTable(ctx, &ddb.DescribeTableInput{TableName: aws.String(opts.TableName)})
		if err != nil {
			return nil, fmt.Errorf("dynamodb: describe table %s: %w", opts.TableName, err)
		}
		if out.Table != nil && out.Table.TableStatus == ddbtypes.TableStatusActive {
			return out.Table, nil
		}
		t := time.NewTimer(poll)
		select {
		case <-ctx.Done():
			t.Stop()
			return nil, fmt.Errorf("dynamodb: table %s is not ACTIVE: %w", opts.TableName, ctx.Err())
		case <-t.C:
		}
	}
}

// checkKeySchema fails unless the table has the string keys pk (HASH) and
// sk (RANGE).
func checkKeySchema(desc *ddbtypes.TableDescription) error {
	types := map[string]ddbtypes.ScalarAttributeType{}
	for _, d := range desc.AttributeDefinitions {
		types[aws.ToString(d.AttributeName)] = d.AttributeType
	}
	want := []ddbtypes.KeySchemaElement{
		{AttributeName: aws.String(AttributePK), KeyType: ddbtypes.KeyTypeHash},
		{AttributeName: aws.String(AttributeSK), KeyType: ddbtypes.KeyTypeRange},
	}
	ok := len(desc.KeySchema) == len(want)
	for i := 0; ok && i < len(want); i++ {
		name := aws.ToString(desc.KeySchema[i].AttributeName)
		ok = name == aws.ToString(want[i].AttributeName) &&
			desc.KeySchema[i].KeyType == want[i].KeyType &&
			types[name] == ddbtypes.ScalarAttributeTypeS
	}
	if !ok {
		return fmt.Errorf("dynamodb: table %s exists with another key schema; want string keys %s (HASH) and %s (RANGE)",
			aws.ToString(desc.TableName), AttributePK, AttributeSK)
	}
	return nil
}

// EnsureTTL enables TTL on expires_at unless it is already enabled or
// being enabled. It fails when TTL is enabled on another attribute.
func EnsureTTL(ctx context.Context, client SchemaAPI, tableName string) error {
	out, err := client.DescribeTimeToLive(ctx, &ddb.DescribeTimeToLiveInput{TableName: aws.String(tableName)})
	if err != nil {
		return fmt.Errorf("dynamodb: describe TTL of %s: %w", tableName, err)
	}
	if d := out.TimeToLiveDescription; d != nil {
		switch d.TimeToLiveStatus {
		case ddbtypes.TimeToLiveStatusEnabled, ddbtypes.TimeToLiveStatusEnabling:
			if attr := aws.ToString(d.AttributeName); attr != AttributeExpiresAt {
				return fmt.Errorf("dynamodb: table %s has TTL on %q; want %q", tableName, attr, AttributeExpiresAt)
			}
			return nil
		}
	}
	_, err = client.UpdateTimeToLive(ctx, &ddb.UpdateTimeToLiveInput{
		TableName: aws.String(tableName),
		TimeToLiveSpecification: &ddbtypes.TimeToLiveSpecification{
			AttributeName: aws.String(AttributeExpiresAt),
			Enabled:       aws.Bool(true),
		},
	})
	if err != nil {
		return fmt.Errorf("dynamodb: enable TTL on %s: %w", tableName, err)
	}
	return nil
}

// EnsurePointInTimeRecovery enables continuous backups with point-in-time
// recovery unless they are already enabled.
func EnsurePointInTimeRecovery(ctx context.Context, client SchemaAPI, tableName string) error {
	out, err := client.DescribeContinuousBackups(ctx, &ddb.DescribeContinuousBackupsInput{TableName: aws.String(tableName)})
	if err != nil {
		return fmt.Errorf("dynamodb: describe continuous backups of %s: %w", tableName, err)
	}
	if d := out.ContinuousBackupsDescription; d != nil && d.PointInTimeRecoveryDescription != nil &&
		d.PointInTimeRecoveryDescription.PointInTimeRecoveryStatus == ddbtypes.PointInTimeRecoveryStatusEnabled {
		return nil
	}
	_, err = client.UpdateContinuousBackups(ctx, &ddb.UpdateContinuousBackupsInput{
		TableName: aws.String(tableName),
		PointInTimeRecoverySpecification: &ddbtypes.PointInTimeRecoverySpecification{
			PointInTimeRecoveryEnabled: aws.Bool(true),
		},
	})
	if err != nil {
		return fmt.Errorf("dynamodb: enable point-in-time recovery on %s: %w", tableName, err)
	}
	return nil
}
