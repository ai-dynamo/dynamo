// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package createtable implements `mcs-admin create-table`. It creates the
// registry DynamoDB table, waits until it is ACTIVE, and enables TTL on
// expires_at and point-in-time recovery. On an existing table it checks the
// key schema and only enables what is missing, so it is safe to run on
// every deployment.
//
// It creates a single-region table. Add Global Table replicas afterwards;
// docs/storage.md shows how.
package createtable

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/dynamodb"
)

// Name is the subcommand name.
const Name = "create-table"

// Summary is the one-line description mcs-admin prints in its usage.
const Summary = "create the DynamoDB table, enable TTL and point-in-time recovery"

// UsageError reports invalid flags or arguments.
type UsageError struct {
	Err error
}

func (e UsageError) Error() string { return e.Err.Error() }

func (e UsageError) Unwrap() error { return e.Err }

const description = `Creates the registry DynamoDB table, waits until it is ACTIVE, and enables TTL
on expires_at and point-in-time recovery. On an existing table it checks
the key schema and enables only what is missing.`

// NewClientFunc returns the client the command uses.
type NewClientFunc func(ctx context.Context, opts dynamodb.ClientOptions) (dynamodb.SchemaAPI, error)

func defaultClient(ctx context.Context, opts dynamodb.ClientOptions) (dynamodb.SchemaAPI, error) {
	return dynamodb.NewClient(ctx, opts)
}

// Defaults are the values of --table, --region and --endpoint when the flag
// is not given: mcs-admin passes the store flags it parsed before the
// subcommand, with their MCS_ environment fallbacks. An empty field falls
// back to the built-in default.
type Defaults struct {
	Table    string
	Region   string
	Endpoint string
}

// Run parses the arguments that follow "create-table" and creates the
// table. Usage goes to stderr, the result to stdout. Invalid flags and
// arguments are a UsageError; -h and --help return an error matching
// flag.ErrHelp.
func Run(ctx context.Context, defaults Defaults, args []string, stdout, stderr io.Writer) error {
	return RunWith(ctx, defaults, args, stdout, stderr, defaultClient)
}

// RunWith is Run with the client constructor injected, for tests.
func RunWith(
	ctx context.Context, defaults Defaults, args []string, stdout, stderr io.Writer, newClient NewClientFunc,
) error {
	if defaults.Table == "" {
		defaults.Table = dynamodb.DefaultTableName
	}
	fs := flag.NewFlagSet(Name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		table  = fs.String("table", defaults.Table, "DynamoDB table name; defaults to the --dynamodb-table store flag")
		region = fs.String("region", defaults.Region,
			"AWS region; defaults to the --region store flag, AWS_REGION or the shared AWS configuration")
		endpoint = fs.String("endpoint", defaults.Endpoint,
			"DynamoDB endpoint override, for dynamodb-local; defaults to the --dynamodb-endpoint store flag")
		deletionProtection = fs.Bool("deletion-protection", true, "create the table with deletion protection")
		pitr               = fs.Bool("pitr", true, "enable point-in-time recovery (dynamodb-local does not support it)")
		ttl                = fs.Bool("ttl", true, "enable TTL on "+dynamodb.AttributeExpiresAt)
		waitTimeout        = fs.Duration("wait-timeout", 5*time.Minute, "how long to wait for the table to become ACTIVE")
	)
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage: mcs-admin %s [flags]\n\n%s\n\nFlags:\n", Name, description)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return UsageError{err}
	}
	if fs.NArg() > 0 {
		fs.Usage()
		return UsageError{fmt.Errorf("unexpected arguments: %q", fs.Args())}
	}
	if *table == "" {
		return UsageError{errors.New("--table must not be empty")}
	}

	client, err := newClient(ctx, dynamodb.ClientOptions{Region: *region, Endpoint: *endpoint})
	if err != nil {
		return err
	}
	created, err := dynamodb.CreateTable(ctx, client, dynamodb.TableOptions{
		TableName:           *table,
		DeletionProtection:  *deletionProtection,
		PointInTimeRecovery: *pitr,
		TimeToLive:          *ttl,
		WaitTimeout:         *waitTimeout,
	})
	if err != nil {
		return err
	}
	if created {
		fmt.Fprintf(stdout, "created table %s\n", *table)
	} else {
		fmt.Fprintf(stdout, "table %s exists; key schema checked\n", *table)
	}
	if *ttl {
		fmt.Fprintf(stdout, "TTL on %s is enabled\n", dynamodb.AttributeExpiresAt)
	}
	if *pitr {
		fmt.Fprintln(stdout, "point-in-time recovery is enabled")
	}
	return nil
}
