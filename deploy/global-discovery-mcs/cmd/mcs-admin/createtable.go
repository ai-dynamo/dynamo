// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"flag"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/cmd/mcs-admin/createtable"
)

// runCreateTable is a variable for tests.
var runCreateTable = createtable.Run

func init() {
	registerCommand(createtable.Name, command{
		summary: createtable.Summary,
		run: func(ctx context.Context, e *cmdEnv, args []string) error {
			err := runCreateTable(ctx, createtable.Defaults{
				Table:    e.cfg.DynamoDBTable,
				Region:   e.cfg.Region,
				Endpoint: e.cfg.DynamoDBEndpoint,
			}, args, e.stdout, e.stderr)
			var usageErr createtable.UsageError
			switch {
			case err == nil:
				return nil
			case errors.Is(err, flag.ErrHelp):
				return nil
			case errors.As(err, &usageErr):
				return usageError{err}
			}
			return err
		},
	})
}
