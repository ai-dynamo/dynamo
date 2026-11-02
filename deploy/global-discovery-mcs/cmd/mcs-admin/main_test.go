// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/cmd/mcs-admin/createtable"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/memory"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/storefactory"
)

var t0 = time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)

func env(vars map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) {
		v, ok := vars[k]
		return v, ok
	}
}

func seeded(t *testing.T) *memory.Store {
	t.Helper()
	ctx := context.Background()
	st := memory.New()
	for _, c := range []string{"a", "b"} {
		_, err := st.PutCluster(ctx, store.ClusterRecord{
			Key:    store.ClusterKey{ClusterSet: "prod", Cluster: c},
			Region: "us-east-1", Lease: store.Lease{TTL: 30 * time.Second},
		}, t0)
		require.NoError(t, err)
		key := store.ExportKey{ClusterSet: "prod", Cluster: c, Namespace: "llm", Name: "router"}
		require.NoError(t, st.PutExport(ctx, store.ExportRecord{Key: key, Generation: 1, Spec: store.ExportSpec{
			Type: store.ServiceTypeClusterSetIP, Labels: map[string]string{"app": "router"}, ExportCreationTimestamp: t0,
		}}))
		eps := store.EndpointSetRecord{Key: key, Generation: 1, Endpoints: []store.Endpoint{
			{Addresses: []string{"10.0.0.1"}, Conditions: store.EndpointConditions{Ready: true}},
			{Addresses: []string{"10.0.0.2"}},
		}}
		require.NoError(t, st.PutEndpointSet(ctx, eps))
	}
	_, err := st.PutCluster(ctx, store.ClusterRecord{Key: store.ClusterKey{ClusterSet: "dev", Cluster: "x"}}, t0)
	require.NoError(t, err)
	return st
}

// keepOpen ignores Close so that one store serves several runs.
type keepOpen struct{ store.Store }

func (keepOpen) Close() error { return nil }

func runWith(t *testing.T, st store.Store, now time.Time, args ...string) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	open := func(context.Context, *config.Config) (store.Store, error) { return keepOpen{st}, nil }
	code := run(context.Background(), args, env(nil), &stdout, &stderr, open, func() time.Time { return now })
	return code, stdout.String(), stderr.String()
}

func TestDescribeText(t *testing.T) {
	st := seeded(t)
	code, out, errOut := runWith(t, st, t0.Add(10*time.Second), "describe")
	require.Equal(t, 0, code, errOut)
	assert.Contains(t, out, "Clusterset dev: 1 clusters")
	assert.Contains(t, out, "Clusterset prod: 2 clusters (2 active), 2 exports, 2 endpoint sets (4 endpoints)\n")
	assert.Regexp(t, `EXPORT\s+CLUSTER\s+TYPE\s+GENERATION\s+CREATED\n`, out)
	assert.Regexp(t, `llm/router\s+a\s+ClusterSetIP\s+1\s+2026-10-26T12:00:00Z\n`, out)
	assert.Regexp(t, `llm/router\s+b\s+2\s+1\s+1\n`, out)
	assert.Contains(t, out, "Clusterset dev: 1 clusters (0 active), 0 exports, 0 endpoint sets (0 endpoints)\n")
	assert.NotContains(t, out[:strings.Index(out, "Clusterset prod")], "ENDPOINT SET")

	code, out, _ = runWith(t, st, t0.Add(time.Hour), "describe", "--clusterset", "prod")
	require.Equal(t, 0, code)
	assert.Contains(t, out, "(0 active)")
	assert.Contains(t, out, "EXPIRED")
	assert.NotContains(t, out, "Clusterset dev")
}

func TestDescribeJSON(t *testing.T) {
	code, out, _ := runWith(t, seeded(t), t0, "describe", "--clusterset=prod", "--output=json")
	require.Equal(t, 0, code)
	var d []description
	require.NoError(t, json.Unmarshal([]byte(out), &d))
	require.Len(t, d, 1)
	assert.Equal(t, "prod", d[0].ClusterSet)
	assert.Len(t, d[0].Clusters, 2)
	assert.Equal(t, "ACTIVE", d[0].Clusters[0].LeaseState)
	assert.Equal(t, 1, d[0].EndpointSets[0].Ready)
	assert.Equal(t, "ClusterSetIP", d[0].Exports[0].Type)
	assert.Equal(t, 4, d[0].Counts["endpoints"])
	assert.Equal(t, 2, d[0].Counts["activeClusters"])

	// Leases are evaluated at the time of the call: after an hour none is
	// active, and the rows remain.
	code, out, _ = runWith(t, seeded(t), t0.Add(time.Hour), "describe", "--clusterset=prod", "--output=json")
	require.Equal(t, 0, code)
	var later []description
	require.NoError(t, json.Unmarshal([]byte(out), &later))
	require.Len(t, later[0].Clusters, 2)
	assert.Equal(t, "EXPIRED", later[0].Clusters[0].LeaseState)
	assert.Zero(t, later[0].Counts["activeClusters"])

	code, out, _ = runWith(t, memory.New(), t0, "describe")
	require.Equal(t, 0, code)
	assert.Contains(t, out, "No clustersets.")
}

func TestDeleteCluster(t *testing.T) {
	st := seeded(t)
	code, _, errOut := runWith(t, st, t0, "delete-cluster", "--clusterset=prod", "--cluster=b")
	assert.Equal(t, 2, code)
	assert.Contains(t, errOut, "would delete cluster prod/b: 1 exports, 1 endpoint sets; pass --yes")

	code, out, _ := runWith(t, st, t0, "delete-cluster", "--clusterset=prod", "--cluster=b", "--yes")
	require.Equal(t, 0, code)
	assert.Contains(t, out, "Deleted cluster prod/b: 1 exports, 1 endpoint sets.")
	snap, err := st.LoadClusterSet(context.Background(), "prod")
	require.NoError(t, err)
	require.Len(t, snap.Clusters, 1)
	assert.Equal(t, "a", snap.Clusters[0].Key.Cluster)
	// The records of cluster a stay.
	require.Len(t, snap.Exports, 1)
	assert.Equal(t, "a", snap.Exports[0].Key.Cluster)
	require.Len(t, snap.EndpointSets, 1)
	assert.Equal(t, "a", snap.EndpointSets[0].Key.Cluster)

	code, _, errOut = runWith(t, st, t0, "delete-cluster", "--clusterset=prod")
	assert.Equal(t, 2, code)
	assert.Contains(t, errOut, "DNS labels")
}

type brokenStore struct{ store.Store }

func (brokenStore) ListClusterSets(context.Context) ([]string, error) {
	return nil, store.ErrUnavailable
}
func (brokenStore) LoadClusterSet(context.Context, string) (store.ClusterSetSnapshot, error) {
	return store.ClusterSetSnapshot{}, store.ErrUnavailable
}
func (brokenStore) Close() error { return nil }

func TestErrors(t *testing.T) {
	broken := brokenStore{}
	for _, args := range [][]string{
		{"describe"},
		{"describe", "--clusterset=prod"},
		{"delete-cluster", "--clusterset=prod", "--cluster=a", "--yes"},
	} {
		code, _, errOut := runWith(t, broken, t0, args...)
		assert.Equal(t, 1, code, args)
		assert.Contains(t, errOut, "unavailable")
	}
	for _, args := range [][]string{
		{},
		{"frobnicate"},
		{"create-table", "--bogus"},
		{"describe", "--output=yaml"},
		{"describe", "--bogus"},
		{"delete-cluster", "--bogus"},
		{"--bogus"},
	} {
		code, _, _ := runWith(t, memory.New(), t0, args...)
		assert.Equal(t, 2, code, args)
	}
	code, _, errOut := runWith(t, memory.New(), t0, "--help")
	assert.Equal(t, 0, code)
	assert.Contains(t, errOut, "delete-cluster  delete a cluster")

	var stdout, stderr bytes.Buffer
	failOpen := func(context.Context, *config.Config) (store.Store, error) { return nil, errors.New("no backend") }
	assert.Equal(t, 1, run(context.Background(), []string{"describe"}, env(nil), &stdout, &stderr, failOpen, time.Now))
	assert.Contains(t, stderr.String(), "no backend")

	// The store flags have MCS_ environment fallbacks.
	stderr.Reset()
	t.Setenv("AWS_REGION", "")
	code = run(context.Background(), []string{"describe"}, env(map[string]string{"MCS_STORE": "dynamodb"}),
		&stdout, &stderr, storefactory.Open, time.Now)
	assert.Equal(t, 1, code)
	assert.Contains(t, stderr.String(), "no AWS region")
	stderr.Reset()
	code = run(context.Background(), []string{"describe"}, env(map[string]string{"MCS_LEASE_SKEW": "soon"}),
		&stdout, &stderr, storefactory.Open, time.Now)
	assert.Equal(t, 2, code)
}

func TestSubcommandHelp(t *testing.T) {
	for name, flags := range map[string][]string{
		"describe":       {"-clusterset", "-output"},
		"delete-cluster": {"-clusterset", "-cluster", "-yes"},
		"create-table":   {"-table", "-region", "-endpoint", "-pitr"},
	} {
		for _, help := range []string{"--help", "-h"} {
			code, out, errOut := runWith(t, memory.New(), t0, name, help)
			assert.Equal(t, 0, code, "%s %s: %s", name, help, errOut)
			assert.Contains(t, errOut, "Usage: mcs-admin ", name)
			assert.Contains(t, errOut, name+" ", name)
			for _, f := range flags {
				assert.Contains(t, errOut, f, "%s %s", name, help)
			}
			assert.NotContains(t, errOut+out, "help requested", name)
			assert.NotContains(t, errOut, "warning", "help does not open the store")
		}
	}

	// A bad flag prints the usage and the error once, and exits 2.
	code, _, errOut := runWith(t, memory.New(), t0, "describe", "--bogus")
	assert.Equal(t, 2, code)
	assert.Contains(t, errOut, "Usage: mcs-admin [store flags] describe")
	assert.Equal(t, 1, strings.Count(errOut, "flag provided but not defined: -bogus"), errOut)
}

func TestMemoryStoreWarning(t *testing.T) {
	const warning = "the memory store is empty in a fresh process; use --store=dynamodb to inspect a deployment"
	for _, args := range [][]string{
		{"describe"},
		{"delete-cluster", "--clusterset=prod", "--cluster=a", "--yes"},
	} {
		_, _, errOut := runWith(t, memory.New(), t0, args...)
		assert.Contains(t, errOut, warning, args)
	}

	var stdout, stderr bytes.Buffer
	open := func(context.Context, *config.Config) (store.Store, error) { return keepOpen{memory.New()}, nil }
	code := run(context.Background(), []string{"--store=dynamodb", "--region=us-east-1", "describe"}, env(nil),
		&stdout, &stderr, open, time.Now)
	assert.Equal(t, 0, code, stderr.String())
	assert.NotContains(t, stderr.String(), "memory store")
}

func TestCreateTableTakesStoreFlags(t *testing.T) {
	orig := runCreateTable
	t.Cleanup(func() { runCreateTable = orig })
	var got createtable.Defaults
	var args []string
	runCreateTable = func(_ context.Context, d createtable.Defaults, a []string, _, _ io.Writer) error {
		got, args = d, a
		return nil
	}
	call := func(vars map[string]string, argv ...string) int {
		got, args = createtable.Defaults{}, nil
		var stdout, stderr bytes.Buffer
		return run(context.Background(), argv, env(vars), &stdout, &stderr, storefactory.Open, time.Now)
	}

	// Store flags before the subcommand.
	require.Equal(t, 0, call(nil, "--dynamodb-table=Y", "--region=X", "--dynamodb-endpoint=http://e:8000",
		"create-table"))
	assert.Equal(t, createtable.Defaults{Table: "Y", Region: "X", Endpoint: "http://e:8000"}, got)
	assert.Empty(t, args)

	// MCS_ environment fallbacks; the store flag wins over them, and
	// MCS_REGION over AWS_REGION.
	vars := map[string]string{
		"MCS_DYNAMODB_TABLE": "envtable", "MCS_REGION": "envregion", "AWS_REGION": "awsregion",
		"MCS_DYNAMODB_ENDPOINT": "http://env:8000",
	}
	require.Equal(t, 0, call(vars, "create-table"))
	assert.Equal(t, createtable.Defaults{Table: "envtable", Region: "envregion", Endpoint: "http://env:8000"}, got)
	require.Equal(t, 0, call(vars, "--dynamodb-table=flagtable", "create-table"))
	assert.Equal(t, "flagtable", got.Table)
	delete(vars, "MCS_REGION")
	require.Equal(t, 0, call(vars, "create-table"))
	assert.Equal(t, "awsregion", got.Region)

	// The built-in default, and the subcommand flags, which createtable
	// itself lets win over the defaults (see its tests).
	require.Equal(t, 0, call(nil, "create-table", "--table=own"))
	assert.Equal(t, "global-discovery-mcs", got.Table)
	assert.Equal(t, []string{"--table=own"}, args)
}
