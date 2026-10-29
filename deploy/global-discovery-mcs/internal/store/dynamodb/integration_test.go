//go:build integration

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamodb_test

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"slices"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	ddb "github.com/aws/aws-sdk-go-v2/service/dynamodb"
	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/dynamodb"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/dynamodb/dynamodblocal"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/storetest"
)

func TestMain(m *testing.M) {
	code := m.Run()
	if err := dynamodblocal.Terminate(); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
	os.Exit(code)
}

var t0 = time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)

// localTableOptions are the production table options minus what
// dynamodb-local lacks (point-in-time recovery) or what would keep the test
// from dropping the table (deletion protection).
func localTableOptions(name string) dynamodb.TableOptions {
	opts := dynamodb.DefaultTableOptions(name)
	opts.DeletionProtection = false
	opts.PointInTimeRecovery = false
	opts.PollInterval = 100 * time.Millisecond
	return opts
}

// newTable creates a table with a unique name and returns the client.
func newTable(t *testing.T) (*ddb.Client, string) {
	t.Helper()
	client := dynamodblocal.Client(t)
	name := "mcs-" + randomSuffix(t)
	opts := localTableOptions(name)
	created, err := dynamodb.CreateTable(t.Context(), client, opts)
	require.NoError(t, err)
	require.True(t, created)
	t.Cleanup(func() {
		_, _ = client.DeleteTable(context.Background(), &ddb.DeleteTableInput{TableName: aws.String(name)})
	})
	return client, name
}

func randomSuffix(t *testing.T) string {
	b := make([]byte, 6)
	_, err := rand.Read(b)
	require.NoError(t, err)
	return hex.EncodeToString(b)
}

func newStore(t *testing.T, client dynamodb.API, table string) *dynamodb.Store {
	t.Helper()
	st, err := dynamodb.New(dynamodb.Options{Client: client, TableName: table, QueryConcurrency: 4})
	require.NoError(t, err)
	return st
}

// TestConformance runs the shared store suite against dynamodb-local. All
// subtests share one table, as replicas share one Global Table.
func TestConformance(t *testing.T) {
	client, table := newTable(t)
	storetest.Run(t, func(t *testing.T) store.Store { return newStore(t, client, table) })
}

// TestMultiReplica checks the conditional writes across two Stores that
// share one table, as two mcs-registry replicas do.
func TestMultiReplica(t *testing.T) {
	client, table := newTable(t)
	a, b := newStore(t, client, table), newStore(t, client, table)
	ctx := t.Context()
	const cs = "multi"

	t.Run("lease epochs", func(t *testing.T) {
		key := store.ClusterKey{ClusterSet: cs, Cluster: "c1"}
		reg, err := a.PutCluster(ctx, store.ClusterRecord{Key: key, Region: "us-east-1", Lease: store.Lease{TTL: 30 * time.Second}}, t0)
		require.NoError(t, err)

		renewed, err := b.RenewClusterLease(ctx, key, reg.Lease.Epoch, t0.Add(5*time.Second))
		require.NoError(t, err, "a heartbeat may land on any replica")
		assert.True(t, renewed.Lease.RenewedAt.Equal(t0.Add(5*time.Second)))

		again, err := b.PutCluster(ctx, store.ClusterRecord{Key: key, Region: "us-west-2", Lease: store.Lease{TTL: 30 * time.Second}}, t0.Add(6*time.Second))
		require.NoError(t, err)
		assert.Greater(t, again.Lease.Epoch, reg.Lease.Epoch)
		assert.Equal(t, uint64(2), again.Generation)

		_, err = a.RenewClusterLease(ctx, key, reg.Lease.Epoch, t0.Add(7*time.Second))
		assert.ErrorIs(t, err, store.ErrLeaseEpochMismatch, "the old incarnation is fenced on the other replica")
		got, err := a.GetCluster(ctx, key)
		require.NoError(t, err)
		assert.Equal(t, "us-west-2", got.Region)
		assert.True(t, got.RegisteredAt.Equal(t0))
	})

	t.Run("stale generations", func(t *testing.T) {
		key := store.ExportKey{ClusterSet: cs, Cluster: "c1", Namespace: "llm", Name: "router"}
		require.NoError(t, a.PutExport(ctx, store.ExportRecord{Key: key, Generation: 10, UpdatedAt: t0}))
		err := b.PutExport(ctx, store.ExportRecord{Key: key, Generation: 9, UpdatedAt: t0})
		current, ok := store.StaleGeneration(err)
		require.True(t, ok, "got %v", err)
		assert.Equal(t, uint64(10), current)
		require.NoError(t, b.PutExport(ctx, store.ExportRecord{Key: key, Generation: 11, UpdatedAt: t0}))
		current, ok = store.StaleGeneration(a.PutExport(ctx, store.ExportRecord{Key: key, Generation: 11, UpdatedAt: t0}))
		require.True(t, ok)
		assert.Equal(t, uint64(11), current)

		require.NoError(t, a.PutEndpointSet(ctx, store.EndpointSetRecord{Key: key, Generation: 20, UpdatedAt: t0}))
		current, ok = store.StaleGeneration(b.PutEndpointSet(ctx, store.EndpointSetRecord{Key: key, Generation: 19, UpdatedAt: t0}))
		require.True(t, ok)
		assert.Equal(t, uint64(20), current)
	})

	t.Run("concurrent registrations", func(t *testing.T) {
		key := store.ClusterKey{ClusterSet: cs, Cluster: "c2"}
		const perReplica = 8
		var mu sync.Mutex
		var epochs []uint64
		var wg sync.WaitGroup
		for _, st := range []*dynamodb.Store{a, b} {
			for range perReplica {
				wg.Go(func() {
					got, err := st.PutCluster(ctx, store.ClusterRecord{Key: key, Lease: store.Lease{TTL: time.Minute}}, t0)
					assert.NoError(t, err)
					mu.Lock()
					epochs = append(epochs, got.Lease.Epoch)
					mu.Unlock()
				})
			}
		}
		wg.Wait()
		slices.Sort(epochs)
		assert.Len(t, slices.Compact(slices.Clone(epochs)), 2*perReplica, "every registration minted its own epoch")
		got, err := a.GetCluster(ctx, key)
		require.NoError(t, err)
		assert.Equal(t, uint64(2*perReplica), got.Generation)
		assert.Equal(t, epochs[len(epochs)-1], got.Lease.Epoch)
	})

	t.Run("delete on one replica", func(t *testing.T) {
		require.NoError(t, b.DeleteCluster(ctx, store.ClusterKey{ClusterSet: cs, Cluster: "c1"}))
		snap, err := a.LoadClusterSet(ctx, cs)
		require.NoError(t, err)
		for _, r := range snap.Clusters {
			assert.NotEqual(t, "c1", r.Key.Cluster)
		}
		assert.Empty(t, snap.Exports)
		assert.Empty(t, snap.EndpointSets)
	})
}

// TestLargeClusterSet loads endpoint partitions larger than one 1 MB query
// page.
func TestLargeClusterSet(t *testing.T) {
	client, table := newTable(t)
	st := newStore(t, client, table)
	ctx := t.Context()
	const cs, sets, endpoints = "large", 12, store.MaxEndpointsPerSet
	for i := range sets {
		rec := store.EndpointSetRecord{
			Key:        store.ExportKey{ClusterSet: cs, Cluster: "c1", Namespace: "ns", Name: "svc-" + strconv.Itoa(i)},
			Generation: 1,
			UpdatedAt:  t0,
		}
		for j := range endpoints {
			rec.Endpoints = append(rec.Endpoints, store.Endpoint{
				Addresses:   []string{fmt.Sprintf("10.%d.%d.%d", i, j/256, j%256)},
				AddressType: store.AddressTypeIPv4,
				Conditions:  store.EndpointConditions{Ready: true, Serving: true},
				Hostname:    fmt.Sprintf("pod-%d-%d", i, j),
				NodeName:    "node-with-a-reasonably-long-name.ec2.internal",
				Zone:        "us-east-1a",
				Ports:       []store.EndpointPort{{Name: "grpc", Protocol: store.ProtocolTCP, Port: 50071}},
			})
		}
		require.NoError(t, st.PutEndpointSet(ctx, rec))
	}
	snap, err := st.LoadClusterSet(ctx, cs)
	require.NoError(t, err)
	require.Len(t, snap.EndpointSets, sets)
	for _, s := range snap.EndpointSets {
		assert.Len(t, s.Endpoints, endpoints)
	}
}

// TestItemAttributes checks the stored layout: top-level attributes, TTL
// and the clusterset index.
func TestItemAttributes(t *testing.T) {
	client, table := newTable(t)
	st, err := dynamodb.New(dynamodb.Options{Client: client, TableName: table, ExpiryAfter: time.Hour})
	require.NoError(t, err)
	ctx := t.Context()
	key := store.ClusterKey{ClusterSet: "attrs", Cluster: "c1"}
	reg, err := st.PutCluster(ctx, store.ClusterRecord{Key: key, Lease: store.Lease{TTL: time.Minute}}, t0)
	require.NoError(t, err)

	get := func(sk string) map[string]ddbtypes.AttributeValue {
		out, err := client.GetItem(ctx, &ddb.GetItemInput{
			TableName: aws.String(table),
			Key: map[string]ddbtypes.AttributeValue{
				"pk": &ddbtypes.AttributeValueMemberS{Value: "CS#attrs"},
				"sk": &ddbtypes.AttributeValueMemberS{Value: sk},
			},
			ConsistentRead: aws.Bool(true),
		})
		require.NoError(t, err)
		return out.Item
	}
	num := func(av ddbtypes.AttributeValue) string {
		n, ok := av.(*ddbtypes.AttributeValueMemberN)
		require.True(t, ok, "%T is not a number", av)
		return n.Value
	}
	it := get("CL#c1")
	assert.Equal(t, strconv.FormatUint(reg.Lease.Epoch, 10), num(it["lease_epoch"]))
	assert.Equal(t, "1", num(it["gen"]))
	assert.Equal(t, strconv.FormatInt(t0.UnixNano(), 10), num(it["lease_renewed_at"]))
	assert.Equal(t, strconv.FormatInt(t0.UnixNano(), 10), num(it["updated_at"]))
	assert.Equal(t, strconv.FormatInt(t0.Add(time.Hour).Unix(), 10), num(it["expires_at"]))
	assert.Contains(t, it, "body")

	_, err = st.RenewClusterLease(ctx, key, reg.Lease.Epoch, t0.Add(time.Minute))
	require.NoError(t, err)
	it = get("CL#c1")
	assert.Equal(t, strconv.FormatInt(t0.Add(time.Hour+time.Minute).Unix(), 10), num(it["expires_at"]), "a heartbeat extends the expiry")
	assert.Equal(t, strconv.FormatInt(t0.UnixNano(), 10), num(it["updated_at"]), "a heartbeat keeps updated_at")

	require.NoError(t, st.PutExport(ctx, store.ExportRecord{Key: store.ExportKey{ClusterSet: "attrs", Cluster: "c1", Namespace: "ns", Name: "svc"}, Generation: 1, UpdatedAt: t0}))
	it = get("EX#c1#ns#svc")
	assert.NotContains(t, it, "expires_at", "exports do not expire")

	out, err := client.DescribeTimeToLive(ctx, &ddb.DescribeTimeToLiveInput{TableName: aws.String(table)})
	require.NoError(t, err)
	assert.Equal(t, "expires_at", aws.ToString(out.TimeToLiveDescription.AttributeName))
}

// TestCreateTable checks the table definition and that CreateTable is
// idempotent.
func TestCreateTable(t *testing.T) {
	client, table := newTable(t)
	ctx := t.Context()
	opts := localTableOptions(table)
	created, err := dynamodb.CreateTable(ctx, client, opts)
	require.NoError(t, err)
	assert.False(t, created, "an existing table is left in place")

	desc, err := client.DescribeTable(ctx, &ddb.DescribeTableInput{TableName: aws.String(table)})
	require.NoError(t, err)
	require.Len(t, desc.Table.LocalSecondaryIndexes, 1)
	lsi := desc.Table.LocalSecondaryIndexes[0]
	assert.Equal(t, dynamodb.IndexByUpdatedAt, aws.ToString(lsi.IndexName))
	assert.Equal(t, ddbtypes.ProjectionTypeKeysOnly, lsi.Projection.ProjectionType)

	other := "mcs-other-" + randomSuffix(t)
	_, err = client.CreateTable(ctx, &ddb.CreateTableInput{
		TableName:            aws.String(other),
		AttributeDefinitions: []ddbtypes.AttributeDefinition{{AttributeName: aws.String("id"), AttributeType: ddbtypes.ScalarAttributeTypeS}},
		KeySchema:            []ddbtypes.KeySchemaElement{{AttributeName: aws.String("id"), KeyType: ddbtypes.KeyTypeHash}},
		BillingMode:          ddbtypes.BillingModePayPerRequest,
	})
	require.NoError(t, err)
	opts.TableName = other
	_, err = dynamodb.CreateTable(ctx, client, opts)
	assert.ErrorContains(t, err, "another key schema")
}
