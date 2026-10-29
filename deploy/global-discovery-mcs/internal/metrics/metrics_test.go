// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package metrics

import (
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNamesAndPreinitialisation(t *testing.T) {
	reg := prometheus.NewRegistry()
	m := New(reg)
	m.SetBuildInfo("v1", "abc")

	families, err := reg.Gather()
	require.NoError(t, err)
	names := map[string]bool{}
	for _, f := range families {
		names[f.GetName()] = true
		assert.True(t, strings.HasPrefix(f.GetName(), "dynamo_mcs_registry_"), f.GetName())
	}
	for _, n := range []string{
		"dynamo_mcs_registry_lease_expirations_total", "dynamo_mcs_registry_watch_events_sent_total",
		"dynamo_mcs_registry_watch_resyncs_total", "dynamo_mcs_registry_watch_rejected_total",
		"dynamo_mcs_registry_view_refresh_failures_total", "dynamo_mcs_registry_store_conditional_failures_total",
		"dynamo_mcs_registry_compaction_deleted_total", "dynamo_mcs_registry_build_info", "dynamo_mcs_registry_watchers",
		"dynamo_mcs_registry_store_request_duration_seconds",
	} {
		assert.True(t, names[n], "%s is missing on the first scrape", n)
	}
	assert.Equal(t, len(Kinds)*len(EventTypes), testutil.CollectAndCount(m.WatchEventsSent))
	assert.Equal(t, len(Kinds)*len(ResyncReasons), testutil.CollectAndCount(m.WatchResyncs))
	assert.Equal(t, 1.0, testutil.ToFloat64(m.BuildInfo.WithLabelValues("v1", "abc")))
}

func TestClusterSetGauges(t *testing.T) {
	m := New(prometheus.NewRegistry())
	m.SetClusterSet("prod", ClusterSetGauges{
		ActiveClusters: 2, ExpiredClusters: 1, Exports: 5, ReadyEndpoints: 7, NotReadyEnds: 1,
		Imports: 3, ConflictedImports: 1,
	})
	assert.Equal(t, 2.0, testutil.ToFloat64(m.Clusters.WithLabelValues("prod", "active")))
	assert.Equal(t, 2.0, testutil.ToFloat64(m.ServiceImports.WithLabelValues("prod", "false")))
	assert.Equal(t, 1.0, testutil.ToFloat64(m.ServiceImports.WithLabelValues("prod", "true")))
	m.DeleteClusterSet("prod")
	assert.Equal(t, 0, testutil.CollectAndCount(m.Clusters))
	assert.Equal(t, 0, testutil.CollectAndCount(m.ServiceImports))
}
