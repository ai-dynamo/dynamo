// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package metrics

import (
	"errors"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPreinitialisedAndRegistered(t *testing.T) {
	reg := prometheus.NewRegistry()
	m := New()
	require.NoError(t, m.Register(reg))
	require.Error(t, m.Register(reg), "registering twice fails")

	families, err := reg.Gather()
	require.NoError(t, err)
	names := map[string]int{}
	for _, f := range families {
		names[f.GetName()] = len(f.GetMetric())
	}
	assert.Equal(t, len(ExportStates), names[Namespace+"_exports"])
	assert.Equal(t, len(Controllers)*len(Results), names[Namespace+"_reconciles_total"])
	assert.Equal(t, len(WriteOps)*len(WriteResults), names[Namespace+"_export_writes_total"])
	assert.Equal(t, len(WriteResults), names[Namespace+"_endpoint_replacements_total"])
	assert.Equal(t, len(EventTypes), names[Namespace+"_import_events_total"])
	assert.Equal(t, len(ResyncReasons), names[Namespace+"_resyncs_total"])
	assert.Equal(t, len(ResyncReasons), names[Namespace+"_resync_attempts_total"])
	assert.Equal(t, len(GCKinds), names[Namespace+"_gc_deletions_total"])
	assert.Equal(t, len(HeartbeatReasons), names[Namespace+"_heartbeat_failures_total"])
	assert.Equal(t, len(Results), names[Namespace+"_registrations_total"])
	for _, n := range []string{"exported_endpoints", "imports", "imports_pending_namespace", "stream_reconnects_total", "cluster_registered"} {
		assert.Equal(t, 1, names[Namespace+"_"+n], n)
	}

	m.ObserveReconcile(ControllerServiceExport, time.Now(), nil)
	m.ObserveReconcile(ControllerServiceExport, time.Now(), errors.New("x"))
	assert.Equal(t, 1.0, testutil.ToFloat64(m.Reconciles.WithLabelValues(ControllerServiceExport, ResultSuccess)))
	assert.Equal(t, 1.0, testutil.ToFloat64(m.Reconciles.WithLabelValues(ControllerServiceExport, ResultError)))
}
