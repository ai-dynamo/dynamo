// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tracing

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestSpans(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec)))
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	_, span := Start(context.Background(), "put_service_export", "cs", "a", "ns", "svc")
	End(span, nil)
	_, span = Start(context.Background(), "heartbeat_cluster", "cs", "a", "", "")
	End(span, errors.New("boom"))

	spans := rec.Ended()
	require.Len(t, spans, 2)
	assert.Equal(t, "mcs-controller.put_service_export", spans[0].Name())
	assert.Contains(t, spans[0].Attributes(), attribute.String(AttrNamespace, "ns"))
	assert.Contains(t, spans[0].Attributes(), attribute.String(AttrCluster, "a"))
	assert.Equal(t, codes.Unset, spans[0].Status().Code)
	assert.Equal(t, codes.Error, spans[1].Status().Code)
	assert.Contains(t, spans[1].Attributes(), attribute.Bool("error", true))
	assert.NotContains(t, spans[1].Attributes(), attribute.String(AttrNamespace, ""))
}

func TestSetup(t *testing.T) {
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
	prev := otel.GetTracerProvider()
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	shutdown, err := Setup(context.Background(), "mcs-controller", "test")
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = shutdown(ctx)
}
