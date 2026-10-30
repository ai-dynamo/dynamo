// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package tracing holds the OpenTelemetry helpers of mcs-controller: the tracer
// provider setup and spans around registry writes and import applies. Without
// Setup the global no-op provider makes every span free.
package tracing

import (
	"context"
	"fmt"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// TracerName names the controller's tracer.
const TracerName = "mcs-controller"

// Span attribute keys.
const (
	AttrClusterSet = "dynamo.clusterset"
	AttrCluster    = "dynamo.cluster.id"
	AttrNamespace  = "k8s.namespace.name"
	AttrName       = "k8s.service.name"
)

// Setup installs a global tracer provider that exports spans over OTLP/gRPC,
// configured by the standard OTEL_* environment variables, and W3C trace
// context propagation. The returned function flushes and stops the
// provider.
func Setup(ctx context.Context, service, version string) (func(context.Context) error, error) {
	exporter, err := otlptracegrpc.New(ctx)
	if err != nil {
		return nil, fmt.Errorf("create OTLP trace exporter: %w", err)
	}
	res, err := resource.New(ctx,
		resource.WithAttributes(attribute.String("service.name", service), attribute.String("service.version", version)),
		resource.WithFromEnv(),
		resource.WithTelemetrySDK(),
	)
	if err != nil {
		return nil, fmt.Errorf("build trace resource: %w", err)
	}
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exporter), sdktrace.WithResource(res))
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	return tp.Shutdown, nil
}

// Start starts a span named mcs-controller.<op> for the object
// namespace/name of cluster in clusterset.
func Start(ctx context.Context, op, clusterset, cluster, namespace, name string) (context.Context, trace.Span) {
	attrs := []attribute.KeyValue{attribute.String(AttrClusterSet, clusterset), attribute.String(AttrCluster, cluster)}
	if namespace != "" {
		attrs = append(attrs, attribute.String(AttrNamespace, namespace))
	}
	if name != "" {
		attrs = append(attrs, attribute.String(AttrName, name))
	}
	return otel.Tracer(TracerName).Start(ctx, TracerName+"."+op, trace.WithAttributes(attrs...))
}

// End ends span, marking it failed when err is set.
func End(span trace.Span, err error) {
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		span.SetAttributes(attribute.Bool("error", true))
	}
	span.End()
}
