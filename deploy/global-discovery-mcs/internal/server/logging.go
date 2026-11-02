// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"fmt"

	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/logging"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
)

// zapLogger adapts zap to the go-grpc-middleware logging interface.
func zapLogger(l *zap.Logger) logging.Logger {
	return logging.LoggerFunc(func(_ context.Context, lvl logging.Level, msg string, fields ...any) {
		zf := make([]zap.Field, 0, len(fields)/2)
		for i := 0; i+1 < len(fields); i += 2 {
			key, ok := fields[i].(string)
			if !ok {
				key = fmt.Sprint(fields[i])
			}
			zf = append(zf, zap.Any(key, fields[i+1]))
		}
		if ce := l.Check(zapLevel(lvl), msg); ce != nil {
			ce.Write(zf...)
		}
	})
}

func zapLevel(lvl logging.Level) zapcore.Level {
	switch {
	case lvl >= logging.LevelError:
		return zapcore.ErrorLevel
	case lvl >= logging.LevelWarn:
		return zapcore.WarnLevel
	case lvl >= logging.LevelInfo:
		return zapcore.InfoLevel
	}
	return zapcore.DebugLevel
}

// codeToLevel logs successful calls at debug (heartbeats are frequent),
// expected client errors at info, server-side trouble at warn and failures
// that need attention at error.
func codeToLevel(code codes.Code) logging.Level {
	switch code {
	case codes.OK:
		return logging.LevelDebug
	case codes.Canceled, codes.InvalidArgument, codes.NotFound, codes.AlreadyExists, codes.FailedPrecondition,
		codes.Aborted, codes.OutOfRange, codes.Unauthenticated, codes.PermissionDenied:
		return logging.LevelInfo
	case codes.DeadlineExceeded, codes.ResourceExhausted, codes.Unavailable:
		return logging.LevelWarn
	}
	return logging.LevelError
}

// requestFields are the request fields logged with every unary call.
type requestFields interface {
	GetClusterset() string
}

// fieldsFromRequest returns the clusterset, cluster, namespace and name of a
// request, when it has them, and the x-request-id header.
func fieldsFromRequest(ctx context.Context, req any) logging.Fields {
	var f logging.Fields
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if ids := md.Get("x-request-id"); len(ids) > 0 {
			f = append(f, "request_id", ids[0])
		}
	}
	if r, ok := req.(requestFields); ok && r.GetClusterset() != "" {
		f = append(f, "clusterset", r.GetClusterset())
	}
	if r, ok := req.(interface{ GetCluster() string }); ok && r.GetCluster() != "" {
		f = append(f, "cluster", r.GetCluster())
	}
	if r, ok := req.(interface{ GetNamespace() string }); ok && r.GetNamespace() != "" {
		f = append(f, "namespace", r.GetNamespace())
	}
	if r, ok := req.(interface{ GetName() string }); ok && r.GetName() != "" {
		f = append(f, "name", r.GetName())
	}
	return f
}

// spanAttributes maps request log fields to span attribute keys.
var spanAttributes = map[string]string{
	"clusterset": "dynamo.clusterset",
	"cluster":    "dynamo.cluster.id",
	"namespace":  "k8s.namespace.name",
	"name":       "k8s.service.name",
	"request_id": "dynamo.request.id",
}

// annotate adds the request fields of req to the call's log fields and to
// its server span.
func annotate(ctx context.Context, req any) {
	f := fieldsFromRequest(ctx, req)
	if len(f) == 0 {
		return
	}
	logging.AddFields(ctx, f)
	span := trace.SpanFromContext(ctx)
	if !span.IsRecording() {
		return
	}
	for i := 0; i+1 < len(f); i += 2 {
		if key, ok := spanAttributes[f[i].(string)]; ok {
			span.SetAttributes(attribute.String(key, fmt.Sprint(f[i+1])))
		}
	}
}

// requestFieldsUnary annotates unary calls. It must run inside the logging
// interceptor.
func requestFieldsUnary(ctx context.Context, req any, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
	annotate(ctx, req)
	return handler(ctx, req)
}

// annotatedStream annotates a server stream with its first request.
type annotatedStream struct {
	grpc.ServerStream
	done bool
}

func (s *annotatedStream) RecvMsg(m any) error {
	err := s.ServerStream.RecvMsg(m)
	if err == nil && !s.done {
		s.done = true
		annotate(s.Context(), m)
	}
	return err
}

// requestFieldsStream annotates streams. It must run inside the logging
// interceptor.
func requestFieldsStream(srv any, ss grpc.ServerStream, _ *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	return handler(srv, &annotatedStream{ServerStream: ss})
}
