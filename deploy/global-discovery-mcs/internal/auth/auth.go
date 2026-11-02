// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package auth is the authentication seam of mcs-registry. v1 ships without
// authentication (NoopAuthenticator); a later Authenticator can bind a caller
// to its clusterset and cluster without API changes. Handlers read the
// Principal with FromContext.
package auth

import (
	"context"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Principal is an authenticated caller.
type Principal struct {
	// Subject identifies the caller, for example a service account. Never
	// a credential.
	Subject string
	// ClusterSet and Cluster bind the caller to a cluster when the
	// authenticator knows it; empty means unbound.
	ClusterSet string
	Cluster    string
	// Anonymous is set when no authentication took place.
	Anonymous bool
}

// Authenticator authenticates an incoming call. It returns an error to
// reject the call; an error without a gRPC status becomes UNAUTHENTICATED.
type Authenticator interface {
	Authenticate(ctx context.Context, fullMethod string) (Principal, error)
}

// NoopAuthenticator accepts every call as anonymous.
type NoopAuthenticator struct{}

// Anonymous is the Principal NoopAuthenticator returns.
var Anonymous = Principal{Subject: "anonymous", Anonymous: true}

// Authenticate implements Authenticator.
func (NoopAuthenticator) Authenticate(context.Context, string) (Principal, error) {
	return Anonymous, nil
}

type principalKey struct{}

// NewContext returns ctx carrying p.
func NewContext(ctx context.Context, p Principal) context.Context {
	return context.WithValue(ctx, principalKey{}, p)
}

// FromContext returns the Principal of a call.
func FromContext(ctx context.Context) (Principal, bool) {
	p, ok := ctx.Value(principalKey{}).(Principal)
	return p, ok
}

// DefaultExempt are method prefixes that skip authentication: health checks
// and reflection.
var DefaultExempt = []string{"/grpc.health.v1.Health/", "/grpc.reflection."}

func exempt(method string, prefixes []string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(method, p) {
			return true
		}
	}
	return false
}

func authenticate(ctx context.Context, a Authenticator, method string) (context.Context, error) {
	p, err := a.Authenticate(ctx, method)
	if err != nil {
		if _, ok := status.FromError(err); ok {
			return nil, err
		}
		return nil, status.Error(codes.Unauthenticated, err.Error())
	}
	return NewContext(ctx, p), nil
}

// UnaryServerInterceptor authenticates unary calls, except methods with an
// exempt prefix.
func UnaryServerInterceptor(a Authenticator, exemptPrefixes ...string) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		if exempt(info.FullMethod, exemptPrefixes) {
			return handler(ctx, req)
		}
		ctx, err := authenticate(ctx, a, info.FullMethod)
		if err != nil {
			return nil, err
		}
		return handler(ctx, req)
	}
}

type wrappedStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (w wrappedStream) Context() context.Context { return w.ctx }

// StreamServerInterceptor authenticates streams, except methods with an
// exempt prefix.
func StreamServerInterceptor(a Authenticator, exemptPrefixes ...string) grpc.StreamServerInterceptor {
	return func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if exempt(info.FullMethod, exemptPrefixes) {
			return handler(srv, ss)
		}
		ctx, err := authenticate(ss.Context(), a, info.FullMethod)
		if err != nil {
			return err
		}
		return handler(srv, wrappedStream{ServerStream: ss, ctx: ctx})
	}
}
