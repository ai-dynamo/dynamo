// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"flag"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/config"
)

func TestParse(t *testing.T) {
	for _, tt := range []struct {
		name string
		args []string
		code int
	}{
		{"help", []string{"-h"}, 0},
		{"unknown flag", []string{"--nope"}, 2},
		{"invalid", []string{"--leader-elect=false"}, 2},
		{"valid", []string{"--registry-url=https://registry.example", "--leader-elect=false", "--zap-log-level=debug"}, -1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			cfg, zapOpts, code := parse(flag.NewFlagSet("test", flag.ContinueOnError), tt.args, &out)
			assert.Equal(t, tt.code, code, out.String())
			assert.NotNil(t, zapOpts)
			if code == -1 {
				assert.Equal(t, []string{"https://registry.example"}, cfg.RegistryURLs)
			}
		})
	}
}

func TestRunWithoutAKubeconfig(t *testing.T) {
	t.Setenv("KUBECONFIG", "/does/not/exist")
	t.Setenv("KUBERNETES_SERVICE_HOST", "")
	t.Setenv("HOME", t.TempDir())
	cfg := config.Config{RegistryURLs: []string{"https://registry.example"}}
	assert.Equal(t, 1, run(context.Background(), cfg))
	cfg.OTelEnabled = true
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
	assert.Equal(t, 1, run(context.Background(), cfg))
}
