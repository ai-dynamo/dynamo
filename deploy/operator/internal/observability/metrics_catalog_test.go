//go:build metriccatalog

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package observability

import (
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/net/html"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

// TestOperatorMetricsCatalog compares real collector output with the operator reference.
// It requires the full checkout rather than the operator-only Docker context.
func TestOperatorMetricsCatalog(t *testing.T) {
	t.Log("Read metric names and types from the operator catalog")
	_, source, _, ok := runtime.Caller(0)
	require.True(t, ok)
	contents, err := os.ReadFile(filepath.Join(filepath.Dir(source),
		"../../../../docs/fern/pages/reference/observability/operator-metrics.mdx"))
	require.NoError(t, err)
	documented, err := operatorMetricCatalog(string(contents))
	require.NoError(t, err)
	require.NotEmpty(t, documented, "the operator catalog must not be empty")

	t.Log("Register the real operator collectors in an isolated registry")
	registry := prometheus.NewPedanticRegistry()
	previous := ctrlmetrics.Registry
	ctrlmetrics.Registry = registry
	t.Cleanup(func() { ctrlmetrics.Registry = previous })
	InitMetrics()

	t.Log("Count all descriptors so an unexercised vector cannot escape the check")
	descriptors := make(chan *prometheus.Desc)
	go func() {
		registry.Describe(descriptors)
		close(descriptors)
	}()
	registered := 0
	for range descriptors {
		registered++
	}

	t.Log("Exercise the resource and webhook metric vectors")
	UpdateResourceCount("DynamoGraphDeployment", "catalog-test", "ready", 1)
	RecordWebhookAdmission("DynamoGraphDeployment", "CREATE", true, time.Millisecond)
	RecordWebhookDenial("DynamoGraphDeployment", "CREATE", k8serrors.NewBadRequest("catalog test"))

	t.Log("Scrape the Prometheus handler and compare exposed metric families")
	response := httptest.NewRecorder()
	promhttp.HandlerFor(registry, promhttp.HandlerOpts{}).ServeHTTP(
		response, httptest.NewRequest("GET", "/metrics", nil))
	require.Equal(t, 200, response.Code, response.Body.String())
	parser := expfmt.NewTextParser(model.UTF8Validation)
	families, err := parser.TextToMetricFamilies(response.Body)
	require.NoError(t, err)
	require.Len(t, families, registered,
		"exercise every registered collector before comparing the catalog")

	emitted := make(map[string]string, len(families))
	for name, family := range families {
		emitted[name] = strings.ToLower(family.GetType().String())
	}
	assert.Equal(t, documented, emitted,
		"operator metric names and types must match operator-metrics.mdx")
}

// operatorMetricCatalog reads literal metric fields outside Markdown examples and comments.
func operatorMetricCatalog(contents string) (map[string]string, error) {
	metrics := make(map[string]string)
	var fence string
	var commentEnd string
	offset := 0
	for _, line := range strings.SplitAfter(contents, "\n") {
		offset += len(line)
		line = strings.TrimSuffix(line, "\n")
		trimmed := strings.TrimLeft(line, " ")
		indent := len(line) - len(trimmed)
		if fence != "" {
			if indent <= 3 && strings.HasPrefix(trimmed, fence) &&
				strings.Trim(trimmed, fence[:1]+" \t\r") == "" {
				fence = ""
			}
			continue
		}
		if commentEnd == "" {
			if indent > 3 {
				continue
			}
			if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
				width := len(trimmed) - len(strings.TrimLeft(trimmed, trimmed[:1]))
				fence = trimmed[:width]
				continue
			}
		}

		// Preserve visible text on either side of comments, including after a closer.
		var visible strings.Builder
		for remaining := line; remaining != ""; {
			if commentEnd != "" {
				end := strings.Index(remaining, commentEnd)
				if end < 0 {
					break
				}
				remaining = remaining[end+len(commentEnd):]
				commentEnd = ""
				continue
			}

			opening, ending := "<!--", "-->"
			start := strings.Index(remaining, opening)
			if jsx := strings.Index(remaining, "{/*"); jsx >= 0 && (start < 0 || jsx < start) {
				start, opening, ending = jsx, "{/*", "*/}"
			}
			if start < 0 {
				visible.WriteString(remaining)
				break
			}
			visible.WriteString(remaining[:start])
			remaining = remaining[start+len(opening):]
			commentEnd = ending
		}
		line = visible.String()
		trimmed = strings.TrimLeft(line, " ")
		if len(line)-len(trimmed) > 3 || !strings.HasPrefix(trimmed, "<ParamField") {
			continue
		}

		// Decode only a block field's opening tag, not Markdown prose or examples.
		tokenizer := html.NewTokenizer(strings.NewReader(trimmed + "\n" + contents[offset:]))
		tokenType := tokenizer.Next()
		if tokenType != html.StartTagToken && tokenType != html.SelfClosingTagToken {
			return nil, fmt.Errorf("parse operator metric field: %v", tokenizer.Err())
		}
		field := tokenizer.Token()
		if field.Data != "paramfield" {
			continue
		}
		var name, kind string
		for _, attribute := range field.Attr {
			switch attribute.Key {
			case "path":
				name = attribute.Val
			case "type":
				kind = attribute.Val
			}
		}
		if !strings.HasPrefix(name, metricsNamespace+"_") {
			continue
		}
		if _, duplicate := metrics[name]; duplicate {
			return nil, fmt.Errorf("duplicate operator metric %q", name)
		}
		switch kind {
		case "counter", "gauge", "histogram", "summary":
			metrics[name] = kind
		default:
			return nil, fmt.Errorf("operator metric %q has unsupported type %q", name, kind)
		}
	}
	return metrics, nil
}

// TestOperatorMetricsCatalogFields covers catalog syntax and validation.
func TestOperatorMetricsCatalogFields(t *testing.T) {
	t.Log("Read literal metric fields while ignoring examples, comments, and label fields")
	documented, err := operatorMetricCatalog(`
<!-- <ParamField path="dynamo_operator_commented" type="counter" /> -->
<!--
<ParamField path="dynamo_operator_commented_block" type="counter" />
-->
{/*
<ParamField path="dynamo_operator_jsx_comment" type="counter" />
*/}
~~~mdx
<ParamField path="dynamo_operator_fenced_example" type="counter" />
~~~
<ParamField type="gauge" deprecated={true}
    path="dynamo_operator_resources_total">Resource count.</ParamField>
<ParamField path="namespace" type="string">A label, not a metric.</ParamField>
<ParamField path="controller_runtime_reconcile_total" type="counter" />
`)
	require.NoError(t, err)
	assert.Equal(t, map[string]string{"dynamo_operator_resources_total": "gauge"}, documented)

	t.Log("Handle comments beside prose and preserve fields before and after comments")
	const field = `<ParamField path="dynamo_operator_resources_total" type="gauge" />`
	const commented = `<ParamField path="dynamo_operator_commented" type="counter" />`
	for _, test := range []struct{ name, contents string }{
		{"jsx_after_prose", "ordinary text {/*\n" + commented + "\n*/}\n" + field},
		{"html_after_prose", "ordinary text <!--\n" + commented + "\n-->\n" + field},
		{"field_before_inline_comment", field + " {/* inline comment */}"},
		{"field_before_multiline_comment", field + " {/*\n" + commented + "\n*/}"},
		{"field_after_comment_closer", "{/*\n" + commented + "\n*/}" + field},
		{"earliest_opener", "ordinary text {/* <!--\n" + commented + "\n*/}" + field},
	} {
		documented, err := operatorMetricCatalog(test.contents)
		require.NoError(t, err, test.name)
		assert.Equal(t, map[string]string{"dynamo_operator_resources_total": "gauge"}, documented, test.name)
	}

	t.Log("Reject duplicate metrics and unsupported Prometheus types")
	for _, contents := range []string{
		`<ParamField path="dynamo_operator_test" type="gauge" />
<ParamField path="dynamo_operator_test" type="gauge" />`,
		`<ParamField path="dynamo_operator_test" type="string" />`,
	} {
		_, err := operatorMetricCatalog(contents)
		require.Error(t, err)
	}
}
