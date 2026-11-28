// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests of the metrics registry that need a live `DistributedRuntime` hierarchy, and their helper.

#[cfg(feature = "integration")]
mod test_helpers {
    /// Injects a `worker_id` label into Prometheus metric data lines.
    /// Prometheus places const labels (like worker_id) before special labels
    /// (like histogram `le`), so for histogram bucket lines we insert before
    /// `,le=`. For all other metric lines, we insert before the closing `}`.
    /// Comment lines and lines without labels are left unchanged.
    pub fn inject_worker_id(expected: &str, wid: &str) -> String {
        let wid_label = format!(",worker_id=\"{}\"", wid);
        expected
            .lines()
            .map(|line| {
                if line.starts_with('#') || line.trim().is_empty() || !line.contains('{') {
                    line.to_string()
                } else if let Some(le_pos) = line.find(",le=") {
                    // Histogram bucket lines: worker_id is a const label, `le` is special,
                    // so worker_id sorts before `le` in Prometheus output.
                    let mut s = line.to_string();
                    s.insert_str(le_pos, &wid_label);
                    s
                } else {
                    line.replacen("}", &format!("{}}}", wid_label), 1)
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(feature = "integration")]
mod test_metricsregistry_prefixes {
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use crate::metrics::*;
    use prometheus::core::Collector;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_hierarchical_prefixes_and_parent_hierarchies() {
        let drt = create_test_drt_async().await;

        const DRT_NAME: &str = "";
        const NAMESPACE_NAME: &str = "ns901";
        const COMPONENT_NAME: &str = "comp901";
        const ENDPOINT_NAME: &str = "ep901";
        let namespace = drt.namespace(NAMESPACE_NAME).unwrap();
        let component = namespace.component(COMPONENT_NAME).unwrap();
        let endpoint = component.endpoint(ENDPOINT_NAME);

        // DRT
        assert_eq!(drt.basename(), DRT_NAME);
        assert_eq!(drt.parent_hierarchies().len(), 0);
        // DRT hierarchy is just its basename (empty string)

        // Namespace
        assert_eq!(namespace.basename(), NAMESPACE_NAME);
        assert_eq!(namespace.parent_hierarchies().len(), 1);
        assert_eq!(namespace.parent_hierarchies()[0].basename(), DRT_NAME);
        // Namespace hierarchy is just its basename since parent is empty

        // Component
        assert_eq!(component.basename(), COMPONENT_NAME);
        assert_eq!(component.parent_hierarchies().len(), 2);
        assert_eq!(component.parent_hierarchies()[0].basename(), DRT_NAME);
        assert_eq!(component.parent_hierarchies()[1].basename(), NAMESPACE_NAME);
        // Component hierarchy structure is validated by the individual assertions above

        // Endpoint
        assert_eq!(endpoint.basename(), ENDPOINT_NAME);
        assert_eq!(endpoint.parent_hierarchies().len(), 3);
        assert_eq!(endpoint.parent_hierarchies()[0].basename(), DRT_NAME);
        assert_eq!(endpoint.parent_hierarchies()[1].basename(), NAMESPACE_NAME);
        assert_eq!(endpoint.parent_hierarchies()[2].basename(), COMPONENT_NAME);
        // Endpoint hierarchy structure is validated by the individual assertions above

        // Relationships
        assert!(
            namespace
                .parent_hierarchies()
                .iter()
                .any(|h| h.basename() == drt.basename())
        );
        assert!(
            component
                .parent_hierarchies()
                .iter()
                .any(|h| h.basename() == namespace.basename())
        );
        assert!(
            endpoint
                .parent_hierarchies()
                .iter()
                .any(|h| h.basename() == component.basename())
        );

        // Depth
        assert_eq!(drt.parent_hierarchies().len(), 0);
        assert_eq!(namespace.parent_hierarchies().len(), 1);
        assert_eq!(component.parent_hierarchies().len(), 2);
        assert_eq!(endpoint.parent_hierarchies().len(), 3);

        // Invalid namespace behavior - sanitizes to "_123" and succeeds
        // @ryanolson intended to enable validation (see TODO comment in component.rs) but didn't turn it on,
        // so invalid characters are sanitized in MetricsRegistry rather than rejected.
        let invalid_namespace = drt.namespace("@@123").unwrap();
        let result =
            invalid_namespace
                .metrics()
                .create_counter("test_counter", "A test counter", &[]);
        assert!(result.is_ok());
        if let Ok(counter) = &result {
            // Verify the namespace was sanitized to "_123" in the label
            let desc = counter.desc();
            let namespace_label = desc[0]
                .const_label_pairs
                .iter()
                .find(|l| l.name() == "dynamo_namespace")
                .expect("Should have dynamo_namespace label");
            assert_eq!(namespace_label.value(), "_123");
        }

        // Valid namespace works
        let valid_namespace = drt.namespace("ns567").unwrap();
        assert!(
            valid_namespace
                .metrics()
                .create_counter("test_counter", "A test counter", &[])
                .is_ok()
        );
    }

    #[tokio::test]
    async fn test_typed_callback_only_registered_on_endpoint_is_collected_once() {
        // The registry tree is traversed once, so a callback registered on the
        // endpoint must not be collected again via its parents. Count
        // invocations rather than emitted series: the merger drops duplicate
        // series, so a double traversal would be invisible in the output.
        let drt = create_test_drt_async().await;
        let namespace = drt.namespace("ns_typed_ep_only").unwrap();
        let component = namespace.component("comp_typed_ep_only").unwrap();
        let endpoint = component.endpoint("ep_typed_ep_only");

        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = calls.clone();
        let callback: PrometheusTypedCallback = Arc::new(move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut family = prometheus::proto::MetricFamily::new();
            family.set_name("dynamo_component_active_decode_blocks".to_string());
            family.set_field_type(prometheus::proto::MetricType::GAUGE);
            let mut metric = prometheus::proto::Metric::new();
            let mut label = prometheus::proto::LabelPair::new();
            label.set_name("dp_rank".to_string());
            label.set_value("0".to_string());
            metric.set_label(vec![label]);
            metric.set_gauge(prometheus::proto::Gauge::new());
            family.mut_metric().push(metric);
            Ok(vec![crate::metrics::prom_typed::BuiltFamily {
                family,
                unit: String::new(),
                prom_type: "gauge".to_string(),
            }])
        });

        endpoint.get_metrics_registry().add_typed_callback(callback);

        // Typed callbacks feed the OTLP collection, not the exposition path.
        let families = drt.get_metrics_registry().metric_families_combined();

        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "endpoint-registered typed callback should be collected once"
        );
        assert!(
            families
                .families
                .iter()
                .any(|f| f.name() == "dynamo_component_active_decode_blocks"),
            "typed family missing from the combined collection"
        );
    }

    #[tokio::test]
    async fn test_recursive_namespace() {
        // Create a distributed runtime for testing
        let drt = create_test_drt_async().await;

        // Create a deeply chained namespace: ns1.ns2.ns3
        let ns1 = drt.namespace("ns1").unwrap();
        let ns2 = ns1.namespace("ns2").unwrap();
        let ns3 = ns2.namespace("ns3").unwrap();

        // Create a component in the deepest namespace
        let component = ns3.component("test-component").unwrap();

        // Verify the hierarchy structure
        assert_eq!(ns1.basename(), "ns1");
        assert_eq!(ns1.parent_hierarchies().len(), 1);
        assert_eq!(ns1.parent_hierarchies()[0].basename(), "");
        // ns1 hierarchy is just its basename since parent is empty

        assert_eq!(ns2.basename(), "ns2");
        assert_eq!(ns2.parent_hierarchies().len(), 2);
        assert_eq!(ns2.parent_hierarchies()[0].basename(), "");
        assert_eq!(ns2.parent_hierarchies()[1].basename(), "ns1");
        // ns2 hierarchy structure validated by parent assertions above

        assert_eq!(ns3.basename(), "ns3");
        assert_eq!(ns3.parent_hierarchies().len(), 3);
        assert_eq!(ns3.parent_hierarchies()[0].basename(), "");
        assert_eq!(ns3.parent_hierarchies()[1].basename(), "ns1");
        assert_eq!(ns3.parent_hierarchies()[2].basename(), "ns2");
        // ns3 hierarchy structure validated by parent assertions above

        assert_eq!(component.basename(), "test-component");
        assert_eq!(component.parent_hierarchies().len(), 4);
        assert_eq!(component.parent_hierarchies()[0].basename(), "");
        assert_eq!(component.parent_hierarchies()[1].basename(), "ns1");
        assert_eq!(component.parent_hierarchies()[2].basename(), "ns2");
        assert_eq!(component.parent_hierarchies()[3].basename(), "ns3");
        // component hierarchy structure validated by parent assertions above

        println!("✓ Chained namespace test passed - all prefixes correct");
    }
}

#[cfg(feature = "integration")]
mod test_metricsregistry_prometheus_fmt_outputs {
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use crate::metrics::prometheus_names::name_prefix;
    use crate::metrics::*;
    use prometheus::Counter;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_prometheusfactory_using_metrics_registry_trait() {
        // Setup real DRT and registry using the test-friendly constructor
        let drt = create_test_drt_async().await;

        // Use a simple constant namespace name
        let namespace_name = "ns345";

        let namespace = drt.namespace(namespace_name).unwrap();
        let component = namespace.component("comp345").unwrap();
        let endpoint = component.endpoint("ep345");

        // Test Counter creation
        let counter = endpoint
            .metrics()
            .create_counter("testcounter", "A test counter", &[])
            .unwrap();
        counter.inc_by(123.456789);
        let epsilon = 0.01;
        assert!((counter.get() - 123.456789).abs() < epsilon);

        let endpoint_output_raw = endpoint.metrics().prometheus_expfmt().unwrap();
        println!("Endpoint output:");
        println!("{}", endpoint_output_raw);

        // worker_id is runtime-generated (etcd lease ID), so we grab it from the DRT
        // and inject it into expected strings via the inject_worker_id helper.
        let wid = format!("{:x}", drt.connection_id());
        use super::test_helpers::inject_worker_id;

        let expected_endpoint_output = inject_worker_id(
            r#"# HELP dynamo_component_testcounter A test counter
# TYPE dynamo_component_testcounter counter
dynamo_component_testcounter{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345"} 123.456789"#,
            &wid,
        );

        assert_eq!(
            endpoint_output_raw.trim_end_matches('\n'),
            expected_endpoint_output.trim_end_matches('\n'),
            "\n=== ENDPOINT COMPARISON FAILED ===\n\
             Actual:\n{}\n\
             Expected:\n{}\n\
             ==============================",
            endpoint_output_raw,
            expected_endpoint_output
        );

        // Test Gauge creation
        let gauge = component
            .metrics()
            .create_gauge("testgauge", "A test gauge", &[])
            .unwrap();
        gauge.set(50000.0);
        assert_eq!(gauge.get(), 50000.0);

        // Test Prometheus format output for Component (gauge + histogram)
        let component_output_raw = component.metrics().prometheus_expfmt().unwrap();
        println!("Component output:");
        println!("{}", component_output_raw);

        let expected_component_output = inject_worker_id(
            r#"# HELP dynamo_component_testcounter A test counter
# TYPE dynamo_component_testcounter counter
dynamo_component_testcounter{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345"} 123.456789
# HELP dynamo_component_testgauge A test gauge
# TYPE dynamo_component_testgauge gauge
dynamo_component_testgauge{dynamo_component="comp345",dynamo_namespace="ns345"} 50000"#,
            &wid,
        );

        assert_eq!(
            component_output_raw.trim_end_matches('\n'),
            expected_component_output.trim_end_matches('\n'),
            "\n=== COMPONENT COMPARISON FAILED ===\n\
             Actual:\n{}\n\
             Expected:\n{}\n\
             ==============================",
            component_output_raw,
            expected_component_output
        );

        let intcounter = namespace
            .metrics()
            .create_intcounter("testintcounter", "A test int counter", &[])
            .unwrap();
        intcounter.inc_by(12345);
        assert_eq!(intcounter.get(), 12345);

        // Test Prometheus format output for Namespace (int_counter + gauge + histogram)
        let namespace_output_raw = namespace.metrics().prometheus_expfmt().unwrap();
        println!("Namespace output:");
        println!("{}", namespace_output_raw);

        let expected_namespace_output = inject_worker_id(
            r#"# HELP dynamo_component_testcounter A test counter
# TYPE dynamo_component_testcounter counter
dynamo_component_testcounter{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345"} 123.456789
# HELP dynamo_component_testgauge A test gauge
# TYPE dynamo_component_testgauge gauge
dynamo_component_testgauge{dynamo_component="comp345",dynamo_namespace="ns345"} 50000
# HELP dynamo_component_testintcounter A test int counter
# TYPE dynamo_component_testintcounter counter
dynamo_component_testintcounter{dynamo_namespace="ns345"} 12345"#,
            &wid,
        );

        assert_eq!(
            namespace_output_raw.trim_end_matches('\n'),
            expected_namespace_output.trim_end_matches('\n'),
            "\n=== NAMESPACE COMPARISON FAILED ===\n\
             Actual:\n{}\n\
             Expected:\n{}\n\
             ==============================",
            namespace_output_raw,
            expected_namespace_output
        );

        // Test IntGauge creation
        let intgauge = namespace
            .metrics()
            .create_intgauge("testintgauge", "A test int gauge", &[])
            .unwrap();
        intgauge.set(42);
        assert_eq!(intgauge.get(), 42);

        // Test IntGaugeVec creation
        let intgaugevec = namespace
            .metrics()
            .create_intgaugevec(
                "testintgaugevec",
                "A test int gauge vector",
                &["instance", "status"],
                &[("service", "api")],
            )
            .unwrap();
        intgaugevec
            .with_label_values(&["server1", "active"])
            .set(10);
        intgaugevec
            .with_label_values(&["server2", "inactive"])
            .set(0);

        // Test CounterVec creation
        let countervec = endpoint
            .metrics()
            .create_countervec(
                "testcountervec",
                "A test counter vector",
                &["method", "status"],
                &[("service", "api")],
            )
            .unwrap();
        countervec.with_label_values(&["GET", "200"]).inc_by(10.0);
        countervec.with_label_values(&["POST", "201"]).inc_by(5.0);

        // Test Histogram creation
        let histogram = component
            .metrics()
            .create_histogram("testhistogram", "A test histogram", &[], None)
            .unwrap();
        histogram.observe(1.0);
        histogram.observe(2.5);
        histogram.observe(4.0);

        // Test Prometheus format output for DRT (all metrics combined)
        let drt_output_raw = drt.metrics().prometheus_expfmt().unwrap();
        println!("DRT output:");
        println!("{}", drt_output_raw);

        // The uptime_seconds value is dynamic (depends on elapsed wall-clock time),
        // so we check all other lines exactly and validate uptime separately.
        let expected_drt_output_without_uptime = inject_worker_id(
            r#"# HELP dynamo_component_testcounter A test counter
# TYPE dynamo_component_testcounter counter
dynamo_component_testcounter{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345"} 123.456789
# HELP dynamo_component_testcountervec A test counter vector
# TYPE dynamo_component_testcountervec counter
dynamo_component_testcountervec{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345",method="GET",service="api",status="200"} 10
dynamo_component_testcountervec{dynamo_component="comp345",dynamo_endpoint="ep345",dynamo_namespace="ns345",method="POST",service="api",status="201"} 5
# HELP dynamo_component_testgauge A test gauge
# TYPE dynamo_component_testgauge gauge
dynamo_component_testgauge{dynamo_component="comp345",dynamo_namespace="ns345"} 50000
# HELP dynamo_component_testhistogram A test histogram
# TYPE dynamo_component_testhistogram histogram
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.005"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.01"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.025"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.05"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.1"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.25"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="0.5"} 0
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="1"} 1
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="2.5"} 2
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="5"} 3
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="10"} 3
dynamo_component_testhistogram_bucket{dynamo_component="comp345",dynamo_namespace="ns345",le="+Inf"} 3
dynamo_component_testhistogram_sum{dynamo_component="comp345",dynamo_namespace="ns345"} 7.5
dynamo_component_testhistogram_count{dynamo_component="comp345",dynamo_namespace="ns345"} 3
# HELP dynamo_component_testintcounter A test int counter
# TYPE dynamo_component_testintcounter counter
dynamo_component_testintcounter{dynamo_namespace="ns345"} 12345
# HELP dynamo_component_testintgauge A test int gauge
# TYPE dynamo_component_testintgauge gauge
dynamo_component_testintgauge{dynamo_namespace="ns345"} 42
# HELP dynamo_component_testintgaugevec A test int gauge vector
# TYPE dynamo_component_testintgaugevec gauge
dynamo_component_testintgaugevec{dynamo_namespace="ns345",instance="server1",service="api",status="active"} 10
dynamo_component_testintgaugevec{dynamo_namespace="ns345",instance="server2",service="api",status="inactive"} 0"#,
            &wid,
        );

        // Split actual output into non-uptime lines and validate the uptime value line.
        // The uptime metric now carries a worker_id label, so we match on the metric name
        // prefix and extract the value as the last whitespace-delimited token.
        // `dynamo_metrics_merge_conflicts_total` is process-wide, so other tests
        // in this binary move it: check that it is present, not its value.
        let mut non_uptime_lines = Vec::new();
        let mut saw_uptime_value = false;
        let mut saw_merge_conflicts = false;
        for line in drt_output_raw.trim_end_matches('\n').lines() {
            if line.contains("dynamo_metrics_merge_conflicts_total") {
                saw_merge_conflicts |=
                    line == "# TYPE dynamo_metrics_merge_conflicts_total counter";
            } else if line.starts_with("dynamo_component_uptime_seconds") && !line.starts_with('#')
            {
                let val_str = line.split_whitespace().last().unwrap();
                val_str.parse::<f64>().expect("uptime should be a float");
                saw_uptime_value = true;
            } else if line.starts_with("# HELP dynamo_component_uptime_seconds")
                || line.starts_with("# TYPE dynamo_component_uptime_seconds")
            {
                // Skip HELP/TYPE lines for uptime (we just verify it exists via the value)
            } else {
                non_uptime_lines.push(line);
            }
        }
        assert!(
            saw_uptime_value,
            "uptime_seconds metric should be present in initial scrape"
        );
        assert!(
            saw_merge_conflicts,
            "the DRT registry must export dynamo_metrics_merge_conflicts_total"
        );

        let actual_without_uptime = non_uptime_lines.join("\n");
        assert_eq!(
            actual_without_uptime,
            expected_drt_output_without_uptime.trim_end_matches('\n'),
            "\n=== DRT COMPARISON FAILED (excluding uptime) ===\n\
             Expected:\n{}\n\
             Actual:\n{}\n\
             ==============================",
            expected_drt_output_without_uptime,
            actual_without_uptime
        );

        // Wait briefly so the uptime gauge is clearly positive on the next scrape.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let drt_output_after = drt.metrics().prometheus_expfmt().unwrap();
        let uptime_line = drt_output_after
            .lines()
            .find(|l| l.starts_with("dynamo_component_uptime_seconds") && !l.starts_with('#'))
            .expect("uptime_seconds metric should be present after sleep");
        let uptime_after: f64 = uptime_line
            .split_whitespace()
            .last()
            .unwrap()
            .parse()
            .expect("uptime should be a float");
        assert!(
            uptime_after > 0.0,
            "uptime_seconds should be > 0 after 10ms sleep, got {}",
            uptime_after
        );

        println!("✓ All Prometheus format outputs verified successfully!");
    }

    #[tokio::test]
    async fn test_same_metric_name_different_endpoints() {
        // Test that the same metric name can exist in different endpoints without collision.
        // This validates the multi-registry approach: each endpoint has its own registry,
        // and metrics are merged at scrape time with distinct labels.
        let drt = create_test_drt_async().await;
        let namespace = drt.namespace("ns_test").unwrap();
        let component = namespace.component("comp_test").unwrap();

        // Create two endpoints with the same metric name
        let ep1 = component.endpoint("ep1");
        let ep2 = component.endpoint("ep2");

        let counter1 = ep1
            .metrics()
            .create_counter("requests_total", "Total requests", &[])
            .unwrap();
        counter1.inc_by(100.0);

        let counter2 = ep2
            .metrics()
            .create_counter("requests_total", "Total requests", &[])
            .unwrap();
        counter2.inc_by(200.0);

        // Get merged Prometheus output from component level
        let output = component.metrics().prometheus_expfmt().unwrap();

        let wid = format!("{:x}", drt.connection_id());
        use super::test_helpers::inject_worker_id;

        let expected_output = inject_worker_id(
            r#"# HELP dynamo_component_requests_total Total requests
# TYPE dynamo_component_requests_total counter
dynamo_component_requests_total{dynamo_component="comp_test",dynamo_endpoint="ep1",dynamo_namespace="ns_test"} 100
dynamo_component_requests_total{dynamo_component="comp_test",dynamo_endpoint="ep2",dynamo_namespace="ns_test"} 200"#,
            &wid,
        );

        assert_eq!(
            output.trim_end_matches('\n'),
            expected_output.trim_end_matches('\n'),
            "\n=== MULTI-REGISTRY COMPARISON FAILED ===\n\
             Actual:\n{}\n\
             Expected:\n{}\n\
             ==============================",
            output,
            expected_output
        );

        println!("✓ Multi-registry prevents Prometheus collisions!");
    }

    #[tokio::test]
    async fn test_duplicate_series_warning() {
        // Test that duplicate series (same metric name + same labels) are detected and deduplicated.
        // This should log a warning and keep only one of the duplicate series.
        let drt = create_test_drt_async().await;
        let namespace = drt.namespace("ns_dup").unwrap();
        let component = namespace.component("comp_dup").unwrap();

        // Create two endpoints with counters that will have identical labels when scraped
        let ep1 = component.endpoint("ep_same");
        let ep2 = component.endpoint("ep_same"); // Same endpoint name = duplicate labels

        let counter1 = ep1
            .metrics()
            .create_counter("dup_metric", "Duplicate metric test", &[])
            .unwrap();
        counter1.inc_by(50.0);

        let counter2 = ep2
            .metrics()
            .create_counter("dup_metric", "Duplicate metric test", &[])
            .unwrap();
        counter2.inc_by(75.0);

        // Get merged output - duplicates should be deduplicated
        let output = component.metrics().prometheus_expfmt().unwrap();

        let wid = format!("{:x}", drt.connection_id());
        use super::test_helpers::inject_worker_id;

        let expected_output = inject_worker_id(
            r#"# HELP dynamo_component_dup_metric Duplicate metric test
# TYPE dynamo_component_dup_metric counter
dynamo_component_dup_metric{dynamo_component="comp_dup",dynamo_endpoint="ep_same",dynamo_namespace="ns_dup"} 50"#,
            &wid,
        );

        assert_eq!(
            output.trim_end_matches('\n'),
            expected_output.trim_end_matches('\n'),
            "\n=== DEDUPLICATION COMPARISON FAILED ===\n\
             Actual:\n{}\n\
             Expected:\n{}\n\
             ==============================",
            output,
            expected_output
        );

        println!("✓ Duplicate series detection and deduplication works!");
    }
}
