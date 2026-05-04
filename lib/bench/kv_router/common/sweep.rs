// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use plotters::prelude::*;

use super::BenchmarkResults;

pub fn plot_sweep(
    all_results: &[(&str, Vec<(u64, BenchmarkResults)>)],
    output_path: &str,
) -> anyhow::Result<()> {
    use plotters::coord::combinators::IntoLogRange;
    use plotters::element::DashedPathElement;
    use plotters::style::ShapeStyle;

    let colors = [
        RGBColor(31, 119, 180),
        RGBColor(255, 127, 14),
        RGBColor(44, 160, 44),
        RGBColor(214, 39, 40),
        RGBColor(148, 103, 189),
        RGBColor(140, 86, 75),
    ];

    let mut global_min = f64::MAX;
    let mut global_max = f64::MIN;
    for (_, results) in all_results {
        for (_, r) in results {
            let offered = r.offered_block_throughput as f64;
            let achieved = r.block_throughput as f64;
            global_min = global_min.min(offered).min(achieved);
            global_max = global_max.max(offered).max(achieved);
        }
    }
    let axis_min = global_min * 0.9;
    let axis_max = global_max * 1.1;

    let root = SVGBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Achieved vs Offered Throughput",
            ("sans-serif", 22).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(
            (axis_min..axis_max).log_scale(),
            (axis_min..axis_max).log_scale(),
        )?;

    chart
        .configure_mesh()
        .x_desc("Offered Throughput (block ops/s)")
        .y_desc("Achieved Throughput (block ops/s)")
        .draw()?;

    let identity_style = ShapeStyle::from(&BLACK.mix(0.4)).stroke_width(1);
    chart.draw_series(std::iter::once(DashedPathElement::new(
        vec![(axis_min, axis_min), (axis_max, axis_max)],
        5,
        3,
        identity_style,
    )))?;

    for (i, (name, results)) in all_results.iter().enumerate() {
        let color = &colors[i % colors.len()];

        let points: Vec<(f64, f64)> = results
            .iter()
            .map(|(_, r)| (r.offered_block_throughput as f64, r.block_throughput as f64))
            .collect();

        let series_color = *color;
        chart
            .draw_series(LineSeries::new(
                points.iter().map(|&(x, y)| (x, y)),
                &series_color,
            ))?
            .label(*name)
            .legend(move |(x, y)| {
                plotters::element::PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    series_color.stroke_width(2),
                )
            });

        chart.draw_series(
            points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 4, series_color.filled())),
        )?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Sweep plot saved to {}", output_path);
    Ok(())
}

/// Compute logarithmically spaced benchmark durations for sweep mode.
pub fn compute_sweep_durations(min_ms: u64, max_ms: u64, steps: usize) -> Vec<u64> {
    let log_min = (min_ms as f64).ln();
    let log_max = (max_ms as f64).ln();
    (0..steps)
        .map(|i| {
            let t = i as f64 / (steps - 1) as f64;
            (log_max * (1.0 - t) + log_min * t).exp().round() as u64
        })
        .collect()
}

/// Print a formatted sweep summary table.
pub fn print_sweep_summary(name: &str, results: &[(u64, BenchmarkResults)]) {
    println!("\n=== Sweep Summary: {} ===", name);
    println!(
        "{:>12} {:>14} {:>14} {:>14} {:>14} {:>10}",
        "duration_ms", "ops/s_off", "ops/s", "blk_ops/s_off", "blk_ops/s", "p99(us)"
    );
    for (dur, r) in results {
        println!(
            "{:>12} {:>14.1} {:>14.1} {:>14.1} {:>14.1} {:>10.1}",
            dur,
            r.offered_ops_throughput,
            r.ops_throughput,
            r.offered_block_throughput,
            r.block_throughput,
            r.latency_p99_us,
        );
    }
}

/// Compute median of durations.
pub fn median(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = durations.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}
