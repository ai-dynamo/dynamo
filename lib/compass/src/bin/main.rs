// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use dynamo_compass::collector::mock::generate_mock_input;
use dynamo_compass::config::{CompassConfig, WeightProfile, COMPASS_VERSION};
use dynamo_compass::engine::run_attribution;
use dynamo_compass::report::{human, json};
use dynamo_compass::sensitivity::{
    counterfactual::{find_minimum_improvement, format_counterfactual},
    sweep::{parse_sweep_spec, run_sensitivity_sweep},
};
use dynamo_compass::types::{
    AttributionReport, CalibrationResult, ComponentDelta, DiffReport, SweepConfig,
};

#[derive(Parser)]
#[command(name = "compass", version = COMPASS_VERSION, about = "System-level bottleneck attribution for Dynamo")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run bottleneck attribution on a deployment
    Diagnose {
        /// Deployment name
        #[arg(long, default_value = "default")]
        deployment: String,
        /// Analysis window in minutes
        #[arg(long, default_value_t = 15)]
        window: i64,
        /// Prometheus endpoint URL
        #[arg(long)]
        prometheus_url: Option<String>,
        /// Use mock data for demo/development
        #[arg(long)]
        mock: bool,
        /// Output human-readable format instead of JSON
        #[arg(long)]
        human: bool,
        /// Weight profile: default, conservative, aggressive
        #[arg(long, default_value = "default")]
        weights: String,
        /// Write report to file
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// Run counterfactual analysis against SLO target
        #[arg(long)]
        slo_ttft_p99_ms: Option<f64>,
    },

    /// Run sensitivity sweep using mocker-based perturbation
    Replay {
        /// Path to recorded trace file
        #[arg(long)]
        trace: Option<PathBuf>,
        /// Use mock data
        #[arg(long)]
        mock: bool,
        /// Sweep specs: component=multiplier1,multiplier2,... (repeatable)
        #[arg(long)]
        sweep: Vec<String>,
        /// Concurrency levels to test
        #[arg(long, value_delimiter = ',', default_values_t = vec![16, 32, 64])]
        concurrency: Vec<usize>,
        /// Output human-readable format
        #[arg(long)]
        human: bool,
        /// Write results to file
        #[arg(long, short)]
        output: Option<PathBuf>,
    },

    /// Compare two attribution reports side-by-side
    Diff {
        /// Path to first report
        #[arg(long)]
        report_a: PathBuf,
        /// Path to second report
        #[arg(long)]
        report_b: PathBuf,
        /// Output human-readable format
        #[arg(long)]
        human: bool,
    },

    /// Calibrate mocker predictions against real measurements
    Calibrate {
        /// Path to recorded trace
        #[arg(long)]
        trace: PathBuf,
        /// Path to real-run measurements
        #[arg(long)]
        real_run: PathBuf,
        /// Residual threshold percentage
        #[arg(long, default_value_t = 15.0)]
        threshold: f64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Diagnose {
            deployment,
            window,
            prometheus_url: _,
            mock,
            human: human_output,
            weights,
            output,
            slo_ttft_p99_ms,
        } => cmd_diagnose(
            &deployment,
            window,
            mock,
            human_output,
            &weights,
            output,
            slo_ttft_p99_ms,
        ),
        Commands::Replay {
            trace: _,
            mock: _,
            sweep,
            concurrency,
            human: human_output,
            output,
        } => cmd_replay(sweep, concurrency, human_output, output),
        Commands::Diff {
            report_a,
            report_b,
            human: human_output,
        } => cmd_diff(report_a, report_b, human_output),
        Commands::Calibrate {
            trace,
            real_run,
            threshold,
        } => cmd_calibrate(trace, real_run, threshold),
    }
}

fn cmd_diagnose(
    deployment: &str,
    window: i64,
    mock: bool,
    human_output: bool,
    weights: &str,
    output: Option<PathBuf>,
    slo_ttft_p99_ms: Option<f64>,
) -> Result<()> {
    if !mock {
        anyhow::bail!(
            "Live Prometheus collection not yet implemented. Use --mock for demo mode."
        );
    }

    let mut config = CompassConfig::default();
    if let Some(wp) = WeightProfile::from_name(weights) {
        config.weights = wp;
    } else {
        anyhow::bail!("Unknown weight profile '{}'. Use: default, conservative, aggressive", weights);
    }

    let input = generate_mock_input(deployment, window);
    let report = run_attribution(input, &config);

    let formatted = if human_output {
        human::format_report(&report)
    } else {
        json::format_report(&report)?
    };

    if let Some(path) = output {
        std::fs::write(&path, &formatted)
            .with_context(|| format!("Failed to write report to {}", path.display()))?;
        eprintln!("Report written to {}", path.display());
    } else {
        println!("{formatted}");
    }

    if let Some(slo) = slo_ttft_p99_ms {
        println!("\n--- Counterfactual Analysis (SLO: {slo:.0}ms) ---\n");
        let ttft_p99 = report.end_to_end.ttft_ms.p99;
        for comp in &report.per_component {
            let label = match &comp.sub_component {
                Some(sub) => format!("{}.{}", comp.component, sub),
                None => comp.component.clone(),
            };
            let cf = find_minimum_improvement(&label, ttft_p99, slo, comp.contribution_pct);
            println!("  {}", format_counterfactual(&cf));
        }
    }

    Ok(())
}

fn cmd_replay(
    sweep_specs: Vec<String>,
    concurrency_levels: Vec<usize>,
    human_output: bool,
    output: Option<PathBuf>,
) -> Result<()> {
    if sweep_specs.is_empty() {
        anyhow::bail!("At least one --sweep spec required. Example: --sweep kvbm-allocate-ms=0.5,1.0,2.0");
    }

    let perturbations: Vec<_> = sweep_specs
        .iter()
        .map(|s| parse_sweep_spec(s))
        .collect::<Result<Vec<_>>>()?;

    let config = SweepConfig {
        trace_source: "mock".to_string(),
        perturbations,
        concurrency_levels,
    };

    let matrix = run_sensitivity_sweep(&config);

    let formatted = if human_output {
        human::format_sensitivity_matrix(&matrix)
    } else {
        json::format_sensitivity_matrix(&matrix)?
    };

    if let Some(path) = output {
        std::fs::write(&path, &formatted)
            .with_context(|| format!("Failed to write results to {}", path.display()))?;
        eprintln!("Results written to {}", path.display());
    } else {
        println!("{formatted}");
    }

    Ok(())
}

fn cmd_diff(report_a_path: PathBuf, report_b_path: PathBuf, human_output: bool) -> Result<()> {
    let a_json = std::fs::read_to_string(&report_a_path)
        .with_context(|| format!("Failed to read {}", report_a_path.display()))?;
    let b_json = std::fs::read_to_string(&report_b_path)
        .with_context(|| format!("Failed to read {}", report_b_path.display()))?;

    let a: AttributionReport = serde_json::from_str(&a_json)
        .with_context(|| format!("Failed to parse {}", report_a_path.display()))?;
    let b: AttributionReport = serde_json::from_str(&b_json)
        .with_context(|| format!("Failed to parse {}", report_b_path.display()))?;

    let verdict_changed = a.verdict.primary_bottleneck != b.verdict.primary_bottleneck;

    let mut component_deltas = Vec::new();
    for comp_a in &a.per_component {
        let label_a = match &comp_a.sub_component {
            Some(sub) => format!("{}.{}", comp_a.component, sub),
            None => comp_a.component.clone(),
        };
        let comp_b = b.per_component.iter().find(|c| {
            let label_b = match &c.sub_component {
                Some(sub) => format!("{}.{}", c.component, sub),
                None => c.component.clone(),
            };
            label_b == label_a
        });

        let (attr_b, lat_b) = comp_b
            .map(|c| (c.contribution_pct, c.latency_ms.p99))
            .unwrap_or((0.0, 0.0));

        component_deltas.push(ComponentDelta {
            component: label_a,
            attribution_pct_a: comp_a.contribution_pct,
            attribution_pct_b: attr_b,
            delta_pct: attr_b - comp_a.contribution_pct,
            latency_p99_a: comp_a.latency_ms.p99,
            latency_p99_b: lat_b,
            latency_delta_ms: lat_b - comp_a.latency_ms.p99,
        });
    }

    let diff = DiffReport {
        report_a: report_a_path.display().to_string(),
        report_b: report_b_path.display().to_string(),
        verdict_changed,
        component_deltas,
    };

    let formatted = if human_output {
        human::format_diff(&diff)
    } else {
        json::format_diff(&diff)?
    };

    println!("{formatted}");
    Ok(())
}

fn cmd_calibrate(trace_path: PathBuf, real_run_path: PathBuf, threshold: f64) -> Result<()> {
    let trace_json = std::fs::read_to_string(&trace_path)
        .with_context(|| format!("Failed to read {}", trace_path.display()))?;
    let real_json = std::fs::read_to_string(&real_run_path)
        .with_context(|| format!("Failed to read {}", real_run_path.display()))?;

    let trace_report: AttributionReport = serde_json::from_str(&trace_json)
        .with_context(|| format!("Failed to parse {}", trace_path.display()))?;
    let real_report: AttributionReport = serde_json::from_str(&real_json)
        .with_context(|| format!("Failed to parse {}", real_run_path.display()))?;

    let mocker_ttft_p99 = trace_report.end_to_end.ttft_ms.p99;
    let real_ttft_p99 = real_report.end_to_end.ttft_ms.p99;
    let residual_pct = ((mocker_ttft_p99 - real_ttft_p99) / real_ttft_p99 * 100.0).abs();

    let result = CalibrationResult {
        trace_source: trace_path.display().to_string(),
        mocker_ttft_p99_ms: mocker_ttft_p99,
        real_ttft_p99_ms: real_ttft_p99,
        residual_pct,
        is_calibrated: residual_pct <= threshold,
        threshold_pct: threshold,
    };

    let formatted = human::format_calibration(&result);
    println!("{formatted}");

    if !result.is_calibrated {
        eprintln!(
            "WARNING: Mocker residual ({:.1}%) exceeds threshold ({:.1}%). Predictions may be unreliable.",
            residual_pct, threshold
        );
    }

    Ok(())
}
