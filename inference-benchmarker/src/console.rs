use crate::benchmark::Event as BenchmarkEvent;
use crate::BenchmarkConfig;
use colored::*;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::broadcast;
use tokio::sync::mpsc::UnboundedReceiver;

#[derive(Clone)]
pub struct ConsoleState {
    pub benchmarks: Vec<BenchmarkInfo>,
    pub messages: Vec<LogMessage>,
}

#[derive(Clone)]
pub struct BenchmarkInfo {
    pub id: String,
    pub status: BenchmarkStatus,
    pub progress: f64,
    pub throughput: String,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_ttft_ms: Option<f64>,
    pub avg_tpot_ms: Option<f64>,
    pub ttft_std_ms: Option<f64>,
    pub tpot_std_ms: Option<f64>,
    pub input_throughput: Option<f64>,
    pub output_throughput: Option<f64>,
    pub total_throughput: Option<f64>,
    pub sent_requests: u64,
    pub in_flight_requests: u64,
    pub completed_requests: u64,
}

#[derive(Clone)]
pub struct LogMessage {
    pub message: String,
    pub level: LogLevel,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, strum_macros::Display)]
pub enum BenchmarkStatus {
    Running,
    Completed,
}

#[derive(Clone, strum_macros::Display)]
pub enum LogLevel {
    Info,
    Warning,
    Error,
}

impl ConsoleState {
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
            messages: Vec::new(),
        }
    }

    pub fn add_benchmark(&mut self, benchmark: BenchmarkInfo) {
        if let Some(existing) = self.benchmarks.iter_mut().find(|b| b.id == benchmark.id) {
            *existing = benchmark;
        } else {
            self.benchmarks.push(benchmark);
        }
    }

    pub fn add_message(&mut self, message: LogMessage) {
        self.messages.push(message);
        // Keep only last 50 messages to avoid memory issues
        if self.messages.len() > 50 {
            self.messages.remove(0);
        }
    }
}

pub async fn run_console(
    benchmark_config: BenchmarkConfig,
    mut receiver: UnboundedReceiver<BenchmarkEvent>,
    stop_sender: broadcast::Sender<()>,
) {
    let state = Arc::new(Mutex::new(ConsoleState::new()));
    let state_clone = state.clone();
    let mut stop_receiver = stop_sender.subscribe();

    // Print initial configuration
    print_config(&benchmark_config);

    // Start event processing task
    let event_task = tokio::spawn(async move {
        while let Some(event) = receiver.recv().await {
            match event {
                BenchmarkEvent::BenchmarkStart(event) => {
                    let benchmark = BenchmarkInfo {
                        id: event.id,
                        status: BenchmarkStatus::Running,
                        progress: 0.0,
                        throughput: "0".to_string(),
                        successful_requests: 0,
                        failed_requests: 0,
                        avg_ttft_ms: event.avg_ttft_ms,
                        avg_tpot_ms: event.avg_tpot_ms,
                        ttft_std_ms: event.ttft_std_ms,
                        tpot_std_ms: event.tpot_std_ms,
                        input_throughput: event.input_throughput,
                        output_throughput: event.output_throughput,
                        total_throughput: event.total_throughput,
                        sent_requests: event.sent_requests,
                        in_flight_requests: event.in_flight_requests,
                        completed_requests: event.completed_requests,
                    };
                    state_clone.lock().unwrap().add_benchmark(benchmark);
                    print_benchmark_update(&state_clone.lock().unwrap());
                }
                BenchmarkEvent::BenchmarkProgress(event) => {
                    let benchmark = BenchmarkInfo {
                        id: event.id,
                        status: BenchmarkStatus::Running,
                        progress: event.progress,
                        throughput: event
                            .request_throughput
                            .map_or("0".to_string(), |e| format!("{e:.2}")),
                        successful_requests: event.successful_requests,
                        failed_requests: event.failed_requests,
                        avg_ttft_ms: event.avg_ttft_ms,
                        avg_tpot_ms: event.avg_tpot_ms,
                        ttft_std_ms: event.ttft_std_ms,
                        tpot_std_ms: event.tpot_std_ms,
                        input_throughput: event.input_throughput,
                        output_throughput: event.output_throughput,
                        total_throughput: event.total_throughput,
                        sent_requests: event.sent_requests,
                        in_flight_requests: event.in_flight_requests,
                        completed_requests: event.completed_requests,
                    };
                    state_clone.lock().unwrap().add_benchmark(benchmark);
                    print_benchmark_update(&state_clone.lock().unwrap());
                }
                BenchmarkEvent::BenchmarkEnd(event) => {
                    let benchmark = BenchmarkInfo {
                        id: event.id,
                        status: BenchmarkStatus::Completed,
                        progress: 100.0,
                        throughput: event
                            .request_throughput
                            .map_or("0".to_string(), |e| format!("{e:.2}")),
                        successful_requests: event
                            .results
                            .as_ref()
                            .map_or(0, |r| r.successful_requests() as u64),
                        failed_requests: event
                            .results
                            .as_ref()
                            .map_or(0, |r| r.failed_requests() as u64),
                        avg_ttft_ms: event.avg_ttft_ms,
                        avg_tpot_ms: event.avg_tpot_ms,
                        ttft_std_ms: event.ttft_std_ms,
                        tpot_std_ms: event.tpot_std_ms,
                        input_throughput: event.input_throughput,
                        output_throughput: event.output_throughput,
                        total_throughput: event.total_throughput,
                        sent_requests: event.sent_requests,
                        in_flight_requests: event.in_flight_requests,
                        completed_requests: event.completed_requests,
                    };
                    state_clone.lock().unwrap().add_benchmark(benchmark);
                    print_benchmark_update(&state_clone.lock().unwrap());
                }
                BenchmarkEvent::Message(event) => {
                    let log_message = LogMessage {
                        message: event.message,
                        level: LogLevel::Info,
                        timestamp: event.timestamp,
                    };
                    state_clone.lock().unwrap().add_message(log_message);
                    print_latest_message(&state_clone.lock().unwrap());
                }
                BenchmarkEvent::BenchmarkReportEnd(path) => {
                    let log_message = LogMessage {
                        message: format!("Benchmark report saved to {}", path),
                        level: LogLevel::Info,
                        timestamp: chrono::Utc::now(),
                    };
                    state_clone.lock().unwrap().add_message(log_message);
                    print_latest_message(&state_clone.lock().unwrap());
                    break;
                }
                BenchmarkEvent::BenchmarkError(error) => {
                    let log_message = LogMessage {
                        message: format!("Error running benchmark: {}", error),
                        level: LogLevel::Error,
                        timestamp: chrono::Utc::now(),
                    };
                    state_clone.lock().unwrap().add_message(log_message);
                    print_latest_message(&state_clone.lock().unwrap());
                    break;
                }
            }
        }
    });

    // Wait for either the event task to complete or a stop signal
    tokio::select! {
        _ = event_task => {},
        _ = stop_receiver.recv() => {
            println!("{}", "Benchmark stopped by user".red().bold());
        }
    }
}

fn print_config(config: &BenchmarkConfig) {
    println!("{}", "=".repeat(80).blue());
    println!("{}", "🚀 INFERENCE BENCHMARKER".blue().bold());
    println!("{}", "=".repeat(80).blue());

    let rate_mode = match config.rates {
        None => "Automatic".to_string(),
        Some(_) => "Manual".to_string(),
    };

    println!("{}", format!("Profile: {} | Benchmark: {} | Max VUs: {} | Duration: {} sec | Rates: {} | Warmup: {} sec",
        config.profile.clone().unwrap_or("N/A".to_string()),
        config.benchmark_kind,
        config.max_vus,
        config.duration.as_secs_f64(),
        rate_mode,
        config.warmup_duration.as_secs_f64()
    ).cyan().bold());

    println!("{}", "=".repeat(80).blue());
    println!();
}

fn print_benchmark_update(state: &ConsoleState) {
    // Clear screen and move cursor to top
    print!("\x1B[2J\x1B[1;1H");

    // Print header
    println!("{}", "BENCHMARK PROGRESS".blue().bold());
    println!("{}", "-".repeat(80).blue());

    // Print benchmark table header
    println!(
        "{:<16} {:<10} {:<5} {:<5} {:<15} {:<18} {:<18} {:<12} {:<12} {:<12} {:<25}",
        "Bench".blue().bold(),
        "Status".blue().bold(),
        "%".blue().bold(),
        "Err".blue().bold(),
        "Throughput".blue().bold(),
        "TTFT".blue().bold(),
        "TPOT".blue().bold(),
        "Input".blue().bold(),
        "Output".blue().bold(),
        "Total".blue().bold(),
        "Requests(S|I|D)".blue().bold()
    );
    println!("{}", "-".repeat(80).blue());

    // Print benchmark data
    for benchmark in &state.benchmarks {
        let error_rate = if benchmark.failed_requests > 0 {
            format!(
                "{:4.0}%",
                benchmark.failed_requests as f64
                    / (benchmark.failed_requests + benchmark.successful_requests) as f64
                    * 100.0
            )
        } else {
            "0%".to_string()
        };

        let ttft_display = match (benchmark.avg_ttft_ms, benchmark.ttft_std_ms) {
            (Some(mean), Some(std)) => format!("{:.1}±{:.1}ms", mean, std),
            (Some(mean), None) => format!("{:.1}ms", mean),
            _ => "N/A".to_string(),
        };

        let tpot_display = match (benchmark.avg_tpot_ms, benchmark.tpot_std_ms) {
            (Some(mean), Some(std)) => format!("{:.1}±{:.1}ms", mean, std),
            (Some(mean), None) => format!("{:.1}ms", mean),
            _ => "N/A".to_string(),
        };

        let input_throughput = benchmark
            .input_throughput
            .map_or("N/A".to_string(), |t| format!("{:.1}", t));
        let output_throughput = benchmark
            .output_throughput
            .map_or("N/A".to_string(), |t| format!("{:.1}", t));
        let total_throughput = benchmark
            .total_throughput
            .map_or("N/A".to_string(), |t| format!("{:.1}", t));

        let request_status = format!(
            "{}|{}|{}",
            benchmark.sent_requests, benchmark.in_flight_requests, benchmark.completed_requests
        );

        let status_color = match benchmark.status {
            BenchmarkStatus::Running => benchmark.status.to_string().yellow().bold(),
            BenchmarkStatus::Completed => benchmark.status.to_string().green().bold(),
        };

        let error_color = if benchmark.failed_requests > 0 {
            error_rate.red().bold()
        } else {
            error_rate.green()
        };

        // Create a simple progress bar
        let progress_bar = create_progress_bar(benchmark.progress);

        println!(
            "{:<16} {:<10} {:<5} {:<5} {:<15} {:<18} {:<18} {:<12} {:<12} {:<12} {:<25}",
            benchmark.id.white(),
            status_color,
            format!("{:4.0}%", benchmark.progress).white(),
            error_color,
            format!("{} req/s", benchmark.throughput).green().bold(),
            ttft_display.cyan(),
            tpot_display.magenta(),
            format!("In: {}", input_throughput).blue(),
            format!("Out: {}", output_throughput).yellow(),
            format!("Total: {}", total_throughput).red(),
            request_status.white()
        );

        // Show progress bar for running benchmarks
        if matches!(benchmark.status, BenchmarkStatus::Running) && benchmark.progress < 100.0 {
            println!("  {} {}", "Progress:".dimmed(), progress_bar);
        }
    }

    println!();

    // Print recent messages
    if !state.messages.is_empty() {
        println!("{}", "RECENT MESSAGES".blue().bold());
        println!("{}", "-".repeat(80).blue());

        for message in state.messages.iter().rev().take(5) {
            let level_color = match message.level {
                LogLevel::Info => message.level.to_string().green().bold(),
                LogLevel::Warning => message.level.to_string().yellow().bold(),
                LogLevel::Error => message.level.to_string().red().bold(),
            };

            println!(
                "{} {} {}",
                message
                    .timestamp
                    .format("%H:%M:%S")
                    .to_string()
                    .bright_black(),
                level_color,
                message.message.white()
            );
        }
        println!();
    }

    io::stdout().flush().unwrap();
}

fn print_latest_message(state: &ConsoleState) {
    if let Some(message) = state.messages.last() {
        let level_color = match message.level {
            LogLevel::Info => message.level.to_string().green(),
            LogLevel::Warning => message.level.to_string().yellow(),
            LogLevel::Error => message.level.to_string().red(),
        };

        println!(
            "{} {} {}",
            message
                .timestamp
                .format("%H:%M:%S")
                .to_string()
                .bright_black(),
            level_color,
            message.message.white()
        );
    }
}

fn create_progress_bar(progress: f64) -> String {
    let width = 20;
    let filled = (progress / 100.0 * width as f64) as usize;
    let empty = width - filled;

    let filled_bar = "█".repeat(filled);
    let empty_bar = "░".repeat(empty);

    format!("[{}{}]", filled_bar.green(), empty_bar.dimmed())
}
