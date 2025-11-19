use crate::requests::{TextGenerationBackend, TextRequestGenerator, TokenizeOptions};
use crate::results::{BenchmarkReport, BenchmarkResults};
use crate::scheduler::{ExecutorType, SchedulerProgress};
use crate::{executors, scheduler};
use log::{debug, info, warn};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{broadcast, mpsc, Mutex};

const THROUGHPUT_BUDGET: f64 = 1.2; // sweep up to 120% of max throughput

#[derive(Clone, Debug, strum_macros::Display, Serialize)]
pub enum BenchmarkKind {
    Throughput,
    Sweep,
    ConcurrencySweep,
    Rate,
}

pub struct MessageEvent {
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: log::Level,
}

pub struct BenchmarkEvent {
    pub id: String,
    pub scheduler_type: ExecutorType,
    pub request_throughput: Option<f64>,
    pub progress: f64,
    pub results: Option<BenchmarkResults>,
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

pub enum Event {
    BenchmarkStart(BenchmarkEvent),
    BenchmarkProgress(BenchmarkEvent),
    BenchmarkEnd(BenchmarkEvent),
    Message(MessageEvent),
    BenchmarkReportEnd(String),
    BenchmarkError(String),
}

pub struct Benchmark {
    start_time: Option<tokio::time::Instant>,
    end_time: Option<tokio::time::Instant>,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    report: BenchmarkReport,
    pub(crate) config: BenchmarkConfig,
    event_bus: mpsc::UnboundedSender<Event>,
    stop_sender: broadcast::Sender<()>,
}

#[serde_with::serde_as]
#[derive(Clone, Serialize)]
pub struct BenchmarkConfig {
    pub max_vus: u64,
    #[serde(rename = "duration_secs")]
    #[serde_as(as = "serde_with::DurationSeconds<u64>")]
    pub duration: Duration,
    pub benchmark_kind: BenchmarkKind,
    #[serde(rename = "warmup_duration_secs")]
    #[serde_as(as = "serde_with::DurationSeconds<u64>")]
    pub warmup_duration: Duration,
    pub rates: Option<Vec<f64>>,
    pub num_rates: u64,
    pub prompt_options: Option<TokenizeOptions>,
    pub decode_options: Option<TokenizeOptions>,
    pub tokenizer: String,
    pub model_name: String,
    pub profile: Option<String>,
    #[serde(rename = "meta")]
    pub extra_metadata: Option<HashMap<String, String>>,
    pub run_id: String,
}

impl BenchmarkConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.max_vus == 0 {
            return Err(anyhow::anyhow!("max_vus must be greater than 0"));
        }
        if self.duration.as_secs() == 0 {
            return Err(anyhow::anyhow!("duration must be greater than 0"));
        }
        if self.warmup_duration.as_secs() == 0 {
            return Err(anyhow::anyhow!("warmup_duration must be greater than 0"));
        }
        match self.benchmark_kind {
            BenchmarkKind::Throughput => {
                if self.rates.is_some() {
                    return Err(anyhow::anyhow!(
                        "rates must not be specified for throughput benchmark"
                    ));
                }
            }
            BenchmarkKind::Sweep => {
                if self.rates.is_some() {
                    return Err(anyhow::anyhow!(
                        "rates must not be specified for sweep benchmark"
                    ));
                }
            }
            BenchmarkKind::ConcurrencySweep => {
                if self.rates.is_some() {
                    return Err(anyhow::anyhow!(
                        "rates must not be specified for concurrency_sweep benchmark"
                    ));
                }
            }
            BenchmarkKind::Rate => {
                if self.rates.is_none() {
                    return Err(anyhow::anyhow!(
                        "rates must be specified for rate benchmark"
                    ));
                }
            }
        }
        Ok(())
    }
}

pub struct BenchmarkProgress {
    id: String,
    progress: SchedulerProgress,
}

impl Benchmark {
    pub fn new(
        config: BenchmarkConfig,
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        event_bus: mpsc::UnboundedSender<Event>,
        stop_sender: broadcast::Sender<()>,
    ) -> Benchmark {
        Benchmark {
            start_time: None,
            end_time: None,
            report: BenchmarkReport::new(),
            config: config.clone(),
            backend,
            requests,
            event_bus,
            stop_sender,
        }
    }

    pub fn get_report(&self) -> BenchmarkReport {
        self.report.clone()
    }

    // Wait for UI update to complete
    async fn wait_for_ui_update(&self) {
        // Give UI some time to update display
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    pub async fn run(&mut self) -> anyhow::Result<BenchmarkReport> {
        self.start_time = Some(tokio::time::Instant::now());
        self.report.start();
        info!("Prewarming backend");
        self.warmup().await?;
        info!("Prewarm complete");
        match self.config.benchmark_kind {
            BenchmarkKind::Throughput => {
                self.run_throughput().await?;
            }
            BenchmarkKind::Sweep => {
                self.run_sweep().await?;
            }
            BenchmarkKind::ConcurrencySweep => {
                self.run_concurrency_sweep().await?;
            }
            BenchmarkKind::Rate => {
                self.run_rates().await?;
            }
        }
        self.end_time = Some(tokio::time::Instant::now());
        self.event_bus.send(Event::Message(MessageEvent {
            message: format!(
                "Benchmark complete in {:?}",
                self.duration().expect("duration exists")
            ),
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }))?;
        self.report.end();
        Ok(self.report.clone())
    }

    pub fn duration(&self) -> Option<std::time::Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    async fn handle_progress(&self, id: String) -> Sender<Option<SchedulerProgress>> {
        let (tx, mut rx): (
            Sender<Option<SchedulerProgress>>,
            Receiver<Option<SchedulerProgress>>,
        ) = mpsc::channel(8);
        let event_bus = self.event_bus.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    None => {
                        break;
                    }
                    Some(progress) => {
                        let progress_evt = BenchmarkProgress {
                            id: id.clone(),
                            progress,
                        };
                        let _ = event_bus.send(Event::BenchmarkProgress(BenchmarkEvent {
                            id: progress_evt.id,
                            scheduler_type: ExecutorType::ConstantVUs,
                            request_throughput: Some(progress_evt.progress.requests_throughput),
                            progress: progress_evt.progress.progress,
                            successful_requests: progress_evt.progress.successful_requests,
                            failed_requests: progress_evt.progress.failed_requests,
                            results: None,
                            avg_ttft_ms: progress_evt.progress.avg_ttft_ms,
                            avg_tpot_ms: progress_evt.progress.avg_tpot_ms,
                            ttft_std_ms: progress_evt.progress.ttft_std_ms,
                            tpot_std_ms: progress_evt.progress.tpot_std_ms,
                            input_throughput: progress_evt.progress.input_throughput,
                            output_throughput: progress_evt.progress.output_throughput,
                            total_throughput: progress_evt.progress.total_throughput,
                            sent_requests: progress_evt.progress.sent_requests,
                            in_flight_requests: progress_evt.progress.in_flight_requests,
                            completed_requests: progress_evt.progress.completed_requests,
                        }));
                    }
                }
            }
        });
        tx
    }

    pub async fn warmup(&mut self) -> anyhow::Result<()> {
        // run a warmup benchmark to prewarm the server

        let id = "warmup".to_string();

        // notify start event
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: id.to_string(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: None,
            progress: 0.0,
            results: None,
            successful_requests: 0,
            failed_requests: 0,
            avg_ttft_ms: None,
            avg_tpot_ms: None,
            ttft_std_ms: None,
            tpot_std_ms: None,
            input_throughput: None,
            output_throughput: None,
            total_throughput: None,
            sent_requests: 0,
            in_flight_requests: 0,
            completed_requests: 0,
        }))?;

        // create progress handler
        let tx = self.handle_progress(id.clone()).await;

        // start scheduler
        let mut scheduler = scheduler::Scheduler::new(
            id,
            self.backend.clone(),
            ExecutorType::ConstantVUs,
            executors::ExecutorConfig {
                max_vus: 1,
                duration: self.config.warmup_duration,
                rate: None,
            },
            self.requests.clone(),
            tx.clone(),
            self.stop_sender.clone(),
        );
        scheduler.run().await?;

        let results = scheduler.get_results().lock().await.clone();
        self.report.add_benchmark_result(results.clone());

        // send None to close the progress handler
        tx.send(None).await.unwrap();

        // Get final request status
        let (sent_requests, in_flight_requests, completed_requests) =
            scheduler.get_final_request_status().await;

        // notify end event
        self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
            id: "warmup".to_string(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: results.successful_request_rate().ok(),
            progress: 100.0,
            results: Some(results.clone()),
            successful_requests: results.successful_requests() as u64,
            failed_requests: results.failed_requests() as u64,
            avg_ttft_ms: results
                .time_to_first_token_avg()
                .ok()
                .map(|d| d.as_millis() as f64),
            avg_tpot_ms: results
                .time_per_output_token_avg()
                .ok()
                .map(|d| d.as_millis() as f64),
            ttft_std_ms: results
                .time_to_first_token_std()
                .ok()
                .map(|d| d.as_millis() as f64),
            tpot_std_ms: results
                .time_per_output_token_std()
                .ok()
                .map(|d| d.as_millis() as f64),
            input_throughput: results.input_token_throughput_secs().ok(),
            output_throughput: results.output_token_throughput_secs().ok(),
            total_throughput: results.total_token_throughput_secs().ok(),
            sent_requests,
            in_flight_requests,
            completed_requests,
        }))?;
        Ok(())
    }

    pub async fn run_throughput(&mut self) -> anyhow::Result<()> {
        info!("Running throughput benchmark");

        let id = "throughput".to_string();

        // notify start event
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: id.clone(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: None,
            progress: 0.0,
            results: None,
            successful_requests: 0,
            failed_requests: 0,
            avg_ttft_ms: None,
            avg_tpot_ms: None,
            ttft_std_ms: None,
            tpot_std_ms: None,
            input_throughput: None,
            output_throughput: None,
            total_throughput: None,
            sent_requests: 0,
            in_flight_requests: 0,
            completed_requests: 0,
        }))?;

        // create progress handler
        let tx = self.handle_progress(id.clone()).await;

        // start scheduler
        let mut scheduler = scheduler::Scheduler::new(
            id.clone(),
            self.backend.clone(),
            ExecutorType::ConstantVUs,
            executors::ExecutorConfig {
                max_vus: self.config.max_vus,
                duration: self.config.duration,
                rate: None,
            },
            self.requests.clone(),
            tx.clone(),
            self.stop_sender.clone(),
        );
        scheduler.run().await?;
        let results = scheduler.get_results().lock().await.clone();
        let rate = results.successful_request_rate().ok();
        self.report.add_benchmark_result(results.clone());

        // send None to close the progress handler
        tx.send(None).await.unwrap();

        // Get final request status
        let (sent_requests, in_flight_requests, completed_requests) =
            scheduler.get_final_request_status().await;

        // notify end event
        self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
            id: id.clone(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: rate,
            progress: 100.0,
            results: Some(results.clone()),
            successful_requests: results.successful_requests() as u64,
            failed_requests: results.failed_requests() as u64,
            avg_ttft_ms: results
                .time_to_first_token_avg()
                .ok()
                .map(|d| d.as_millis() as f64),
            avg_tpot_ms: results
                .time_per_output_token_avg()
                .ok()
                .map(|d| d.as_millis() as f64),
            ttft_std_ms: results
                .time_to_first_token_std()
                .ok()
                .map(|d| d.as_millis() as f64),
            tpot_std_ms: results
                .time_per_output_token_std()
                .ok()
                .map(|d| d.as_millis() as f64),
            input_throughput: results.input_token_throughput_secs().ok(),
            output_throughput: results.output_token_throughput_secs().ok(),
            total_throughput: results.total_token_throughput_secs().ok(),
            sent_requests,
            in_flight_requests,
            completed_requests,
        }))?;
        Ok(())
    }

    pub async fn run_sweep(&mut self) -> anyhow::Result<()> {
        // run a throughput benchmark to retrieve the maximum throughput of server
        self.run_throughput().await?;
        // get the max throughput from the second benchmark result (first is warmup)
        let throughput_results = &self.report.get_results()[1];
        let max_throughput = throughput_results.successful_request_rate()?;
        let max_tokens_throughput = throughput_results.token_throughput_secs()?;
        // notify event bus
        self.event_bus.send(Event::Message(MessageEvent {
            message: format!(
                "Max throughput detected at: {:.2} req/s | {:.2} tokens/s",
                max_throughput, max_tokens_throughput
            ),
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }))?;
        // run a sweep benchmark for 10 different rates from 1req/s to max throughput
        let mut rates = Vec::new();
        let num_rates = self.config.num_rates;
        for i in 1..=num_rates {
            rates.push(i as f64 * max_throughput * THROUGHPUT_BUDGET / num_rates as f64);
        }
        for rate in rates {
            self.run_rate(rate).await?;
        }
        Ok(())
    }

    pub async fn run_rates(&mut self) -> anyhow::Result<()> {
        let rates = self.config.rates.clone().expect("config already validated");
        for rate in rates {
            self.run_rate(rate).await?;
        }
        Ok(())
    }

    pub async fn run_concurrency_sweep(&mut self) -> anyhow::Result<()> {
        info!("Running concurrency sweep benchmark");

        // Find optimal concurrency level
        let optimal_concurrency = self.find_optimal_concurrency().await?;

        // Analyze concurrency sweep results and generate report
        self.analyze_concurrency_sweep_results(optimal_concurrency)
            .await?;

        Ok(())
    }

    async fn analyze_concurrency_sweep_results(
        &mut self,
        optimal_concurrency: u64,
    ) -> anyhow::Result<()> {
        info!("Analyzing concurrency sweep results");

        let results = self.report.get_results();

        // Collect concurrency test results
        let mut concurrency_results = Vec::new();

        for result in results {
            if result.id.starts_with("concurrency#") {
                concurrency_results.push(result);
            }
        }

        if concurrency_results.is_empty() {
            warn!("No concurrency test results found for analysis");
            return Ok(());
        }

        // Analyze concurrency test results
        let mut concurrency_throughput_map = std::collections::HashMap::new();
        let mut best_throughput = 0.0;

        for result in &concurrency_results {
            if let Ok(throughput) = result.successful_request_rate() {
                let concurrency = result
                    .id
                    .strip_prefix("concurrency#")
                    .and_then(|s: &str| s.strip_suffix("vus"))
                    .and_then(|s: &str| s.parse::<u64>().ok())
                    .unwrap_or(1);

                concurrency_throughput_map.insert(concurrency, throughput);

                // Find throughput corresponding to optimal concurrency
                if concurrency == optimal_concurrency {
                    best_throughput = throughput;
                }
            }
        }

        // Generate concurrency analysis report
        let mut concurrency_analysis = format!(
            "\n=== Concurrency Sweep Analysis Report ===\n\
            Tested concurrency levels: {}\n\
            Optimal concurrency: {} VUs\n\
            Maximum throughput: {:.2} req/s\n\n\
            Concurrency Level vs Throughput:\n",
            concurrency_throughput_map.len(),
            optimal_concurrency,
            best_throughput
        );

        let mut sorted_concurrency: Vec<_> = concurrency_throughput_map.iter().collect();
        sorted_concurrency.sort_by_key(|(concurrency, _)| **concurrency);

        for (concurrency, throughput) in &sorted_concurrency {
            let efficiency = if **concurrency > 0 {
                **throughput / **concurrency as f64
            } else {
                0.0
            };
            concurrency_analysis.push_str(&format!(
                "  {} VUs: {:.2} req/s (efficiency: {:.3} req/s/VU)\n",
                concurrency, throughput, efficiency
            ));
        }

        // Analyze throughput trend
        if concurrency_throughput_map.len() > 1 {
            let throughputs: Vec<f64> = sorted_concurrency.iter().map(|(_, t)| **t).collect();
            let mut increasing = true;
            let mut decreasing = true;

            for i in 1..throughputs.len() {
                if throughputs[i] <= throughputs[i - 1] {
                    increasing = false;
                }
                if throughputs[i] >= throughputs[i - 1] {
                    decreasing = false;
                }
            }

            concurrency_analysis.push_str("\nThroughput Trend Analysis:\n");
            if increasing {
                concurrency_analysis
                    .push_str("  - Throughput continues to increase with higher concurrency\n");
                concurrency_analysis.push_str(
                    "  - Recommendation: Try higher concurrency levels for better throughput\n",
                );
            } else if decreasing {
                concurrency_analysis.push_str("  - Throughput decreases with higher concurrency\n");
                concurrency_analysis.push_str(
                    "  - Recommendation: System may have reached performance bottleneck\n",
                );
            } else {
                concurrency_analysis.push_str("  - Throughput trend is complex\n");
                concurrency_analysis.push_str("  - Recommendation: Current optimal concurrency may be near system's best performance point\n");
            }
        }

        // Generate recommendations
        concurrency_analysis.push_str("\n=== Recommendations ===\n");
        concurrency_analysis.push_str(&format!(
            "1. Recommended gateway concurrency limit: {} concurrent requests\n",
            optimal_concurrency
        ));
        concurrency_analysis.push_str(&format!(
            "2. Expected maximum throughput: {:.2} req/s\n",
            best_throughput
        ));

        if optimal_concurrency > 1 {
            let efficiency = best_throughput / optimal_concurrency as f64;
            concurrency_analysis.push_str(&format!(
                "3. Average efficiency per VU: {:.3} req/s/VU\n",
                efficiency
            ));
        }

        // Send analysis results to event bus
        self.event_bus.send(Event::Message(MessageEvent {
            message: concurrency_analysis,
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }))?;

        info!("Concurrency sweep analysis completed");
        Ok(())
    }

    async fn test_concurrency_level(&mut self, concurrency: u64) -> anyhow::Result<f64> {
        debug!("Testing concurrency level: {}", concurrency);

        let id = format!("concurrency#{}vus", concurrency);

        // notify start event
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: id.clone(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: None,
            progress: 0.0,
            results: None,
            successful_requests: 0,
            failed_requests: 0,
            avg_ttft_ms: None,
            avg_tpot_ms: None,
            ttft_std_ms: None,
            tpot_std_ms: None,
            input_throughput: None,
            output_throughput: None,
            total_throughput: None,
            sent_requests: 0,
            in_flight_requests: 0,
            completed_requests: 0,
        }))?;

        // create progress handler
        let tx = self.handle_progress(id.clone()).await;

        // Use user-configured duration parameter
        let test_duration = self.config.duration;

        info!(
            "Testing concurrency {} with duration {:?}",
            concurrency, test_duration
        );

        // start scheduler with specific concurrency level
        let mut scheduler = scheduler::Scheduler::new(
            id.clone(),
            self.backend.clone(),
            ExecutorType::ConstantVUs,
            executors::ExecutorConfig {
                max_vus: concurrency,
                duration: test_duration,
                rate: None,
            },
            self.requests.clone(),
            tx.clone(),
            self.stop_sender.clone(),
        );

        let start_time = std::time::Instant::now();
        scheduler.run().await?;
        let elapsed = start_time.elapsed();
        let results = scheduler.get_results().lock().await.clone();

        info!(
            "Concurrency {} test completed in {:?}",
            concurrency, elapsed
        );
        info!(
            "Results: {} successful, {} failed, {} total requests",
            results.successful_requests(),
            results.failed_requests(),
            results.successful_requests() + results.failed_requests()
        );

        // Check if there are any successful requests
        if results.successful_requests() == 0 {
            warn!(
                "No successful requests for concurrency {} after {:?}, this might indicate:",
                concurrency, elapsed
            );
            warn!("1. Backend is not responding");
            warn!("2. Test duration is too short for requests to complete");
            warn!("3. Backend is overloaded or has issues");
            return Ok(0.0); // Return 0 throughput instead of error
        }

        let throughput = results.successful_request_rate()?;

        self.report.add_benchmark_result(results.clone());

        // send None to close the progress handler
        tx.send(None).await.unwrap();

        // Get final request status
        let (sent_requests, in_flight_requests, completed_requests) =
            scheduler.get_final_request_status().await;

        // notify end event
        self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
            id: id.clone(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: Some(throughput),
            progress: 100.0,
            results: Some(results.clone()),
            successful_requests: results.successful_requests() as u64,
            failed_requests: results.failed_requests() as u64,
            avg_ttft_ms: results
                .time_to_first_token_avg()
                .ok()
                .map(|d| d.as_millis() as f64),
            avg_tpot_ms: results
                .time_per_output_token_avg()
                .ok()
                .map(|d| d.as_millis() as f64),
            ttft_std_ms: results
                .time_to_first_token_std()
                .ok()
                .map(|d| d.as_millis() as f64),
            tpot_std_ms: results
                .time_per_output_token_std()
                .ok()
                .map(|d| d.as_millis() as f64),
            input_throughput: results.input_token_throughput_secs().ok(),
            output_throughput: results.output_token_throughput_secs().ok(),
            total_throughput: results.total_token_throughput_secs().ok(),
            sent_requests,
            in_flight_requests,
            completed_requests,
        }))?;

        // Wait for UI update to complete
        self.wait_for_ui_update().await;

        Ok(throughput)
    }

    async fn find_optimal_concurrency(&mut self) -> anyhow::Result<u64> {
        info!("Finding optimal concurrency level");

        let max_concurrency = self.config.max_vus;
        let mut best_concurrency = 1;
        let mut best_throughput = 0.0;

        // Generate concurrency test sequence: 1, 2, 5, 10, 20, 50, 100, 200, ...
        let mut concurrency_levels = Vec::new();
        concurrency_levels.push(1);

        let mut level = 2;
        while level <= max_concurrency {
            concurrency_levels.push(level);
            if level < 10 {
                level += 1;
            } else if level < 50 {
                level += 5;
            } else if level < 100 {
                level += 10;
            } else {
                level += 20;
            }
        }

        // Ensure maximum concurrency is included
        if !concurrency_levels.contains(&max_concurrency) {
            concurrency_levels.push(max_concurrency);
        }
        concurrency_levels.sort();
        concurrency_levels.dedup();

        info!("Testing concurrency levels: {:?}", concurrency_levels);

        for concurrency in concurrency_levels {
            let throughput = self.test_concurrency_level(concurrency).await?;

            info!("Concurrency {}: {:.2} req/s", concurrency, throughput);

            if throughput > best_throughput {
                best_throughput = throughput;
                best_concurrency = concurrency;
            }

            // If no successful requests at low concurrency levels, backend might have issues
            if concurrency <= 5 && throughput == 0.0 {
                warn!(
                    "No successful requests at low concurrency {}, checking backend connectivity",
                    concurrency
                );
            }

            // If throughput starts declining, can stop early (optional optimization)
            if concurrency > 10 && throughput < best_throughput * 0.9 {
                warn!(
                    "Throughput declining, stopping early at concurrency {}",
                    concurrency
                );
                break;
            }
        }

        // If no successful requests found across all tests, return error
        if best_throughput == 0.0 {
            return Err(anyhow::anyhow!(
                "No successful requests found across all concurrency levels. Please check:\n\
                1. Backend is running and accessible\n\
                2. Backend URL is correct\n\
                3. API key is valid (if required)\n\
                4. Test duration is long enough for requests to complete"
            ));
        }

        info!(
            "Optimal concurrency: {} (throughput: {:.2} req/s)",
            best_concurrency, best_throughput
        );

        // Notify event bus
        self.event_bus.send(Event::Message(MessageEvent {
            message: format!(
                "Optimal concurrency found: {} VUs (throughput: {:.2} req/s)",
                best_concurrency, best_throughput
            ),
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }))?;

        Ok(best_concurrency)
    }

    pub async fn run_rate(&mut self, rate: f64) -> anyhow::Result<()> {
        debug!("Running benchmark with rate: {} req/s", rate);

        let id = format!("rate@{:.1}reqs", rate);

        // notify start event
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: id.clone(),
            scheduler_type: ExecutorType::ConstantArrivalRate,
            request_throughput: None,
            progress: 0.0,
            results: None,
            successful_requests: 0,
            failed_requests: 0,
            avg_ttft_ms: None,
            avg_tpot_ms: None,
            ttft_std_ms: None,
            tpot_std_ms: None,
            input_throughput: None,
            output_throughput: None,
            total_throughput: None,
            sent_requests: 0,
            in_flight_requests: 0,
            completed_requests: 0,
        }))?;

        // create progress handler
        let tx = self.handle_progress(id.clone()).await;

        // start scheduler
        let mut scheduler = scheduler::Scheduler::new(
            id,
            self.backend.clone(),
            scheduler::ExecutorType::ConstantArrivalRate,
            executors::ExecutorConfig {
                max_vus: self.config.max_vus,
                duration: self.config.duration,
                rate: Some(rate),
            },
            self.requests.clone(),
            tx.clone(),
            self.stop_sender.clone(),
        );
        scheduler.run().await?;
        let results = scheduler.get_results().lock().await.clone();
        self.report.add_benchmark_result(results.clone());

        // send None to close the progress handler
        tx.send(None).await.unwrap();

        // Get final request status
        let (sent_requests, in_flight_requests, completed_requests) =
            scheduler.get_final_request_status().await;

        // notify end event
        self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
            id: format!("rate@{:.1}reqs", rate),
            scheduler_type: ExecutorType::ConstantArrivalRate,
            request_throughput: results.successful_request_rate().ok(),
            progress: 100.0,
            results: Some(results.clone()),
            successful_requests: results.successful_requests() as u64,
            failed_requests: results.failed_requests() as u64,
            avg_ttft_ms: results
                .time_to_first_token_avg()
                .ok()
                .map(|d| d.as_millis() as f64),
            avg_tpot_ms: results
                .time_per_output_token_avg()
                .ok()
                .map(|d| d.as_millis() as f64),
            ttft_std_ms: results
                .time_to_first_token_std()
                .ok()
                .map(|d| d.as_millis() as f64),
            tpot_std_ms: results
                .time_per_output_token_std()
                .ok()
                .map(|d| d.as_millis() as f64),
            input_throughput: results.input_token_throughput_secs().ok(),
            output_throughput: results.output_token_throughput_secs().ok(),
            total_throughput: results.total_token_throughput_secs().ok(),
            sent_requests,
            in_flight_requests,
            completed_requests,
        }))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requests::DummyTextGenerationBackend;
    use crate::requests::DummyTextRequestGenerator;
    use std::time::Duration;

    #[tokio::test]
    async fn test_sweep_benchmark_timings() {
        let generation_time = Duration::from_secs(2);
        let (event_tx, mut _event_rx) = tokio::sync::mpsc::unbounded_channel();
        let (stop_sender, _) = tokio::sync::broadcast::channel(1);
        let backend = Box::new(DummyTextGenerationBackend::new(Duration::from_secs(
            generation_time.as_secs(),
        )));
        let requests_generator = Arc::from(Mutex::from(DummyTextRequestGenerator::new()));
        let mut benchmark = Benchmark::new(
            BenchmarkConfig {
                max_vus: 100,
                duration: Duration::from_secs(10),
                benchmark_kind: BenchmarkKind::Sweep,
                warmup_duration: Duration::from_secs(1),
                rates: None,
                num_rates: 2,
                prompt_options: None,
                decode_options: None,
                tokenizer: "gpt2".to_string(),
                model_name: "gpt2".to_string(),
                profile: None,
                extra_metadata: None,
                run_id: "test".to_string(),
            },
            backend,
            requests_generator,
            event_tx,
            stop_sender,
        );
        let report = benchmark.run().await.unwrap();
        assert_eq!(report.get_results().len(), 4);
        let generation_time_per_token_milli = generation_time.as_millis() as i128 / 10;
        for result in report.get_results() {
            let delta_ttft = result.time_to_first_token_avg().unwrap().as_millis() as i128
                - generation_time_per_token_milli; // Dummy backends generates 10 tokens
            let delta_itl = result.inter_token_latency_avg().unwrap().as_millis() as i128
                - generation_time_per_token_milli;
            let delta_e2e = result.e2e_latency_avg().unwrap().as_millis() as i128
                - generation_time.as_millis() as i128;
            let allowed_error_ms = 3; // allow error margin for timing tests
            assert!(
                delta_ttft.abs() <= allowed_error_ms,
                "time_to_first_token_delta: {:?}, expected {:?}",
                delta_ttft.abs(),
                allowed_error_ms
            );
            assert!(
                delta_itl.abs() <= allowed_error_ms,
                "inter_token_latency_delta: {:?}, expected {:?}",
                delta_itl.abs(),
                allowed_error_ms
            );
            assert!(
                delta_e2e.abs() <= allowed_error_ms * 10, // Cumulative error for 10 tokens
                "e2e_latency_delta: {:?}, expected {:?}",
                delta_e2e.abs(),
                allowed_error_ms * 10
            );
        }
    }
}
