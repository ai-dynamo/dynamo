use crate::executors::{
    ConstantArrivalRateExecutor, ConstantVUsExecutor, Executor, ExecutorConfig,
};
use crate::requests::{
    TextGenerationAggregatedResponse, TextGenerationBackend, TextRequestGenerator,
};
use crate::results::BenchmarkErrors::NoResponses;
use crate::results::BenchmarkResults;
use log::{debug, trace, warn};
use std::sync::Arc;
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use tokio::sync::{broadcast, Mutex};

#[derive(Clone, strum_macros::Display)]
pub enum ExecutorType {
    ConstantVUs,
    ConstantArrivalRate,
}

pub struct Scheduler {
    id: String,
    executor: Arc<Mutex<dyn Executor + Send>>,
    requests_generator: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    results: Arc<Mutex<BenchmarkResults>>,
    progress_tx: Sender<Option<SchedulerProgress>>,
    stop_sender: broadcast::Sender<()>,
}

pub struct SchedulerProgress {
    pub progress: f64,
    pub requests_throughput: f64,
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

impl Scheduler {
    pub fn new(
        id: String,
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        executor_type: ExecutorType,
        config: ExecutorConfig,
        requests_generator: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        progress_tx: Sender<Option<SchedulerProgress>>,
        stop_sender: broadcast::Sender<()>,
    ) -> Scheduler {
        match executor_type {
            ExecutorType::ConstantVUs => Scheduler {
                id: id.clone(),
                executor: Arc::from(Mutex::from(ConstantVUsExecutor::new(
                    backend.clone(),
                    config.max_vus,
                    config.duration,
                ))),
                results: Arc::from(Mutex::from(BenchmarkResults::new(
                    id.clone(),
                    ExecutorType::ConstantVUs,
                    config,
                ))),
                requests_generator,
                progress_tx,
                stop_sender,
            },
            ExecutorType::ConstantArrivalRate => {
                if config.rate.is_none() {
                    panic!("Rate must be specified for ConstantArrivalRateExecutor");
                }
                let rate = config.rate.unwrap();
                Scheduler {
                    id: id.clone(),
                    executor: Arc::from(Mutex::from(ConstantArrivalRateExecutor::new(
                        backend.clone(),
                        config.max_vus,
                        config.duration,
                        rate,
                    ))),
                    results: Arc::from(Mutex::from(BenchmarkResults::new(
                        id.clone(),
                        ExecutorType::ConstantArrivalRate,
                        config,
                    ))),
                    requests_generator,
                    progress_tx,
                    stop_sender,
                }
            }
        }
    }

    pub async fn run(&mut self) -> anyhow::Result<BenchmarkResults> {
        debug!("Starting scheduler '{}'", self.id);
        // add responses to the benchmark result as they arrive
        let (tx, mut rx): (
            UnboundedSender<TextGenerationAggregatedResponse>,
            UnboundedReceiver<TextGenerationAggregatedResponse>,
        ) = tokio::sync::mpsc::unbounded_channel();
        let results = self.results.clone();
        let progress_tx = self.progress_tx.clone();
        let mut stop_receiver = self.stop_sender.subscribe();
        let req_gen = self.requests_generator.clone();

        // Request status tracking
        let sent_requests = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let in_flight_requests = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let completed_requests = Arc::new(std::sync::atomic::AtomicU64::new(0));

        // Clone for the executor
        let sent_requests_for_executor = sent_requests.clone();
        let in_flight_requests_for_executor = in_flight_requests.clone();

        let response_handler = tokio::spawn(async move {
            tokio::select! {
                _ = stop_receiver.recv() => {
                    debug!("Received stop signal, stopping benchmark");
                }
                _ = async{
                    while let Some(response) = rx.recv().await{
                        // call generator callback
                        let response_txt=response.response.clone();
                        if let Some(request)= response.request.clone(){
                            req_gen.lock().await.callback(request, response_txt.unwrap_or_default().as_str());
                        }
                        let result = results.clone();
                        let progress_tx = progress_tx.clone();
                        trace!("Received response: {:?}", response);
                        let response_ended = response.ended;
                        let mut result = result.lock().await;
                        result.add_response(response);
                        let expected_duration = result.executor_config().duration.as_secs_f64();
                        let start_time = result.start_time().unwrap_or(tokio::time::Instant::now());
                        // Calculate real-time TTFT and TPOT with standard deviation
                        let avg_ttft_ms = result.time_to_first_token_avg().ok().map(|d| d.as_millis() as f64);
                        let avg_tpot_ms = result.time_per_output_token_avg().ok().map(|d| d.as_millis() as f64);
                        let ttft_std_ms = result.time_to_first_token_std().ok().map(|d| d.as_millis() as f64);
                        let tpot_std_ms = result.time_per_output_token_std().ok().map(|d| d.as_millis() as f64);

                        // Calculate throughput metrics
                        let input_throughput = result.input_token_throughput_secs().ok();
                        let output_throughput = result.output_token_throughput_secs().ok();
                        let total_throughput = result.total_token_throughput_secs().ok();

                        // Update request status - only count as completed if the response has ended
                        if response_ended {
                            completed_requests.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            in_flight_requests.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                        }

                        let current_sent = sent_requests.load(std::sync::atomic::Ordering::SeqCst);
                        let current_in_flight = in_flight_requests.load(std::sync::atomic::Ordering::SeqCst);
                        let current_completed = completed_requests.load(std::sync::atomic::Ordering::SeqCst);

                        let _ = progress_tx.send(Some(SchedulerProgress {
                            progress: (100.0 * (1.0 - (expected_duration - start_time.elapsed().as_secs_f64()) / expected_duration)).min(100.0),
                            requests_throughput: result.successful_request_rate().unwrap_or_default(),
                            successful_requests: result.successful_requests() as u64,
                            failed_requests: result.failed_requests() as u64,
                            avg_ttft_ms,
                            avg_tpot_ms,
                            ttft_std_ms,
                            tpot_std_ms,
                            input_throughput,
                            output_throughput,
                            total_throughput,
                            sent_requests: current_sent,
                            in_flight_requests: current_in_flight,
                            completed_requests: current_completed,
                        })).await;
                    }
                }=>{}
            }
        });
        self.executor
            .lock()
            .await
            .run(
                self.requests_generator.clone(),
                tx,
                self.stop_sender.clone(),
                Some(sent_requests_for_executor),
                Some(in_flight_requests_for_executor),
            )
            .await;

        // Wait for the response handler to finish processing all responses
        let _ = response_handler.await;

        warn!("{:?}", self.results.clone());
        if self.results.lock().await.successful_requests() == 0 {
            Err(anyhow::anyhow!(NoResponses))
        } else {
            Ok(self.results.lock().await.clone())
        }
    }

    pub fn get_results(&self) -> Arc<Mutex<BenchmarkResults>> {
        self.results.clone()
    }

    pub async fn get_final_request_status(&self) -> (u64, u64, u64) {
        // This method would need to be implemented to return the final status
        // For now, we'll use the results to calculate the final status
        let results = self.results.lock().await;
        let successful = results.successful_requests() as u64;
        let failed = results.failed_requests() as u64;
        let total = successful + failed;
        (total, 0, total) // (sent, in_flight, completed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requests::OpenAITextGenerationBackend;
    use std::time::Duration;
    use tokenizers::Tokenizer;
    use tokio::time;

    #[tokio::test]
    async fn test_constant_arrival_rate_scheduler() {
        let (progress_tx, _) = tokio::sync::mpsc::channel(10000);
        let (stop_sender, _) = tokio::sync::broadcast::channel(1);
        let backend = Box::new(crate::requests::DummyTextGenerationBackend::new(
            Duration::from_secs(1),
        ));
        let requests_generator = Arc::from(Mutex::from(
            crate::requests::DummyTextRequestGenerator::new(),
        ));
        let mut scheduler = Scheduler::new(
            "test".to_string(),
            backend,
            ExecutorType::ConstantArrivalRate,
            ExecutorConfig {
                max_vus: 800,
                duration: std::time::Duration::from_secs(10),
                rate: Some(20.0),
            },
            requests_generator,
            progress_tx,
            stop_sender,
        );
        let results = scheduler.run().await.unwrap();
        assert_eq!(results.successful_requests(), 180); // 20 requests per second for 10 seconds - 20 requests for last second as the backend has a 1 second delay
    }

    #[tokio::test]
    async fn test_constant_vus_scheduler() {
        let (progress_tx, _) = tokio::sync::mpsc::channel(10000);
        let (stop_sender, _) = broadcast::channel(1);
        let backend = Box::new(crate::requests::DummyTextGenerationBackend::new(
            Duration::from_secs(1),
        ));
        let requests_generator = Arc::from(Mutex::from(
            crate::requests::DummyTextRequestGenerator::new(),
        ));
        let mut scheduler = Scheduler::new(
            "test".to_string(),
            backend,
            ExecutorType::ConstantVUs,
            ExecutorConfig {
                max_vus: 800,
                duration: Duration::from_secs(10),
                rate: None,
            },
            requests_generator,
            progress_tx,
            stop_sender,
        );
        let results = scheduler.run().await.unwrap();
        assert!(
            results.successful_requests() > 7200,
            "Expected at least 7200 requests, got {}",
            results.successful_requests()
        );
    }

    #[tokio::test]
    async fn test_constant_arrival_rate_openai_backend() {
        let (progress_tx, _) = tokio::sync::mpsc::channel(10000);
        let (stop_sender, _) = tokio::sync::broadcast::channel(1);
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| {
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                std::thread::sleep(Duration::from_millis(500));
                w.write_all(b"data: {\"choices\": [{\"message\": {\"content\": \"Hello, world!Hello, world!Hello, world!Hello, world!\", \"role\": \"user\"}, \"finish_reason\": \"stop\", \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: [DONE]\n\n")
            })
            .create_async().await;
        let url = s.url().parse().unwrap();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let requests_generator = Arc::from(Mutex::from(
            crate::requests::DummyTextRequestGenerator::new(),
        ));
        let mut scheduler = Scheduler::new(
            "test".to_string(),
            Box::new(backend),
            ExecutorType::ConstantArrivalRate,
            ExecutorConfig {
                max_vus: 800,
                duration: Duration::from_secs(10),
                rate: Some(50.0),
            },
            requests_generator,
            progress_tx,
            stop_sender,
        );
        let results = scheduler.run().await.unwrap();
        assert_eq!(results.successful_requests(), 475); // 25 expected missing requests due to the 500ms delay in the backend
    }
}
