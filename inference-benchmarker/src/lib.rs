use std::collections::HashMap;
use std::fs::File;
// Removed unused import
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

pub use crate::benchmark::{BenchmarkConfig, BenchmarkKind};
use crate::benchmark::{Event, MessageEvent};
pub use crate::console::run_console;
pub use crate::profiles::apply_profile;
use crate::requests::OpenAITextGenerationBackend;
pub use crate::requests::TokenizeOptions;
use chrono::Local;
// Removed crossterm dependency
use log::{debug, error, info, warn, Level, LevelFilter};
use reqwest::Url;
use tokenizers::{FromPretrainedParameters, Tokenizer};
use tokio::sync::broadcast::Sender;
use tokio::sync::Mutex;
use writers::BenchmarkReportWriter;

mod benchmark;
mod console;
mod executors;
mod profiles;
mod requests;
mod results;
mod scheduler;
mod table;
mod writers;

pub struct RunConfiguration {
    pub url: Url,
    pub api_key: String,
    pub tokenizer_name: String,
    pub profile: Option<String>,
    pub max_vus: u64,
    pub duration: std::time::Duration,
    pub rates: Option<Vec<f64>>,
    pub num_rates: u64,
    pub benchmark_kind: String,
    pub warmup_duration: std::time::Duration,
    pub prompt_options: Option<TokenizeOptions>,
    pub decode_options: Option<TokenizeOptions>,
    pub dataset: String,
    pub dataset_file: String,
    pub hf_token: Option<String>,
    pub extra_metadata: Option<HashMap<String, String>>,
    pub model_name: String,
    pub run_id: String,
}

pub async fn run(mut run_config: RunConfiguration, stop_sender: Sender<()>) -> anyhow::Result<()> {
    info!("Starting benchmark");
    // set process system limits
    sysinfo::set_open_files_limit(0);
    // apply profile if needed
    run_config = match run_config.profile.clone() {
        None => run_config,
        Some(profile) => match apply_profile(profile.as_str(), run_config) {
            Ok(config) => {
                info!("Profile applied: {}", profile);
                config
            }
            Err(e) => {
                error!("Failed to apply profile: {:?}", e);
                return Err(e);
            }
        },
    };
    // initialize tokenizer
    let params = FromPretrainedParameters {
        token: run_config.hf_token.clone(),
        ..Default::default()
    };
    let tokenizer =
        match Tokenizer::from_pretrained(run_config.tokenizer_name.clone(), Some(params)) {
            Ok(tokenizer) => tokenizer,
            Err(e) => {
                return Err(anyhow::anyhow!("Error loading tokenizer: {e}"));
            }
        };
    let tokenizer = Arc::new(tokenizer);
    let backend = OpenAITextGenerationBackend::try_new(
        run_config.api_key,
        run_config.url,
        run_config.model_name.clone(),
        tokenizer,
        run_config.duration,
    )?;

    let config = BenchmarkConfig {
        max_vus: run_config.max_vus,
        duration: run_config.duration,
        benchmark_kind: match run_config.benchmark_kind.to_lowercase().as_str() {
            "throughput" => BenchmarkKind::Throughput,
            "sweep" => BenchmarkKind::Sweep,
            "csweep" => BenchmarkKind::ConcurrencySweep,
            "rate" => BenchmarkKind::Rate,
            _ => BenchmarkKind::Sweep,
        },
        warmup_duration: run_config.warmup_duration,
        rates: run_config.rates,
        num_rates: run_config.num_rates,
        prompt_options: run_config.prompt_options.clone(),
        decode_options: run_config.decode_options.clone(),
        tokenizer: run_config.tokenizer_name.clone(),
        model_name: run_config.model_name.clone(),
        profile: run_config.profile.clone(),
        extra_metadata: run_config.extra_metadata.clone(),
        run_id: run_config.run_id.clone(),
    };
    config.validate()?;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    // Always use console interface, send logs to file for debugging
    let target = Box::new(File::create("log.txt").expect("Can't create file"));
    env_logger::Builder::new()
        .target(env_logger::Target::Pipe(target))
        .filter(Some("inference_benchmarker"), LevelFilter::Debug)
        .format(|buf, record| {
            writeln!(
                buf,
                "[{} {} {}:{}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                record.level(),
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args()
            )
        })
        .init();
    let config_clone = config.clone();
    let mut stop_receiver = stop_sender.subscribe();
    let stop_sender_clone = stop_sender.clone();
    let ui_thread = tokio::spawn(async move {
        tokio::select! {
            _ = stop_receiver.recv() => {
                debug!("Received stop signal, stopping benchmark");
            }
            _ = async{
                run_console(config_clone, rx, stop_sender_clone).await;
            } => {}
        }
    });

    // download prompts dataset
    info!("Downloading dataset");
    let _ = tx.send(Event::Message(MessageEvent {
        message: "Downloading dataset".to_string(),
        timestamp: chrono::Utc::now(),
        level: Level::Info,
    }));
    let filepath = requests::ConversationTextRequestGenerator::download_dataset(
        run_config.dataset,
        run_config.dataset_file,
        run_config.hf_token.clone(),
    )
    .expect("Can't download dataset");
    let requests = requests::ConversationTextRequestGenerator::load(
        filepath,
        run_config.tokenizer_name.clone(),
        run_config.prompt_options,
        run_config.decode_options,
        run_config.hf_token,
    )?;

    let mut benchmark = benchmark::Benchmark::new(
        config.clone(),
        Box::new(backend),
        Arc::from(Mutex::from(requests)),
        tx.clone(),
        stop_sender.clone(),
    );
    let mut stop_receiver = stop_sender.subscribe();
    tokio::select! {
        report = benchmark.run() => {
            match report {
                Ok(_) => {
                    let report = benchmark.get_report();
                    let path = format!("results/{}_{}.json",run_config.tokenizer_name.replace("/","_").replace(".","_"), chrono::Utc::now().format("%Y-%m-%d-%H-%M-%S"));
                    let path=Path::new(&path);
                    let writer=BenchmarkReportWriter::try_new(config.clone(), report)?;
                    writer.json(path).await?;
                    info!("Report saved to {:?}",path);
                    let _ = tx.send(Event::BenchmarkReportEnd(format!("{:?}", path)));
                },
                Err(e) => {
                    error!("Error running benchmark: {:?}", e.to_string());
                    let _ = tx.send(Event::BenchmarkError(e.to_string()));
                }
            };
        }
        _ = stop_receiver.recv() => {
            debug!("Received stop signal, stopping benchmark");
        }
    }
    info!("Benchmark finished");
    // Always keep the console interface running
    ui_thread.await?;

    // No need to revert terminal since we're not using TUI anymore

    let report = benchmark.get_report();
    match BenchmarkReportWriter::try_new(config.clone(), report) {
        Ok(writer) => {
            writer.stdout().await?;
        }
        Err(_) => {
            warn!("No results to report.");
        }
    };

    Ok(())
}
