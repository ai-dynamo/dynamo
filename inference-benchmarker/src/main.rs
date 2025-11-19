use clap::error::ErrorKind::InvalidValue;
use clap::{ArgGroup, Error, Parser};
use inference_benchmarker::{run, RunConfiguration, TokenizeOptions};
use log::{debug, error};
use reqwest::Url;
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::broadcast;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None, group(ArgGroup::new("group_profile").multiple(true)),group(ArgGroup::new("group_manual").multiple(true).conflicts_with("group_profile"))
)]
struct Args {
    /// The name of the tokenizer to use
    #[clap(short, long, env)]
    tokenizer_name: String,

    /// The name of the model to use. If not provided, the same name as the tokenizer will be used.
    #[clap(long, env)]
    model_name: Option<String>,

    /// The maximum number of virtual users to use
    #[clap(default_value = "128", short, long, env, group = "group_manual")]
    max_vus: u64,
    /// The duration of each benchmark step
    #[clap(default_value = "120s", short, long, env, group = "group_manual")]
    #[arg(value_parser = parse_duration)]
    duration: Duration,
    /// A list of rates of requests to send per second (only valid for the ConstantArrivalRate benchmark).
    #[clap(short, long, env)]
    rates: Option<Vec<f64>>,
    /// The number of rates to sweep through (only valid for the "sweep" benchmark)
    /// The rates will be linearly spaced up to the detected maximum rate
    #[clap(default_value = "10", long, env)]
    num_rates: u64,
    /// A benchmark profile to use
    #[clap(long, env, group = "group_profile")]
    profile: Option<String>,
    /// The kind of benchmark to run (throughput, sweep, csweep, rate)
    #[clap(default_value = "sweep", short, long, env, group = "group_manual")]
    benchmark_kind: String,
    /// The duration of the prewarm step ran before the benchmark to warm up the backend (JIT, caches, etc.)
    #[clap(default_value = "30s", short, long, env, group = "group_manual")]
    #[arg(value_parser = parse_duration)]
    warmup: Duration,
    /// The URL of the backend to benchmark. Must be compatible with OpenAI Message API
    #[clap(default_value = "http://localhost:8000", short, long, env)]
    url: Url,

    /// The api key send to the [`url`] as Header "Authorization: Bearer {API_KEY}".
    #[clap(default_value = "", short, long, env)]
    api_key: String,

    /// Constraints for prompt length.
    /// No value means use the input prompt as defined in input dataset.
    /// We sample the number of tokens to generate from a normal distribution.
    /// Specified as a comma-separated list of key=value pairs.
    /// * num_tokens: target number of prompt tokens
    /// * min_tokens: minimum number of prompt tokens
    /// * max_tokens: maximum number of prompt tokens
    /// * variance: variance in the number of prompt tokens
    ///
    /// Example: num_tokens=200,max_tokens=210,min_tokens=190,variance=10
    #[clap(
        long,
        env,
        value_parser(parse_tokenizer_options),
        group = "group_manual"
    )]
    prompt_options: Option<TokenizeOptions>,
    /// Constraints for the generated text.
    /// We sample the number of tokens to generate from a normal distribution.
    /// Specified as a comma-separated list of key=value pairs.
    /// * num_tokens: target number of generated tokens
    /// * min_tokens: minimum number of generated tokens
    /// * max_tokens: maximum number of generated tokens
    /// * variance: variance in the number of generated tokens
    ///
    /// Example: num_tokens=200,max_tokens=210,min_tokens=190,variance=10
    #[clap(
        long,
        env,
        value_parser(parse_tokenizer_options),
        group = "group_manual"
    )]
    decode_options: Option<TokenizeOptions>,
    /// Hugging Face dataset to use for prompt generation
    #[clap(
        default_value = "hlarcher/inference-benchmarker",
        long,
        env,
        group = "group_manual"
    )]
    dataset: String,
    /// File to use in the Dataset
    #[clap(
        default_value = "share_gpt_filtered_small.json",
        long,
        env,
        group = "group_manual"
    )]
    dataset_file: String,
    /// Extra metadata to include in the benchmark results file, comma-separated key-value pairs.
    /// It can be, for example, used to include information about the configuration of the
    /// benched server.
    /// Example: --extra-meta "key1=value1,key2=value2"
    #[clap(long, env, value_parser(parse_key_val))]
    extra_meta: Option<HashMap<String, String>>,
    // A run identifier to use for the benchmark. This is used to identify the benchmark in the
    // results file.
    #[clap(long, env)]
    run_id: Option<String>,
}

fn parse_duration(s: &str) -> Result<Duration, Error> {
    humantime::parse_duration(s).map_err(|_| Error::new(InvalidValue))
}

fn parse_key_val(s: &str) -> Result<HashMap<String, String>, Error> {
    let mut key_val_map = HashMap::new();
    let items = s.split(",").collect::<Vec<&str>>();
    for item in items.iter() {
        let key_value = item.split("=").collect::<Vec<&str>>();
        if key_value.len() % 2 != 0 {
            return Err(Error::new(InvalidValue));
        }
        for i in 0..key_value.len() / 2 {
            key_val_map.insert(
                key_value[i * 2].to_string(),
                key_value[i * 2 + 1].to_string(),
            );
        }
    }
    Ok(key_val_map)
}

fn parse_tokenizer_options(s: &str) -> Result<TokenizeOptions, Error> {
    let mut tokenizer_options = TokenizeOptions::new();
    let items = s.split(",").collect::<Vec<&str>>();
    for item in items.iter() {
        let key_value = item.split("=").collect::<Vec<&str>>();
        if key_value.len() != 2 {
            return Err(Error::new(InvalidValue));
        }
        match key_value[0] {
            "num_tokens" => {
                tokenizer_options.num_tokens = Some(key_value[1].parse::<u64>().unwrap())
            }
            "min_tokens" => tokenizer_options.min_tokens = key_value[1].parse::<u64>().unwrap(),
            "max_tokens" => tokenizer_options.max_tokens = key_value[1].parse::<u64>().unwrap(),
            "variance" => tokenizer_options.variance = key_value[1].parse::<u64>().unwrap(),
            _ => return Err(Error::new(InvalidValue)),
        }
    }
    if tokenizer_options.num_tokens.is_some()
        && (tokenizer_options.num_tokens.unwrap() == 0
            || tokenizer_options.min_tokens == 0
            || tokenizer_options.max_tokens == 0)
    {
        return Err(Error::new(InvalidValue));
    }
    if tokenizer_options.min_tokens > tokenizer_options.max_tokens {
        return Err(Error::new(InvalidValue));
    }
    Ok(tokenizer_options)
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let git_sha = option_env!("VERGEN_GIT_SHA").unwrap_or("unknown");
    println!(
        "Text Generation Inference Benchmark {} ({})",
        env!("CARGO_PKG_VERSION"),
        git_sha
    );

    let (stop_sender, _) = broadcast::channel(1);
    // handle ctrl-c
    let stop_sender_clone = stop_sender.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for ctrl-c");
        debug!("Received stop signal, stopping benchmark");
        stop_sender_clone
            .send(())
            .expect("Failed to send stop signal");
    });

    let stop_sender_clone = stop_sender.clone();
    // get HF token
    let token_env_key = "HF_TOKEN".to_string();
    let cache = hf_hub::Cache::from_env();
    let hf_token = match std::env::var(token_env_key).ok() {
        Some(token) => Some(token),
        None => cache.token(),
    };
    let model_name = args
        .model_name
        .clone()
        .unwrap_or(args.tokenizer_name.clone());
    let run_id = args
        .run_id
        .unwrap_or(uuid::Uuid::new_v4().to_string()[..7].to_string());
    let run_config = RunConfiguration {
        url: args.url,
        api_key: args.api_key,
        profile: args.profile.clone(),
        tokenizer_name: args.tokenizer_name.clone(),
        max_vus: args.max_vus,
        duration: args.duration,
        rates: args.rates,
        num_rates: args.num_rates,
        benchmark_kind: args.benchmark_kind.clone(),
        warmup_duration: args.warmup,
        prompt_options: args.prompt_options.clone(),
        decode_options: args.decode_options.clone(),
        dataset: args.dataset.clone(),
        dataset_file: args.dataset_file.clone(),
        extra_metadata: args.extra_meta.clone(),
        hf_token,
        model_name,
        run_id,
    };
    let main_thread = tokio::spawn(async move {
        match run(run_config, stop_sender_clone).await {
            Ok(_) => {}
            Err(e) => {
                error!("Fatal: {:?}", e);
                println!("Fatal: {:?}", e)
            }
        };
    });
    let _ = main_thread.await;
}
