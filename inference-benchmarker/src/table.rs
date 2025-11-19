use crate::results::BenchmarkReport;
use crate::BenchmarkConfig;
use tabled::builder::Builder;

pub fn parameters_table(benchmark: BenchmarkConfig) -> anyhow::Result<tabled::Table> {
    let mut builder = Builder::default();
    let rates = benchmark
        .rates
        .map_or("N/A".to_string(), |e| format!("{:?}", e));
    let prompt_options = benchmark
        .prompt_options
        .map_or("N/A".to_string(), |e| format!("{}", e));
    let decode_options = benchmark
        .decode_options
        .map_or("N/A".to_string(), |e| format!("{}", e));
    let extra_metadata = benchmark
        .extra_metadata
        .map_or("N/A".to_string(), |e| format!("{:?}", e));
    builder.set_header(vec!["Parameter", "Value"]);
    builder.push_record(vec!["Max VUs", benchmark.max_vus.to_string().as_str()]);
    builder.push_record(vec![
        "Duration",
        benchmark.duration.as_secs().to_string().as_str(),
    ]);
    builder.push_record(vec![
        "Warmup Duration",
        benchmark.warmup_duration.as_secs().to_string().as_str(),
    ]);
    builder.push_record(vec![
        "Benchmark Kind",
        benchmark.benchmark_kind.to_string().as_str(),
    ]);
    builder.push_record(vec!["Rates", rates.as_str()]);
    builder.push_record(vec!["Num Rates", benchmark.num_rates.to_string().as_str()]);
    builder.push_record(vec!["Prompt Options", prompt_options.as_str()]);
    builder.push_record(vec!["Decode Options", decode_options.as_str()]);
    builder.push_record(vec!["Tokenizer", benchmark.tokenizer.to_string().as_str()]);
    builder.push_record(vec!["Extra Metadata", extra_metadata.as_str()]);
    let mut table = builder.build();
    table.with(tabled::settings::Style::sharp());
    Ok(table)
}

pub fn results_table(benchmark: BenchmarkReport) -> anyhow::Result<tabled::Table> {
    let mut builder = Builder::default();
    builder.set_header(vec![
        "Benchmark",
        "QPS",
        "E2E Latency (avg)",
        "TTFT (avg)",
        "ITL (avg)",
        "Throughput",
        "Error Rate",
        "Successful Requests",
        "Prompt tokens per req (avg)",
        "Decoded tokens per req (avg)",
    ]);
    let results = benchmark.get_results();
    for result in results {
        let qps = format!("{:.2} req/s", result.successful_request_rate()?);
        let e2e = format!("{:.2} sec", result.e2e_latency_avg()?.as_secs_f64());
        let ttft = format!(
            "{:.2} ms",
            result.time_to_first_token_avg()?.as_micros() as f64 / 1000.0
        );
        let itl = format!(
            "{:.2} ms",
            result.inter_token_latency_avg()?.as_micros() as f64 / 1000.0
        );
        let throughput = format!("{:.2} tokens/sec", result.token_throughput_secs()?);
        let error_rate = result.failed_requests() as f64 / result.total_requests() as f64 * 100.0;
        let error_rate = format!("{:.2}%", error_rate);
        builder.push_record(vec![
            result.id.as_str(),
            qps.as_str(),
            e2e.as_str(),
            ttft.as_str(),
            itl.as_str(),
            throughput.as_str(),
            error_rate.as_str(),
            format!(
                "{}/{}",
                result.successful_requests(),
                result.total_requests()
            )
            .as_str(),
            format!("{:.2}", result.prompt_tokens_avg()?).as_str(),
            format!(
                "{:.2}",
                result.total_tokens() as f64 / result.successful_requests() as f64
            )
            .as_str(),
        ]);
    }
    let mut table = builder.build();
    table.with(tabled::settings::Style::sharp());
    Ok(table)
}
