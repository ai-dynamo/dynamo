from contextlib import ExitStack
from dataclasses import dataclass
from typing import List

import click
import gradio as gr
import pandas as pd

from parse_results import build_results


@dataclass
class PlotConfig:
    x_title: str
    y_title: str
    title: str
    percentiles: List[float] = None


def run(from_results_dir, datasource, port):
    css = '''
    .summary span {
        font-size: 10px;
        padding-top:0;
        padding-bottom:0;
    }
    '''

    summary_desc = '''
    ## Summary
    This table shows the average of the metrics for each model and QPS rate.
    
    The metrics are:
    * Inter token latency: Time to generate a new output token for each user querying the system. 
      It translates as the “speed” perceived by the end-user. We aim for at least 300 words per minute (average reading speed), so ITL<150ms
    * Time to First Token: Time the user has to wait before seeing the first token of its answer. 
      Lower waiting time are essential for real-time interactions, less so for offline workloads.
    * End-to-end latency: The overall time the system took to generate the full response to the user.
    * Throughput: The number of tokens per second the system can generate across all requests
    * Successful requests: The number of requests the system was able to honor in the benchmark timeframe
    * Error rate: The percentage of requests that ended up in error, as the system could not process them in time or failed to process them. 
          
    '''

    df_bench = pd.DataFrame()
    line_plots_bench = []
    column_mappings = {'inter_token_latency_ms_p90': 'ITL P90 (ms)', 'time_to_first_token_ms_p90': 'TTFT P90 (ms)',
                       'e2e_latency_ms_p90': 'E2E P90 (ms)', 'token_throughput_secs': 'Throughput (tokens/s)',
                       'successful_requests': 'Successful requests', 'error_rate': 'Error rate (%)', 'model': 'Model',
                       'rate': 'QPS', 'run_id': 'Run ID'}
    default_df = pd.DataFrame.from_dict(
        {"rate": [1, 2], "inter_token_latency_ms_p90": [10, 20],
         "version": ["default", "default"],
         "model": ["default", "default"]})

    def load_demo(model_bench, percentiles):
        return update_bench(model_bench, percentiles)

    def update_bench(model, percentiles):
        res = []
        for plot in line_plots_bench:
            if plot['config'].percentiles:
                k = plot['metric'] + '_' + str(percentiles)
                df_bench[plot['metric']] = df_bench[k] if k in df_bench.columns else 0
            res.append(df_bench[(df_bench['model'] == model)])

        return res + [summary_table()]

    def summary_table() -> pd.DataFrame:
        data = df_bench.groupby(['model', 'run_id', 'rate']).agg(
            {'inter_token_latency_ms_p90': 'mean', 'time_to_first_token_ms_p90': 'mean',
             'e2e_latency_ms_p90': 'mean', 'token_throughput_secs': 'mean',
             'successful_requests': 'mean', 'error_rate': 'mean'}).reset_index()
        data = data[
            ['run_id', 'model', 'rate', 'inter_token_latency_ms_p90', 'time_to_first_token_ms_p90',
             'e2e_latency_ms_p90',
             'token_throughput_secs']]
        for metric in ['inter_token_latency_ms_p90', 'time_to_first_token_ms_p90', 'e2e_latency_ms_p90',
                       'token_throughput_secs']:
            data[metric] = data[metric].apply(lambda x: f"{x:.2f}")
        data = data.rename(
            columns=column_mappings)
        return data

    def load_bench_results(source) -> pd.DataFrame:
        data = pd.read_parquet(source)
        # remove warmup and throughput
        data = data[(data['id'] != 'warmup') & (data['id'] != 'throughput')]
        # only keep constant rate
        data = data[data['executor_type'] == 'ConstantArrivalRate']
        return data

    def select_region(selection: gr.SelectData, model):
        min_w, max_w = selection.index
        data = df_bench[(df_bench['model'] == model) & (df_bench['rate'] >= min_w) & (
                df_bench['rate'] <= max_w)]
        res = []
        for plot in line_plots_bench:
            # find the y values for the selected region
            metric = plot["metric"]
            y_min = data[metric].min()
            y_max = data[metric].max()
            res.append(gr.LinePlot(x_lim=[min_w, max_w], y_lim=[y_min, y_max]))
        return res

    def reset_region():
        res = []
        for _ in line_plots_bench:
            res.append(gr.LinePlot(x_lim=None, y_lim=None))
        return res

    def load_datasource(datasource, fn):
        if datasource.startswith('file://'):
            return fn(datasource)
        elif datasource.startswith('s3://'):
            return fn(datasource)
        else:
            raise ValueError(f"Unknown datasource: {datasource}")

    if from_results_dir is not None:
        build_results(from_results_dir, 'benchmarks.parquet', None)
    # Load data
    df_bench = load_datasource(datasource, load_bench_results)

    # Define metrics
    metrics = {
        "inter_token_latency_ms": PlotConfig(title="Inter Token Latency (lower is better)", x_title="QPS",
                                             y_title="Time (ms)", percentiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]),
        "time_to_first_token_ms": PlotConfig(title="TTFT (lower is better)", x_title="QPS",
                                             y_title="Time (ms)", percentiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]),
        "e2e_latency_ms": PlotConfig(title="End to End Latency (lower is better)", x_title="QPS",
                                     y_title="Time (ms)", percentiles=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]),
        "token_throughput_secs": PlotConfig(title="Request Output Throughput (higher is better)", x_title="QPS",
                                            y_title="Tokens/s"),
        "successful_requests": PlotConfig(title="Successful requests (higher is better)", x_title="QPS",
                                          y_title="Count"),
        "error_rate": PlotConfig(title="Error rate", x_title="QPS", y_title="%"),
        "prompt_tokens": PlotConfig(title="Prompt tokens", x_title="QPS", y_title="Count"),
        "decoded_tokens": PlotConfig(title="Decoded tokens", x_title="QPS", y_title="Count")
    }

    models = df_bench["model"].unique()
    run_ids = df_bench["run_id"].unique()

    # get all available percentiles
    percentiles = set()
    for k, v in metrics.items():
        if v.percentiles:
            percentiles.update(v.percentiles)
    percentiles = map(lambda p: f'p{int(float(p) * 100)}', percentiles)
    percentiles = sorted(list(percentiles))
    percentiles.append('avg')
    with gr.Blocks(css=css, title="Inference Benchmarker") as demo:
        with gr.Row():
            gr.Markdown("# Inference-benchmarker 🤗\n## Benchmarks results")
        with gr.Row():
            gr.Markdown(summary_desc)
        with gr.Row():
            table = gr.DataFrame(
                pd.DataFrame(),
                elem_classes=["summary"],
            )
        with gr.Row():
            details_desc = gr.Markdown("## Details")
        with gr.Row():
            model = gr.Dropdown(list(models), label="Select model", value=models[0])
        with gr.Row():
            percentiles_bench = gr.Radio(percentiles, label="", value="avg")
        i = 0
        with ExitStack() as stack:
            for k, v in metrics.items():
                if i % 2 == 0:
                    stack.close()
                    gs = stack.enter_context(gr.Row())
                line_plots_bench.append(
                    {"component": gr.LinePlot(default_df, label=f'{v.title}', x="rate", y=k,
                                              y_title=v.y_title, x_title=v.x_title,
                                              color="run_id"
                                              ),
                     "model": model.value,
                     "metric": k,
                     "config": v
                     },
                )
                i += 1

        for component in [model, percentiles_bench]:
            component.change(update_bench, [model, percentiles_bench],
                             [item["component"] for item in line_plots_bench] + [table])
        gr.on([plot["component"].select for plot in line_plots_bench], select_region, [model],
              outputs=[item["component"] for item in line_plots_bench])
        gr.on([plot["component"].double_click for plot in line_plots_bench], reset_region, None,
              outputs=[item["component"] for item in line_plots_bench])
        demo.load(load_demo, [model, percentiles_bench],
                  [item["component"] for item in line_plots_bench] + [table])

    demo.launch(server_port=port, server_name="0.0.0.0")


@click.command()
@click.option('--from-results-dir', default=None, help='Load inference-benchmarker results from a directory')
@click.option('--datasource', default='file://benchmarks.parquet', help='Load a Parquet file already generated')
@click.option('--port', default=7860, help='Port to run the dashboard')
def main(from_results_dir, datasource, port):
    run(from_results_dir, datasource, port)


if __name__ == '__main__':
    main(auto_envvar_prefix='DASHBOARD')
