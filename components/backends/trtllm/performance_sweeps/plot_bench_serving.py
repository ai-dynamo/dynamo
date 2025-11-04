import argparse
import os
from pathlib import Path
import json
import plotly.graph_objects as go
from tqdm import tqdm

def load_data(directory: str):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")

    results = []

    base = Path(directory)
    
    for subdir in tqdm(base.iterdir()):
        if not subdir.is_dir():
            print(f"Warning: {subdir} is not a directory. Skipping...")
            continue

        if not (subdir / "deployment_config.json").exists():
            print(f"Warning: {subdir} does not have a deployment_config.json. Skipping...")
            continue
        
        with open(subdir / "deployment_config.json", "r") as f:
            deployment_config = json.load(f)
            if "total_gpus" not in deployment_config:
                print(f"Warning: {subdir} does not have a total_gpus. Skipping...")
                continue
            total_gpus = int(deployment_config["total_gpus"])

        deployment_config["name"] = subdir.name
        results_dir = subdir / "results"
        if not results_dir.exists():
            print(f"Warning: {subdir} does not have a results directory. Skipping...")
            continue


        for result_file in results_dir.iterdir():
            if not result_file.is_file() or not result_file.name.startswith("results_concurrency"):
                print(f"Warning: {result_file} is not a file or does not start with 'concurrency'. Skipping...")
                continue

            with open(result_file, "r") as f:
                result_data = json.load(f)

                throughput_per_gpu = result_data["total_token_throughput"] / total_gpus
                throughput_per_user = 1000 / result_data["mean_tpot_ms"]
                
                results.append({
                    "per_gpu": throughput_per_gpu,
                    "per_user": throughput_per_user,
                    "ttft": result_data["mean_ttft_ms"] / 1000,
                    "config": deployment_config,
                })
    print(f"Gathered results from {len(results)} benchmarks")

    return results

def get_pareto_optimal(x, y):
    pareto_points = []

    for (x_point, y_point) in zip(x, y):
        if not any(x_val > x_point and y_val > y_point for x_val, y_val in zip(x, y)):
            pareto_points.append((x_point, y_point))
    
    pareto_points.sort(key=lambda p: p[0])
    return pareto_points
    

def format_config_tooltip(config):
    """Format the configuration dictionary into a readable string for tooltips"""
    
    return f"<br>{config['name']}"

def plot_pareto(results, output_file):
    # Group results by config name
    config_groups = {}
    for result in results:
        config_name = result["config"]["name"]
        if config_name not in config_groups:
            config_groups[config_name] = []
        config_groups[config_name].append(result)
    
    # Get all x and y values for Pareto calculation
    x = [result["per_user"] for result in results]
    y = [result["per_gpu"] for result in results]
    
    pareto_points = get_pareto_optimal(x, y)
    pareto_x, pareto_y = zip(*pareto_points) if pareto_points else ([], [])

    # Create the plot
    fig = go.Figure()
    
    # Use Plotly's color palette - cycle through colors
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    # Add scatter plot for each config with different color
    config_names = sorted(config_groups.keys())
    for i, config_name in enumerate(config_names):
        config_results = config_groups[config_name]
        config_x = [r["per_user"] for r in config_results]
        config_y = [r["per_gpu"] for r in config_results]
        
        # Create hover text for each point in this config
        hover_texts = []
        for result in config_results:
            config_text = format_config_tooltip(result["config"])
            hover_text = (
                f"tokens/s/user: {result['per_user']:.2f}<br>"
                f"tokens/s/gpu: {result['per_gpu']:.2f}<br>"
                f"ttft: {result['ttft']:.3f}s<br>"
                f"<br>Config:<br>{config_text}"
            )
            hover_texts.append(hover_text)
        
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=config_x,
            y=config_y,
            mode='markers',
            marker=dict(
                color=color,
                size=10,
                opacity=0.6
            ),
            name=config_name,
            hovertext=hover_texts,
            hoverinfo='text',
        ))
    
    # Add Pareto frontier line (always red, on top)
    if pareto_x and pareto_y:
        fig.add_trace(go.Scatter(
            x=list(pareto_x),
            y=list(pareto_y),
            mode='lines+markers',
            marker=dict(
                color='red',
                size=10
            ),
            line=dict(
                color='red',
                width=2
            ),
            name='New Disagg Results',
            hoverinfo='skip',
        ))

    prior_pareto_points = [
        (493.86, 1063.01),
        (435.31, 1872.98),
        (398.8, 3422.14),
        (337.06, 5582.79),
        (294.11, 9130.98),
        (276.3, 9145.31),
        (220.62, 14040.17),
        (164.237, 21223.8),
        (114.73, 30164.56),
        (75.136, 39755.45),
        (47.922, 50762.26),
        (32.617, 51570.95),
        (30.52, 64598.6),
        (23.792, 66235.53)
    ]

    reference_pareto_x, reference_pareto_y = zip(*prior_pareto_points)

    fig.add_trace(go.Scatter(
        x=list(reference_pareto_x),
        y=list(reference_pareto_y),
        mode='lines+markers',
        marker=dict(
            color='black',
            size=10
        ),
        line=dict(
            color='blue',
            width=2
        ),
        name='Current SA submission',
        hoverinfo='skip',
    ))
    
    fig.update_layout(
        title="GPT-OSS GB200 Pareto Frontier, 8K/1K",
        xaxis_title="tokens/s/user",
        yaxis_title="Total tokens/s/gpu",
        hovermode='closest',
        template='plotly_white',
    )
    
    # Save as HTML
    fig.write_html(output_file)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("directory")
    parser.add_argument("output_file")

    args = parser.parse_args()

    data = load_data(args.directory)

    plot_pareto(data, args.output_file)


if __name__ == "__main__":
    main()