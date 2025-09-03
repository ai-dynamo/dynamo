import json

import matplotlib.pyplot as plt


def load_results(file_path):
    """Load results from JSON file"""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_data(results, exclude_ratio=None):
    """Extract and optionally filter data from results"""
    prefix_ratios = results["prefix_ratios"]
    try:
        ttft_values = results["ttft_values"]
        throughput_values = results["throughput_values"]
    except KeyError:
        ttft_values = results["ttft_avg_values"]
        throughput_values = results["throughput_avg_values"]

    if exclude_ratio is not None:
        filtered_indices = [
            i for i, ratio in enumerate(prefix_ratios) if ratio != exclude_ratio
        ]
        return (
            [prefix_ratios[i] for i in filtered_indices],
            [ttft_values[i] for i in filtered_indices],
            [throughput_values[i] for i in filtered_indices],
        )

    return prefix_ratios, ttft_values, throughput_values


def plot_dataset(
    ax1, ax2, prefix_ratios, ttft_values, throughput_values, label, color_marker
):
    """Helper function to plot a dataset on both TTFT and throughput axes"""
    ax1.plot(
        prefix_ratios, ttft_values, color_marker, label=label, linewidth=2, markersize=6
    )
    ax2.plot(
        prefix_ratios,
        throughput_values,
        color_marker,
        label=label,
        linewidth=2,
        markersize=6,
    )


def configure_axis(ax, xlabel, ylabel, title, xlim=(0.05, 0.95), yscale="log"):
    """Configure axis with common settings"""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(*xlim)
    # ax.set_yscale(yscale)


def print_dataset_stats(datasets):
    """Print summary statistics for all datasets"""
    for name, (_, ttft_values, throughput_values, _) in datasets.items():
        print(
            f"\n{name} - TTFT range: {min(ttft_values):.1f} - {max(ttft_values):.1f} ms"
        )
        print(
            f"{name} - Throughput range: {min(throughput_values):.1f} - {max(throughput_values):.1f} tokens/s"
        )


def create_comparison_plots():
    title = "Concurrency 20"
    output_fn = "plots"

    # Dataset configurations: (file_path, label, style, exclude_ratio)
    dataset_configs = [
        (
            "push_load/kv/results_summary.json",
            "Kv Router",
            "c-s",
            None,
        ),
        # (
        #     "push_load/kv_2/results_summary.json",
        #     "Kv Router (2 replicas)",
        #     "r-o",
        #     None,
        # ),
        # (
        #     "push_load/kv_2_shared/results_summary.json",
        #     "Kv Routers (2 replicas, states shared)",
        #     "b-D",
        #     None,
        # ),
        # (
        #     "kv_router_overlap_weight_0.5/results_summary.json",
        #     "Kv Router (overlap weight 0.5)",
        #     "m-D",  # Changed from "r-D" to "m-D" (magenta with diamond markers)
        #     None,
        # ),
    ]

    # Load and extract data for all datasets
    datasets = {}
    reference_results = None

    for file_path, label, style, exclude_ratio in dataset_configs:
        results = load_results(file_path)
        if reference_results is None:
            reference_results = results  # Use first dataset for config info

        data = extract_data(results, exclude_ratio)
        datasets[label] = (*data, style)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot all datasets
    for label, (
        prefix_ratios,
        ttft_values,
        throughput_values,
        style,
    ) in datasets.items():
        plot_dataset(
            ax1, ax2, prefix_ratios, ttft_values, throughput_values, label, style
        )

    # Configure both axes
    configure_axis(
        ax1, "Prefix Ratio", "TTFT (ms)", "Time to First Token vs Prefix Ratio"
    )
    configure_axis(
        ax2, "Prefix Ratio", "Throughput (tokens/s)", "Throughput vs Prefix Ratio"
    )

    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Adjust layout and save
    plt.tight_layout()

    # Save plots
    for ext, dpi in [("png", 300), ("pdf", None)]:
        output_path = f"{output_fn}.{ext}"
        save_kwargs = {"bbox_inches": "tight"}
        if dpi:
            save_kwargs["dpi"] = dpi
        plt.savefig(output_path, **save_kwargs)
        print(f"Plot saved as: {output_path}")

    # Print summary statistics
    config = reference_results["config"]
    print("\nSummary Statistics:")
    print(f"Model: {config['model']}")
    print(f"Input Sequence Length: {config['isl']}")
    print(f"Output Sequence Length: {config['osl']}")
    print(f"Requests: {config['requests']}")
    print(f"Concurrency: {config['concurrency']}")

    print_dataset_stats(datasets)


if __name__ == "__main__":
    create_comparison_plots()
