import argparse
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
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
            data = json.load(f)
            if "total_gpus" not in data:
                print(f"Warning: {subdir} does not have a total_gpus. Skipping...")
                continue
            total_gpus = int(data["total_gpus"])

        results_dir = subdir / "results"
        if not results_dir.exists():
            print(f"Warning: {subdir} does not have a results directory. Skipping...")
            continue


        for result_file in results_dir.iterdir():
            if not result_file.is_file() or not result_file.name.startswith("results_concurrency"):
                print(f"Warning: {result_file} is not a file or does not start with 'concurrency'. Skipping...")
                continue

            with open(result_file, "r") as f:
                data = json.load(f)

                throughput_per_gpu = data["output_throughput"] / total_gpus
                throughput_per_user = 1000 / data["mean_tpot_ms"]
                
                results.append({
                    "per_gpu": throughput_per_gpu,
                    "per_user": throughput_per_user,
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
    

def plot_pareto(results, output_file):
    x = [result["per_user"] for result in results]
    y = [result["per_gpu"] for result in results]

    pareto_points = get_pareto_optimal(x, y)
    pareto_x, pareto_y = zip(*pareto_points)

    plt.scatter(x, y, zorder=1, color='blue', marker='o', alpha=0.5, s=10)
    plt.plot(pareto_x, pareto_y, zorder=2, color='red', marker='o')
    
    plt.xlabel("tokens/s/user")
    plt.ylabel("tokens/s/gpu")
    plt.savefig(output_file)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("directory")
    parser.add_argument("output_file")

    args = parser.parse_args()

    data = load_data(args.directory)

    plot_pareto(data, args.output_file)


if __name__ == "__main__":
    main()