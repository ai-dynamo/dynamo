import argparse
import json
import os

import pandas as pd


def build_df(model: str, data_files: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame()
    # Load the results
    for key, filename in data_files.items():
        with open(filename, 'r') as f:
            data = json.load(f)
            if data['config']['meta'] is None:
                data['config']['meta'] = {}
            for result in data['results']:
                entry = pd.json_normalize(result).to_dict(orient='records')[0]
                if 'engine' in data['config']['meta']:
                    entry['engine'] = data['config']['meta']['engine']
                if 'tp' in data['config']['meta']:
                    entry['tp'] = data['config']['meta']['tp']
                if 'version' in data['config']['meta']:
                    entry['version'] = data['config']['meta']['version']
                if 'device' in data['config']['meta']:
                    entry['device'] = data['config']['meta']['device']
                entry['model'] = data['config']['model_name']
                entry['run_id'] = data['config']['run_id']
                df_tmp = pd.DataFrame(entry, index=[0])
                # rename columns that start with 'config.'
                df_tmp = df_tmp.rename(columns={c: c.split('config.')[-1] for c in df_tmp.columns})
                # replace . with _ in column names
                df_tmp.columns = [c.replace('.', '_') for c in df_tmp.columns]

                df = pd.concat([df, df_tmp])
    return df


def build_results_df(results_dir) -> pd.DataFrame:
    df = pd.DataFrame()
    # list directories
    directories = [f'{results_dir}/{d}' for d in os.listdir(results_dir) if os.path.isdir(f'{results_dir}/{d}')] + [results_dir]
    for directory in directories:
        # list json files in results directory
        data_files = {}
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                data_files[filename.split('.')[-2]] = f'{directory}/{filename}'
        df = pd.concat([df, build_df(directory.split('/')[-1], data_files)])
    return df


def build_results(results_dir, results_file, device):
    df = build_results_df(results_dir)
    if 'device' not in df.columns:
        df['device'] = df['model'].apply(lambda x: device)
    df['error_rate'] = df['failed_requests'] / (df['failed_requests'] + df['successful_requests']) * 100.0
    df['prompt_tokens'] = df['total_tokens_sent'] / df['successful_requests']
    df['decoded_tokens'] = df['total_tokens'] / df['successful_requests']
    df.to_parquet(results_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results', type=str, required=True,
                        help='Path to the source directory containing the results')
    parser.add_argument('--results-file', type=str, required=True,
                        help='Path to the results file to write to. Can be a S3 path')
    parser.add_argument('--device', type=str, required=True, help='GPU name used for benchmarking')
    args = parser.parse_args()
    build_results(args.results_dir, args.results_file, args.device)
