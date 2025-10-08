# BurstGPT Load Generator Converter

A tool to convert CSV files containing ChatGPT/GPT-4 conversation logs into mooncake-style JSONL format for load testing and simulation.

> [!NOTE]
> Currently, KV reuse is not considered in the output. We will update the script once [BurstGPT](https://github.com/HPMLL/BurstGPT) adds user session information.

## Input Format

The input CSV can be downloaded from [BurstGPT Release v1.1](https://github.com/HPMLL/BurstGPT/releases/tag/v1.1):
- `Timestamp`: Request timestamp in seconds
- `Model`: Model name (e.g., "ChatGPT", "GPT-4")
- `Request tokens`: Number of input tokens
- `Response tokens`: Number of output tokens
- `Total tokens`: Total tokens (not used)
- `Log Type`: Type of log (e.g., "Conversation log", "API log")

Example:
```csv
Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type
5,ChatGPT,472,18,490,Conversation log
45,ChatGPT,1087,230,1317,Conversation log
118,GPT-4,417,276,693,Conversation log
```

## Output Format

The output is a JSONL file where each line is a JSON object:
```json
{"timestamp": 5000, "input_length": 472, "output_length": 18, "hash_ids": [123, 456, 789, ...]}
```

Fields:
- `timestamp`: Request time in milliseconds (integer)
- `input_length`: Number of input tokens
- `output_length`: Number of output tokens
- `hash_ids`: Array of random hash IDs simulating KV cache blocks

## Usage

### Basic Usage

```bash
python convert.py --input-file <BurstGPT CSV data>
```

If `--output-file` is not specified, the output will use the input filename with `.jsonl` extension.

### Command Line Arguments

#### Required Arguments
- `--input-file`: Path to the input CSV file

#### Optional Arguments

**Filtering:**
- `--model`: Filter by model (`ChatGPT` or `GPT-4`), None for no filtering
- `--log-type`: Filter by log type (`Conversation log` or `API log`), None for no filtering
- `--num-prompt`: Limit number of rows in the final output, None for no filtering

**Timestamp Adjustment:**
- `--speed-ratio`: Adjust request timing (default: 1.0)
  - Values > 1: Speed up (e.g., 2.0 = 2x faster)
  - Values < 1: Slow down (e.g., 0.5 = 2x slower)
  - Formula: `new_timestamp = old_timestamp / speed_ratio`

**Hash Generation:**
- `--block-size`: Block size in mooncake traces (default: 128)
- `--num-hash-blocks`: Maximum hash ID value (default: 10000). Hash IDs are randomly chosen from 0 to this value for each block.
**Output:**
- `--output-file`: Path to output JSONL file (default: input filename with .jsonl extension)

## Statistics Output

After conversion, the script displays statistics about the generated workload:

```
============================================================
STATISTICS
============================================================

Input Length (ISL):
  Min: 37
  Max: 1528
  Avg: 705.89
  Std: 524.33

Output Length (OSL):
  Min: 18
  Max: 1656
  Avg: 494.67
  Std: 513.21

Sequence Length (ISL + OSL):
  Max: 3184

Request Rate:
  Total requests: 9
  Duration: 405.00 seconds
  Average RPS: 0.02
============================================================
```
