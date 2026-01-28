<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sinusoidal Load Generator

`sin_synth.py` is a simple script to generate synthetic load with sinusoidal request rate and isl/osl ratio. The output is in [mooncake-style](https://github.com/kvcache-ai/Mooncake) jsonl format, which can be directly used in [AIPerf](https://github.com/ai-dynamo/aiperf).

## Usage

```bash
cd benchmarks/sin_load_generator
python sin_synth.py [OPTIONS]
```

### Basic Options

- `--block-size INT` (default: 512)
  - Block size for hashing, since there is no prefix caching, the block size does not need to be the same as the engine's KV block size.

- `--total-blocks INT` (default: 10000)
  - ISL prompt blocks are randomly sampled from this range. Use a larger number to reduce the chance of duplicated prompts.

- `--output-file STR` (default: auto-generated)
  - Output file name (in jsonl format)
  - If not specified, the script will generate a filename based on parameters

- `--time-duration INT` (default: 100)
  - Total time duration of the dataset in seconds

- `--process-interval INT` (default: 1)
  - Sampling interval used to generate the dataset
  - Smaller interval leads to more precise changes in request rate and isl/osl ratio but longer generation time.

### Request Rate Parameters

The request rate follows a sinusoidal pattern:
```
request_rate(t) = (min + max) / 2 + (max - min) / 2 * sin(2 * π / period * t - π / 2)
```

Note the phase shift of `-π/2` is to make the request rate start from the minimum at `t = 0`.

- `--request-rate-min FLOAT` (default: 5)
  - Minimum request rate in requests per second

- `--request-rate-max FLOAT` (default: 10)
  - Maximum request rate in requests per second

- `--request-rate-period FLOAT` (default: 10)
  - Period of the sinusoidal request rate in seconds

### Input/Output Sequence Length Parameters

The script will generate load with requests sampled from two preset ISL/OSL combinations.
The ISL/OSL ratio defines how much of requests follow the first preset ISL/OSL pattern. ISl/OSL 0 means all requests follow the first preset ISL/OSL pattern, while ISL/OSL 1 means all requests follow the second preset ISL/OSL pattern.

The ISL/OSL ratio follows a sinusoidal pattern:
```
isl-osl-ratio(t) = (min + max) / 2 + (max - min) / 2 * sin(2 * π / period * t - π / 2)
```

Similarly, the phase shift of `-π/2` is to make the ISL/OSL ratio start from the minimum at `t = 0`.

- `--isl1 INT` (default: 100)
  - Minimum input sequence length

- `--osl1 INT` (default: 2000)
  - Minimum output sequence length

- `--isl2 INT` (default: 5000)
  - Maximum input sequence length

- `--osl2 INT` (default: 100)
  - Maximum output sequence length

- `--isl-osl-ratio-min FLOAT` (default: 0.2)
  - Minimum ratio of input sequence length to output sequence length

- `--isl-osl-ratio-max FLOAT` (default: 0.8)
  - Maximum ratio of input sequence length to output sequence length

- `--isl-osl-ratio-period FLOAT` (default: 10)
  - Period of the sinusoidal input/output sequence length ratio

### Examples

#### Varying Request Rate with Fixed ISL/OSL Ratio

```bash
python sin_synth.py \
  --time-duration 60 \
  --request-rate-min 2 \
  --request-rate-max 8 \
  --request-rate-period 20 \
  --isl1 3000 \
  --osl1 150 \
  --isl2 3000 \
  --osl2 150 \
  --output-file dataset.jsonl
```

This generates a 60-second dataset with request rates varying between 2-8 requests/second over a 20-second period, with 3000 ISL and 150 OSL. The ISL/OSL ratio is fixed at 0.2.

#### Varying ISL/OSL Ratio with Fixed Request Rate

```bash
python sin_synth.py \
  --time-duration 60 \
  --request-rate-min 5 \
  --request-rate-max 5 \
  --isl1 3000 \
  --osl1 150 \
  --isl2 500 \
  --osl2 2000 \
  --isl-osl-ratio-min 0.2 \
  --isl-osl-ratio-max 0.8 \
  --isl-osl-ratio-period 20 \
  --output-file dataset.jsonl
```

This generates a 60-second dataset with request rate fixed at 5 requests/second, with ISL/OSL ratio varying between 0.2 and 0.8 between I3000O150 and I500O2000over a 20-second period.