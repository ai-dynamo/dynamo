# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MMLU evaluation logic for Dynamo correctness comparisons."""

import argparse
import json
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

CHOICES = ("A", "B", "C", "D")


def get_llm_response(args: argparse.Namespace, prompt: str) -> str:
    data = {
        "model": args.model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 3,
        "stream": False,
        "seed": 42,
    }
    url = f"http://{args.host}:{args.port}/v1/completions"
    response = requests.post(url, json=data, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} {response.text}")
    return response.json()["choices"][0]["text"]


def prompt_string(df: pd.DataFrame, idx: int, include_answer: bool = True) -> str:
    prompt = df.iloc[idx, 0]
    option_count = df.shape[1] - 2
    for option_index in range(option_count):
        prompt += f"\n{CHOICES[option_index]}. {df.iloc[idx, option_index + 1]}"
    prompt += (
        "\nRespond with **only the letter** (A, B, C, D).  Do **not** output "
        "any explanation, analysis, or extra words. Answer:"
    )
    if include_answer:
        prompt += f" {df.iloc[idx, option_count + 1]}\n\n"
    return prompt


def evaluate(
    args: argparse.Namespace,
    subject: str,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
) -> float:
    shared_multi_shot_prefix = [
        f"The following are multiple choice questions (with answers) "
        f"                                about {subject}. \n\n"
    ]
    shared_multi_shot_prefix_length = 0
    for index in range(dev_df.shape[0]):
        example = prompt_string(dev_df, index)
        shared_multi_shot_prefix.append(example)
        token_ids = tokenizer(example, add_special_tokens=True)["input_ids"]
        shared_multi_shot_prefix_length += len(token_ids)
        if shared_multi_shot_prefix_length > 4000:
            break

    shared_multi_shot_prefix_str = "".join(shared_multi_shot_prefix)
    prompts = []
    labels = []
    for index in range(test_df.shape[0]):
        query_prompt = prompt_string(test_df, index, include_answer=False)
        prompts.append(f"{shared_multi_shot_prefix_str}\n\n{query_prompt}")
        labels.append(test_df.iloc[index, test_df.shape[1] - 1])

    predictions = [
        _extract_choice(get_llm_response(args, prompt)) for prompt in prompts
    ]
    return float(np.mean(np.array(predictions) == np.array(labels)))


def _extract_choice(response: str) -> str:
    stripped = response.strip()
    if stripped and stripped[0] in CHOICES:
        return stripped[0]
    return next((char for char in stripped if char in CHOICES), "A")


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    test_files = [
        name for name in os.listdir("data/test") if name.endswith("_test.csv")
    ]
    subjects = sorted(name.split("_test.csv")[0] for name in test_files)

    accuracies = []
    num_questions = []
    output_dict = {}
    for subject_raw in tqdm(
        subjects[: args.number_of_subjects], desc="Processing subjects"
    ):
        subject = " ".join(subject_raw.split("_"))
        dev_df = pd.read_csv(
            os.path.join("data/dev", subject_raw + "_dev.csv"), header=None
        )
        test_df = pd.read_csv(
            os.path.join("data/test", subject_raw + "_test.csv"), header=None
        )
        accuracy = evaluate(args, subject, dev_df, test_df, tokenizer)
        accuracies.append(accuracy)
        num_questions.append(len(test_df))
        output_dict[subject_raw] = {
            "accuracy": accuracy,
            "num_questions": len(test_df),
        }

    output_dict["total"] = {
        "accuracy": float(np.mean(accuracies)),
        "num_questions": sum(num_questions),
    }

    with open(args.result_file, "w") as result_file:
        for subject, value in output_dict.items():
            result_file.write(json.dumps({subject: value}) + "\n")


def parse_args(
    default_result_prefix: str, argv: Sequence[str] | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=False)
    parser.add_argument("--number-of-subjects", type=int, required=True)
    parser.add_argument("--host", type=str, default="localhost", help="Dynamo host")
    parser.add_argument("--port", type=int, default=8000, help="Dynamo port")

    args = parser.parse_args(argv)
    if args.result_file is None:
        model_name = args.model.split("/")[-1]
        args.result_file = f"{default_result_prefix}-{model_name}.jsonl"
    return args


def run(default_result_prefix: str) -> None:
    set_seed(42)
    main(parse_args(default_result_prefix))
