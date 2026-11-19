# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MMLU evaluation logic for Dynamo correctness comparisons."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from tests.lmcache.mmlu_utils import extract_choice, prompt_string

MAX_FEW_SHOT_TOKENS = 4000


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
    response.raise_for_status()
    return response.json()["choices"][0]["text"]


def evaluate(
    args: argparse.Namespace,
    subject: str,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
) -> float:
    header = (
        "The following are multiple choice questions (with answers) "
        f"about {subject}.\n\n"
    )
    shared_multi_shot_prefix = [header]
    shared_multi_shot_prefix_length = len(
        tokenizer(header, add_special_tokens=True)["input_ids"]
    )
    for index in range(dev_df.shape[0]):
        example = prompt_string(dev_df, index)
        token_ids = tokenizer(example, add_special_tokens=False)["input_ids"]
        if shared_multi_shot_prefix_length + len(token_ids) > MAX_FEW_SHOT_TOKENS:
            break
        shared_multi_shot_prefix.append(example)
        shared_multi_shot_prefix_length += len(token_ids)

    shared_multi_shot_prefix_str = "".join(shared_multi_shot_prefix)
    prompts = []
    labels = []
    for index in range(test_df.shape[0]):
        query_prompt = prompt_string(test_df, index, include_answer=False)
        prompts.append(f"{shared_multi_shot_prefix_str}\n\n{query_prompt}")
        labels.append(test_df.iloc[index, test_df.shape[1] - 1])

    predictions = [extract_choice(get_llm_response(args, prompt)) for prompt in prompts]
    return float(np.mean(np.array(predictions) == np.array(labels)))


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    data_dir = Path("data")
    subjects = sorted(
        path.name.removesuffix("_test.csv")
        for path in (data_dir / "test").glob("*_test.csv")
    )

    accuracies = []
    num_questions = []
    output_dict = {}
    for subject_raw in tqdm(
        subjects[: args.number_of_subjects], desc="Processing subjects"
    ):
        subject = " ".join(subject_raw.split("_"))
        dev_df = pd.read_csv(data_dir / "dev" / f"{subject_raw}_dev.csv", header=None)
        test_df = pd.read_csv(
            data_dir / "test" / f"{subject_raw}_test.csv", header=None
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

    with Path(args.result_file).open("w", encoding="utf-8") as result_file:
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
