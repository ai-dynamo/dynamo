# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Download MMLU dataset using HuggingFace datasets library.
This is a simpler alternative to the bash script.
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset


def download_mmlu():
    """Download and prepare MMLU dataset."""
    print("Downloading MMLU dataset...")

    # Create data directories
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("data/dev", exist_ok=True)

    try:
        # Download MMLU dataset
        print("🔄 Loading MMLU dataset from HuggingFace...")
        dataset = load_dataset("cais/mmlu", "all")

        print("📊 Dataset information:")
        print(f"  - Training set: {len(dataset['auxiliary_train'])} samples")
        print(f"  - Validation set: {len(dataset['validation'])} samples")
        print(f"  - Test set: {len(dataset['test'])} samples")

        # Get all subjects
        subjects_set = set()
        for split in ["validation", "test"]:
            for item in dataset[split]:
                subjects_set.add(item["subject"])

        subjects = sorted(list(subjects_set))
        print(f"Contains {len(subjects)} subjects")

        # Process each subject
        print("Organizing data files...")
        for subject in subjects:
            # Filter data for this subject
            test_data = [item for item in dataset["test"] if item["subject"] == subject]
            val_data = [
                item for item in dataset["validation"] if item["subject"] == subject
            ]

            if test_data:
                # Convert to DataFrame and save as CSV
                test_df = pd.DataFrame(test_data)

                # Create the expected CSV format: question, A, B, C, D, answer
                csv_data = []
                for _, row in test_df.iterrows():
                    csv_row = [
                        row["question"],
                        row["choices"][0],  # A
                        row["choices"][1],  # B
                        row["choices"][2],  # C
                        row["choices"][3],  # D
                        chr(ord("A") + row["answer"]),  # Convert 0,1,2,3 to A,B,C,D
                    ]
                    csv_data.append(csv_row)

                # Save test CSV
                test_csv = pd.DataFrame(csv_data)
                test_file = f"data/test/{subject}_test.csv"
                test_csv.to_csv(test_file, header=False, index=False)

            if val_data:
                # Convert validation data (used as dev set)
                val_df = pd.DataFrame(val_data)

                csv_data = []
                for _, row in val_df.iterrows():
                    csv_row = [
                        row["question"],
                        row["choices"][0],  # A
                        row["choices"][1],  # B
                        row["choices"][2],  # C
                        row["choices"][3],  # D
                        chr(ord("A") + row["answer"]),  # Convert 0,1,2,3 to A,B,C,D
                    ]
                    csv_data.append(csv_row)

                # Save dev CSV
                dev_csv = pd.DataFrame(csv_data)
                dev_file = f"data/dev/{subject}_dev.csv"
                dev_csv.to_csv(dev_file, header=False, index=False)

        # Verify the download
        test_files = [f for f in os.listdir("data/test") if f.endswith("_test.csv")]
        dev_files = [f for f in os.listdir("data/dev") if f.endswith("_dev.csv")]

        print("\nDownload completed statistics:")
        print(f"  Test files: {len(test_files)} files")
        print(f"  Dev files: {len(dev_files)} files")

        if test_files and dev_files:
            print("✅ MMLU dataset download completed!")
            print("Data locations:")
            print("  - Test set: data/test/")
            print("  - Dev set: data/dev/")

            print("\nAvailable subject examples:")
            for i, f in enumerate(sorted(test_files)[:10]):
                subject = f.replace("_test.csv", "")
                print(f"  - {subject}")
            if len(test_files) > 10:
                print(f"  ... and {len(test_files) - 10} more subjects")

            print("\n🚀 You can now run MMLU tests:")
            print('   ./deploy-1-dynamo.sh "Qwen/Qwen3-0.6B"')
            print(
                '   python3 1-mmlu-dynamo.py --model "Qwen/Qwen3-0.6B" --number-of-subjects 15'
            )
        else:
            print("❌ Data download incomplete")

    except Exception as e:
        print(f"❌ Download failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MMLU dataset")
    args = parser.parse_args()
    download_mmlu()
