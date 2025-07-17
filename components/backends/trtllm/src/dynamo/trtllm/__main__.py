# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main entry point for the dynamo.trtllm module.

This module can be run as: python -m dynamo.trtllm
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import utils
#parent_dir = Path(__file__).parent
#if str(parent_dir) not in sys.path:
#    sys.path.insert(0, str(parent_dir))

import uvloop
from dynamo.trtllm.worker import worker


def main():
    """Main entry point for the dynamo.trtllm module."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run the worker
    try:
        uvloop.install()
        asyncio.run(worker())
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logging.error(f"Error running dynamo.trtllm: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 