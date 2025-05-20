# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import json
import logging
import os
import sys
import yaml
from argparse import ArgumentTypeError
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DynamoArgumentParser(argparse.ArgumentParser):
    """
    A flexible argument parser for Dynamo components.
    
    Supports:
    - Command line arguments in standard format (--arg value)
    - Command line arguments in colon format (--arg:value)
    - Configuration from YAML files (-f/--config file.yaml)
    - Environment variables from DYNAMO_SERVICE_CONFIG
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the parser with common Dynamo arguments"""
        super().__init__(*args, **kwargs)
        
        # Add common arguments
        self.add_argument(
            "--namespace", "-ns",
            type=str,
            default="dynamo",
            help="Dynamo namespace for the component"
        )
        self.add_argument(
            "--component", "-c",
            type=str,
            help="Component name"
        )
        self.add_argument(
            "--config", "-f",
            type=str,
            help="Path to a YAML configuration file"
        )
    
    def parse_args(self, args=None, namespace=None):
        """Parse arguments with extended capabilities"""
        if args is None:
            args = sys.argv[1:]
        
        # Process colon format arguments
        args = self._process_colon_args(args)
        
        # Process config files
        args = self._process_config_files(args)
        
        # Process env vars from DYNAMO_SERVICE_CONFIG
        env_args = self._get_env_args()
        if env_args:
            # CLI args take precedence over env vars
            args = env_args + args
        
        # Parse with the processed arguments
        return super().parse_args(args, namespace)
    
    def _process_colon_args(self, args):
        """Convert --arg:value to --arg value"""
        processed_args = []
        for arg in args:
            if arg.startswith("-") and ":" in arg:
                # Split the argument at the colon
                parts = arg.split(":", 1)
                flag, value = parts
                processed_args.append(flag)
                processed_args.append(value)
            else:
                processed_args.append(arg)
        return processed_args
    
    def _process_config_files(self, args):
        """Process -f/--config file.yaml arguments"""
        processed_args = list(args)
        
        # Find config file arguments
        i = 0
        while i < len(processed_args):
            arg = processed_args[i]
            if arg in ["-f", "--config"] and i + 1 < len(processed_args):
                # Get the config file path
                config_file = processed_args[i + 1]
                # Remove the -f/--config and filename from args
                processed_args.pop(i)
                processed_args.pop(i)
                # Load config file arguments
                config_args = self._load_config_args(config_file)
                # Add config args at the current position
                processed_args[i:i] = config_args
            else:
                i += 1
        
        return processed_args
    
    def _load_config_args(self, file_path):
        """Load arguments from a YAML configuration file"""
        extension = file_path.split('.')[-1].lower()
        if extension not in ('yaml', 'yml'):
            raise ValueError(f"Config file must be of a yaml/yml type: {extension}")
        
        processed_args = []
        
        try:
            with open(file_path) as config_file:
                config = yaml.safe_load(config_file)
            
            for key, value in config.items():
                if isinstance(value, bool):
                    if value:
                        processed_args.append(f"--{key}")
                else:
                    processed_args.append(f"--{key}")
                    processed_args.append(str(value))
                    
        except Exception as ex:
            logger.error(f"Unable to read the config file at {file_path}: {ex}")
            raise
        
        return processed_args
    
    def _get_env_args(self):
        """Get arguments from DYNAMO_SERVICE_CONFIG environment variable"""
        env_args = []
        
        env_config = os.environ.get("DYNAMO_SERVICE_CONFIG")
        if not env_config:
            return env_args
        
        try:
            configs = json.loads(env_config)
            
            # Try to determine the service name from existing args
            service_name = None
            for arg in sys.argv:
                if arg.startswith("--component="):
                    service_name = arg.split("=")[1]
                    break
                elif arg.startswith("--component"):
                    idx = sys.argv.index(arg)
                    if idx + 1 < len(sys.argv):
                        service_name = sys.argv[idx + 1]
                    break
            
            if service_name and service_name in configs:
                # Convert service config to CLI args
                service_config = configs[service_name]
                for key, value in service_config.items():
                    # Skip special keys
                    if key in ["ServiceArgs", "common-configs"]:
                        continue
                    
                    # Convert to CLI args
                    if isinstance(value, bool):
                        if value:
                            env_args.append(f"--{key}")
                    else:
                        env_args.append(f"--{key}")
                        env_args.append(str(value))
        
        except json.JSONDecodeError:
            logger.warning("Failed to parse DYNAMO_SERVICE_CONFIG")
        
        return env_args
    
