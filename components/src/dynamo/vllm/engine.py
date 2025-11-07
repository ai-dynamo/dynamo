# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified vLLM engine interface that supports both regular AsyncLLM
and CheckpointableAsyncLLM based on configuration.
"""

import errno
import logging
import os
import shutil
import time
import uuid
from typing import Optional

from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerFactory

from dynamo.vllm.checkpoint import CheckpointableAsyncLLM


logger = logging.getLogger(__name__)


def create_vllm_engine(
    vllm_config,
    usage_context: UsageContext,
    stat_loggers: Optional[list[StatLoggerFactory]] = None,
    disable_log_requests: bool = False,
    disable_log_stats: bool = False,
    enable_checkpointing: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> AsyncLLM:
    """
    Factory function to create either AsyncLLM or CheckpointableAsyncLLM
    based on configuration.
    
    When enable_checkpointing=False, returns standard AsyncLLM (no overhead).
    When enable_checkpointing=True, returns CheckpointableAsyncLLM (subprocess + ZMQ).
    
    Args:
        vllm_config: vLLM configuration
        usage_context: Usage context for telemetry
        stat_loggers: Optional stat loggers
        disable_log_requests: Whether to disable request logging
        disable_log_stats: Whether to disable stats logging
        enable_checkpointing: Whether to enable CRIU checkpointing support
        checkpoint_dir: Optional checkpoint directory for restore
        
    Returns:
        AsyncLLM instance (standard or checkpointable variant)
    """
    factory = []
    if stat_loggers:
        factory.extend(stat_loggers)
    
    if not enable_checkpointing:
        # Standard path: use AsyncLLM directly (no subprocess, no overhead)
        logger.info("Creating standard AsyncLLM engine (checkpointing disabled)")
        return AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=factory,
            disable_log_requests=disable_log_requests,
            disable_log_stats=disable_log_stats,
        )
    else:
        # Checkpointing enabled: use CheckpointableAsyncLLM
        # Use vLLM's built-in config hash for checkpoint directory naming
        config_hash = vllm_config.compute_hash()[:8]  # Use first 8 chars for brevity
        
        if checkpoint_dir:
            # Create checkpoint path with config hash subfolder
            checkpoint_dir_with_hash = os.path.join(checkpoint_dir, config_hash)
            logger.info(f"Checkpoint directory with config hash: {checkpoint_dir_with_hash}")
        else:
            checkpoint_dir_with_hash = None
        
        # Only auto_start if checkpoint doesn't exist yet
        checkpoint_exists = checkpoint_dir_with_hash and os.path.exists(checkpoint_dir_with_hash)
        auto_start = not checkpoint_exists
        
        if checkpoint_exists:
            logger.info(f"Creating CheckpointableAsyncLLM engine for checkpoint restore (auto_start=False, config_hash={config_hash})")
        else:
            logger.info(f"Creating CheckpointableAsyncLLM engine (auto_start=True, will create checkpoint, config_hash={config_hash})")
        
        vllm_config.model_config.enable_sleep_mode = True
        vllm_config.parallel_config.disable_custom_all_reduce = True

        engine = CheckpointableAsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=factory,
            enable_log_requests=not disable_log_requests,
            disable_log_stats=disable_log_stats,
            auto_start=auto_start,
        )
        
        # Store the full checkpoint path on the engine for later use
        if checkpoint_dir_with_hash:
            engine.checkpoint_dir = checkpoint_dir_with_hash
        
        return engine


async def _sync_vllm_config_from_subprocess(engine: CheckpointableAsyncLLM):
    """Helper to sync vLLM config from restored subprocess.
    
    Args:
        engine: CheckpointableAsyncLLM instance
        
    Returns:
        The synced VllmConfig
        
    Raises:
        RuntimeError: If get_vllm_config method is not available
    """
    if hasattr(engine, 'get_vllm_config'):
        logger.info("Syncing vLLM config from restored subprocess...")
        engine.vllm_config = await engine.get_vllm_config()
        logger.info(f"Synced config - num_gpu_blocks: {engine.vllm_config.cache_config.num_gpu_blocks}")
        return engine.vllm_config
    else:
        raise RuntimeError("get_vllm_config method not found on engine")


async def initialize_engine(engine: AsyncLLM, enable_checkpointing: bool, checkpoint_dir: Optional[str] = None):
    """
    Initialize the engine after creation.
    
    Handles three cases:
    1. Checkpointing disabled: No-op (engine already ready)
    2. Checkpointing enabled + checkpoint exists: Restore from checkpoint
    3. Checkpointing enabled + checkpoint doesn't exist: Wait for ready, checkpoint, 
       atomic move (race with other workers), then restore
    
    Args:
        engine: Engine instance to initialize
        enable_checkpointing: Whether checkpointing is enabled
        checkpoint_dir: Optional checkpoint directory (destination path).
                       If engine already has checkpoint_dir set (from create_vllm_engine),
                       that takes precedence.
    
    Returns:
        The vllm_config (synced from subprocess if checkpointing is enabled)
    """
    # Case 1: Checkpointing disabled - return the config from the engine
    if not enable_checkpointing:
        logger.debug("Standard AsyncLLM is ready (no initialization needed)")
        return engine.vllm_config if hasattr(engine, 'vllm_config') else None
    
    if not isinstance(engine, CheckpointableAsyncLLM):
        logger.warning("Engine is not CheckpointableAsyncLLM, skipping initialization")
        return engine.vllm_config if hasattr(engine, 'vllm_config') else None
    
    # Use checkpoint_dir from engine if already set (includes config hash)
    # Otherwise use the provided checkpoint_dir
    if hasattr(engine, 'checkpoint_dir') and engine.checkpoint_dir:
        checkpoint_dir = engine.checkpoint_dir
        logger.debug(f"Using checkpoint_dir from engine: {checkpoint_dir}")
    
    if not checkpoint_dir:
        raise ValueError("checkpoint_dir must be provided when enable_checkpointing=True")
    
    # Case 2: Checkpoint already exists - restore from it
    if os.path.exists(checkpoint_dir):
        logger.info(f"Checkpoint directory exists: {checkpoint_dir}")
        logger.info(f"Starting checkpoint restore from: {checkpoint_dir}")
        start_time = time.perf_counter()
        
        engine.checkpoint_dir = checkpoint_dir
        await engine.criu_resume()
        
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Checkpoint restore completed successfully in {elapsed_time:.2f} seconds")
        
        # Sync config from restored subprocess
        return await _sync_vllm_config_from_subprocess(engine)
    
    # Case 3: No checkpoint exists - create one with atomic move
    logger.info("No checkpoint found, will create one")
    logger.info("Waiting for CheckpointableAsyncLLM engine to initialize...")
    start_time = time.perf_counter()
    
    await engine.wait_until_ready()
    
    elapsed_time = time.perf_counter() - start_time
    logger.info(f"CheckpointableAsyncLLM engine is ready (initialization time: {elapsed_time:.2f}s)")
    
    # Create checkpoint in temporary directory (same filesystem as checkpoint_dir)
    temp_dir = f"{checkpoint_dir}.tmp.{uuid.uuid4().hex[:8]}"
    logger.info(f"Creating checkpoint in temporary directory: {temp_dir}")
    
    checkpoint_start = time.perf_counter()
    await engine.criu_checkpoint(temp_dir)
    checkpoint_elapsed = time.perf_counter() - checkpoint_start
    logger.info(f"Checkpoint created in {checkpoint_elapsed:.2f} seconds")
    
    # Attempt atomic move to final destination using os.rename()
    # os.rename() is atomic and will fail with EEXIST if destination already exists
    logger.info(f"Attempting atomic move to final destination: {checkpoint_dir}")
    try:
        os.rename(temp_dir, checkpoint_dir)
        logger.info(f"Successfully moved checkpoint to {checkpoint_dir} (this worker won the race)")
    except OSError as e:
        if e.errno == errno.EEXIST:
            logger.info(f"Another worker already created checkpoint at {checkpoint_dir} (race lost)")
            # Clean up our temp directory
            logger.info(f"Cleaning up temporary checkpoint directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        else:
            raise
    
    # Now restore from the final checkpoint (whether we created it or another worker did)
    logger.info(f"Restoring from checkpoint: {checkpoint_dir}")
    restore_start = time.perf_counter()
    
    engine.checkpoint_dir = checkpoint_dir
    await engine.criu_resume()
    
    restore_elapsed = time.perf_counter() - restore_start
    logger.info(f"Checkpoint restore completed in {restore_elapsed:.2f} seconds")
    
    # Sync config from restored subprocess
    return await _sync_vllm_config_from_subprocess(engine)
