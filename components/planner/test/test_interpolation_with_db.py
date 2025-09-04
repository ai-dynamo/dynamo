#!/usr/bin/env python3
"""
Test script for testing prefill and decode interpolators with real database data.
This script reads data points that were previously submitted to the database and
tests the interpolation results.
"""

import sys
import os
import asyncio
import numpy as np
from typing import Optional

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from dynamo.planner.utils.perf_interpolation import (
    create_prefill_interpolator,
    create_decode_interpolator
)
from dynamo.planner.db.worker_profile_client import WorkerProfileClient


class InterpolationDBTester:
    """Test class for testing interpolators with database data."""
    
    def __init__(self):
        """Initialize the tester with a WorkerProfileClient."""
        self.profile_client = WorkerProfileClient()
        
    async def test_prefill_interpolator_with_db(self, 
                                               model_name: str = "nvidia/Llama-3.3-70B-Instruct-FP8",
                                               tp_size: int = 2,
                                               backend: str = "vllm",
                                               gpu_type: str = "h100"):
        """Test prefill interpolator with data from database."""
        print(f"\n{'='*60}")
        print(f"Testing Prefill Interpolator with Database Data")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"TP Size: {tp_size}")
        print(f"Backend: {backend}")
        print(f"GPU Type: {gpu_type}")
        
        try:
            # Create prefill interpolator from database
            print("\nCreating prefill interpolator from database...")
            prefill_interpolator = await create_prefill_interpolator(
                model_name=model_name,
                tp_size=tp_size,
                backend=backend,
                gpu_type=gpu_type,
                gpu_count=8,
                node_count=1,
                max_context_length=128000
            )
            
            print(f"✓ Prefill interpolator created successfully!")
            print(f"  Data points: {len(prefill_interpolator.prefill_isl)}")
            print(f"  ISL range: {prefill_interpolator.min_isl} - {prefill_interpolator.max_isl}")
            
            # Test interpolation at various input sequence lengths
            test_isl_values = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
            
            print(f"\nTesting TTFT interpolation:")
            print(f"{'ISL':<10} {'TTFT (s)':<12} {'Throughput (tokens/s/gpu)':<25}")
            print(f"{'-'*50}")
            
            for isl in test_isl_values:
                if isl <= prefill_interpolator.max_isl:
                    ttft = prefill_interpolator.interpolate_ttft(isl)
                    throughput = prefill_interpolator.interpolate_thpt_per_gpu(isl)
                    print(f"{isl:<10} {ttft:<12.4f} {throughput:<25.2f}")
                else:
                    print(f"{isl:<10} {'N/A':<12} {'N/A':<25}")
            
            # Test some specific interpolation scenarios
            print(f"\nDetailed interpolation tests:")
            
            # Test at minimum ISL
            min_isl = prefill_interpolator.min_isl
            min_ttft = prefill_interpolator.interpolate_ttft(min_isl)
            min_throughput = prefill_interpolator.interpolate_thpt_per_gpu(min_isl)
            print(f"  Min ISL ({min_isl}): TTFT={min_ttft:.4f}s, Throughput={min_throughput:.2f} tokens/s/gpu")
            
            # Test at maximum ISL
            max_isl = prefill_interpolator.max_isl
            max_ttft = prefill_interpolator.interpolate_ttft(max_isl)
            max_throughput = prefill_interpolator.interpolate_thpt_per_gpu(max_isl)
            print(f"  Max ISL ({max_isl}): TTFT={max_ttft:.4f}s, Throughput={max_throughput:.2f} tokens/s/gpu")
            
            # Test at middle ISL
            mid_isl = (min_isl + max_isl) // 2
            mid_ttft = prefill_interpolator.interpolate_ttft(mid_isl)
            mid_throughput = prefill_interpolator.interpolate_thpt_per_gpu(mid_isl)
            print(f"  Mid ISL ({mid_isl}): TTFT={mid_ttft:.4f}s, Throughput={mid_throughput:.2f} tokens/s/gpu")
            
            return prefill_interpolator
            
        except ValueError as e:
            if "Insufficient data points" in str(e):
                print(f"⚠ Insufficient data points for prefill interpolator: {e}")
                print(f"  This usually means you need to submit more prefill data points to the database.")
                print(f"  Try running batch-submit with more prefill data.")
            else:
                print(f"✗ Error creating prefill interpolator: {e}")
            return None
        except Exception as e:
            print(f"✗ Error creating prefill interpolator: {e}")
            return None
    
    async def test_decode_interpolator_with_db(self,
                                              model_name: str = "nvidia/Llama-3.3-70B-Instruct-FP8",
                                              tp_size: int = 2,
                                              backend: str = "vllm",
                                              gpu_type: str = "h100"):
        """Test decode interpolator with data from database."""
        print(f"\n{'='*60}")
        print(f"Testing Decode Interpolator with Database Data")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"TP Size: {tp_size}")
        print(f"Backend: {backend}")
        print(f"GPU Type: {gpu_type}")
        
        try:
            # Create decode interpolator from database
            print("\nCreating decode interpolator from database...")
            decode_interpolator = await create_decode_interpolator(
                model_name=model_name,
                tp_size=tp_size,
                backend=backend,
                gpu_type=gpu_type,
                gpu_count=8,
                node_count=1,
                max_context_length=128000,
                resolution=50  # Lower resolution for faster testing
            )
            
            print(f"✓ Decode interpolator created successfully!")
            print(f"  KV usage range: {decode_interpolator.min_kv_usage:.2f} - {decode_interpolator.max_kv_usage:.2f}")
            print(f"  Context length range: {decode_interpolator.min_context_length} - {decode_interpolator.max_context_length}")
            
            # Test interpolation at various KV usage and context length combinations
            test_kv_usage_values = [0.1, 0.3, 0.5, 0.7, 0.9]
            test_context_lengths = [1000, 5000, 10000, 50000, 100000]
            
            print(f"\nTesting ITL interpolation:")
            print(f"{'KV Usage':<10} {'Context Length':<15} {'ITL (s)':<12} {'Throughput (tokens/s/gpu)':<25}")
            print(f"{'-'*70}")
            
            for kv_usage in test_kv_usage_values:
                for context_length in test_context_lengths:
                    if (kv_usage <= 1.0 and 
                        context_length <= decode_interpolator.max_context_length):
                        # Calculate concurrency from kv_usage and context_length
                        concurrency = kv_usage * decode_interpolator.max_kv_tokens / context_length
                        itl = decode_interpolator.interpolate_itl(concurrency, context_length)
                        throughput = decode_interpolator.interpolate_thpt_per_gpu(concurrency, context_length)
                        print(f"{kv_usage:<10.2f} {context_length:<15} {itl:<12.4f} {throughput:<25.2f}")
                    else:
                        print(f"{kv_usage:<10.2f} {context_length:<15} {'N/A':<12} {'N/A':<25}")
            
            # Test some specific interpolation scenarios
            print(f"\nDetailed interpolation tests:")
            
            # Test at minimum values
            min_kv = 0.1  # Minimum KV usage
            min_ctx = decode_interpolator.min_context_length
            min_concurrency = min_kv * decode_interpolator.max_kv_tokens / min_ctx
            min_itl = decode_interpolator.interpolate_itl(min_concurrency, min_ctx)
            min_throughput = decode_interpolator.interpolate_thpt_per_gpu(min_concurrency, min_ctx)
            print(f"  Min values (KV={min_kv:.2f}, CTX={min_ctx}): ITL={min_itl:.4f}s, Throughput={min_throughput:.2f} tokens/s/gpu")
            
            # Test at maximum values
            max_kv = 0.9  # Maximum KV usage
            max_ctx = decode_interpolator.max_context_length
            max_concurrency = max_kv * decode_interpolator.max_kv_tokens / max_ctx
            max_itl = decode_interpolator.interpolate_itl(max_concurrency, max_ctx)
            max_throughput = decode_interpolator.interpolate_thpt_per_gpu(max_concurrency, max_ctx)
            print(f"  Max values (KV={max_kv:.2f}, CTX={max_ctx}): ITL={max_itl:.4f}s, Throughput={max_throughput:.2f} tokens/s/gpu")
            
            # Test at middle values
            mid_kv = 0.5  # Middle KV usage
            mid_ctx = (min_ctx + max_ctx) // 2
            mid_concurrency = mid_kv * decode_interpolator.max_kv_tokens / mid_ctx
            mid_itl = decode_interpolator.interpolate_itl(mid_concurrency, mid_ctx)
            mid_throughput = decode_interpolator.interpolate_thpt_per_gpu(mid_concurrency, mid_ctx)
            print(f"  Mid values (KV={mid_kv:.2f}, CTX={mid_ctx}): ITL={mid_itl:.4f}s, Throughput={mid_throughput:.2f} tokens/s/gpu")
            
            return decode_interpolator
            
        except ValueError as e:
            if "Insufficient data points" in str(e):
                print(f"⚠ Insufficient data points for decode interpolator: {e}")
                print(f"  This usually means you need to submit more decode data points to the database.")
                print(f"  Try running batch-submit with more decode data.")
                print(f"  For 2D interpolation, you need at least 4 data points with different KV usage and context length combinations.")
            else:
                print(f"✗ Error creating decode interpolator: {e}")
            return None
        except Exception as e:
            print(f"✗ Error creating decode interpolator: {e}")
            return None
    
    def list_available_configurations(self):
        """List available configurations in the database."""
        print(f"\n{'='*60}")
        print(f"Available Configurations in Database")
        print(f"{'='*60}")
        
        try:
            # List available models
            models = self.profile_client.list_available_models()
            print(f"Available models: {models}")
            
            # Get configurations for a specific model
            if models:
                test_model = models[0]
                print(f"\nTesting with model: {test_model}")
                
                # Get prefill configurations
                prefill_configs = self.profile_client.get_engine_configurations(
                    hf_model_name=test_model,
                    hardware_sku="h100",
                    context_length=4096,
                    mode="p"
                )
                print(f"Prefill configurations: {prefill_configs}")
                
                # Get decode configurations
                decode_configs = self.profile_client.get_engine_configurations(
                    hf_model_name=test_model,
                    hardware_sku="h100",
                    context_length=4096,
                    mode="d"
                )
                print(f"Decode configurations: {decode_configs}")
                
        except Exception as e:
            print(f"✗ Error listing configurations: {e}")
    
    async def run_comprehensive_test(self):
        """Run comprehensive tests with different configurations."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE INTERPOLATION TEST WITH DATABASE DATA")
        print(f"{'='*80}")
        
        # List available configurations first
        self.list_available_configurations()
        
        # Test configurations to try
        test_configs = [
            {
                "model_name": "nvidia/Llama-3.3-70B-Instruct-FP8",
                "tp_size": 8,
                "backend": "vllm",
                "gpu_type": "h100"
            }
        ]
        
        for config in test_configs:
            print(f"\n{'='*80}")
            print(f"Testing Configuration: {config}")
            print(f"{'='*80}")
            
            # Test prefill interpolator
            prefill_interpolator = await self.test_prefill_interpolator_with_db(**config)
            
            # Test decode interpolator
            decode_interpolator = await self.test_decode_interpolator_with_db(**config)
            
            if prefill_interpolator and decode_interpolator:
                print(f"\n✓ Both interpolators created successfully for {config['model_name']} with TP={config['tp_size']}")
            else:
                print(f"\n✗ Some interpolators failed for {config['model_name']} with TP={config['tp_size']}")


async def main():
    """Main function to run the interpolation tests."""
    tester = InterpolationDBTester()
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
