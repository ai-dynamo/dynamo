#!/usr/bin/env python3
"""
Test script to verify HTTP/2 cleartext support in Dynamo frontend.

This script demonstrates how to enable HTTP/2 without TLS in the Dynamo frontend
and tests the functionality using curl with HTTP/2 support.
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_command(cmd, timeout=10):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def test_http2_frontend():
    """Test HTTP/2 cleartext support in Dynamo frontend."""
    print("🧪 Testing HTTP/2 cleartext support in Dynamo frontend")

    # Create a temporary model directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "test_model"
        model_dir.mkdir()

        # Create a minimal model config for testing
        config = {
            "model_type": "text-generation",
            "tokenizer_config": {"tokenizer_class": "GPT2Tokenizer"},
        }

        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Start the frontend with HTTP/2 enabled
        print("🚀 Starting Dynamo frontend with HTTP/2 enabled...")

        frontend_cmd = [
            "python",
            "-m",
            "dynamo.frontend",
            "--model-path",
            str(model_dir),
            "--model-name",
            "test-model",
            "--http-port",
            "8787",
            "--enable-http2",
        ]

        # Start the frontend process
        frontend_process = None
        try:
            frontend_process = subprocess.Popen(
                frontend_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Wait for the server to start
            print("⏳ Waiting for server to start...")
            time.sleep(8)  # Increased wait time

            # Check if process is still running
            if frontend_process.poll() is not None:
                # Process has terminated
                stdout, stderr = frontend_process.communicate()
                print(f"❌ Frontend process terminated unexpectedly!")
                print(f"   stdout: {stdout[:500]}")
                print(f"   stderr: {stderr[:500]}")
                return False

            # Test HTTP/2 connection using curl
            print("🔍 Testing HTTP/2 connection...")

            # Test 1: Check if server is running
            health_cmd = "curl -v --http2-prior-knowledge http://localhost:8787/health"
            returncode, stdout, stderr = run_command(health_cmd, timeout=15)

            if returncode == 0:
                print("✅ Health check passed!")
                print(f"   Response: {stdout[:200]}")
            else:
                print(f"❌ Health check failed!")
                print(f"   Return code: {returncode}")
                print(f"   stdout: {stdout[:300]}")
                print(f"   stderr: {stderr[:300]}")

                # Try to read frontend logs
                if frontend_process and frontend_process.poll() is None:
                    # Try simple HTTP/1.1 to see if server is up at all
                    simple_cmd = "curl -v http://localhost:8787/health"
                    rc2, out2, err2 = run_command(simple_cmd, timeout=5)
                    print(f"\n   HTTP/1.1 fallback test:")
                    print(f"   Return code: {rc2}")
                    print(f"   stderr: {err2[:300]}")

                return False

            # Test 2: Check HTTP/2 protocol
            protocol_cmd = "curl -s --http2-prior-knowledge -w '%{http_version}' http://localhost:8787/health"
            returncode, stdout, stderr = run_command(protocol_cmd)

            if returncode == 0 and "2" in stdout:
                print("✅ HTTP/2 protocol confirmed!")
                print(f"   HTTP Version: {stdout}")
            else:
                print(f"❌ HTTP/2 protocol test failed: {stderr}")
                print(f"   Output: {stdout}")
                return False

            # Test 3: Test models endpoint
            models_cmd = (
                "curl -s --http2-prior-knowledge http://localhost:8787/v1/models"
            )
            returncode, stdout, stderr = run_command(models_cmd)

            if returncode == 0:
                print("✅ Models endpoint accessible!")
                try:
                    models_data = json.loads(stdout)
                    print(f"   Found {len(models_data.get('data', []))} models")
                except json.JSONDecodeError:
                    print(f"   Response: {stdout[:100]}...")
            else:
                print(f"❌ Models endpoint failed: {stderr}")
                return False

            print("🎉 All HTTP/2 tests passed!")
            return True

        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            return False

        finally:
            # Clean up the frontend process
            if frontend_process:
                print("🧹 Cleaning up...")
                frontend_process.terminate()
                try:
                    frontend_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    frontend_process.kill()


def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")

    # Check if curl supports HTTP/2
    returncode, stdout, stderr = run_command("curl --version")
    if returncode != 0:
        print("❌ curl is not available")
        return False

    if "HTTP2" not in stdout:
        print("❌ curl does not support HTTP/2")
        return False

    print("✅ curl with HTTP/2 support found")

    # Check if dynamo.frontend module is available
    returncode, stdout, stderr = run_command("python -c 'import dynamo.frontend'")
    if returncode != 0:
        print("❌ dynamo.frontend module not available")
        print(f"   Error: {stderr}")
        return False

    print("✅ dynamo.frontend module found")
    return True


def main():
    """Main test function."""
    print("🧪 Dynamo Frontend HTTP/2 Test Suite")
    print("=" * 50)

    if not check_dependencies():
        print("❌ Dependency check failed")
        sys.exit(1)

    if test_http2_frontend():
        print("🎉 All tests passed! HTTP/2 cleartext is working correctly.")
        sys.exit(0)
    else:
        print("❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
