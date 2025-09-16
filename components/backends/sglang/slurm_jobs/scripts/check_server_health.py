# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pytest: skip-file

import json
import sys

"""
A file that parses the response of `curl <host_ip>:<host_port>/health` endpoint
to check whether the server is ready to be benchmarked. 

Usage: 

```bash
curl_result=$(curl "${host_ip}:${host_port}/health" 2> /dev/null)
check_result=$(python3 check_server_health.py $N_PREFILL $N_DECODE <<< $curl_result)

# ... then do subsequent processing for check_result ...
```
"""

def check_server_health(
    expected_n_prefill, 
    expected_n_decode, 
    response):
    
    try: 
        decoded_response = json.loads(response)
    except json.JSONDecodeError:
        return f"Got invalid response from server that leads to JSON Decode error: {response}"

    if "instances" not in decoded_response:
        return f"Key 'instances' not found in response: {response}"
    
    for instance in decoded_response['instances']:
        if instance.get('endpoint') == 'generate':
            if instance.get('component') == "prefill":
                expected_n_prefill -= 1
            if instance.get('component') == 'backend':
                expected_n_decode -= 1
    
    if expected_n_prefill == 0 and expected_n_decode == 0:
        return f"Model is ready. Response: {response}"
    else:
        return f"Model is not ready, waiting for {expected_n_prefill} prefills and {expected_n_decode} decodes to spin up. Response: {response}"

if __name__ == "__main__":
    expected_n_prefill = sys.argv[1]
    expected_n_decode = sys.argv[2]

    response = sys.stdin.read()
    print(
        check_server_health(
            expected_n_prefill=expected_n_prefill, 
            expected_n_decode=expected_n_decode, 
            response=response
        )
    )
