#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo "commit id: $TRT_LLM_GIT_COMMIT"
echo "ucx info: $(ucx_info -v)"
echo "hostname: $(hostname)"

hostname=$(hostname)
short_hostname=$(echo "$hostname" | awk -F'.' '{print $1}')
echo "short_hostname: ${short_hostname}"

config_file=$1

# Check and replace the hostname setting in config_file
if [ -f "$config_file" ]; then
    # Use sed to find the hostname line and check if replacement is needed
    if grep -q "hostname:" "$config_file"; then
        # Extract the current hostname value from the config
        current_hostname=$(grep "hostname:" "$config_file" | sed 's/.*hostname:[ ]*//' | awk '{print $1}')

        if [ "$current_hostname" != "$short_hostname" ]; then
            echo "Replacing hostname '$current_hostname' with '$short_hostname' in $config_file"
            # Use sed to replace the hostname value
            sed -i "s/hostname:[ ]*[^ ]*/hostname: $short_hostname/" "$config_file"
        else
            echo "Hostname '$current_hostname' already matches '$short_hostname', no change needed"
        fi
    else
        echo "No hostname setting found in $config_file"
    fi
else
    echo "Config file $config_file not found"
fi

# Start NATS
nats-server -js &

# Start etcd
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &

# Wait for NATS/etcd to startup
sleep 3

# Start OpenAI Frontend which will dynamically discover workers when they startup
# NOTE: This is a blocking call.
python3 -m dynamo.frontend --http-port 8000

