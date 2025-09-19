# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from dynamo._core import DistributedRuntime

# Todo add support for launching etcd
# pytestmark = pytest.mark.pre_merge


async def test_simple_put_get():
    # Initialize runtime
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, False)

    # Get etcd client
    etcd = runtime.do_not_use_etcd_client()

    # Write some key-value pairs
    test_keys = {
        "test/key1": b"value1",
        "test/key2": b"value2",
        "test/nested/key3": b"value3",
    }
    # Write each key-value pair
    for key, value in test_keys.items():
        print(f"Writing {key} = {value!r}")
        await etcd.kv_create_or_validate(key, value, None)

    print("Successfully wrote all keys to etcd")

    # Test kv_put
    put_key = "test/put_key"
    put_value = b"put_value"
    test_keys[put_key] = put_value
    print(f"Using kv_put to write {put_key} = {put_value!r}")
    await etcd.kv_put(put_key, put_value, None)

    # Test kv_get_prefix to read all keys
    print("\nReading all keys with prefix 'test/':")
    keys_values = await etcd.kv_get_prefix("test/")
    for item in keys_values:
        print(f"Retrieved {item['key']} = {item['value']!r}")
        assert test_keys[item["key"]] == item["value"]

    # Verify prefix filtering works
    print("\nReading keys with prefix 'test/nested/':")
    nested_keys_values = await etcd.kv_get_prefix("test/nested/")
    for item in nested_keys_values:
        print(f"Retrieved {item['key']} = {item['value']!r}")
        assert test_keys[item["key"]] == item["value"]

    # Shutdown runtime
    runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(test_simple_put_get())
