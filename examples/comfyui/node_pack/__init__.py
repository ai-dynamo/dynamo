# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .nodes_dynamo import (
    DynamoEndpointConfig,
    DynamoImageEdit,
    DynamoImageToVideo,
    DynamoListModels,
    DynamoTextToImage,
    DynamoTextToVideo,
)

NODE_CLASS_MAPPINGS = {
    "DynamoEndpointConfig": DynamoEndpointConfig,
    "DynamoTextToImage": DynamoTextToImage,
    "DynamoImageEdit": DynamoImageEdit,
    "DynamoTextToVideo": DynamoTextToVideo,
    "DynamoImageToVideo": DynamoImageToVideo,
    "DynamoListModels": DynamoListModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamoEndpointConfig": "Dynamo Endpoint Config",
    "DynamoTextToImage": "Dynamo Text-to-Image",
    "DynamoImageEdit": "Dynamo Image Edit",
    "DynamoTextToVideo": "Dynamo Text-to-Video",
    "DynamoImageToVideo": "Dynamo Image-to-Video",
    "DynamoListModels": "Dynamo List Models",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
