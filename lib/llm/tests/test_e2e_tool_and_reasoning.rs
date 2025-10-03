// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use std::path::Path;
use std::fs;
use serde_json::Value;
use dynamo_llm::protocols::{DataStream, codec::{Message, SseCodecError, create_message_stream}};

const BASE_DATA_DIR: &str = "data/e2e/";

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_e2e_vllm_tool_and_reasoning_flow_gpt_oss_vllm(){
        let _data_dir = format!("{}/vllm/gpt-oss", BASE_DATA_DIR);
        
        // For now, use stream_recordings directory as test data
        let test_data_dir = "stream_recordings";
        
        if !Path::new(test_data_dir).exists() {
            println!("Test data directory not found: {}", test_data_dir);
            return;
        }
        
        // Iterate over JSON files in the directory
        let entries = fs::read_dir(test_data_dir).unwrap();
        
        for entry in entries {
            let entry = entry.unwrap();
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                println!("Processing file: {:?}", path);
                
                // Read and convert JSON file to stream up to line 801
                let stream = create_stream_from_json_file(&path).await;
                
                // Apply preprocessor transform up to line 801
                let processed_stream = apply_preprocessor_transform(stream).await;
                
                // Collect a few items to verify the stream works
                let mut stream_pin = std::pin::pin!(processed_stream);
                let mut count = 0;
                while let Some(item) = stream_pin.next().await {
                    println!("Processed item {}: {:?}", count, item);
                    count += 1;
                    if count >= 5 { // Just process first 5 items for testing
                        break;
                    }
                }
            }
        }
    }
    
    async fn create_stream_from_json_file(file_path: &Path) -> impl futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> {
        let data = fs::read_to_string(file_path).unwrap();
        
        // Parse JSON array format (like stream_recordings)
        let json_array: Vec<Value> = serde_json::from_str(&data).unwrap();
        
        // Convert to SSE format that create_message_stream expects
        let mut sse_data = String::new();
        for entry in json_array {
            if let Some(response_data) = entry.get("response_data") {
                if let Some(data_obj) = response_data.get("data") {
                    sse_data.push_str(&format!("data: {}\n\n", serde_json::to_string(data_obj).unwrap()));
                }
            }
        }
        
        // Create message stream from SSE data
        let message_stream = create_message_stream(&sse_data);
        
        // Convert to NvCreateChatCompletionStreamResponse stream
        convert_to_response_stream(message_stream)
    }
    
    fn convert_to_response_stream(
        message_stream: DataStream<Result<Message, SseCodecError>>
    ) -> impl futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> {
        message_stream.filter_map(|msg_result| async move {
            match msg_result {
                Ok(msg) => {
                    if let Some(data) = msg.data {
                        match serde_json::from_str::<NvCreateChatCompletionStreamResponse>(&data) {
                            Ok(response) => Some(Annotated::from_data(response)),
                            Err(e) => {
                                println!("Failed to parse response: {}", e);
                                None
                            }
                        }
                    } else {
                        None
                    }
                }
                Err(e) => {
                    println!("Message parsing error: {:?}", e);
                    None
                }
            }
        })
    }
    
    async fn apply_preprocessor_transform(
        stream: impl futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>
    ) -> impl futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> {
        // This simulates the preprocessing up to line 801
        // In a real implementation, you would:
        // 1. Create an OpenAIPreprocessor instance
        // 2. Create a mock AsyncEngineContext 
        // 3. Apply transform_postprocessor_stream
        
        // For now, just pass through the stream as a demonstration
        stream.map(|item| {
            println!("Transform applied to: {:?}", item);
            item
        })
    }
}