use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::types::responses::{OutputContent, OutputItem, Response, ResponseLogProb, SummaryPart};

/// Event types for streaming responses from the Responses API
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
#[schema(no_recursion)]
#[serde(tag = "type")]
pub enum ResponseStreamEvent {
    /// An event that is emitted when a response is created.
    #[serde(rename = "response.created")]
    ResponseCreated(ResponseCreatedEvent),
    /// Emitted when the response is in progress.
    #[serde(rename = "response.in_progress")]
    ResponseInProgress(ResponseInProgressEvent),
    /// Emitted when the model response is complete.
    #[serde(rename = "response.completed")]
    ResponseCompleted(ResponseCompletedEvent),
    /// An event that is emitted when a response fails.
    #[serde(rename = "response.failed")]
    ResponseFailed(ResponseFailedEvent),
    /// An event that is emitted when a response finishes as incomplete.
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete(ResponseIncompleteEvent),
    /// Emitted when a new output item is added.
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded(ResponseOutputItemAddedEvent),
    /// Emitted when an output item is marked done.
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone(ResponseOutputItemDoneEvent),
    /// Emitted when a new content part is added.
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded(ResponseContentPartAddedEvent),
    /// Emitted when a content part is done.
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone(ResponseContentPartDoneEvent),
    /// Emitted when there is an additional text delta.
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta(ResponseTextDeltaEvent),
    /// Emitted when text content is finalized.
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone(ResponseTextDoneEvent),
    /// Emitted when there is a partial refusal text.
    #[serde(rename = "response.refusal.delta")]
    ResponseRefusalDelta(ResponseRefusalDeltaEvent),
    #[serde(rename = "response.refusal.done")]
    /// Emitted when refusal text is finalized.
    ResponseRefusalDone(ResponseRefusalDoneEvent),
    /// Emitted when there is a partial function-call arguments delta.
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta(ResponseFunctionCallArgumentsDeltaEvent),
    /// Emitted when function-call arguments are finalized.
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone(ResponseFunctionCallArgumentsDoneEvent),
    /// Emitted when a file search call is initiated.
    #[serde(rename = "response.file_search_call.in_progress")]
    ResponseFileSearchCallInProgress(ResponseFileSearchCallInProgressEvent),
    /// Emitted when a file search is currently searching.
    #[serde(rename = "response.file_search_call.searching")]
    ResponseFileSearchCallSearching(ResponseFileSearchCallSearchingEvent),
    /// Emitted when a file search call is completed (results found).
    #[serde(rename = "response.file_search_call.completed")]
    ResponseFileSearchCallCompleted(ResponseFileSearchCallCompletedEvent),
    /// Emitted when a web search call is initiated.
    #[serde(rename = "response.web_search_call.in_progress")]
    ResponseWebSearchCallInProgress(ResponseWebSearchCallInProgressEvent),
    /// Emitted when a web search call is executing.
    #[serde(rename = "response.web_search_call.searching")]
    ResponseWebSearchCallSearching(ResponseWebSearchCallSearchingEvent),
    /// Emitted when a web search call is completed.
    #[serde(rename = "response.web_search_call.completed")]
    ResponseWebSearchCallCompleted(ResponseWebSearchCallCompletedEvent),
    /// Emitted when a new reasoning summary part is added.
    #[serde(rename = "response.reasoning_summary_part.added")]
    ResponseReasoningSummaryPartAdded(ResponseReasoningSummaryPartAddedEvent),
    /// Emitted when a reasoning summary part is completed.
    #[serde(rename = "response.reasoning_summary_part.done")]
    ResponseReasoningSummaryPartDone(ResponseReasoningSummaryPartDoneEvent),
    /// Emitted when a delta is added to a reasoning summary text.
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ResponseReasoningSummaryTextDelta(ResponseReasoningSummaryTextDeltaEvent),
    /// Emitted when a reasoning summary text is completed.
    #[serde(rename = "response.reasoning_summary_text.done")]
    ResponseReasoningSummaryTextDone(ResponseReasoningSummaryTextDoneEvent),
    /// Emitted when a delta is added to a reasoning text.
    #[serde(rename = "response.reasoning_text.delta")]
    ResponseReasoningTextDelta(ResponseReasoningTextDeltaEvent),
    /// Emitted when a reasoning text is completed.
    #[serde(rename = "response.reasoning_text.done")]
    ResponseReasoningTextDone(ResponseReasoningTextDoneEvent),
    /// Emitted when an image generation tool call has completed and the final image is available.
    #[serde(rename = "response.image_generation_call.completed")]
    ResponseImageGenerationCallCompleted(ResponseImageGenCallCompletedEvent),
    /// Emitted when an image generation tool call is actively generating an image (intermediate state).
    #[serde(rename = "response.image_generation_call.generating")]
    ResponseImageGenerationCallGenerating(ResponseImageGenCallGeneratingEvent),
    /// Emitted when an image generation tool call is in progress.
    #[serde(rename = "response.image_generation_call.in_progress")]
    ResponseImageGenerationCallInProgress(ResponseImageGenCallInProgressEvent),
    /// Emitted when a partial image is available during image generation streaming.
    #[serde(rename = "response.image_generation_call.partial_image")]
    ResponseImageGenerationCallPartialImage(ResponseImageGenCallPartialImageEvent),
    /// Emitted when there is a delta (partial update) to the arguments of an MCP tool call.
    #[serde(rename = "response.mcp_call_arguments.delta")]
    ResponseMCPCallArgumentsDelta(ResponseMCPCallArgumentsDeltaEvent),
    /// Emitted when the arguments for an MCP tool call are finalized.
    #[serde(rename = "response.mcp_call_arguments.done")]
    ResponseMCPCallArgumentsDone(ResponseMCPCallArgumentsDoneEvent),
    /// Emitted when an MCP tool call has completed successfully.
    #[serde(rename = "response.mcp_call.completed")]
    ResponseMCPCallCompleted(ResponseMCPCallCompletedEvent),
    /// Emitted when an MCP tool call has failed.
    #[serde(rename = "response.mcp_call.failed")]
    ResponseMCPCallFailed(ResponseMCPCallFailedEvent),
    /// Emitted when an MCP tool call is in progress.
    #[serde(rename = "response.mcp_call.in_progress")]
    ResponseMCPCallInProgress(ResponseMCPCallInProgressEvent),
    /// Emitted when the list of available MCP tools has been successfully retrieved.
    #[serde(rename = "response.mcp_list_tools.completed")]
    ResponseMCPListToolsCompleted(ResponseMCPListToolsCompletedEvent),
    /// Emitted when the attempt to list available MCP tools has failed.
    #[serde(rename = "response.mcp_list_tools.failed")]
    ResponseMCPListToolsFailed(ResponseMCPListToolsFailedEvent),
    /// Emitted when the system is in the process of retrieving the list of available MCP tools.
    #[serde(rename = "response.mcp_list_tools.in_progress")]
    ResponseMCPListToolsInProgress(ResponseMCPListToolsInProgressEvent),
    /// Emitted when a code interpreter call is in progress.
    #[serde(rename = "response.code_interpreter_call.in_progress")]
    ResponseCodeInterpreterCallInProgress(ResponseCodeInterpreterCallInProgressEvent),
    /// Emitted when the code interpreter is actively interpreting the code snippet.
    #[serde(rename = "response.code_interpreter_call.interpreting")]
    ResponseCodeInterpreterCallInterpreting(ResponseCodeInterpreterCallInterpretingEvent),
    /// Emitted when the code interpreter call is completed.
    #[serde(rename = "response.code_interpreter_call.completed")]
    ResponseCodeInterpreterCallCompleted(ResponseCodeInterpreterCallCompletedEvent),
    /// Emitted when a partial code snippet is streamed by the code interpreter.
    #[serde(rename = "response.code_interpreter_call_code.delta")]
    ResponseCodeInterpreterCallCodeDelta(ResponseCodeInterpreterCallCodeDeltaEvent),
    /// Emitted when the code snippet is finalized by the code interpreter.
    #[serde(rename = "response.code_interpreter_call_code.done")]
    ResponseCodeInterpreterCallCodeDone(ResponseCodeInterpreterCallCodeDoneEvent),
    /// Emitted when an annotation is added to output text content.
    #[serde(rename = "response.output_text.annotation.added")]
    ResponseOutputTextAnnotationAdded(ResponseOutputTextAnnotationAddedEvent),
    /// Emitted when a response is queued and waiting to be processed.
    #[serde(rename = "response.queued")]
    ResponseQueued(ResponseQueuedEvent),
    /// Event representing a delta (partial update) to the input of a custom tool call.
    #[serde(rename = "response.custom_tool_call_input.delta")]
    ResponseCustomToolCallInputDelta(ResponseCustomToolCallInputDeltaEvent),
    /// Event indicating that input for a custom tool call is complete.
    #[serde(rename = "response.custom_tool_call_input.done")]
    ResponseCustomToolCallInputDone(ResponseCustomToolCallInputDoneEvent),
    /// Emitted when an error occurs.
    #[serde(rename = "error")]
    ResponseError(ResponseErrorEvent),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseCreatedEvent {
    pub sequence_number: u64,
    pub response: Response,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseInProgressEvent {
    pub sequence_number: u64,
    pub response: Response,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseCompletedEvent {
    pub sequence_number: u64,
    pub response: Response,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseFailedEvent {
    pub sequence_number: u64,
    pub response: Response,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseIncompleteEvent {
    pub sequence_number: u64,
    pub response: Response,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseOutputItemAddedEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item: OutputItem,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseOutputItemDoneEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item: OutputItem,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseContentPartAddedEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub part: OutputContent,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseContentPartDoneEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub part: OutputContent,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseTextDeltaEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub delta: String,
    pub logprobs: Option<Vec<ResponseLogProb>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseTextDoneEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub text: String,
    pub logprobs: Option<Vec<ResponseLogProb>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseRefusalDeltaEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseRefusalDoneEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub refusal: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseFunctionCallArgumentsDeltaEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseFunctionCallArgumentsDoneEvent {
    /// <https://github.com/64bit/async-openai/issues/472>
    pub name: Option<String>,
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseFileSearchCallInProgressEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseFileSearchCallSearchingEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseFileSearchCallCompletedEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseWebSearchCallInProgressEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseWebSearchCallSearchingEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseWebSearchCallCompletedEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseReasoningSummaryPartAddedEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub summary_index: u32,
    pub part: SummaryPart,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseReasoningSummaryPartDoneEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub summary_index: u32,
    pub part: SummaryPart,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseReasoningSummaryTextDeltaEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub summary_index: u32,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseReasoningSummaryTextDoneEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub summary_index: u32,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseReasoningTextDeltaEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseReasoningTextDoneEvent {
    pub sequence_number: u64,
    pub item_id: String,
    pub output_index: u32,
    pub content_index: u32,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseImageGenCallCompletedEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseImageGenCallGeneratingEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseImageGenCallInProgressEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseImageGenCallPartialImageEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
    pub partial_image_index: u32,
    pub partial_image_b64: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseMCPCallArgumentsDeltaEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseMCPCallArgumentsDoneEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseMCPCallCompletedEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseMCPCallFailedEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseMCPCallInProgressEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseMCPListToolsCompletedEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseMCPListToolsFailedEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseMCPListToolsInProgressEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseCodeInterpreterCallInProgressEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseCodeInterpreterCallInterpretingEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseCodeInterpreterCallCompletedEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseCodeInterpreterCallCodeDeltaEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseCodeInterpreterCallCodeDoneEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
    pub code: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseOutputTextAnnotationAddedEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub content_index: u32,
    pub annotation_index: u32,
    pub item_id: String,
    pub annotation: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseQueuedEvent {
    pub sequence_number: u64,
    pub response: Response,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseCustomToolCallInputDeltaEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseCustomToolCallInputDoneEvent {
    pub sequence_number: u64,
    pub output_index: u32,
    pub item_id: String,
    pub input: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ResponseErrorEvent {
    pub sequence_number: u64,
    pub code: Option<String>,
    pub message: String,
    pub param: Option<String>,
}
