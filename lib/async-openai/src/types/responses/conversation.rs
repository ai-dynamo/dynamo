use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    error::OpenAIError,
    types::responses::{
        AnyItemReference, CodeInterpreterToolCall, ComputerToolCall, CustomToolCall,
        CustomToolCallOutput, FileSearchToolCall, ImageGenToolCall, InputFileContent,
        InputImageContent, InputItem, InputTextContent, LocalShellToolCall,
        LocalShellToolCallOutput, MCPApprovalRequest, MCPApprovalResponse, MCPListTools,
        MCPToolCall, OutputTextContent, ReasoningItem, ReasoningTextContent, RefusalContent,
        WebSearchToolCall,
    },
};

/// Represents a conversation object.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, ToSchema)]
pub struct ConversationResource {
    /// The unique ID of the conversation.
    pub id: String,
    /// The object type, which is always `conversation`.
    pub object: String,
    /// Set of 16 key-value pairs that can be attached to an object.
    #[schema(value_type = Object)]
    pub metadata: Option<serde_json::Value>,
    /// The time at which the conversation was created, measured in seconds since the Unix epoch.
    pub created_at: u64,
}

/// Request to create a conversation.
/// openapi spec type: CreateConversationBody
#[derive(Clone, Serialize, Default, Debug, Deserialize, Builder, PartialEq, ToSchema)]
#[builder(name = "CreateConversationRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CreateConversationRequest {
    /// Set of 16 key-value pairs that can be attached to an object.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub metadata: Option<serde_json::Value>,

    /// Initial items to include in the conversation context. You may add up to 20 items at a time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<InputItem>>,
}

/// Request to update a conversation.
#[derive(Clone, Serialize, Default, Debug, Deserialize, Builder, PartialEq, ToSchema)]
#[builder(name = "UpdateConversationRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct UpdateConversationRequest {
    /// Set of 16 key-value pairs that can be attached to an object.
    #[schema(value_type = Object)]
    pub metadata: Option<serde_json::Value>,
}

/// Represents a deleted conversation.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, ToSchema)]
pub struct DeleteConversationResponse {
    /// The unique ID of the deleted conversation.
    pub id: String,
    /// The object type, which is always `conversation.deleted`.
    pub object: String,
    /// Whether the conversation was successfully deleted.
    pub deleted: bool,
}

/// Request to create conversation items.
#[derive(Clone, Serialize, Default, Debug, Deserialize, Builder, PartialEq, ToSchema)]
#[builder(name = "CreateConversationItemsRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CreateConversationItemsRequest {
    /// The items to add to the conversation. You may add up to 20 items at a time.
    pub items: Vec<InputItem>,
}

/// A list of Conversation items.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, ToSchema)]
pub struct ConversationItemList {
    /// The type of object returned, must be `list`.
    pub object: String,
    /// A list of conversation items.
    pub data: Vec<ConversationItem>,
    /// Whether there are more items available.
    pub has_more: bool,
    /// The ID of the first item in the list.
    pub first_id: Option<String>,
    /// The ID of the last item in the list.
    pub last_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum MessageStatus {
    InProgress,
    Incomplete,
    Completed,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    Unknown,
    User,
    Assistant,
    System,
    Critic,
    Discriminator,
    Developer,
    Tool,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct TextContent {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct SummaryTextContent {
    /// A summary of the reasoning output from the model so far.
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct ComputerScreenContent {
    /// The URL of the screenshot image.
    pub image_url: Option<String>,
    ///  The identifier of an uploaded file that contains the screenshot.
    pub file_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    InputText(InputTextContent),
    OutputText(OutputTextContent),
    Text(TextContent),
    SummaryText(SummaryTextContent),
    ReasoningText(ReasoningTextContent),
    Refusal(RefusalContent),
    InputImage(InputImageContent),
    ComputerScreen(ComputerScreenContent),
    InputFile(InputFileContent),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
pub struct Message {
    /// The unique ID of the message.
    pub id: String,
    /// The status of item. One of `in_progress`, `completed`, or `incomplete`. Populated when items are
    /// returned via API.
    pub status: MessageStatus,
    /// The role of the message. One of `unknown`, `user`, `assistant`, `system`, `critic`,
    /// `discriminator`, `developer`, or `tool`.
    pub role: MessageRole,
    /// The content of the message.
    pub content: Vec<MessageContent>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationItem {
    Message(Message),
    FileSearchCall(FileSearchToolCall),
    WebSearchCall(WebSearchToolCall),
    ImageGenerationCall(ImageGenToolCall),
    ComputerCall(ComputerToolCall),
    Reasoning(ReasoningItem),
    CodeInterpreterCall(CodeInterpreterToolCall),
    LocalShellCall(LocalShellToolCall),
    LocalShellCallOutput(LocalShellToolCallOutput),
    McpListTools(MCPListTools),
    McpApprovalRequest(MCPApprovalRequest),
    McpApprovalResponse(MCPApprovalResponse),
    McpCall(MCPToolCall),
    CustomToolCall(CustomToolCall),
    CustomToolCallOutput(CustomToolCallOutput),
    #[serde(untagged)]
    ItemReference(AnyItemReference),
}

/// Additional fields to include in the response.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum IncludeParam {
    /// Include the sources of the web search tool call.
    #[serde(rename = "web_search_call.action.sources")]
    WebSearchCallActionSources,
    /// Include the outputs of python code execution in code interpreter tool call items.
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
    /// Include image urls from the computer call output.
    #[serde(rename = "computer_call_output.output.image_url")]
    ComputerCallOutputOutputImageUrl,
    /// Include the search results of the file search tool call.
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    /// Include image urls from the input message.
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageImageUrl,
    /// Include logprobs with assistant messages.
    #[serde(rename = "message.output_text.logprobs")]
    MessageOutputTextLogprobs,
    /// Include an encrypted version of reasoning tokens in reasoning item outputs.
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
}

/// The order to return items in.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ListOrder {
    /// Return items in ascending order.
    Asc,
    /// Return items in descending order.
    Desc,
}
