use crate::types::mcp::MCPTool;
use crate::types::responses::{
    ApplyPatchToolCallItemParam, ApplyPatchToolCallOutputItemParam, CodeInterpreterContainerAuto,
    CodeInterpreterTool, CodeInterpreterToolCall, CodeInterpreterToolContainer,
    ComputerCallOutputItemParam, ComputerToolCall, ComputerUsePreviewTool, ConversationParam,
    CustomToolCall, CustomToolCallOutput, CustomToolParam, EasyInputContent, EasyInputMessage,
    FileSearchTool, FileSearchToolCall, FunctionCallOutput, FunctionCallOutputItemParam,
    FunctionShellCallItemParam, FunctionShellCallOutputItemParam, FunctionTool, FunctionToolCall,
    ImageGenTool, ImageGenToolCall, InputContent, InputFileContent, InputImageContent, InputItem,
    InputMessage, InputParam, InputTextContent, Item, ItemReference, ItemReferenceType,
    LocalShellToolCall, LocalShellToolCallOutput, MCPApprovalRequest, MCPApprovalResponse,
    MCPListTools, MCPToolCall, MessageItem, MessageType, OutputMessage, OutputMessageContent,
    OutputTextContent, Prompt, Reasoning, ReasoningEffort, ReasoningItem, ReasoningSummary,
    RefusalContent, ResponseFormatJsonSchema, ResponsePromptVariables, ResponseStreamOptions,
    ResponseTextParam, Role, TextResponseFormatConfiguration, Tool, ToolChoiceCustom,
    ToolChoiceFunction, ToolChoiceMCP, ToolChoiceOptions, ToolChoiceParam, ToolChoiceTypes,
    WebSearchTool, WebSearchToolCall,
};

impl<S: Into<String>> From<S> for EasyInputMessage {
    fn from(value: S) -> Self {
        EasyInputMessage {
            r#type: MessageType::Message,
            role: Role::User,
            content: EasyInputContent::Text(value.into()),
        }
    }
}

impl From<EasyInputMessage> for InputItem {
    fn from(msg: EasyInputMessage) -> Self {
        InputItem::EasyMessage(msg)
    }
}

// InputItem ergonomics

impl From<InputMessage> for InputItem {
    fn from(msg: InputMessage) -> Self {
        InputItem::Item(Item::Message(MessageItem::Input(msg)))
    }
}

impl From<Item> for InputItem {
    fn from(item: Item) -> Self {
        InputItem::Item(item)
    }
}

impl From<ItemReference> for InputItem {
    fn from(item: ItemReference) -> Self {
        InputItem::ItemReference(item)
    }
}

// InputParam ergonomics: from InputItem

impl From<InputItem> for InputParam {
    fn from(item: InputItem) -> Self {
        InputParam::Items(vec![item])
    }
}

impl From<Item> for InputParam {
    fn from(item: Item) -> Self {
        InputParam::Items(vec![InputItem::Item(item)])
    }
}

impl From<MessageItem> for InputParam {
    fn from(item: MessageItem) -> Self {
        InputParam::Items(vec![InputItem::Item(Item::Message(item))])
    }
}

impl From<InputMessage> for InputParam {
    fn from(msg: InputMessage) -> Self {
        InputParam::Items(vec![InputItem::Item(Item::Message(MessageItem::Input(
            msg,
        )))])
    }
}

impl<I: Into<InputItem>> From<Vec<I>> for InputParam {
    fn from(items: Vec<I>) -> Self {
        InputParam::Items(items.into_iter().map(|item| item.into()).collect())
    }
}

impl<I: Into<InputItem>, const N: usize> From<[I; N]> for InputParam {
    fn from(items: [I; N]) -> Self {
        InputParam::Items(items.into_iter().map(|item| item.into()).collect())
    }
}

// InputParam ergonomics: from string "family"

impl From<&str> for InputParam {
    fn from(value: &str) -> Self {
        InputParam::Text(value.into())
    }
}

impl From<String> for InputParam {
    fn from(value: String) -> Self {
        InputParam::Text(value)
    }
}

impl From<&String> for InputParam {
    fn from(value: &String) -> Self {
        InputParam::Text(value.clone())
    }
}

// InputParam ergonomics: from vector family

macro_rules! impl_inputparam_easy_from_collection {
    // Vec<T>
    ($t:ty, $map:expr, $clone:expr) => {
        impl From<Vec<$t>> for InputParam {
            fn from(values: Vec<$t>) -> Self {
                InputParam::Items(
                    values
                        .into_iter()
                        .map(|value| {
                            InputItem::EasyMessage(EasyInputMessage {
                                r#type: MessageType::Message,
                                role: Role::User,
                                content: EasyInputContent::Text($map(value)),
                            })
                        })
                        .collect(),
                )
            }
        }
        // &[T; N]
        impl<const N: usize> From<[$t; N]> for InputParam {
            fn from(values: [$t; N]) -> Self {
                InputParam::Items(
                    values
                        .into_iter()
                        .map(|value| {
                            InputItem::EasyMessage(EasyInputMessage {
                                r#type: MessageType::Message,
                                role: Role::User,
                                content: EasyInputContent::Text($map(value)),
                            })
                        })
                        .collect(),
                )
            }
        }
        // &Vec<T>
        impl From<&Vec<$t>> for InputParam {
            fn from(values: &Vec<$t>) -> Self {
                InputParam::Items(
                    values
                        .iter()
                        .map(|value| {
                            InputItem::EasyMessage(EasyInputMessage {
                                r#type: MessageType::Message,
                                role: Role::User,
                                content: EasyInputContent::Text($clone(value)),
                            })
                        })
                        .collect(),
                )
            }
        }
    };
}

// Apply for &str
impl_inputparam_easy_from_collection!(&str, |v: &str| v.to_string(), |v: &str| v.to_string());
// Apply for String
impl_inputparam_easy_from_collection!(String, |v: String| v, |v: &String| v.clone());
// Apply for &String
impl_inputparam_easy_from_collection!(&String, |v: &String| v.clone(), |v: &String| v.clone());

// ConversationParam ergonomics

impl<S: Into<String>> From<S> for ConversationParam {
    fn from(id: S) -> Self {
        ConversationParam::ConversationID(id.into())
    }
}

// ToolChoiceParam ergonomics

impl From<ToolChoiceOptions> for ToolChoiceParam {
    fn from(mode: ToolChoiceOptions) -> Self {
        ToolChoiceParam::Mode(mode)
    }
}

impl From<ToolChoiceTypes> for ToolChoiceParam {
    fn from(tool_type: ToolChoiceTypes) -> Self {
        ToolChoiceParam::Hosted(tool_type)
    }
}

impl<S: Into<String>> From<S> for ToolChoiceParam {
    fn from(name: S) -> Self {
        ToolChoiceParam::Function(ToolChoiceFunction { name: name.into() })
    }
}

impl From<ToolChoiceFunction> for ToolChoiceParam {
    fn from(function: ToolChoiceFunction) -> Self {
        ToolChoiceParam::Function(function)
    }
}

impl From<ToolChoiceMCP> for ToolChoiceParam {
    fn from(mcp: ToolChoiceMCP) -> Self {
        ToolChoiceParam::Mcp(mcp)
    }
}

impl From<ToolChoiceCustom> for ToolChoiceParam {
    fn from(custom: ToolChoiceCustom) -> Self {
        ToolChoiceParam::Custom(custom)
    }
}

// ResponseTextParam ergonomics

impl From<TextResponseFormatConfiguration> for ResponseTextParam {
    fn from(format: TextResponseFormatConfiguration) -> Self {
        ResponseTextParam {
            format,
            verbosity: None,
        }
    }
}

impl From<ResponseFormatJsonSchema> for ResponseTextParam {
    fn from(schema: ResponseFormatJsonSchema) -> Self {
        ResponseTextParam {
            format: TextResponseFormatConfiguration::JsonSchema(schema),
            verbosity: None,
        }
    }
}

// ResponseStreamOptions ergonomics

impl From<bool> for ResponseStreamOptions {
    fn from(include_obfuscation: bool) -> Self {
        ResponseStreamOptions {
            include_obfuscation: Some(include_obfuscation),
        }
    }
}

// Reasoning ergonomics

impl From<ReasoningEffort> for Reasoning {
    fn from(effort: ReasoningEffort) -> Self {
        Reasoning {
            effort: Some(effort),
            summary: None,
        }
    }
}

impl From<ReasoningSummary> for Reasoning {
    fn from(summary: ReasoningSummary) -> Self {
        Reasoning {
            effort: None,
            summary: Some(summary),
        }
    }
}

// Prompt ergonomics

impl<S: Into<String>> From<S> for Prompt {
    fn from(id: S) -> Self {
        Prompt {
            id: id.into(),
            version: None,
            variables: None,
        }
    }
}

// InputTextContent ergonomics

impl<S: Into<String>> From<S> for InputTextContent {
    fn from(text: S) -> Self {
        InputTextContent { text: text.into() }
    }
}

// InputContent ergonomics

impl From<InputTextContent> for InputContent {
    fn from(content: InputTextContent) -> Self {
        InputContent::InputText(content)
    }
}

impl From<InputImageContent> for InputContent {
    fn from(content: InputImageContent) -> Self {
        InputContent::InputImage(content)
    }
}

impl From<InputFileContent> for InputContent {
    fn from(content: InputFileContent) -> Self {
        InputContent::InputFile(content)
    }
}

impl<S: Into<String>> From<S> for InputContent {
    fn from(text: S) -> Self {
        InputContent::InputText(InputTextContent { text: text.into() })
    }
}

// ResponsePromptVariables ergonomics

impl From<InputContent> for ResponsePromptVariables {
    fn from(content: InputContent) -> Self {
        ResponsePromptVariables::Content(content)
    }
}

impl<S: Into<String>> From<S> for ResponsePromptVariables {
    fn from(text: S) -> Self {
        ResponsePromptVariables::String(text.into())
    }
}

// MessageItem ergonomics

impl From<InputMessage> for MessageItem {
    fn from(msg: InputMessage) -> Self {
        MessageItem::Input(msg)
    }
}

impl From<OutputMessage> for MessageItem {
    fn from(msg: OutputMessage) -> Self {
        MessageItem::Output(msg)
    }
}

// FunctionCallOutput ergonomics

impl From<&str> for FunctionCallOutput {
    fn from(text: &str) -> Self {
        FunctionCallOutput::Text(text.to_string())
    }
}

impl From<String> for FunctionCallOutput {
    fn from(text: String) -> Self {
        FunctionCallOutput::Text(text)
    }
}

impl From<Vec<InputContent>> for FunctionCallOutput {
    fn from(content: Vec<InputContent>) -> Self {
        FunctionCallOutput::Content(content)
    }
}

// RefusalContent ergonomics

impl<S: Into<String>> From<S> for RefusalContent {
    fn from(refusal: S) -> Self {
        RefusalContent {
            refusal: refusal.into(),
        }
    }
}

// OutputMessageContent ergonomics

impl From<OutputTextContent> for OutputMessageContent {
    fn from(content: OutputTextContent) -> Self {
        OutputMessageContent::OutputText(content)
    }
}

impl From<RefusalContent> for OutputMessageContent {
    fn from(content: RefusalContent) -> Self {
        OutputMessageContent::Refusal(content)
    }
}

// Item ergonomics

impl From<MessageItem> for Item {
    fn from(item: MessageItem) -> Self {
        Item::Message(item)
    }
}

impl From<FileSearchToolCall> for Item {
    fn from(call: FileSearchToolCall) -> Self {
        Item::FileSearchCall(call)
    }
}

impl From<ComputerToolCall> for Item {
    fn from(call: ComputerToolCall) -> Self {
        Item::ComputerCall(call)
    }
}

impl From<ComputerCallOutputItemParam> for Item {
    fn from(output: ComputerCallOutputItemParam) -> Self {
        Item::ComputerCallOutput(output)
    }
}

impl From<WebSearchToolCall> for Item {
    fn from(call: WebSearchToolCall) -> Self {
        Item::WebSearchCall(call)
    }
}

impl From<FunctionToolCall> for Item {
    fn from(call: FunctionToolCall) -> Self {
        Item::FunctionCall(call)
    }
}

impl From<FunctionCallOutputItemParam> for Item {
    fn from(output: FunctionCallOutputItemParam) -> Self {
        Item::FunctionCallOutput(output)
    }
}

impl From<ReasoningItem> for Item {
    fn from(item: ReasoningItem) -> Self {
        Item::Reasoning(item)
    }
}

impl From<ImageGenToolCall> for Item {
    fn from(call: ImageGenToolCall) -> Self {
        Item::ImageGenerationCall(call)
    }
}

impl From<CodeInterpreterToolCall> for Item {
    fn from(call: CodeInterpreterToolCall) -> Self {
        Item::CodeInterpreterCall(call)
    }
}

impl From<LocalShellToolCall> for Item {
    fn from(call: LocalShellToolCall) -> Self {
        Item::LocalShellCall(call)
    }
}

impl From<LocalShellToolCallOutput> for Item {
    fn from(output: LocalShellToolCallOutput) -> Self {
        Item::LocalShellCallOutput(output)
    }
}

impl From<FunctionShellCallItemParam> for Item {
    fn from(call: FunctionShellCallItemParam) -> Self {
        Item::ShellCall(call)
    }
}

impl From<FunctionShellCallOutputItemParam> for Item {
    fn from(output: FunctionShellCallOutputItemParam) -> Self {
        Item::ShellCallOutput(output)
    }
}

impl From<ApplyPatchToolCallItemParam> for Item {
    fn from(call: ApplyPatchToolCallItemParam) -> Self {
        Item::ApplyPatchCall(call)
    }
}

impl From<ApplyPatchToolCallOutputItemParam> for Item {
    fn from(output: ApplyPatchToolCallOutputItemParam) -> Self {
        Item::ApplyPatchCallOutput(output)
    }
}

impl From<MCPListTools> for Item {
    fn from(tools: MCPListTools) -> Self {
        Item::McpListTools(tools)
    }
}

impl From<MCPApprovalRequest> for Item {
    fn from(request: MCPApprovalRequest) -> Self {
        Item::McpApprovalRequest(request)
    }
}

impl From<MCPApprovalResponse> for Item {
    fn from(response: MCPApprovalResponse) -> Self {
        Item::McpApprovalResponse(response)
    }
}

impl From<MCPToolCall> for Item {
    fn from(call: MCPToolCall) -> Self {
        Item::McpCall(call)
    }
}

impl From<CustomToolCallOutput> for Item {
    fn from(output: CustomToolCallOutput) -> Self {
        Item::CustomToolCallOutput(output)
    }
}

impl From<CustomToolCall> for Item {
    fn from(call: CustomToolCall) -> Self {
        Item::CustomToolCall(call)
    }
}

// Tool ergonomics

impl From<FunctionTool> for Tool {
    fn from(tool: FunctionTool) -> Self {
        Tool::Function(tool)
    }
}

impl From<FileSearchTool> for Tool {
    fn from(tool: FileSearchTool) -> Self {
        Tool::FileSearch(tool)
    }
}

impl From<ComputerUsePreviewTool> for Tool {
    fn from(tool: ComputerUsePreviewTool) -> Self {
        Tool::ComputerUsePreview(tool)
    }
}

impl From<WebSearchTool> for Tool {
    fn from(tool: WebSearchTool) -> Self {
        Tool::WebSearch(tool)
    }
}

impl From<MCPTool> for Tool {
    fn from(tool: MCPTool) -> Self {
        Tool::Mcp(tool)
    }
}

impl From<CodeInterpreterTool> for Tool {
    fn from(tool: CodeInterpreterTool) -> Self {
        Tool::CodeInterpreter(tool)
    }
}

impl From<ImageGenTool> for Tool {
    fn from(tool: ImageGenTool) -> Self {
        Tool::ImageGeneration(tool)
    }
}

impl From<CustomToolParam> for Tool {
    fn from(tool: CustomToolParam) -> Self {
        Tool::Custom(tool)
    }
}

// Vec<Tool> ergonomics

impl From<Tool> for Vec<Tool> {
    fn from(tool: Tool) -> Self {
        vec![tool]
    }
}

impl From<FunctionTool> for Vec<Tool> {
    fn from(tool: FunctionTool) -> Self {
        vec![Tool::Function(tool)]
    }
}

impl From<FileSearchTool> for Vec<Tool> {
    fn from(tool: FileSearchTool) -> Self {
        vec![Tool::FileSearch(tool)]
    }
}

impl From<ComputerUsePreviewTool> for Vec<Tool> {
    fn from(tool: ComputerUsePreviewTool) -> Self {
        vec![Tool::ComputerUsePreview(tool)]
    }
}

impl From<WebSearchTool> for Vec<Tool> {
    fn from(tool: WebSearchTool) -> Self {
        vec![Tool::WebSearch(tool)]
    }
}

impl From<MCPTool> for Vec<Tool> {
    fn from(tool: MCPTool) -> Self {
        vec![Tool::Mcp(tool)]
    }
}

impl From<CodeInterpreterTool> for Vec<Tool> {
    fn from(tool: CodeInterpreterTool) -> Self {
        vec![Tool::CodeInterpreter(tool)]
    }
}

impl From<ImageGenTool> for Vec<Tool> {
    fn from(tool: ImageGenTool) -> Self {
        vec![Tool::ImageGeneration(tool)]
    }
}

impl From<CustomToolParam> for Vec<Tool> {
    fn from(tool: CustomToolParam) -> Self {
        vec![Tool::Custom(tool)]
    }
}

// EasyInputContent ergonomics

impl Default for EasyInputContent {
    fn default() -> Self {
        Self::Text("".to_string())
    }
}

impl From<String> for EasyInputContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for EasyInputContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

// Defaults

impl Default for CodeInterpreterToolContainer {
    fn default() -> Self {
        Self::Auto(CodeInterpreterContainerAuto::default())
    }
}

impl Default for InputParam {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl ItemReference {
    /// Create a new item reference with the given ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            r#type: Some(ItemReferenceType::ItemReference),
            id: id.into(),
        }
    }
}
