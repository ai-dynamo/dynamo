use crate::types::mcp::{
    MCPToolAllowedTools, MCPToolApprovalFilter, MCPToolApprovalSetting, MCPToolFilter,
    MCPToolRequireApproval,
};

// MCPToolRequireApproval ergonomics

impl From<MCPToolApprovalSetting> for MCPToolRequireApproval {
    fn from(setting: MCPToolApprovalSetting) -> Self {
        MCPToolRequireApproval::ApprovalSetting(setting)
    }
}

impl From<MCPToolApprovalFilter> for MCPToolRequireApproval {
    fn from(filter: MCPToolApprovalFilter) -> Self {
        MCPToolRequireApproval::Filter(filter)
    }
}

// MCPToolAllowedTools ergonomics

impl From<MCPToolFilter> for MCPToolAllowedTools {
    fn from(filter: MCPToolFilter) -> Self {
        MCPToolAllowedTools::Filter(filter)
    }
}

impl From<Vec<String>> for MCPToolAllowedTools {
    fn from(tools: Vec<String>) -> Self {
        MCPToolAllowedTools::List(tools)
    }
}

impl From<Vec<&str>> for MCPToolAllowedTools {
    fn from(tools: Vec<&str>) -> Self {
        MCPToolAllowedTools::List(tools.into_iter().map(|s| s.to_string()).collect())
    }
}

impl From<&[&str]> for MCPToolAllowedTools {
    fn from(tools: &[&str]) -> Self {
        MCPToolAllowedTools::List(tools.iter().map(|s| s.to_string()).collect())
    }
}

impl<const N: usize> From<[&str; N]> for MCPToolAllowedTools {
    fn from(tools: [&str; N]) -> Self {
        MCPToolAllowedTools::List(tools.iter().map(|s| s.to_string()).collect())
    }
}

impl From<&Vec<String>> for MCPToolAllowedTools {
    fn from(tools: &Vec<String>) -> Self {
        MCPToolAllowedTools::List(tools.clone())
    }
}

impl From<&Vec<&str>> for MCPToolAllowedTools {
    fn from(tools: &Vec<&str>) -> Self {
        MCPToolAllowedTools::List(tools.iter().map(|s| s.to_string()).collect())
    }
}

impl From<&str> for MCPToolAllowedTools {
    fn from(tool: &str) -> Self {
        MCPToolAllowedTools::List(vec![tool.to_string()])
    }
}

impl From<String> for MCPToolAllowedTools {
    fn from(tool: String) -> Self {
        MCPToolAllowedTools::List(vec![tool])
    }
}
