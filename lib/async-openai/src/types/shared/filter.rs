use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Filters for file search.
#[derive(ToSchema, Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum Filter {
    /// A filter used to compare a specified attribute key to a given value using a defined
    /// comparison operation.
    Comparison(ComparisonFilter),
    /// Combine multiple filters using and or or.
    Compound(CompoundFilter),
}

/// Single comparison filter.
#[derive(ToSchema, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ComparisonFilter {
    /// Specifies the comparison operator
    #[serde(rename = "type")]
    pub op: ComparisonType,
    /// The key to compare against the value.
    pub key: String,
    /// The value to compare against the attribute key; supports string, number, or boolean types.
    pub value: serde_json::Value,
}

#[derive(ToSchema, Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum ComparisonType {
    #[serde(rename = "eq")]
    Equals,
    #[serde(rename = "ne")]
    NotEquals,
    #[serde(rename = "gt")]
    GreaterThan,
    #[serde(rename = "gte")]
    GreaterThanOrEqualTo,
    #[serde(rename = "lt")]
    LessThan,
    #[serde(rename = "lte")]
    LessThanOrEqualTo,
}

/// Combine multiple filters.
#[derive(ToSchema, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CompoundFilter {
    /// Type of operation
    #[serde(rename = "type")]
    pub op: CompoundType,
    /// Array of filters to combine. Items can be ComparisonFilter or CompoundFilter.
    #[schema(no_recursion)]
    pub filters: Vec<Filter>,
}

#[derive(ToSchema, Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CompoundType {
    And,
    Or,
}
