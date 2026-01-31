use crate::types::responses::{OutputItem, OutputMessageContent, Response};

impl Response {
    /// SDK-only convenience property that contains the aggregated text output from all
    /// `output_text` items in the `output` array, if any are present.
    pub fn output_text(&self) -> Option<String> {
        let output = self
            .output
            .iter()
            .filter_map(|item| match item {
                OutputItem::Message(msg) => Some(
                    msg.content
                        .iter()
                        .filter_map(|content| match content {
                            OutputMessageContent::OutputText(ot) => Some(ot.text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<String>>(),
                ),
                _ => None,
            })
            .flatten()
            .collect::<Vec<String>>()
            .join("");
        if output.is_empty() {
            None
        } else {
            Some(output)
        }
    }
}
