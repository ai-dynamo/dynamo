//! Model-specific audio preprocessing implementations.

mod inkling;
mod qwen3_audio;

pub use inkling::{InklingAudioEncoderParams, InklingAudioProcessor};
pub use qwen3_audio::{Qwen3AudioParams, Qwen3AudioProcessor};
