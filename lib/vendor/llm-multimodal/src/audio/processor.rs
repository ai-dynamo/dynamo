use std::sync::Arc;

use crate::{encoder_inputs::PreprocessedEncoderInputs, error::TransformError, types::AudioClip};

/// Audio preprocessing contract for a model family.
///
/// The concrete processor for a model is selected by its
/// [`ModelProcessorSpec::audio_processor`](crate::registry::ModelProcessorSpec::audio_processor),
/// which owns audio-processor selection alongside the model's prompt/placeholder
/// logic.
pub trait AudioPreProcessor: Send + Sync {
    fn preprocess(
        &self,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError>;
}
