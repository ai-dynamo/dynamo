//! Model-specific vision processors.
//!
//! This module contains implementations of `VisionPreProcessor` for various
//! vision-language model families.
//!
//! # Supported Models
//!
//! - **LLaVA 1.5** (`llava`): CLIP-based preprocessing with configurable aspect ratio
//! - **LLaVA-NeXT** (`llava`): Multi-crop anyres processing
//! - **Qwen2-VL** (`qwen2_vl`): Dynamic resolution with smart resizing
//! - **Qwen2.5-VL** (`qwen2_vl`): Same processor as Qwen2-VL (identical preprocessing)
//! - **Qwen3-VL** (`qwen3_vl`): Similar to Qwen2-VL but with patch_size=16 and [0.5,0.5,0.5] normalization
//! - **Qwen3-Omni** (`qwen3_omni_vision`): Qwen3 vision preprocessing with Omni video limits and timing metadata
//! - **Inkling** (`inkling`): Optional aspect-preserving resize with CLIP normalization and padded patch columns
//! - **Kimi-K2.5** (`kimi_k25`): MoonViT resize and zero-padding to patch alignment
//! - **Kimi-K3** (`kimi_k3`): Same MoonViT stack with a 4x patch budget and
//!   chessboard alpha compositing
//! - **Phi3-Vision** (`phi3_vision`): Dynamic HD transform with 336x336 tiles
//! - **Phi4-Vision** (`phi4_vision`): Dynamic HD transform with 448x448 tiles and SiGLIP encoder
//! - **LLaMA 4 Vision** (`llama4_vision`): Tile-based processing with 336x336 tiles and global tile
//! - **Pixtral/Mistral3** (`pixtral`): CLIP-based preprocessing with dynamic resolution

pub mod inkling;
pub mod kimi_k25;
pub mod kimi_k3;
pub mod llama4_vision;
pub mod llava;
pub mod moonvit;
pub mod phi3_vision;
pub mod phi4_vision;
pub mod pixtral;
pub mod qwen2_vl;
pub mod qwen3_omni_vision;
pub mod qwen3_vl;
pub mod qwen_vl_base;

pub use inkling::InklingImageProcessor;
pub use kimi_k25::KimiK25Processor;
pub use kimi_k3::KimiK3Processor;
pub use llama4_vision::Llama4VisionProcessor;
pub use llava::{ImageAspectRatio, LlavaNextProcessor, LlavaProcessor};
pub use phi3_vision::Phi3VisionProcessor;
pub use phi4_vision::Phi4VisionProcessor;
pub use pixtral::PixtralProcessor;
pub use qwen2_vl::Qwen2VLProcessor;
pub use qwen3_omni_vision::Qwen3OmniVisionProcessor;
pub use qwen3_vl::Qwen3VLProcessor;
