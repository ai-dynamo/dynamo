//! HTTP middleware for Dynamo
//!
//! This module contains middleware components for request processing,
//! including session extraction from trusted upstream headers.

pub mod session;

pub use session::{extract_session_middleware, RequestSession};
