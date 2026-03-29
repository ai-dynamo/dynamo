/// Unwrap an `Option<T>` in a `const` context.
///
/// If it fails, panics with the given message.
macro_rules! unwrap {
    ($val:expr, $msg:expr$(,)?) => {
        match $val {
            Some(val) => val,
            None => panic!($msg),
        }
    };
}

/// Unwrap a `Result<T, E>` in a `const` context.
///
/// If it fails, panics with the given message.
macro_rules! unwrapr {
    ($val:expr, $msg:expr$(,)?) => {
        match $val {
            Ok(val) => val,
            Err(_) => panic!($msg),
        }
    };
}

pub(crate) use {unwrap, unwrapr};
