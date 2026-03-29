use crate::{error, util::escape};

#[derive(Clone, Debug)]
pub(crate) enum Error {
    EndOfInputAnnotation,
    EndOfInputAnnotationClose,
    EndOfInputAnnotationKey,
    EndOfInputAnnotationSeparator,
    EndOfInputAnnotationValue,
    EndOfInputTzAnnotationClose,
    UnexpectedByteAnnotation { byte: u8 },
    UnexpectedByteAnnotationClose { byte: u8 },
    UnexpectedByteAnnotationKey { byte: u8 },
    UnexpectedByteAnnotationValue { byte: u8 },
    UnexpectedByteAnnotationSeparator { byte: u8 },
    UnexpectedByteTzAnnotationClose { byte: u8 },
    UnexpectedSlashAnnotationSeparator,
    UnsupportedAnnotationCritical,
}

impl error::IntoError for Error {
    fn into_error(self) -> error::Error {
        self.into()
    }
}

impl From<Error> for error::Error {
    #[cold]
    #[inline(never)]
    fn from(err: Error) -> error::Error {
        error::ErrorKind::FmtRfc9557(err).into()
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        use self::Error::*;

        match *self {
            EndOfInputAnnotation => f.write_str(
                "expected the start of an RFC 9557 annotation or IANA \
                 time zone component name, but found end of input instead",
            ),
            EndOfInputAnnotationClose => f.write_str(
                "expected an `]` after parsing an RFC 9557 annotation key \
                 and value, but found end of input instead",
            ),
            EndOfInputAnnotationKey => f.write_str(
                "expected the start of an RFC 9557 annotation key, \
                 but found end of input instead",
            ),
            EndOfInputAnnotationSeparator => f.write_str(
                "expected an `=` after parsing an RFC 9557 annotation key, \
                 but found end of input instead",
            ),
            EndOfInputAnnotationValue => f.write_str(
                "expected the start of an RFC 9557 annotation value, \
                 but found end of input instead",
            ),
            EndOfInputTzAnnotationClose => f.write_str(
                "expected an `]` after parsing an RFC 9557 time zone \
                 annotation, but found end of input instead",
            ),
            UnexpectedByteAnnotation { byte } => write!(
                f,
                "expected ASCII alphabetic byte (or underscore or period) \
                 at the start of an RFC 9557 annotation or time zone \
                 component name, but found `{byte}` instead",
                byte = escape::Byte(byte),
            ),
            UnexpectedByteAnnotationClose { byte } => write!(
                f,
                "expected an `]` after parsing an RFC 9557 annotation key \
                 and value, but found `{byte}` instead",
                byte = escape::Byte(byte),
            ),
            UnexpectedByteAnnotationKey { byte } => write!(
                f,
                "expected lowercase alphabetic byte (or underscore) \
                 at the start of an RFC 9557 annotation key, \
                 but found `{byte}` instead",
                byte = escape::Byte(byte),
            ),
            UnexpectedByteAnnotationValue { byte } => write!(
                f,
                "expected alphanumeric ASCII byte \
                 at the start of an RFC 9557 annotation value, \
                 but found `{byte}` instead",
                byte = escape::Byte(byte),
            ),
            UnexpectedByteAnnotationSeparator { byte } => write!(
                f,
                "expected an `=` after parsing an RFC 9557 annotation \
                 key, but found `{byte}` instead",
                byte = escape::Byte(byte),
            ),
            UnexpectedByteTzAnnotationClose { byte } => write!(
                f,
                "expected an `]` after parsing an RFC 9557 time zone \
                 annotation, but found `{byte}` instead",
                byte = escape::Byte(byte),
            ),
            UnexpectedSlashAnnotationSeparator => f.write_str(
                "expected an `=` after parsing an RFC 9557 annotation \
                 key, but found `/` instead (time zone annotations must \
                 come first)",
            ),
            UnsupportedAnnotationCritical => f.write_str(
                "found unsupported RFC 9557 annotation \
                 with the critical flag (`!`) set",
            ),
        }
    }
}
