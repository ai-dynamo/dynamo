// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A stream that edits each item where the stream below returned it.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{Stream, StreamExt};

/// `stream`, with `edit` applied to each item through `&mut`.
///
/// In place of `stream.map(|mut item| { ...; item })`: that moves each item into the closure and
/// out again, and a response item is about 1 KB, so each such stage costs a few `memcpy` calls
/// per item. Here the item is edited in the value the stream below returned.
pub(super) fn edit_items<S, F>(stream: S, edit: F) -> EditItems<S, F>
where
    S: Stream + Unpin,
    F: FnMut(&mut S::Item),
{
    EditItems { stream, edit }
}

/// The stream of [`edit_items`].
pub(super) struct EditItems<S, F> {
    stream: S,
    edit: F,
}

impl<S, F> Stream for EditItems<S, F>
where
    S: Stream + Unpin,
    F: FnMut(&mut S::Item) + Unpin,
{
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        let this = &mut *self;
        // Edit the item inside the `Poll` the stream below returned, and return that same
        // value: taking the item out of it and wrapping it again would move it twice more.
        let mut poll = this.stream.poll_next_unpin(cx);
        if let Poll::Ready(Some(item)) = &mut poll {
            (this.edit)(item);
        }
        poll
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.stream.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{StreamExt, stream};

    #[tokio::test]
    async fn edits_every_item_in_order_and_ends_with_the_stream_below() {
        let mut seen = 0;
        let edited: Vec<u32> = edit_items(stream::iter([1, 2, 3]), |item| {
            seen += 1;
            *item *= 10;
        })
        .collect()
        .await;
        assert_eq!(edited, [10, 20, 30]);
        assert_eq!(seen, 3);
    }

    #[test]
    fn keeps_the_size_hint_of_the_stream_below() {
        let edited = edit_items(stream::iter([1, 2, 3]), |_: &mut u32| {});
        assert_eq!(edited.size_hint(), (3, Some(3)));
    }
}
