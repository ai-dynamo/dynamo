use dynamo_async_openai::types::CreateTranslationRequestArgs;
use dynamo_async_openai::{types::CreateTranscriptionRequestArgs, Client};
use tokio_test::assert_err;

#[tokio::test]
async fn transcribe_test() {
    let client = Client::new();

    let request = CreateTranscriptionRequestArgs::default().build().unwrap();

    let response = client.audio().transcribe(request).await;

    assert_err!(response); // FileReadError("cannot extract file name from ")
}

#[tokio::test]
async fn transcribe_sendable_test() {
    let client = Client::new();

    // https://github.com/64bit/async-openai/issues/140
    let transcribe = tokio::spawn(async move {
        let request = CreateTranscriptionRequestArgs::default().build().unwrap();

        client.audio().transcribe(request).await
    });

    let response = transcribe.await.unwrap();

    assert_err!(response); // FileReadError("cannot extract file name from ")
}

#[tokio::test]
async fn translate_test() {
    let client = Client::new();

    let request = CreateTranslationRequestArgs::default().build().unwrap();

    let response = client.audio().translate(request).await;

    assert_err!(response); // FileReadError("cannot extract file name from ")
}

#[tokio::test]
async fn translate_sendable_test() {
    let client = Client::new();

    // https://github.com/64bit/async-openai/issues/140
    let translate = tokio::spawn(async move {
        let request = CreateTranslationRequestArgs::default().build().unwrap();

        client.audio().translate(request).await
    });

    let response = translate.await.unwrap();

    assert_err!(response); // FileReadError("cannot extract file name from ")
}
