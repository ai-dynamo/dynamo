//! This test is primarily to make sure that macros_rules for From traits are correct.
use dynamo_async_openai::types::Prompt;

fn prompt_input<T>(input: T) -> Prompt
where
    Prompt: From<T>,
{
    input.into()
}

#[test]
fn create_prompt_input() {
    let prompt = "This is &str prompt";
    let _ = prompt_input(prompt);

    let prompt = "This is String".to_string();
    let _ = prompt_input(&prompt);
    let _ = prompt_input(prompt);

    let prompt = vec!["This is first", "This is second"];
    let _ = prompt_input(&prompt);
    let _ = prompt_input(prompt);

    let prompt = vec!["First string".to_string(), "Second string".to_string()];
    let _ = prompt_input(&prompt);
    let _ = prompt_input(prompt);

    let first = "First".to_string();
    let second = "Second".to_string();
    let prompt = vec![&first, &second];
    let _ = prompt_input(&prompt);
    let _ = prompt_input(prompt);

    let prompt = ["first", "second"];
    let _ = prompt_input(&prompt);
    let _ = prompt_input(prompt);

    let prompt = ["first".to_string(), "second".to_string()];
    let _ = prompt_input(&prompt);
    let _ = prompt_input(prompt);

    let first = "First".to_string();
    let second = "Second".to_string();
    let prompt = [&first, &second];
    let _ = prompt_input(&prompt);
    let _ = prompt_input(prompt);
}
