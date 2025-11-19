use async_trait::async_trait;
use futures_util::StreamExt;
use hf_hub::api::sync::ApiBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, trace, warn};
use rand_distr::Distribution;
use rayon::iter::split;
use rayon::prelude::*;
use reqwest::Url;
use reqwest_eventsource::{Error, Event, EventSource};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time;
use tokenizers::{FromPretrainedParameters, Tokenizer};
use tokio::sync::mpsc::Sender;
use tokio::time::{sleep, Instant};
use uuid::Uuid;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TextGenerationRequest {
    pub id: Option<Uuid>,
    pub prompt: String,
    pub num_prompt_tokens: u64,
    pub num_decode_tokens: Option<u64>,
}

#[async_trait]
pub trait TextGenerationBackend: TextGenerationBackendClone {
    async fn generate(
        &self,
        request: Arc<TextGenerationRequest>,
        sender: Sender<TextGenerationAggregatedResponse>,
    );
}

pub trait TextGenerationBackendClone {
    fn clone_box(&self) -> Box<dyn TextGenerationBackend + Send + Sync>;
}

impl<T> TextGenerationBackendClone for T
where
    T: 'static + TextGenerationBackend + Clone + Send + Sync,
{
    fn clone_box(&self) -> Box<dyn TextGenerationBackend + Send + Sync> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn TextGenerationBackend + Send + Sync> {
    fn clone(&self) -> Box<dyn TextGenerationBackend + Send + Sync> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct OpenAITextGenerationBackend {
    pub api_key: String,
    pub base_url: Url,
    pub model_name: String,
    pub client: reqwest::Client,
    pub tokenizer: Arc<Tokenizer>,
    pub timeout: time::Duration,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAITextGenerationMessage {
    pub content: String,
    pub role: String,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAITextGenerationDelta {
    pub content: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAITextGenerationChoice {
    pub message: Option<OpenAITextGenerationMessage>,
    pub finish_reason: Option<String>,
    pub delta: Option<OpenAITextGenerationDelta>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct OpenAITextGenerationResponse {
    pub choices: Vec<OpenAITextGenerationChoice>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct OpenAITextGenerationRequest {
    pub model: String,
    pub messages: Vec<OpenAITextGenerationMessage>,
    pub max_tokens: Option<u64>,
    pub stream: bool,
    pub stop: Option<String>,
    pub temperature: f64,
}

impl OpenAITextGenerationBackend {
    pub fn try_new(
        api_key: String,
        base_url: Url,
        model_name: String,
        tokenizer: Arc<Tokenizer>,
        timeout: time::Duration,
    ) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| anyhow::anyhow!("Error creating HTTP client: {e}"))?;
        Ok(Self {
            client,
            api_key,
            base_url,
            model_name,
            tokenizer,
            timeout,
        })
    }
}

#[async_trait]
impl TextGenerationBackend for OpenAITextGenerationBackend {
    async fn generate(
        &self,
        request: Arc<TextGenerationRequest>,
        sender: Sender<TextGenerationAggregatedResponse>,
    ) {
        let mut url = self.base_url.clone();
        url.set_path("/v1/chat/completions");
        // let url = format!("{base_url}", base_url = self.base_url);
        let mut aggregated_response = TextGenerationAggregatedResponse::new(request.clone());
        let messages = vec![OpenAITextGenerationMessage {
            role: "user".to_string(),
            content: request.prompt.clone(),
        }];
        let body = OpenAITextGenerationRequest {
            model: self.model_name.clone(),
            messages,
            max_tokens: request.num_decode_tokens,
            stream: true,
            stop: None,
            temperature: 0.0,
        };
        let req = self
            .client
            .post(url)
            .header(
                "Authorization",
                format!("Bearer {token}", token = self.api_key),
            )
            .json(&serde_json::json!(body))
            .timeout(self.timeout);
        // start timer
        aggregated_response.start();
        let mut es = EventSource::new(req).unwrap();
        let mut final_response = "".to_string();
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => trace!("SSE connection opened"),
                Ok(Event::Message(message)) => {
                    if message.data == "\n" || message.data == "[DONE]" {
                        aggregated_response.stop();
                        continue;
                    }
                    if message.data.starts_with("{\"error\":") {
                        error!("Error from OpenAI API: {message}", message = message.data);
                        aggregated_response.fail();
                        es.close();
                        break;
                    }
                    // deserialize message data
                    let oai_response: OpenAITextGenerationResponse =
                        match serde_json::from_str(&message.data) {
                            Ok(response) => response,
                            Err(e) => {
                                error!("Error deserializing OpenAI API response: {e}", e = e);
                                aggregated_response.fail();
                                es.close();
                                break;
                            }
                        };
                    let choices = oai_response.choices;
                    let content = choices[0]
                        .clone()
                        .delta
                        .unwrap()
                        .content
                        .unwrap_or("".to_string());
                    if content.is_empty() {
                        // skip empty responses
                        continue;
                    }
                    // Count tokens in the current content delta
                    // Note: Each SSE message contains a content delta that may be:
                    // 1. A single token (most common)
                    // 2. Multiple tokens (vLLM chunked prefill or speculative decoding)
                    // 3. Part of a token (rare, but possible with streaming)
                    let num_tokens =
                        self.tokenizer.encode(content.clone(), false).unwrap().len() as u64;

                    // Log when we receive multiple tokens in one delta
                    if num_tokens > 1 {
                        debug!(
                            "Received {num_tokens} tokens in one delta: '{content}'",
                            num_tokens = num_tokens,
                            content = content
                        );
                    }

                    // Additional validation: check if content makes sense
                    if content.len() > 0 && num_tokens == 0 {
                        warn!(
                            "Tokenizer returned 0 tokens for non-empty content: '{content}'",
                            content = content
                        );
                    }
                    match choices[0].clone().finish_reason {
                        None => {
                            aggregated_response.add_tokens(num_tokens);
                            final_response += content.as_str();
                        }
                        Some(_) => {
                            aggregated_response.add_tokens(num_tokens);
                            aggregated_response.stop();
                            trace!("Generated text using OpenAI API | prompt: {prompt}, max tokens: {max_tokens:?}, response: {message}", prompt = request.prompt, max_tokens = request.num_decode_tokens,message = &content);
                        }
                    };
                }
                Err(e) => {
                    match e {
                        Error::Utf8(_) => {
                            aggregated_response.fail();
                        }
                        Error::Parser(_) => {
                            aggregated_response.fail();
                        }
                        Error::Transport(_) => {
                            aggregated_response.fail();
                        }
                        Error::InvalidContentType(_, _) => {
                            aggregated_response.fail();
                        }
                        Error::InvalidStatusCode(_, _) => {
                            aggregated_response.fail();
                        }
                        Error::InvalidLastEventId(_) => {
                            aggregated_response.fail();
                        }
                        Error::StreamEnded => {
                            if aggregated_response.num_generated_tokens == 0 {
                                // server sent no data
                                aggregated_response.fail();
                            }
                            if aggregated_response.end_time.is_none() {
                                // server closed the connection before we received the final response
                                warn!("Connection closed before completion. Received :: {num_tokens}/{max_tokens} tokens. Response: {final_response}", num_tokens = aggregated_response.num_generated_tokens, max_tokens = request.num_decode_tokens.unwrap_or(0));
                                aggregated_response.fail();
                            }
                        }
                    }
                    es.close();
                }
            };
        }

        // Ensure the response is marked as ended if it has an end_time
        if aggregated_response.end_time.is_some() {
            aggregated_response.ended = true;
        }

        sender
            .send(aggregated_response.clone())
            .await
            .expect("Error sending response to channel");
        //debug!("Final response: {response}", response = final_response);
    }
}

#[derive(Debug, Clone)]
pub struct DummyTextGenerationBackend {
    time_to_generate: time::Duration,
}

impl DummyTextGenerationBackend {
    pub fn new(time_to_generate: time::Duration) -> Self {
        Self { time_to_generate }
    }
}

impl Default for DummyTextGenerationBackend {
    fn default() -> Self {
        Self::new(time::Duration::from_secs(1))
    }
}

#[async_trait]
impl TextGenerationBackend for DummyTextGenerationBackend {
    async fn generate(
        &self,
        request: Arc<TextGenerationRequest>,
        sender: Sender<crate::requests::TextGenerationAggregatedResponse>,
    ) {
        let mut response = TextGenerationAggregatedResponse::new(request.clone());
        response.start();
        let num_tokens = request.num_decode_tokens.unwrap_or(10);
        let time_per_token = self
            .time_to_generate
            .checked_div(num_tokens as u32)
            .unwrap();
        for _ in 0..num_tokens {
            sleep(time_per_token).await;
            response.add_tokens(1);
        }
        response.stop();
        sender
            .send(response.clone())
            .await
            .expect("Error sending response to channel");
    }
}

pub trait TextRequestGenerator: Sync {
    fn generate_request(&mut self) -> TextGenerationRequest;
    /// callback can be used by generators to add new requests to the queue based on the response (e.g. for multi-turn conversation generation)
    fn callback(&mut self, request: Arc<TextGenerationRequest>, response: &str);
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Conversation {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ConversationEntry {
    pub id: String,
    pub conversations: Vec<Conversation>,
}

#[derive(Clone, Serialize, Debug)]
pub struct TokenizeOptions {
    pub num_tokens: Option<u64>,
    pub min_tokens: u64,
    pub max_tokens: u64,
    pub variance: u64,
}

impl TokenizeOptions {
    pub fn new() -> Self {
        Self {
            num_tokens: None,
            min_tokens: 0,
            max_tokens: u64::MAX,
            variance: 0,
        }
    }
}

impl Default for TokenizeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for TokenizeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "num_tokens={num_tokens:?},min_tokens={min_tokens},max_tokens={max_tokens},variance={variance}",
            num_tokens = self.num_tokens,
            min_tokens = self.min_tokens,
            max_tokens = self.max_tokens,
            variance = self.variance
        )
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ConversationTurnRequest {
    id: Uuid,
    priority: u64,
    tie: Instant,
    request: TextGenerationRequest,
}

impl Ord for ConversationTurnRequest {
    // order by increasing priority and decreasing tie-breaking
    // this way, we can pop the item with the highest priority and oldest tie-breaking
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.tie.cmp(&other.tie).reverse())
    }
}

impl PartialOrd for ConversationTurnRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone)]
pub struct ConversationTextRequestGenerator {
    pub requests: HashMap<Uuid, ConversationTurnRequest>,
    pub queue: BinaryHeap<ConversationTurnRequest>,
    pub tokenizer: Arc<Tokenizer>,
}

impl ConversationTextRequestGenerator {
    /// Load a conversation dataset from a JSON file
    /// The JSON file should be an array of objects, each object representing a conversation entry
    /// Each conversation entry should have an `id` field and a `conversations` field
    /// The `conversations` field should be an array of objects, each object representing a turn in the conversation
    /// Each turn should have a `role` field and a `content` field
    /// The `role` field should be either "user" or "system"
    /// The `content` field should be the text of the turn
    /// All conversation turns are tokenized and converted into `TextGenerationRequest`. The `id` field is used to link turns in the conversation,
    /// so that each `TextGenerationRequest` has a reference to the next turn in the conversation.
    pub fn load(
        filepath: PathBuf,
        tokenizer: String,
        prompt_tokenize_opts: Option<TokenizeOptions>,
        decode_tokenize_opts: Option<TokenizeOptions>,
        hf_token: Option<String>,
    ) -> anyhow::Result<Self> {
        let params = FromPretrainedParameters {
            token: hf_token,
            ..Default::default()
        };
        let tokenizer = match Tokenizer::from_pretrained(tokenizer, Some(params)) {
            Ok(tokenizer) => tokenizer,
            Err(e) => {
                return Err(anyhow::anyhow!("Error loading tokenizer: {e}"));
            }
        };
        let tokenizer = Arc::new(tokenizer);
        // load json file
        let input = std::fs::read_to_string(&filepath)?;
        let data: Vec<ConversationEntry> = serde_json::from_str(&input).expect("Unable to parse input file. Check that it is valid JSON and matches the expected format.");
        // generate requests
        let requests: Arc<Mutex<HashMap<Uuid, ConversationTurnRequest>>> =
            Arc::from(Mutex::from(HashMap::new()));
        info!(
            "Generating requests from {filepath}",
            filepath = filepath.display()
        );
        let bar = ProgressBar::new(data.len() as u64);
        bar.set_style(ProgressStyle::with_template(
            "Tokenizing prompts [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )?);
        split(data, entry_splitter).for_each(|subrange| {
            for entry in subrange {
                bar.inc(1);
                if entry.conversations.is_empty() {
                    continue;
                }
                let ids = (0..entry.conversations.len())
                    .map(|_| Uuid::new_v4())
                    .collect::<Vec<Uuid>>();
                let filtered_conversations = entry
                    .conversations
                    .iter()
                    .filter(|c| c.role == "user" || c.role == "system")
                    .collect::<Vec<&Conversation>>();
                for (turn_idx, c) in filtered_conversations.iter().enumerate() {
                    let prompt = c.content.clone();
                    let num_decode_tokens = decode_tokenize_opts.clone().map_or_else(
                        || None,
                        |opts| {
                            opts.num_tokens.map(|num_tokens| {
                                sample_num_tokens(
                                    num_tokens,
                                    opts.min_tokens,
                                    opts.max_tokens,
                                    opts.variance,
                                )
                            })
                        },
                    );
                    let next_id = if turn_idx == entry.conversations.len() - 1 {
                        None
                    } else {
                        Some(ids[turn_idx + 1]) // link to next turn in the conversation
                    };
                    debug!("Prompt: {prompt}", prompt = prompt);
                    match &prompt_tokenize_opts {
                        None => {
                            let (_, num_tokens) = match tokenize_prompt(
                                prompt.clone(),
                                tokenizer.clone(),
                                &TokenizeOptions::default(),
                            ) {
                                Ok((prompt, num_tokens)) => (prompt, num_tokens),
                                Err(e) => {
                                    debug!("Error tokenizing prompt: {e}");
                                    return;
                                }
                            };
                            requests.lock().unwrap().insert(
                                ids[turn_idx],
                                ConversationTurnRequest {
                                    id: ids[turn_idx],
                                    priority: turn_idx as u64,
                                    tie: Instant::now(),
                                    request: TextGenerationRequest {
                                        id: next_id,
                                        prompt,
                                        num_prompt_tokens: num_tokens,
                                        num_decode_tokens,
                                    },
                                },
                            );
                        }
                        Some(options) => {
                            // compute number of tokens to generate using a Gaussian distribution
                            let (sampled_prompt, prompt_tokens) =
                                match tokenize_prompt(prompt.clone(), tokenizer.clone(), options) {
                                    Ok(prompt) => prompt,
                                    Err(e) => {
                                        debug!("Error tokenizing prompt: {e}");
                                        return;
                                    }
                                };
                            requests.lock().unwrap().insert(
                                ids[turn_idx],
                                ConversationTurnRequest {
                                    id: ids[turn_idx],
                                    tie: Instant::now(),
                                    priority: turn_idx as u64,
                                    request: TextGenerationRequest {
                                        id: next_id,
                                        prompt: sampled_prompt,
                                        num_prompt_tokens: prompt_tokens,
                                        num_decode_tokens,
                                    },
                                },
                            );
                        }
                    }
                }
                // TODO: check that we have enough requests
            }
        });
        let requests = requests.lock().unwrap();
        info!(
            "Generated {num_requests} requests",
            num_requests = requests.len()
        );
        // create the queue from the hashmap. Only queue first turns in the conversation
        let queue = BinaryHeap::from(
            requests
                .values()
                .filter(|item| item.priority == 0)
                .cloned()
                .collect::<Vec<ConversationTurnRequest>>(),
        );
        Ok(Self {
            requests: requests.clone(),
            tokenizer,
            queue,
        })
    }

    pub fn download_dataset(
        repo_name: String,
        filename: String,
        hf_token: Option<String>,
    ) -> anyhow::Result<PathBuf> {
        let api = ApiBuilder::from_env().with_token(hf_token).build()?;
        let repo = api.dataset(repo_name);
        let dataset = repo.get(&filename)?;
        Ok(dataset)
    }
}

fn sample_num_tokens(num_tokens: u64, min_tokens: u64, max_tokens: u64, variance: u64) -> u64 {
    let normal = rand_distr::Normal::new(num_tokens as f64, variance as f64).unwrap();
    let mut num_tokens = normal.sample(&mut rand::rng()) as u64;
    if num_tokens < min_tokens {
        num_tokens = min_tokens;
    }
    if num_tokens > max_tokens {
        num_tokens = max_tokens;
    }
    num_tokens
}

fn entry_splitter(
    gen: Vec<ConversationEntry>,
) -> (Vec<ConversationEntry>, Option<Vec<ConversationEntry>>) {
    if gen.len() <= 2 {
        return (gen, None);
    }
    let middle = gen.len() / 2;
    let (left, right) = gen.split_at(middle);
    let left = left.to_vec();
    let right = right.to_vec();
    (left, Some(right))
}

impl TextRequestGenerator for ConversationTextRequestGenerator {
    fn generate_request(&mut self) -> TextGenerationRequest {
        let item = self.queue.pop().expect("Queue is empty");
        // add the item back to the end of the queue if it is a first turn in the conversation
        if item.priority == 0 {
            let mut cloned_item = item.clone();
            cloned_item.tie = Instant::now(); // update the tie-breaking for intra-priority sorting
            self.queue.push(cloned_item);
        }
        item.request
    }

    /// Use callback to add a new chat turn to the queue.
    /// The turn is generated from the `TextGenerationRequest`, using the `id` field to link it to
    /// the next turn in the conversation.
    /// Those turns must be scheduled as soon as possible so that we may benefit from
    /// KV cache hits. The `priority` field is used to move the turn to the front of the queue.
    fn callback(&mut self, request: Arc<TextGenerationRequest>, response: &str) {
        // retrieve current turn id
        let id = match request.id {
            None => {
                return;
            }
            Some(id) => id,
        };
        // retrieve next turn from id
        let next_request = match self.requests.get(&id) {
            None => {
                return;
            }
            Some(request) => request,
        };
        // create a new turn with the prompt concatenated with the response and next turn's prompt
        // and add the next turn id to the new turn
        let new_prompt =
            request.prompt.clone() + "\n" + response + "\n" + next_request.request.prompt.as_str();
        // tokenize the prompt
        let (prompt, num_tokens) = match tokenize_prompt(
            new_prompt.to_string(),
            self.tokenizer.clone(),
            &TokenizeOptions::default(),
        ) {
            Ok((prompt, num_tokens)) => (prompt, num_tokens),
            Err(_) => {
                return;
            }
        };
        let next_id = next_request.request.id;
        let turn = ConversationTurnRequest {
            id,
            priority: 100,       // move to the front of the queue
            tie: Instant::now(), // use the current time as tie-breaking (older turns have higher priority)
            request: TextGenerationRequest {
                id: next_id,
                prompt,
                num_prompt_tokens: num_tokens,
                num_decode_tokens: request.num_decode_tokens, // decode tokens do not change between turns
            },
        };
        //debug!("Adding new turn to queue: {turn}", turn = turn.request.prompt);
        self.queue.push(turn);
    }
}

pub struct DummyTextRequestGenerator {}

impl DummyTextRequestGenerator {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DummyTextRequestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TextRequestGenerator for DummyTextRequestGenerator {
    fn generate_request(&mut self) -> TextGenerationRequest {
        TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(10),
        }
    }
    fn callback(&mut self, _request: Arc<TextGenerationRequest>, _response: &str) {}
}

fn tokenize_prompt(
    prompt: String,
    tokenizer: Arc<Tokenizer>,
    options: &TokenizeOptions,
) -> anyhow::Result<(String, u64)> {
    let prompt_tokens = tokenizer
        .encode(prompt.clone(), false)
        .map_err(|_| anyhow::anyhow!("Error tokenizing prompt"))?;
    match options.num_tokens {
        None => {
            // check if we have a min/max number of tokens, skip prompts that are too short or too long
            if prompt_tokens.len() > options.max_tokens as usize
                || prompt_tokens.len() < options.min_tokens as usize
            {
                return Err(anyhow::anyhow!(format!(
                    "Prompt is too short or too long, skipping: {}<{}<{}",
                    options.min_tokens,
                    prompt_tokens.len(),
                    options.max_tokens
                )));
            }
            Ok((prompt, prompt_tokens.len() as u64))
        }
        Some(num_tokens) => {
            if prompt_tokens.len() < num_tokens as usize {
                return Err(anyhow::anyhow!(format!(
                    "Prompt is too short to tokenize: {}<{}",
                    prompt_tokens.len(),
                    num_tokens
                )));
            }
            let tokens = prompt_tokens
                .get_ids()
                .iter()
                .take(num_tokens as usize)
                .copied()
                .collect::<Vec<u32>>();
            let prompt = tokenizer.decode(&tokens, true).unwrap();
            Ok((prompt, num_tokens))
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextGenerationAggregatedResponse {
    pub start_time: Option<tokio::time::Instant>,
    pub end_time: Option<tokio::time::Instant>,
    pub num_generated_tokens: u64,
    pub times_to_tokens: Vec<time::Duration>,
    pub token_arrival_times: Vec<tokio::time::Instant>,
    last_received_token_time: tokio::time::Instant,
    pub failed: bool,
    pub ended: bool,
    pub request: Option<Arc<TextGenerationRequest>>,
    pub response: Option<String>,
}

impl TextGenerationAggregatedResponse {
    pub fn new(request: Arc<TextGenerationRequest>) -> Self {
        Self {
            start_time: None,
            end_time: None,
            num_generated_tokens: 0,
            times_to_tokens: Vec::new(),
            token_arrival_times: Vec::new(),
            last_received_token_time: tokio::time::Instant::now(),
            failed: false,
            ended: false,
            request: Some(request),
            response: None,
        }
    }

    pub fn new_as_ended() -> Self {
        Self {
            start_time: None,
            end_time: None,
            num_generated_tokens: 0,
            times_to_tokens: Vec::new(),
            token_arrival_times: Vec::new(),
            last_received_token_time: tokio::time::Instant::now(),
            failed: false,
            ended: true,
            request: None,
            response: None,
        }
    }
    fn start(&mut self) {
        self.start_time = Some(tokio::time::Instant::now());
        self.last_received_token_time = tokio::time::Instant::now();
    }

    fn stop(&mut self) {
        self.end_time = Some(tokio::time::Instant::now());
    }

    fn fail(&mut self) {
        self.end_time = Some(tokio::time::Instant::now());
        self.failed = true;
    }

    fn add_tokens(&mut self, num_tokens: u64) {
        self.num_generated_tokens += num_tokens;
        let current_time = tokio::time::Instant::now();
        let time_since_last_batch = self.last_received_token_time.elapsed();

        // More accurate token timing: distribute time across tokens in this batch
        if num_tokens > 0 {
            let time_per_token = if num_tokens == 1 {
                time_since_last_batch
            } else {
                // For multiple tokens, assume they were generated sequentially
                // with equal time intervals between them
                time_since_last_batch / num_tokens as u32
            };

            // Record arrival time for each token with distributed timing
            for i in 0..num_tokens {
                let token_arrival_time = if i == 0 {
                    current_time
                } else {
                    // Simulate sequential token generation
                    current_time - time_per_token * (num_tokens - i - 1) as u32
                };
                self.token_arrival_times.push(token_arrival_time);
            }
        }

        self.last_received_token_time = current_time;
        self.times_to_tokens.push(time_since_last_batch);
    }

    pub fn time_to_first_token(&self) -> Option<std::time::Duration> {
        match self.start_time {
            Some(_) => self.times_to_tokens.first().copied(),
            None => None,
        }
    }

    pub fn inter_token_latency(&self) -> Option<std::time::Duration> {
        match self.times_to_tokens.len() {
            0 => None,
            1 => Some(std::time::Duration::new(0, 0)),
            _ => {
                let mut total_time = std::time::Duration::new(0, 0);
                for i in 1..self.times_to_tokens.len() {
                    total_time += self.times_to_tokens[i];
                }
                Some(total_time / (self.num_generated_tokens as u32 - 1))
            }
        }
    }
    pub fn e2e_latency(&self) -> Option<std::time::Duration> {
        match self.start_time {
            Some(start_time) => self.end_time.map(|end_time| end_time - start_time),
            None => None,
        }
    }

    /// Calculate Time Per Output Token (TPOT) - average time between consecutive tokens
    pub fn time_per_output_token(&self) -> Option<std::time::Duration> {
        if self.token_arrival_times.len() < 2 {
            return None;
        }

        let mut total_intervals = std::time::Duration::new(0, 0);
        for i in 1..self.token_arrival_times.len() {
            total_intervals += self.token_arrival_times[i] - self.token_arrival_times[i - 1];
        }

        Some(total_intervals / (self.token_arrival_times.len() - 1) as u32)
    }

    /// Get the arrival time of a specific token (0-indexed)
    pub fn token_arrival_time(&self, token_index: usize) -> Option<tokio::time::Instant> {
        self.token_arrival_times.get(token_index).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executors::ExecutorConfig;
    use crate::results::BenchmarkResults;
    use crate::scheduler::ExecutorType;
    use std::sync::atomic::AtomicU64;
    use std::thread::sleep;
    use std::time::Duration;
    use tokio::sync::RwLock;

    #[tokio::test]
    async fn test_openai_token_count() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| {
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": {\"content\": \"Hello, world!Hello, world!Hello, world!Hello, world!\", \"role\": \"user\"}, \"finish_reason\": \"stop\", \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: [DONE]\n\n")
            })
            .create_async().await;
        let url = s.url().parse().unwrap();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(10),
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx).await;
        });
        let num_tokens = Arc::new(AtomicU64::new(0));
        let num_tokens_clone = num_tokens.clone();
        let t = tokio::spawn(async move {
            while let Some(item) = rx.recv().await {
                let response = item;
                assert_eq!(response.failed, false);
                num_tokens_clone.fetch_add(
                    response.num_generated_tokens,
                    std::sync::atomic::Ordering::SeqCst,
                );
            }
        });
        t.await.unwrap();
        assert_eq!(
            num_tokens.load(std::sync::atomic::Ordering::SeqCst),
            16 as u64
        );
    }

    /// Test that the timings are correct
    /// The tests may be flaky due to the nature of the SSE connection (it may depend on the testing environment)
    /// We need to account for the time it takes to establish the connection
    /// and the time it takes to receive the first message
    #[tokio::test]
    #[ignore] // flaky test, don't run it on CI
    async fn test_openai_timings() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| {
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                // sleep for 500ms
                sleep(std::time::Duration::from_millis(500));
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": {\"content\": \"Hello, world!Hello, world!Hello, world!Hello, world!\", \"role\": \"user\"}, \"finish_reason\": \"stop\", \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: [DONE]\n\n")
            })
            .create_async().await;
        let url = s.url().parse().unwrap();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let results = BenchmarkResults::new(
            "test".to_string(),
            ExecutorType::ConstantArrivalRate,
            ExecutorConfig {
                max_vus: 1,
                duration: Duration::from_secs(10),
                rate: None,
            },
        );
        let results = Arc::new(RwLock::new(results));
        let results_clone = results.clone();
        let t = tokio::spawn(async move {
            let mut handle = results_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.add_response(response);
            }
        });
        t.await.unwrap();
        let results = results.read().await;
        let e2e_latency_avg = results.e2e_latency_avg().unwrap();
        let inter_token_latency_avg = results.inter_token_latency_avg().unwrap();
        let ttft = results.time_to_first_token_avg().unwrap();

        let e2e_timing_overhead = Duration::from_millis(10);
        let expected_e2e_latency_avg = Duration::from_millis(500);
        let expected_inter_token_latency_avg = Duration::from_millis(33); // 16-1 tokens with a 500ms delay
        let inter_token_latency_overhead = Duration::from_millis(3);
        let expected_ttft = Duration::from_millis(3); // account for http overhead
        assert!(
            e2e_latency_avg > expected_e2e_latency_avg
                && e2e_latency_avg < expected_e2e_latency_avg + e2e_timing_overhead,
            "e2e_latency_avg: {:?} < {:?} < {:?}",
            expected_e2e_latency_avg,
            e2e_latency_avg,
            expected_e2e_latency_avg + e2e_timing_overhead
        );
        assert!(
            inter_token_latency_avg > expected_inter_token_latency_avg
                && inter_token_latency_avg
                    < expected_inter_token_latency_avg + inter_token_latency_overhead,
            "inter_token_latency_avg: {:?} < {:?} < {:?}",
            expected_inter_token_latency_avg,
            inter_token_latency_avg,
            expected_inter_token_latency_avg + inter_token_latency_overhead
        );
        assert!(
            ttft < expected_ttft,
            "TTFT: {:?} < {:?}",
            ttft,
            expected_ttft
        );
    }

    /// Test that server errors are handled correctly
    #[tokio::test]
    async fn test_openai_fails_on_error() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| w.write_all(b"data: {\"error\": \"Internal server error\"}\n\n"))
            .create_async()
            .await;
        let url = s.url().parse().unwrap();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let responses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = responses.clone();
        let t = tokio::spawn(async move {
            let mut handle = responses_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.push(response);
            }
        });
        t.await.unwrap();
        let responses = responses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that bad responses are handled correctly
    #[tokio::test]
    async fn test_openai_fails_on_bad_response() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| w.write_all(b"this is wrong\n\n"))
            .create_async()
            .await;
        let url = s.url().parse().unwrap();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let responses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = responses.clone();
        let t = tokio::spawn(async move {
            let mut handle = responses_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.push(response);
            }
        });
        t.await.unwrap();
        let responses = responses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that malformed JSON responses are handled correctly
    #[tokio::test]
    async fn test_openai_fails_on_malformed_json() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| w.write_all(b"data: {\"foo\": \"bar\"}\n\n"))
            .create_async()
            .await;
        let url = s.url().parse().unwrap();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let responses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = responses.clone();
        let t = tokio::spawn(async move {
            let mut handle = responses_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.push(response);
            }
        });
        t.await.unwrap();
        let responses = responses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that request timeout is handled correctly
    #[tokio::test]
    async fn test_timeout_should_fail_request() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| {
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                // sleep for 5s
                sleep(Duration::from_secs(5));
                w.write_all(b"data: [DONE]\n\n")
            })
            .create_async().await;
        let url = s.url().parse().unwrap();
        let tokenizer = Arc::new(Tokenizer::from_pretrained("gpt2", None).unwrap());
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            Duration::from_secs(1),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let reponses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = reponses.clone();
        let t = tokio::spawn(async move {
            while let Some(item) = rx.recv().await {
                responses_clone.write().await.push(item);
            }
        });
        t.await.unwrap();
        let responses = reponses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that conversations are correctly loaded
    #[tokio::test]
    async fn test_load_conversations_from_file() {
        let filepath = PathBuf::from("test_data/conversations.json");
        let tokenizer = "gpt2".to_string();
        let prompt_tokenize_opts = TokenizeOptions::default();
        let decode_tokenize_opts = TokenizeOptions::default();
        let hf_token = None;
        let generator = ConversationTextRequestGenerator::load(
            filepath,
            tokenizer,
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
            hf_token,
        )
        .unwrap();
        assert_eq!(generator.requests.len(), 17016);
    }

    /// Test that conversations are bounded by the min/max number of tokens
    #[tokio::test]
    async fn test_load_conversations_bounded() {
        let filepath = PathBuf::from("test_data/conversations.json");
        let tokenizer = "gpt2".to_string();
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: None,
            min_tokens: 4,
            max_tokens: 1024,
            variance: 0,
        };
        let decode_tokenize_opts = TokenizeOptions::default();
        let hf_token = None;
        let generator = ConversationTextRequestGenerator::load(
            filepath,
            tokenizer,
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
            hf_token,
        )
        .unwrap();
        let min_tokens = generator
            .requests
            .iter()
            .map(|r| r.1.request.num_prompt_tokens)
            .min()
            .unwrap();
        let max_tokens = generator
            .requests
            .iter()
            .map(|r| r.1.request.num_prompt_tokens)
            .max()
            .unwrap();
        assert!(min_tokens >= 4, "Min tokens: {}", min_tokens);
        assert!(max_tokens <= 1024, "Max tokens: {}", max_tokens);
    }

    /// Test that conversations prompts have the correct number of tokens
    #[tokio::test]
    async fn test_load_conversations_fixed_tokens() {
        let filepath = PathBuf::from("test_data/conversations.json");
        let tokenizer = "gpt2".to_string();
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: Some(200),
            min_tokens: 200,
            max_tokens: 200,
            variance: 0,
        };
        let decode_tokenize_opts = TokenizeOptions::default();
        let hf_token = None;
        let generator = ConversationTextRequestGenerator::load(
            filepath,
            tokenizer,
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
            hf_token,
        )
        .unwrap();
        for r in generator.requests.iter() {
            assert_eq!(r.1.request.num_prompt_tokens, 200);
        }
    }

    /// Test that multi-turn conversations are correctly loaded
    /// The test data contains 2 conversations first with 6 turns and the second with 1 turn and a system prompt
    #[tokio::test]
    async fn test_load_conversations_multi_turn() {
        let filepath = PathBuf::from("test_data/chat.json");
        let tokenizer = "gpt2".to_string();
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: None,
            min_tokens: 1,
            max_tokens: 200,
            variance: 0,
        };
        let hf_token = None;
        let decode_tokenize_opts = TokenizeOptions::default();
        let generator = ConversationTextRequestGenerator::load(
            filepath,
            tokenizer,
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
            hf_token,
        )
        .unwrap();
        let turns = generator
            .requests
            .into_iter()
            .map(|r| r.1.clone())
            .collect::<Vec<ConversationTurnRequest>>();
        assert_eq!(turns.len(), 8);
        let first_turns = turns.clone().into_iter().filter(|t| t.priority == 0);
        // we expect to have 2 None values for the first turn in each conversation
        assert_eq!(first_turns.count(), 2);
        let first_conversation = turns.clone().into_iter().filter(|t| t.request.prompt == "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...").collect::<Vec<ConversationTurnRequest>>();
        // rebuild the conversation from first turn
        let mut conversation = Vec::new();
        let mut current_turn = first_conversation[0].clone();
        loop {
            conversation.push(current_turn.clone());
            match turns
                .iter()
                .find(|t| t.id == current_turn.request.id.unwrap_or_default())
            {
                Some(t) => current_turn = t.clone(),
                None => break,
            }
        }
        assert_eq!(conversation.len(), 6);
        let got = conversation
            .iter()
            .map(|t| t.request.prompt.clone())
            .collect::<Vec<String>>()
            .join("\n");
        let expect = "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...\nSummarize the main ideas of Brendon Burchard's Experts Academy into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...\nWhat are the mental triggers in Jeff Walker's Product Launch Formula and \"Launch\" book?\nWrite a summary of why scarcity and urgency are the strongest mental triggers and have been the driving force behind many of our best performing campaigns over the last 8 years.\nSummarize Russell Brunson's Perfect Webinar Script...\nSummarize the 6 human needs as Tony Robbins explains...";
        assert_eq!(expect, got);
    }

    /// Test that only first turns of multi-turn conversations are queued
    #[tokio::test]
    async fn test_conversation_queue() {
        let filepath = PathBuf::from("test_data/chat.json");
        let tokenizer = "gpt2".to_string();
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: None,
            min_tokens: 1,
            max_tokens: 200,
            variance: 0,
        };
        let hf_token = None;
        let decode_tokenize_opts = TokenizeOptions::default();
        let mut generator = ConversationTextRequestGenerator::load(
            filepath,
            tokenizer,
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
            hf_token,
        )
        .unwrap();
        for i in 0..20 {
            let req = generator.generate_request();
            if i % 2 == 0 {
                // first turn of the first conversation
                assert_eq!(req.prompt, "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...");
            }
            if i % 2 == 1 {
                // first turn of the second conversation
                assert_eq!(req.prompt, "You are a helpful assistant.");
            }
        }
    }

    /// Test that multi-turn conversations are correctly queued when we add responses
    #[tokio::test]
    async fn test_conversation_turns_queue() {
        let filepath = PathBuf::from("test_data/chat.json");
        let tokenizer = "gpt2".to_string();
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: None,
            min_tokens: 1,
            max_tokens: 200,
            variance: 0,
        };
        let hf_token = None;
        let decode_tokenize_opts = TokenizeOptions::default();
        let mut generator = ConversationTextRequestGenerator::load(
            filepath,
            tokenizer,
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
            hf_token,
        )
        .unwrap();
        // generate the first user turn
        let req = generator.generate_request();
        let response = "This is my response".to_string();
        generator.callback(Arc::from(req), &response);
        // now try to generate the next user turn
        let req = generator.generate_request();
        // we expect to have all the turns concatenated into the prompt
        assert_eq!(req.prompt, "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...\nThis is my response\nSummarize the main ideas of Brendon Burchard's Experts Academy into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...");
        assert_eq!(req.num_prompt_tokens, 76);
        // now add a response to the second turn
        let response = "This is my second response".to_string();
        generator.callback(Arc::from(req), &response);
        // now try to generate the next user turn
        let req = generator.generate_request();
        assert_eq!(req.prompt, "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...\nThis is my response\nSummarize the main ideas of Brendon Burchard's Experts Academy into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...\nThis is my second response\nWhat are the mental triggers in Jeff Walker's Product Launch Formula and \"Launch\" book?");
        // check that next turn is a first turn
        let req = generator.generate_request();
        assert_eq!(req.prompt, "You are a helpful assistant.");
    }
}
