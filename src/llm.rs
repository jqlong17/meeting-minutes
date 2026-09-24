use crate::config::LlmConfig;
use crate::error::{AppError, AppResult};
use serde::{Deserialize, Serialize};

pub struct OpenAiCompatibleClient {
    http: reqwest::Client,
    config: LlmConfig,
}

impl OpenAiCompatibleClient {
    pub fn new(config: LlmConfig) -> AppResult<Self> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(180))
            .build()
            .map_err(|error| AppError::llm(format!("初始化 HTTP 客户端失败: {error}")))?;
        Ok(Self { http, config })
    }

    pub async fn summarize(&self, prompt: &str) -> AppResult<String> {
        self.complete(
            "你负责根据会议文本输出简洁、可靠、结构化的中文会议纪要。",
            prompt,
        )
        .await
    }

    pub async fn polish_asr(&self, prompt: &str) -> AppResult<String> {
        self.complete(
            "你负责在尽量保留原文信息、顺序和措辞的前提下，轻度清洗中文会议 ASR 文本。",
            prompt,
        )
        .await
    }

    async fn complete(&self, system_prompt: &str, prompt: &str) -> AppResult<String> {
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let payload = ChatCompletionsRequest {
            model: self.config.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                },
            ],
            temperature: 0.2,
        };

        let response = self
            .http
            .post(url)
            .bearer_auth(&self.config.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|error| AppError::llm(format!("请求大模型服务失败: {error}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::llm(format!(
                "大模型服务返回失败状态 {}: {}",
                status, body
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|error| AppError::llm(format!("读取大模型响应失败: {error}")))?;
        let content = extract_message_content(&body)?;

        Ok(content)
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessage,
}

fn extract_message_content(body: &str) -> AppResult<String> {
    if let Ok(parsed) = serde_json::from_str::<ChatCompletionsResponse>(body) {
        if let Some(content) = parsed
            .choices
            .first()
            .map(|choice| choice.message.content.trim().to_string())
            .filter(|content| !content.is_empty())
        {
            return Ok(content);
        }
    }

    let value: serde_json::Value = serde_json::from_str(body).map_err(|error| {
        AppError::llm(format!(
            "解析大模型响应失败: {error}; body={}",
            truncate_for_error(body)
        ))
    })?;

    if let Some(content) = value
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(value_to_string)
        .filter(|content| !content.trim().is_empty())
    {
        return Ok(content.trim().to_string());
    }

    if let Some(content) = value
        .get("output_text")
        .and_then(|content| content.as_str())
        .filter(|content| !content.trim().is_empty())
    {
        return Ok(content.trim().to_string());
    }

    Err(AppError::llm(format!(
        "大模型响应中没有可用内容: {}",
        truncate_for_error(body)
    )))
}

fn value_to_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(content) => Some(content.clone()),
        serde_json::Value::Array(items) => {
            let mut fragments = Vec::new();
            for item in items {
                if let Some(text) = item
                    .get("text")
                    .and_then(|text| text.as_str())
                    .or_else(|| item.get("content").and_then(|text| text.as_str()))
                {
                    fragments.push(text.to_string());
                }
            }
            if fragments.is_empty() {
                None
            } else {
                Some(fragments.join("\n"))
            }
        }
        _ => None,
    }
}

fn truncate_for_error(body: &str) -> String {
    let truncated: String = body.chars().take(500).collect();
    if body.chars().count() > 500 {
        format!("{truncated}...")
    } else {
        truncated
    }
}
