use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscribeOptions {
    pub language: LanguageHint,
    pub use_itn: bool,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: LanguageHint::Auto,
            use_itn: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageHint {
    Auto,
    Zh,
    En,
    Ja,
    Ko,
    Yue,
    Nospeech,
}

impl LanguageHint {
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_lowercase().as_str() {
            "zh" | "chinese" | "中文" => Self::Zh,
            "en" | "english" | "英文" => Self::En,
            "ja" | "japanese" | "日文" => Self::Ja,
            "ko" | "korean" | "韩文" => Self::Ko,
            "yue" | "cantonese" | "粤语" => Self::Yue,
            "nospeech" => Self::Nospeech,
            _ => Self::Auto,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Zh => "zh",
            Self::En => "en",
            Self::Ja => "ja",
            Self::Ko => "ko",
            Self::Yue => "yue",
            Self::Nospeech => "nospeech",
        }
    }

    pub fn model_id(self) -> i32 {
        match self {
            Self::Auto => 0,
            Self::Zh => 3,
            Self::En => 4,
            Self::Yue => 7,
            Self::Ja => 11,
            Self::Ko => 12,
            Self::Nospeech => 13,
        }
    }
}

pub fn textnorm_model_id(use_itn: bool) -> i32 {
    if use_itn {
        14
    } else {
        15
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SenseVoiceAnnotation {
    pub language: Option<String>,
    pub events: Vec<String>,
    pub emotion: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedChunk {
    pub text: String,
    pub formatted_text: String,
    pub annotation: SenseVoiceAnnotation,
    pub start_ms: u64,
    pub end_ms: u64,
}

pub fn format_annotated_text(content: &str, annotation: &SenseVoiceAnnotation) -> String {
    let trimmed = content.trim();
    if trimmed.is_empty() && annotation.events.is_empty() && annotation.emotion.is_none() {
        return String::new();
    }

    let mut parts = Vec::new();
    for event in &annotation.events {
        parts.push(format!("(事件:{event})"));
    }
    if !trimmed.is_empty() {
        parts.push(trimmed.to_string());
    }
    if let Some(emotion) = &annotation.emotion {
        parts.push(format!("(情感:{emotion})"));
    }
    if let Some(language) = &annotation.language {
        parts.push(format!("(语言:{language})"));
    }
    parts.join(" ")
}

pub fn tag_label(token: &str) -> Option<String> {
    let trimmed = token.trim();
    if !trimmed.starts_with("<|") || !trimmed.ends_with("|>") {
        return None;
    }
    Some(
        trimmed
            .trim_matches(|c| c == '<' || c == '|' || c == '>')
            .to_string(),
    )
}

pub fn classify_tag(label: &str) -> TagKind {
    match label {
        "withitn" | "woitn" => TagKind::TextNorm,
        "HAPPY" | "SAD" | "ANGRY" | "NEUTRAL" | "FEARFUL" | "DISGUSTED" | "SURPRISED" | "OTHER"
        | "EMO_UNKNOWN" => TagKind::Emotion,
        "Speech" | "BGM" | "Applause" | "Laughter" | "Cry" | "Sneeze" | "Breath" | "Cough"
        | "Sing" | "Speech_Noise" | "Event_UNK" | "GBG" => TagKind::Event,
        "ASR" | "AED" | "SER" | "SPECIAL_TOKEN_1" | "SPECIAL_TOKEN_2" | "SPECIAL_TOKEN_3"
        | "SPECIAL_TOKEN_4" | "SPECIAL_TOKEN_5" | "SPECIAL_TOKEN_6" | "SPECIAL_TOKEN_7"
        | "SPECIAL_TOKEN_8" => TagKind::Ignore,
        "zh" | "en" | "yue" | "ja" | "ko" | "nospeech" | "zh/en" | "en/zh" | "dialect"
        | "minnan" | "wuyu" => TagKind::Language,
        _ if label.chars().all(|c| c.is_ascii_alphabetic()) && label.len() <= 8 => {
            TagKind::Language
        }
        _ => TagKind::Ignore,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKind {
    Language,
    Emotion,
    Event,
    TextNorm,
    Ignore,
}
