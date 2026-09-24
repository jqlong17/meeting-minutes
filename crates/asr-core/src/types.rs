use serde::{Deserialize, Serialize};

use crate::sensevoice::{AnnotatedChunk, SenseVoiceAnnotation};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimedToken {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrOutput {
    pub text: String,
    pub confidence: f32,
    pub language: Option<String>,
    pub duration_ms: u64,
    pub tokens: Vec<TimedToken>,
    pub chunks: Vec<AnnotatedChunk>,
}

impl AsrOutput {
    pub fn empty() -> Self {
        Self {
            text: String::new(),
            confidence: 0.0,
            language: None,
            duration_ms: 0,
            tokens: Vec::new(),
            chunks: Vec::new(),
        }
    }
}

pub type SenseVoiceAnnotationExport = SenseVoiceAnnotation;
