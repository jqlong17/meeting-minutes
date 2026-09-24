use serde::{Deserialize, Serialize};

pub const DIARIZATION_SCHEMA_VERSION: u32 = 1;
pub const EMBEDDING_DIM: usize = 192;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiarizationSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker_index: usize,
    pub speaker_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiarizationSpeaker {
    pub id: String,
    pub display_name: String,
    pub total_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiarizationOutput {
    pub version: u32,
    pub model: String,
    pub num_speakers: usize,
    pub segments: Vec<DiarizationSegment>,
    pub speakers: Vec<DiarizationSpeaker>,
}

impl DiarizationOutput {
    pub fn empty(model: impl Into<String>) -> Self {
        Self {
            version: DIARIZATION_SCHEMA_VERSION,
            model: model.into(),
            num_speakers: 0,
            segments: Vec::new(),
            speakers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechInterval {
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiarizationOptions {
    pub min_speakers: usize,
    pub max_speakers: usize,
    pub min_segment_ms: u64,
}

impl Default for DiarizationOptions {
    fn default() -> Self {
        Self {
            min_speakers: 1,
            max_speakers: 8,
            min_segment_ms: 500,
        }
    }
}
