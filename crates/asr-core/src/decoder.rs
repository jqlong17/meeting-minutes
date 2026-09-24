use anyhow::Result;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;

use super::sensevoice::{
    classify_tag, format_annotated_text, tag_label, SenseVoiceAnnotation, TagKind,
};
use super::types::TimedToken;

fn is_ignored_token(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    let lower = s.to_lowercase();
    matches!(
        lower.as_str(),
        "<unk>" | "<s>" | "</s>" | "<blank>" | "<blk>" | "<space>"
    )
}

fn is_structural_token(s: &str) -> bool {
    if is_ignored_token(s) {
        return true;
    }
    if let Some(label) = tag_label(s) {
        return !matches!(classify_tag(&label), TagKind::Ignore);
    }
    false
}

fn normalize_token_text(token: &str) -> String {
    token.replace('▁', " ").replace("<space>", " ")
}

fn postprocess_tokens(tokens: &[String]) -> String {
    let sentence = tokens
        .iter()
        .map(|token| normalize_token_text(token))
        .collect::<Vec<_>>()
        .join("");
    sentence
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn annotation_from_tokens(tokens: &[String]) -> SenseVoiceAnnotation {
    let mut annotation = SenseVoiceAnnotation::default();
    for token in tokens {
        let Some(label) = tag_label(token) else {
            continue;
        };
        match classify_tag(&label) {
            TagKind::Language => annotation.language = Some(label),
            TagKind::Emotion => annotation.emotion = Some(label),
            TagKind::Event => {
                if !annotation.events.contains(&label) {
                    annotation.events.push(label);
                }
            }
            TagKind::TextNorm | TagKind::Ignore => {}
        }
    }
    annotation
}

#[derive(Debug, Clone)]
pub struct DecodedTranscript {
    pub text: String,
    pub formatted_text: String,
    pub annotation: SenseVoiceAnnotation,
    pub tokens: Vec<TimedToken>,
}

pub struct CtcDecoder {
    _token_to_id: HashMap<String, i32>,
    id_to_token: HashMap<i32, String>,
    blank_id: i32,
    has_explicit_blank: bool,
}

impl CtcDecoder {
    pub fn from_tokens_file(tokens_path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(tokens_path)?;
        let tokens: serde_json::Value = serde_json::from_str(&content)?;

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        if let Some(obj) = tokens.as_object() {
            for (token, id_val) in obj {
                if let Some(id) = id_val.as_i64() {
                    let id = id as i32;
                    token_to_id.insert(token.clone(), id);
                    id_to_token.insert(id, token.clone());
                }
            }
        } else if let Some(arr) = tokens.as_array() {
            for (id, token_val) in arr.iter().enumerate() {
                if let Some(token) = token_val.as_str() {
                    let id = id as i32;
                    token_to_id.insert(token.to_string(), id);
                    id_to_token.insert(id, token.to_string());
                }
            }
        }

        if token_to_id.is_empty() {
            anyhow::bail!("tokens.json 解析失败");
        }

        let blank_token_id = token_to_id
            .get("<blank>")
            .copied()
            .or_else(|| token_to_id.get("<blk>").copied());
        let has_explicit_blank = blank_token_id.is_some();
        let blank_id = blank_token_id.unwrap_or(0);

        Ok(Self {
            _token_to_id: token_to_id,
            id_to_token,
            blank_id,
            has_explicit_blank,
        })
    }

    pub fn decode_with_timestamps(
        &self,
        logits: &Array2<f32>,
        total_duration_ms: u64,
        debug: bool,
    ) -> DecodedTranscript {
        let (frame_ids, _) = self.argmax_frames(logits, debug);
        let blank_id_use = if self.has_explicit_blank {
            self.blank_id
        } else {
            0
        };

        let mut ordered_tokens = Vec::new();
        let mut current_id: Option<i32> = None;

        for frame_id in frame_ids {
            if frame_id == blank_id_use {
                current_id = None;
                continue;
            }

            if current_id == Some(frame_id) {
                continue;
            }

            current_id = Some(frame_id);
            if let Some(token) = self.id_to_token.get(&frame_id) {
                ordered_tokens.push(token.clone());
            }
        }

        let annotation = annotation_from_tokens(&ordered_tokens);
        let content_tokens: Vec<String> = ordered_tokens
            .iter()
            .filter(|token| !is_structural_token(token))
            .cloned()
            .collect();
        let text = postprocess_tokens(&content_tokens);
        let formatted_text = format_annotated_text(&text, &annotation);
        let timed_tokens = self.build_timed_tokens(logits, total_duration_ms, debug);

        DecodedTranscript {
            text,
            formatted_text,
            annotation,
            tokens: timed_tokens,
        }
    }

    fn argmax_frames(&self, logits: &Array2<f32>, debug: bool) -> (Vec<i32>, usize) {
        let num_frames = logits.nrows();
        let num_classes = logits.ncols();

        let mut frame_ids = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let mut max_prob = f32::NEG_INFINITY;
            let mut max_id = 0i32;
            for j in 0..num_classes {
                let prob = logits[[i, j]];
                if prob > max_prob {
                    max_prob = prob;
                    max_id = j as i32;
                }
            }
            frame_ids.push(max_id);
        }

        if debug && logits.nrows() > 0 {
            let k = 5.min(logits.ncols());
            let frames_to_show = [0, logits.nrows() / 2, logits.nrows().saturating_sub(1)];
            for &frame_idx in &frames_to_show {
                if frame_idx >= logits.nrows() {
                    continue;
                }
                let mut probs: Vec<(usize, f32)> = (0..logits.ncols())
                    .map(|j| (j, logits[[frame_idx, j]]))
                    .collect();
                probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top: Vec<String> = probs
                    .iter()
                    .take(k)
                    .map(|(id, p)| {
                        let token = self
                            .id_to_token
                            .get(&(*id as i32))
                            .cloned()
                            .unwrap_or_else(|| "?".to_string());
                        format!("{}:{:.2}", token, p)
                    })
                    .collect();
                tracing::info!(
                    "[fastcut-asr] logits frame {} top-{}: {}",
                    frame_idx,
                    k,
                    top.join(", ")
                );
            }
        }

        (frame_ids, num_frames)
    }

    fn build_timed_tokens(
        &self,
        logits: &Array2<f32>,
        total_duration_ms: u64,
        debug: bool,
    ) -> Vec<TimedToken> {
        let (frame_ids, total_frames) = self.argmax_frames(logits, debug);
        let blank_id_use = if self.has_explicit_blank {
            self.blank_id
        } else {
            0
        };

        let total_frames = total_frames.max(1);
        let mut current_id: Option<i32> = None;
        let mut current_start = 0usize;
        let mut raw_tokens = Vec::new();
        let mut timed_tokens = Vec::new();

        for frame_idx in 0..=frame_ids.len() {
            let frame_id = if frame_idx < frame_ids.len() {
                frame_ids[frame_idx]
            } else {
                blank_id_use
            };

            if let Some(active_id) = current_id {
                if frame_id != active_id {
                    if let Some(token) = self.id_to_token.get(&active_id) {
                        if !is_structural_token(token) {
                            let normalized = normalize_token_text(token);
                            if !normalized.trim().is_empty() {
                                raw_tokens.push(token.clone());
                                let start_ms = (total_duration_ms as u128 * current_start as u128
                                    / total_frames as u128)
                                    as u64;
                                let mut end_ms = (total_duration_ms as u128 * frame_idx as u128
                                    / total_frames as u128)
                                    as u64;
                                if end_ms <= start_ms {
                                    end_ms = start_ms + 1;
                                }
                                timed_tokens.push(TimedToken {
                                    text: normalized,
                                    start_ms,
                                    end_ms,
                                });
                            }
                        }
                    }
                    current_id = None;
                }
            }

            if frame_idx == frame_ids.len() {
                break;
            }

            if frame_id == blank_id_use {
                continue;
            }

            let token = match self.id_to_token.get(&frame_id) {
                Some(token) if !is_structural_token(token) => token,
                _ => continue,
            };

            let _ = token;
            if current_id.is_none() {
                current_id = Some(frame_id);
                current_start = frame_idx;
            }
        }

        let _ = postprocess_tokens(&raw_tokens);
        timed_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_simple_ctc_sequence() {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        token_to_id.insert("<blank>".to_string(), 0);
        token_to_id.insert("a".to_string(), 1);
        token_to_id.insert("b".to_string(), 2);
        id_to_token.insert(0, "<blank>".to_string());
        id_to_token.insert(1, "a".to_string());
        id_to_token.insert(2, "b".to_string());

        let decoder = CtcDecoder {
            _token_to_id: token_to_id,
            id_to_token,
            blank_id: 0,
            has_explicit_blank: true,
        };

        let logits = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, //
            ],
        )
        .unwrap();

        let decoded = decoder.decode_with_timestamps(&logits, 1000, false);
        assert_eq!(decoded.text, "ab");
    }
}
