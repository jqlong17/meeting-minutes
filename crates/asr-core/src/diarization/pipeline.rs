use anyhow::{Context, Result};
use std::path::Path;

use super::audio::{load_wav_mono, pad_or_trim_min_duration, slice_ms, TARGET_SAMPLE_RATE};
use super::cluster::cluster_embeddings;
use super::embedder::CampplusEmbedder;
use super::types::{
    DiarizationOptions, DiarizationOutput, DiarizationSegment, DiarizationSpeaker, SpeechInterval,
    DIARIZATION_SCHEMA_VERSION,
};

pub struct DiarizationEngine {
    embedder: CampplusEmbedder,
}

impl DiarizationEngine {
    pub fn new(model_dir: impl AsRef<Path>, intra_threads: usize) -> Result<Self> {
        Ok(Self {
            embedder: CampplusEmbedder::new(model_dir, intra_threads)?,
        })
    }

    pub fn diarize_file(
        &mut self,
        wav_path: &Path,
        speech_segments: &[SpeechInterval],
        options: DiarizationOptions,
    ) -> Result<DiarizationOutput> {
        let (waveform, sample_rate) = load_wav_mono(wav_path)?;
        self.diarize_pcm(&waveform, sample_rate, speech_segments, options)
    }

    pub fn diarize_pcm(
        &mut self,
        waveform: &[f32],
        sample_rate: u32,
        speech_segments: &[SpeechInterval],
        options: DiarizationOptions,
    ) -> Result<DiarizationOutput> {
        let model_name = self.embedder.model_name().to_string();
        if speech_segments.is_empty() {
            return Ok(DiarizationOutput::empty(model_name));
        }

        let mut usable_segments = Vec::new();
        let mut embeddings = Vec::new();

        for segment in speech_segments {
            if segment.end_ms <= segment.start_ms {
                continue;
            }
            let slice = slice_ms(waveform, sample_rate, segment.start_ms, segment.end_ms);
            let slice = pad_or_trim_min_duration(slice, TARGET_SAMPLE_RATE, options.min_segment_ms);
            if slice.is_empty() {
                continue;
            }
            let embedding = self
                .embedder
                .embed_pcm(&slice, TARGET_SAMPLE_RATE)
                .with_context(|| {
                    format!(
                        "CAM++ embedding failed for {}-{} ms",
                        segment.start_ms, segment.end_ms
                    )
                })?;
            usable_segments.push(*segment);
            embeddings.push(embedding);
        }

        if usable_segments.is_empty() {
            return Ok(DiarizationOutput::empty(model_name));
        }

        let cluster_ids = cluster_embeddings(&embeddings, options);
        let mut output_segments = Vec::with_capacity(usable_segments.len());
        for (segment, cluster_id) in usable_segments.iter().zip(cluster_ids.iter()) {
            output_segments.push(DiarizationSegment {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                speaker_index: *cluster_id,
                speaker_id: format!("spk_{cluster_id}"),
                confidence: None,
            });
        }

        let speakers = build_speakers(&output_segments, &embeddings, &cluster_ids);
        Ok(DiarizationOutput {
            version: DIARIZATION_SCHEMA_VERSION,
            model: model_name,
            num_speakers: speakers.len(),
            segments: output_segments,
            speakers,
        })
    }
}

fn build_speakers(
    segments: &[DiarizationSegment],
    embeddings: &[Vec<f32>],
    cluster_ids: &[usize],
) -> Vec<DiarizationSpeaker> {
    let mut totals = std::collections::BTreeMap::<usize, u64>::new();
    for segment in segments {
        let duration = segment.end_ms.saturating_sub(segment.start_ms);
        *totals.entry(segment.speaker_index).or_insert(0) += duration;
    }

    totals
        .into_iter()
        .map(|(speaker_index, total_ms)| DiarizationSpeaker {
            id: format!("spk_{speaker_index}"),
            display_name: format!("说话人 {}", speaker_index + 1),
            total_ms,
            embedding: centroid_embedding(speaker_index, embeddings, cluster_ids),
        })
        .collect()
}

fn centroid_embedding(
    speaker_index: usize,
    embeddings: &[Vec<f32>],
    cluster_ids: &[usize],
) -> Option<Vec<f32>> {
    let mut count = 0usize;
    let mut centroid: Vec<f32> = Vec::new();

    for (embedding, cluster_id) in embeddings.iter().zip(cluster_ids.iter()) {
        if *cluster_id != speaker_index {
            continue;
        }
        if centroid.is_empty() {
            centroid = vec![0.0; embedding.len()];
        }
        if centroid.len() != embedding.len() {
            continue;
        }
        for (current, value) in centroid.iter_mut().zip(embedding.iter()) {
            *current += *value;
        }
        count += 1;
    }

    if count == 0 || centroid.is_empty() {
        return None;
    }

    for value in &mut centroid {
        *value /= count as f32;
    }
    let norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in &mut centroid {
            *value /= norm;
        }
    }
    Some(centroid)
}
