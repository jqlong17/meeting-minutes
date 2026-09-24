use anyhow::{Context, Result};
use half::f16;
use ndarray::Axis;
use ort::{
    logging::LogLevel,
    session::{builder::GraphOptimizationLevel, Session},
    value::{TensorRef, Value},
};
use std::path::{Path, PathBuf};

use super::fbank::campplus_fbank;
use super::types::EMBEDDING_DIM;

pub struct CampplusEmbedder {
    model_path: PathBuf,
    session: Session,
}

impl CampplusEmbedder {
    pub fn new(model_dir: impl AsRef<Path>, intra_threads: usize) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let model_path = find_model_file(&model_dir)?;
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_log_level(LogLevel::Error)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_intra_threads(intra_threads.max(1))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        Ok(Self {
            model_path,
            session,
        })
    }

    pub fn model_name(&self) -> &str {
        self.model_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("campplus-onnx")
    }

    pub fn embed_pcm(&mut self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let features = campplus_fbank(audio, sample_rate)?;
        if features.nrows() == 0 {
            anyhow::bail!("CAM++ fbank 特征为空");
        }

        let input = features.insert_axis(Axis(0));
        let speech =
            TensorRef::from_array_view(input.view()).map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs! { "feats" => speech })
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let embedding = extract_embedding(&outputs[0])?;
        if embedding.len() != EMBEDDING_DIM {
            tracing::warn!(
                "CAM++ embedding dim {} != expected {}",
                embedding.len(),
                EMBEDDING_DIM
            );
        }
        Ok(l2_normalize(embedding))
    }
}

fn find_model_file(model_dir: &Path) -> Result<PathBuf> {
    for name in [
        "campplus_cn_en_common_200k.onnx",
        "model_quant.onnx",
        "model.onnx",
    ] {
        let path = model_dir.join(name);
        if path.exists() {
            return Ok(path);
        }
    }
    anyhow::bail!("CAM++ 模型文件不存在: {}", model_dir.display());
}

fn extract_embedding(value: &Value) -> Result<Vec<f32>> {
    if let Ok(view) = value.try_extract_array::<f32>() {
        return Ok(view.iter().copied().collect());
    }
    let view = value
        .try_extract_array::<f16>()
        .map_err(|e: ort::Error| anyhow::anyhow!(e.to_string()))?;
    Ok(view.iter().map(|x| x.to_f32()).collect())
}

fn l2_normalize(mut embedding: Vec<f32>) -> Vec<f32> {
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    embedding
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarization::types::EMBEDDING_DIM;

    #[test]
    fn l2_normalize_unit_length() {
        let normalized = l2_normalize(vec![3.0, 4.0]);
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn campplus_embeds_short_audio() {
        let model_dir = std::path::Path::new(env!("HOME")).join("projects/models/campplus-onnx");
        if !model_dir.join("campplus_cn_en_common_200k.onnx").exists() {
            eprintln!("skip: CAM++ model missing at {}", model_dir.display());
            return;
        }

        let mut embedder = CampplusEmbedder::new(&model_dir, 2).expect("load embedder");
        let audio = vec![0.01f32; 16_000];
        let embedding = embedder.embed_pcm(&audio, 16_000).expect("embed");
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }
}
