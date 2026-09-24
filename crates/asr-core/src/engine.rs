use super::decoder::CtcDecoder;
use super::model_store::resolve_model_dir;
use super::onnx::OnnxInference;
use super::preprocess::{AudioPreprocessor, TARGET_SAMPLE_RATE};
use super::sensevoice::{textnorm_model_id, AnnotatedChunk, LanguageHint, TranscribeOptions};
use super::types::AsrOutput;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub struct AsrEngine {
    model_path: PathBuf,
    inference_threads: usize,
    options: TranscribeOptions,
    preprocessor: Option<AudioPreprocessor>,
    inference: Option<OnnxInference>,
    decoder: Option<CtcDecoder>,
    ready: bool,
    load_error: Option<String>,
}

impl AsrEngine {
    pub fn from_resolved_model_dir(
        model_override: Option<PathBuf>,
        inference_threads: usize,
    ) -> Result<Self> {
        let model_path = resolve_model_dir(model_override)?;
        Ok(Self::new(model_path, inference_threads))
    }

    pub fn new(model_path: PathBuf, inference_threads: usize) -> Self {
        let mut engine = Self {
            model_path,
            inference_threads: inference_threads.max(1),
            options: TranscribeOptions::default(),
            preprocessor: None,
            inference: None,
            decoder: None,
            ready: false,
            load_error: None,
        };

        if let Err(error) = engine.load_model() {
            engine.load_error = Some(error.to_string());
            tracing::warn!("[fastcut-asr] 模型加载失败: {}", error);
        }

        engine
    }

    pub fn set_options(&mut self, options: TranscribeOptions) {
        self.options = options;
    }

    pub fn warmup(&mut self) {
        if !self.ready {
            return;
        }

        let silence = vec![0.0f32; TARGET_SAMPLE_RATE as usize];
        let preprocessor = self.preprocessor.as_ref().unwrap();
        if let Ok(features) = preprocessor.process(&silence, TARGET_SAMPLE_RATE) {
            let inference = self.inference.as_mut().unwrap();
            let _ = inference.infer(
                &features,
                LanguageHint::Auto.model_id(),
                textnorm_model_id(self.options.use_itn),
            );
        }
    }

    pub fn transcribe_file_chunked(
        &mut self,
        audio_path: &Path,
        chunk_duration_ms: u64,
    ) -> Result<AsrOutput> {
        self.transcribe_file_chunked_with_options(audio_path, chunk_duration_ms, self.options)
    }

    pub fn transcribe_file_chunked_with_options(
        &mut self,
        audio_path: &Path,
        chunk_duration_ms: u64,
        options: TranscribeOptions,
    ) -> Result<AsrOutput> {
        self.transcribe_file_chunked_with_options_and_progress(
            audio_path,
            chunk_duration_ms,
            options,
            |_, _, _| {},
        )
    }

    pub fn transcribe_file_chunked_with_options_and_progress(
        &mut self,
        audio_path: &Path,
        chunk_duration_ms: u64,
        options: TranscribeOptions,
        mut on_progress: impl FnMut(usize, usize, f64),
    ) -> Result<AsrOutput> {
        self.options = options;

        if !audio_path.exists() {
            anyhow::bail!("音频文件不存在: {}", audio_path.display());
        }

        let mut reader = hound::WavReader::open(audio_path)
            .with_context(|| format!("无法打开 wav 文件: {}", audio_path.display()))?;
        let spec = reader.spec();
        let sample_rate = spec.sample_rate.max(1);
        let channels = spec.channels.max(1) as usize;
        let total_samples = (reader.duration() as usize / channels.max(1)).max(1);
        let chunk_size = ((sample_rate as u64 * chunk_duration_ms.max(1_000)) / 1000) as usize;
        let mut merged = AsrOutput::empty();
        let mut chunk_index = 0usize;
        let mut processed_samples = 0usize;
        let mut chunk_samples = Vec::<f32>::with_capacity(chunk_size.max(1));
        let mut channel_sum = 0.0f32;
        let mut channel_count = 0usize;

        match spec.sample_format {
            hound::SampleFormat::Float => {
                for sample in reader.samples::<f32>() {
                    let sample = sample?;
                    channel_sum += sample;
                    channel_count += 1;
                    if channel_count == channels {
                        chunk_samples.push(channel_sum / channels as f32);
                        channel_sum = 0.0;
                        channel_count = 0;
                        if chunk_samples.len() >= chunk_size.max(1) {
                            self.process_chunk(
                                &chunk_samples,
                                sample_rate,
                                processed_samples,
                                &mut merged,
                                &mut chunk_index,
                            )?;
                            processed_samples += chunk_samples.len();
                            on_progress(
                                chunk_index,
                                processed_samples,
                                (processed_samples as f64 / total_samples as f64).clamp(0.0, 1.0),
                            );
                            chunk_samples.clear();
                        }
                    }
                }
            }
            hound::SampleFormat::Int => {
                let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                for sample in reader.samples::<i32>() {
                    let sample = sample? as f32 / max_val;
                    channel_sum += sample;
                    channel_count += 1;
                    if channel_count == channels {
                        chunk_samples.push(channel_sum / channels as f32);
                        channel_sum = 0.0;
                        channel_count = 0;
                        if chunk_samples.len() >= chunk_size.max(1) {
                            self.process_chunk(
                                &chunk_samples,
                                sample_rate,
                                processed_samples,
                                &mut merged,
                                &mut chunk_index,
                            )?;
                            processed_samples += chunk_samples.len();
                            on_progress(
                                chunk_index,
                                processed_samples,
                                (processed_samples as f64 / total_samples as f64).clamp(0.0, 1.0),
                            );
                            chunk_samples.clear();
                        }
                    }
                }
            }
        }

        if channel_count > 0 {
            chunk_samples.push(channel_sum / channel_count as f32);
        }

        if !chunk_samples.is_empty() {
            self.process_chunk(
                &chunk_samples,
                sample_rate,
                processed_samples,
                &mut merged,
                &mut chunk_index,
            )?;
            processed_samples += chunk_samples.len();
            on_progress(
                chunk_index,
                processed_samples,
                (processed_samples as f64 / total_samples as f64).clamp(0.0, 1.0),
            );
        }

        if merged.text.trim().is_empty() && merged.chunks.is_empty() {
            anyhow::bail!("ASR 分块转写未返回有效文本");
        }

        Ok(merged)
    }

    pub fn transcribe_pcm(&mut self, samples: &[f32], sample_rate: u32) -> Result<AsrOutput> {
        if !self.ready {
            anyhow::bail!(
                "ASR 模型未就绪: {}",
                self.load_error.as_deref().unwrap_or("unknown")
            );
        }

        let features = self
            .preprocessor
            .as_ref()
            .unwrap()
            .process(samples, sample_rate)?;
        let (logits, _) = self.inference.as_mut().unwrap().infer(
            &features,
            self.options.language.model_id(),
            textnorm_model_id(self.options.use_itn),
        )?;
        let audio_duration_ms =
            ((samples.len() as f64 / sample_rate.max(1) as f64) * 1000.0).round() as u64;
        let decoded = self.decoder.as_ref().unwrap().decode_with_timestamps(
            &logits,
            audio_duration_ms,
            false,
        );

        let chunk = AnnotatedChunk {
            text: decoded.text.clone(),
            formatted_text: decoded.formatted_text.clone(),
            annotation: decoded.annotation.clone(),
            start_ms: 0,
            end_ms: audio_duration_ms,
        };

        Ok(AsrOutput {
            text: decoded.formatted_text,
            confidence: 0.95,
            language: decoded.annotation.language.clone(),
            duration_ms: audio_duration_ms,
            tokens: decoded.tokens,
            chunks: vec![chunk],
        })
    }

    fn load_model(&mut self) -> Result<()> {
        let model_file = find_model_file(&self.model_path).ok_or_else(|| {
            anyhow::anyhow!(
                "模型文件不存在（已查找 model.onnx / model_quant.onnx）: {}",
                self.model_path.display()
            )
        })?;
        let tokens_file = self.model_path.join("tokens.json");

        if !tokens_file.exists() {
            anyhow::bail!("tokens 文件不存在: {}", tokens_file.display());
        }

        let mut preprocessor = AudioPreprocessor::new(TARGET_SAMPLE_RATE);
        let cmvn_file = self.model_path.join("am.mvn");
        if cmvn_file.exists() {
            preprocessor.load_cmvn_from_file(&cmvn_file)?;
        }

        self.preprocessor = Some(preprocessor);
        self.inference = Some(OnnxInference::new(&model_file, self.inference_threads)?);
        self.decoder = Some(CtcDecoder::from_tokens_file(&tokens_file)?);
        self.ready = true;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk_samples: &[f32],
        sample_rate: u32,
        processed_samples: usize,
        merged: &mut AsrOutput,
        chunk_index: &mut usize,
    ) -> Result<()> {
        let mut chunk_output = self.transcribe_pcm(chunk_samples, sample_rate)?;
        let offset_ms = ((processed_samples as f64 / sample_rate as f64) * 1000.0).round() as u64;

        for token in &mut chunk_output.tokens {
            token.start_ms += offset_ms;
            token.end_ms += offset_ms;
        }

        if let Some(chunk) = chunk_output.chunks.first().cloned() {
            let annotated = AnnotatedChunk {
                text: chunk.text,
                formatted_text: chunk.formatted_text,
                annotation: chunk.annotation,
                start_ms: offset_ms,
                end_ms: offset_ms + chunk_output.duration_ms,
            };

            if !annotated.formatted_text.trim().is_empty() {
                if !merged.text.is_empty() {
                    merged.text.push('\n');
                }
                merged.text.push_str(annotated.formatted_text.trim());
            }
            merged.chunks.push(annotated);
        }

        merged.duration_ms += chunk_output.duration_ms;
        merged.tokens.extend(chunk_output.tokens);
        merged.confidence = merged.confidence.max(chunk_output.confidence);
        if merged.language.is_none() {
            merged.language = chunk_output.language;
        }
        *chunk_index += 1;
        tracing::info!("[meeting-minutes] 已完成 ASR 分块 {}", chunk_index);
        Ok(())
    }
}

fn find_model_file(model_path: &Path) -> Option<PathBuf> {
    for name in ["model.onnx", "model_quant.onnx"] {
        let path = model_path.join(name);
        if path.exists() {
            return Some(path);
        }
    }
    None
}
