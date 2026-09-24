use super::e2e_vad::{E2EVadModel, VadPostConfig};
use super::vad_preprocess::VadPreprocessor;
use anyhow::{Context, Result};
use half::f16;
use ndarray::{Array4, ArrayView2, Axis};
use ort::{
    logging::LogLevel,
    session::{builder::GraphOptimizationLevel, Session},
    value::{TensorRef, Value},
};
use serde::Deserialize;
use std::path::{Path, PathBuf};

const CHUNK_FRAMES: usize = 6000;
const FRAME_SHIFT_SAMPLES: usize = 160;
const FRAME_LENGTH_SAMPLES: usize = 400;

#[derive(Debug, Deserialize)]
struct VadYamlConfig {
    frontend_conf: FrontendConf,
    encoder_conf: EncoderConf,
    vad_post_conf: VadPostConfig,
}

#[derive(Debug, Deserialize)]
struct FrontendConf {
    fs: u32,
}

#[derive(Debug, Deserialize)]
struct EncoderConf {
    fsmn_layers: usize,
    proj_dim: usize,
    lorder: usize,
}

pub struct VadEngine {
    model_dir: PathBuf,
    preprocessor: VadPreprocessor,
    session: Session,
    encoder_conf: EncoderConf,
    vad_post_conf: VadPostConfig,
    max_end_sil: i32,
}

impl VadEngine {
    pub fn new(model_dir: impl AsRef<Path>, intra_threads: usize) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let config_path = model_dir.join("vad.yaml");
        let cmvn_path = model_dir.join("vad.mvn");
        let model_path = find_vad_model_file(&model_dir)?;

        let yaml_text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("无法读取 VAD 配置: {}", config_path.display()))?;
        let config: VadYamlConfig = serde_yaml::from_str(&yaml_text)
            .with_context(|| format!("无法解析 VAD 配置: {}", config_path.display()))?;

        let sample_rate = config.frontend_conf.fs.max(1);
        let mut preprocessor = VadPreprocessor::new(sample_rate);
        if cmvn_path.exists() {
            preprocessor.load_cmvn_from_file(&cmvn_path)?;
        }

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

        let max_end_sil = config.vad_post_conf.max_end_silence_time;

        Ok(Self {
            model_dir,
            preprocessor,
            session,
            encoder_conf: config.encoder_conf,
            vad_post_conf: config.vad_post_conf,
            max_end_sil,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Detect speech segments in a WAV file; returns `(start_ms, end_ms)` pairs.
    pub fn detect_segments(&mut self, wav_path: &Path) -> Result<Vec<(u64, u64)>> {
        let (waveform, sample_rate) = load_wav(wav_path)?;
        self.detect_segments_from_pcm(&waveform, sample_rate)
    }

    pub fn detect_segments_from_pcm(
        &mut self,
        waveform: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<(u64, u64)>> {
        let features = self.preprocessor.process(waveform, sample_rate)?;
        let feats_len = features.nrows();
        if feats_len == 0 {
            return Ok(Vec::new());
        }

        let mut vad_scorer = E2EVadModel::new(self.vad_post_conf.clone());
        let mut in_cache = prepare_cache(&self.encoder_conf);
        let mut all_segments = Vec::new();

        let mut t_offset = 0usize;
        while t_offset < feats_len {
            let step = (CHUNK_FRAMES.min(feats_len - t_offset)).max(1);
            let is_final = t_offset + step >= feats_len;

            let feat_chunk = features.slice(ndarray::s![t_offset..t_offset + step, ..]);
            let wave_start = t_offset * FRAME_SHIFT_SAMPLES;
            let wave_end = if is_final {
                waveform.len()
            } else {
                ((t_offset + step).saturating_sub(1)) * FRAME_SHIFT_SAMPLES + FRAME_LENGTH_SAMPLES
            }
            .min(waveform.len());
            let wave_chunk = &waveform[wave_start..wave_end];

            let scores = self.infer_chunk(&feat_chunk, &mut in_cache)?;
            let chunk_segments =
                vad_scorer.process_chunk(&scores, wave_chunk, is_final, self.max_end_sil);
            all_segments.extend(chunk_segments);

            if is_final {
                break;
            }
            t_offset += step;
        }

        Ok(all_segments)
    }

    fn infer_chunk(
        &mut self,
        features: &ArrayView2<f32>,
        in_cache: &mut [Array4<f32>],
    ) -> Result<Vec<Vec<f64>>> {
        let speech = TensorRef::from_array_view(features.view().insert_axis(Axis(0)))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let mut outputs = self
            .session
            .run(ort::inputs! {
                "speech" => speech,
                "in_cache0" => TensorRef::from_array_view(in_cache[0].view())
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?,
                "in_cache1" => TensorRef::from_array_view(in_cache[1].view())
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?,
                "in_cache2" => TensorRef::from_array_view(in_cache[2].view())
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?,
                "in_cache3" => TensorRef::from_array_view(in_cache[3].view())
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?,
            })
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let logits = extract_logits_as_f64(
            &outputs
                .remove("logits")
                .unwrap_or_else(|| outputs.remove("output").expect("missing logits output")),
        )?;

        for (i, key) in ["out_cache0", "out_cache1", "out_cache2", "out_cache3"]
            .iter()
            .enumerate()
        {
            if let Some(value) = outputs.remove(*key) {
                in_cache[i] = extract_array4_f32(&value)?;
            }
        }

        Ok(logits)
    }
}

fn prepare_cache(encoder_conf: &EncoderConf) -> Vec<Array4<f32>> {
    let cache_shape = (1, encoder_conf.proj_dim, encoder_conf.lorder - 1, 1);
    (0..encoder_conf.fsmn_layers)
        .map(|_| Array4::<f32>::zeros(cache_shape))
        .collect()
}

fn find_vad_model_file(model_dir: &Path) -> Result<PathBuf> {
    for name in ["model_quant.onnx", "model.onnx"] {
        let path = model_dir.join(name);
        if path.exists() {
            return Ok(path);
        }
    }
    anyhow::bail!(
        "VAD ONNX 模型不存在（已查找 model_quant.onnx / model.onnx）: {}",
        model_dir.display()
    )
}

fn load_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("无法打开 wav: {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let sample_rate = spec.sample_rate.max(1);

    let samples: Result<Vec<f32>> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(Into::into))
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| Ok(s? as f32 / max_val))
                .collect()
        }
    };

    let raw = samples?;
    if channels == 1 {
        return Ok((raw, sample_rate));
    }

    let mut mono = Vec::with_capacity(raw.len() / channels);
    for chunk in raw.chunks(channels) {
        let sum: f32 = chunk.iter().sum();
        mono.push(sum / channels as f32);
    }
    Ok((mono, sample_rate))
}

fn extract_logits_as_f64(value: &Value) -> Result<Vec<Vec<f64>>> {
    let array = if let Ok(view) = value.try_extract_array::<f32>() {
        view.to_owned()
    } else {
        let view = value
            .try_extract_array::<f16>()
            .map_err(|e: ort::Error| anyhow::anyhow!(e.to_string()))?;
        view.mapv(|x| x.to_f32())
    };

    let view = match array.ndim() {
        3 => array.index_axis(Axis(0), 0).to_owned(),
        2 => array,
        ndim => anyhow::bail!("未预期的 logits 维度: {ndim}"),
    };

    Ok(view
        .outer_iter()
        .map(|row| row.iter().map(|&x| f64::from(x)).collect())
        .collect())
}

fn extract_array4_f32(value: &Value) -> Result<Array4<f32>> {
    if let Ok(view) = value.try_extract_array::<f32>() {
        return Ok(view.to_owned().into_dimensionality()?);
    }
    let view = value
        .try_extract_array::<f16>()
        .map_err(|e: ort::Error| anyhow::anyhow!(e.to_string()))?;
    Ok(view.mapv(|x| x.to_f32()).into_dimensionality()?)
}
