use anyhow::{Context, Result};
use std::path::Path;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

pub fn load_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
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
            let max_val = (1i64 << (spec.bits_per_sample.saturating_sub(1))) as f32;
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

pub fn resample_audio(audio: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if audio.is_empty() {
        return Ok(Vec::new());
    }
    if from_rate == to_rate {
        return Ok(audio.to_vec());
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = (i as f64) / ratio;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(audio.len().saturating_sub(1));
        let frac = (src_pos - left as f64) as f32;
        let sample = if left == right {
            audio[left]
        } else {
            audio[left] * (1.0 - frac) + audio[right] * frac
        };
        out.push(sample);
    }

    Ok(out)
}

pub fn slice_ms(waveform: &[f32], sample_rate: u32, start_ms: u64, end_ms: u64) -> Vec<f32> {
    if waveform.is_empty() || end_ms <= start_ms {
        return Vec::new();
    }

    let start_sample = ((start_ms as u64 * sample_rate as u64) / 1_000) as usize;
    let end_sample =
        ((end_ms as u64 * sample_rate as u64) / 1_000).min(waveform.len() as u64) as usize;
    if start_sample >= end_sample {
        return Vec::new();
    }
    waveform[start_sample..end_sample].to_vec()
}

pub fn pad_or_trim_min_duration(samples: Vec<f32>, sample_rate: u32, min_ms: u64) -> Vec<f32> {
    let min_samples = ((min_ms as u64 * sample_rate as u64) / 1_000).max(1) as usize;
    if samples.len() >= min_samples {
        return samples;
    }
    let mut padded = samples;
    padded.resize(min_samples, 0.0);
    padded
}
