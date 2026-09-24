use anyhow::Result;
use kaldi_native_fbank::fbank::{FbankComputer, FbankOptions};
use kaldi_native_fbank::online::{FeatureComputer, OnlineFeature};
use kaldi_native_fbank::window::FrameOptions;
use ndarray::Array2;
use std::path::Path;

pub const TARGET_SAMPLE_RATE: u32 = 16000;
pub const N_MELS: usize = 80;
pub const FRAME_LENGTH_MS: f32 = 25.0;
pub const FRAME_SHIFT_MS: f32 = 10.0;
pub const LFR_M: usize = 5;
pub const LFR_N: usize = 1;
pub const INPUT_DIM: usize = N_MELS * LFR_M;

const WAVEFORM_SCALE: f32 = 32768.0;

pub struct VadPreprocessor {
    fbank_opts: FbankOptions,
    cmvn_shift: Option<Vec<f32>>,
    cmvn_scale: Option<Vec<f32>>,
}

impl VadPreprocessor {
    pub fn new(sample_rate: u32) -> Self {
        let mut frame_opts = FrameOptions::default();
        frame_opts.samp_freq = sample_rate as f32;
        frame_opts.dither = 0.0;
        frame_opts.window_type = "hamming".to_string();
        frame_opts.frame_shift_ms = FRAME_SHIFT_MS;
        frame_opts.frame_length_ms = FRAME_LENGTH_MS;
        frame_opts.snip_edges = true;

        let mut fbank_opts = FbankOptions::default();
        fbank_opts.frame_opts = frame_opts;
        fbank_opts.mel_opts.num_bins = N_MELS;
        fbank_opts.use_energy = false;
        fbank_opts.energy_floor = 0.0;
        fbank_opts.use_log_fbank = true;
        fbank_opts.use_power = true;

        Self {
            fbank_opts,
            cmvn_shift: None,
            cmvn_scale: None,
        }
    }

    pub fn load_cmvn_from_file(&mut self, path: &Path) -> Result<()> {
        let (shift, scale) = parse_kaldi_cmvn(path)?;
        if shift.len() != INPUT_DIM || scale.len() != INPUT_DIM {
            anyhow::bail!(
                "VAD CMVN 维度不匹配: shift={}, scale={}, expected={}",
                shift.len(),
                scale.len(),
                INPUT_DIM
            );
        }
        self.cmvn_shift = Some(shift);
        self.cmvn_scale = Some(scale);
        Ok(())
    }

    /// FunASR WavFrontend: scale waveform by 2^15, kaldi fbank, LFR (5/1), CMVN.
    pub fn process(&self, audio: &[f32], source_sample_rate: u32) -> Result<Array2<f32>> {
        let resampled = if source_sample_rate != TARGET_SAMPLE_RATE {
            resample_audio(audio, source_sample_rate, TARGET_SAMPLE_RATE)?
        } else {
            audio.to_vec()
        };

        let scaled: Vec<f32> = resampled.iter().map(|s| s * WAVEFORM_SCALE).collect();
        let mel_spec = compute_kaldi_fbank(&scaled, &self.fbank_opts)?;
        let mut lfr_features = apply_lfr_vad(&mel_spec, LFR_M, LFR_N)?;
        if let (Some(shift), Some(scale)) = (&self.cmvn_shift, &self.cmvn_scale) {
            apply_cmvn(&mut lfr_features, shift, scale)?;
        }

        Ok(lfr_features)
    }
}

fn compute_kaldi_fbank(waveform: &[f32], opts: &FbankOptions) -> Result<Array2<f32>> {
    let computer =
        FeatureComputer::Fbank(FbankComputer::new(opts.clone()).map_err(|e| anyhow::anyhow!(e))?);
    let mut online = OnlineFeature::new(computer);
    online.accept_waveform(opts.frame_opts.samp_freq, waveform);
    online.input_finished();

    let num_frames = online.num_frames_ready();
    let dim = opts.mel_opts.num_bins;
    let mut mel_spec = Array2::<f32>::zeros((num_frames, dim));
    for i in 0..num_frames {
        if let Some(frame) = online.get_frame(i) {
            for (j, &value) in frame.iter().enumerate() {
                mel_spec[[i, j]] = value;
            }
        }
    }
    Ok(mel_spec)
}

fn parse_kaldi_cmvn(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
    let content = std::fs::read_to_string(path)?;

    fn extract_vec_after(content: &str, marker: &str) -> Option<Vec<f32>> {
        let marker_start = content.find(marker)?;
        let start = content[marker_start..].find('[')? + marker_start + 1;
        let end = content[start..].find(']')? + start;
        let values = content[start..end]
            .split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect::<Vec<_>>();

        if values.is_empty() {
            None
        } else {
            Some(values)
        }
    }

    let shift = extract_vec_after(&content, "<AddShift>")
        .ok_or_else(|| anyhow::anyhow!("无法解析 <AddShift> 向量"))?;
    let scale = extract_vec_after(&content, "<Rescale>")
        .ok_or_else(|| anyhow::anyhow!("无法解析 <Rescale> 向量"))?;
    Ok((shift, scale))
}

fn apply_cmvn(features: &mut Array2<f32>, shift: &[f32], scale: &[f32]) -> Result<()> {
    let dim = features.ncols();
    if shift.len() != dim || scale.len() != dim {
        anyhow::bail!(
            "CMVN 维度不匹配: feat_dim={}, shift={}, scale={}",
            dim,
            shift.len(),
            scale.len()
        );
    }

    for mut row in features.outer_iter_mut() {
        for index in 0..dim {
            row[index] = (row[index] + shift[index]) * scale[index];
        }
    }

    Ok(())
}

/// FunASR `apply_lfr`: left-pad (lfr_m-1)//2 frames, T_lfr = ceil(T/lfr_n).
fn apply_lfr_vad(features: &Array2<f32>, lfr_m: usize, lfr_n: usize) -> Result<Array2<f32>> {
    let t = features.nrows();
    let feat_dim = features.ncols();
    let left_pad = (lfr_m - 1) / 2;
    let t_lfr = t.div_ceil(lfr_n);
    let output_dim = feat_dim * lfr_m;
    let mut output = Array2::zeros((t_lfr, output_dim));

    let padded_t = t + left_pad;
    for i in 0..t_lfr {
        let start = i * lfr_n;
        if lfr_m <= padded_t - start {
            for j in 0..lfr_m {
                let src_idx = if start + j < left_pad {
                    0
                } else {
                    (start + j - left_pad).min(t.saturating_sub(1))
                };
                for k in 0..feat_dim {
                    output[[i, j * feat_dim + k]] = features[[src_idx, k]];
                }
            }
        } else {
            let mut frame = vec![0.0f32; output_dim];
            let available = padded_t.saturating_sub(start);
            for j in 0..available {
                let src_idx = if start + j < left_pad {
                    0
                } else {
                    (start + j - left_pad).min(t.saturating_sub(1))
                };
                for k in 0..feat_dim {
                    frame[j * feat_dim + k] = features[[src_idx, k]];
                }
            }
            let last_start = (available.saturating_sub(1)) * feat_dim;
            for j in available..lfr_m {
                for k in 0..feat_dim {
                    frame[j * feat_dim + k] = frame[last_start + k];
                }
            }
            for (k, &v) in frame.iter().enumerate() {
                output[[i, k]] = v;
            }
        }
    }

    Ok(output)
}

fn resample_audio(audio: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
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
        let right = (left + 1).min(audio.len() - 1);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vad_lfr_output_dim_is_400() {
        let feats = Array2::from_shape_fn((10, N_MELS), |(i, j)| (i + j) as f32);
        let lfr = apply_lfr_vad(&feats, LFR_M, LFR_N).unwrap();
        assert_eq!(lfr.ncols(), INPUT_DIM);
        assert_eq!(lfr.nrows(), 10);
    }
}
