use anyhow::Result;
use ndarray::Array2;
use std::f32::consts::PI;
use std::path::Path;

pub const TARGET_SAMPLE_RATE: u32 = 16000;
pub const N_MELS: usize = 80;
pub const FRAME_LENGTH_MS: f32 = 25.0;
pub const FRAME_SHIFT_MS: f32 = 10.0;
pub const LFR_M: usize = 7;
pub const LFR_N: usize = 6;

const N_FFT: usize = 512;
const MEL_FMIN_HZ: f32 = 20.0;
const MEL_FMAX_HZ: f32 = 8000.0;

pub struct AudioPreprocessor {
    n_fft: usize,
    hop_length: usize,
    frame_length: usize,
    mel_filterbank: Array2<f32>,
    cmvn_shift: Option<Vec<f32>>,
    cmvn_scale: Option<Vec<f32>>,
}

impl AudioPreprocessor {
    pub fn new(sample_rate: u32) -> Self {
        let frame_length = (FRAME_LENGTH_MS / 1000.0 * sample_rate as f32) as usize;
        let hop_length = (FRAME_SHIFT_MS / 1000.0 * sample_rate as f32) as usize;
        let mel_filterbank =
            create_mel_filterbank(sample_rate, N_FFT, N_MELS, MEL_FMIN_HZ, MEL_FMAX_HZ);

        Self {
            n_fft: N_FFT,
            hop_length,
            frame_length,
            mel_filterbank,
            cmvn_shift: None,
            cmvn_scale: None,
        }
    }

    pub fn load_cmvn_from_file(&mut self, path: &Path) -> Result<()> {
        let (shift, scale) = parse_kaldi_cmvn(path)?;
        if shift.len() != N_MELS * LFR_M || scale.len() != N_MELS * LFR_M {
            anyhow::bail!(
                "CMVN 维度不匹配: shift={}, scale={}, expected={}",
                shift.len(),
                scale.len(),
                N_MELS * LFR_M
            );
        }
        self.cmvn_shift = Some(shift);
        self.cmvn_scale = Some(scale);
        Ok(())
    }

    pub fn process(&self, audio: &[f32], source_sample_rate: u32) -> Result<Array2<f32>> {
        let resampled = if source_sample_rate != TARGET_SAMPLE_RATE {
            resample_audio(audio, source_sample_rate, TARGET_SAMPLE_RATE)?
        } else {
            audio.to_vec()
        };

        let preemphasized = preemphasis(&resampled, 0.97);
        let frames = frame_audio(&preemphasized, self.frame_length, self.hop_length)?;
        let power_spec = compute_power_spectrum(&frames, self.n_fft)?;
        let mel_spec = apply_mel_filterbank(&power_spec, &self.mel_filterbank)?;
        let log_mel_spec = mel_spec.mapv(|x| x.max(f32::EPSILON).ln());

        let mut lfr_features = apply_lfr(&log_mel_spec, LFR_M, LFR_N)?;
        if let (Some(shift), Some(scale)) = (&self.cmvn_shift, &self.cmvn_scale) {
            apply_cmvn(&mut lfr_features, shift, scale)?;
        }

        Ok(lfr_features)
    }
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

fn preemphasis(audio: &[f32], coeff: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(audio.len());
    result.push(audio[0]);
    for i in 1..audio.len() {
        result.push(audio[i] - coeff * audio[i - 1]);
    }
    result
}

fn frame_audio(audio: &[f32], frame_size: usize, hop_length: usize) -> Result<Array2<f32>> {
    if audio.len() < frame_size {
        return Ok(Array2::zeros((1, frame_size)));
    }

    let num_frames = (audio.len() - frame_size) / hop_length + 1;
    let mut frames = Array2::zeros((num_frames, frame_size));
    let window: Vec<f32> = (0..frame_size)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (frame_size - 1) as f32).cos())
        .collect();

    for (i, mut frame) in frames.outer_iter_mut().enumerate() {
        let start = i * hop_length;
        for (j, value) in frame.iter_mut().enumerate() {
            if start + j < audio.len() {
                *value = audio[start + j] * window[j];
            }
        }
    }

    Ok(frames)
}

fn compute_power_spectrum(frames: &Array2<f32>, n_fft: usize) -> Result<Array2<f32>> {
    use realfft::RealFftPlanner;

    let num_frames = frames.nrows();
    let n_freqs = n_fft / 2 + 1;
    let mut power_spec = Array2::zeros((num_frames, n_freqs));

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    for (i, frame) in frames.outer_iter().enumerate() {
        let mut input = frame.to_vec();
        input.resize(n_fft, 0.0);
        let mut output = fft.make_output_vec();
        fft.process(&mut input, &mut output)?;

        for (j, complex) in output.iter().enumerate() {
            power_spec[[i, j]] = complex.norm_sqr();
        }
    }

    Ok(power_spec)
}

fn create_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    f_min_hz: f32,
    f_max_hz: f32,
) -> Array2<f32> {
    let n_freqs = n_fft / 2 + 1;
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);

    let mel_min = hz_to_mel(f_min_hz);
    let mel_max = hz_to_mel(f_max_hz);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + i as f32 * (mel_max - mel_min) / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|m| mel_to_hz(*m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|hz| (n_fft as f32 * hz / sample_rate as f32).round() as usize)
        .collect();

    let mut filterbank = Array2::zeros((n_mels, n_freqs));

    for i in 0..n_mels {
        let left = bin_points[i].min(n_freqs.saturating_sub(1));
        let center = bin_points[i + 1].min(n_freqs);
        let right = bin_points[i + 2].min(n_freqs);
        for j in left..center {
            if center > left {
                filterbank[[i, j]] = (j as f32 - left as f32) / (center as f32 - left as f32);
            }
        }
        for j in center..right {
            if right > center {
                filterbank[[i, j]] = (right as f32 - j as f32) / (right as f32 - center as f32);
            }
        }
    }

    filterbank
}

fn apply_mel_filterbank(
    power_spec: &Array2<f32>,
    mel_filterbank: &Array2<f32>,
) -> Result<Array2<f32>> {
    let num_frames = power_spec.nrows();
    let n_mels = mel_filterbank.nrows();
    let mut mel_spec = Array2::zeros((num_frames, n_mels));

    for i in 0..num_frames {
        for j in 0..n_mels {
            let mut sum = 0.0;
            for k in 0..power_spec.ncols() {
                sum += power_spec[[i, k]] * mel_filterbank[[j, k]];
            }
            mel_spec[[i, j]] = sum;
        }
    }

    Ok(mel_spec)
}

fn apply_lfr(features: &Array2<f32>, m: usize, n: usize) -> Result<Array2<f32>> {
    let t = features.nrows();
    let feat_dim = features.ncols();
    let left_pad = (m - 1) / 2;
    let t_eff = t + left_pad;
    let t_lfr = t_eff.div_ceil(n);
    let output_dim = feat_dim * m;
    let mut output = Array2::zeros((t_lfr, output_dim));

    for i in 0..t_lfr {
        for j in 0..m {
            let global_idx = i * n + j;
            let src_idx = if global_idx < left_pad {
                0
            } else if global_idx - left_pad < t {
                global_idx - left_pad
            } else {
                t.saturating_sub(1)
            };
            for k in 0..feat_dim {
                output[[i, j * feat_dim + k]] = features[[src_idx, k]];
            }
        }
    }

    Ok(output)
}
