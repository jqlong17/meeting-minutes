use anyhow::Result;
use kaldi_native_fbank::fbank::{FbankComputer, FbankOptions};
use kaldi_native_fbank::online::{FeatureComputer, OnlineFeature};
use kaldi_native_fbank::window::FrameOptions;
use ndarray::Array2;

use super::audio::TARGET_SAMPLE_RATE;

pub const N_MELS: usize = 80;
const WAVEFORM_SCALE: f32 = 32_768.0;

pub fn campplus_fbank(audio: &[f32], source_sample_rate: u32) -> Result<Array2<f32>> {
    let resampled = if source_sample_rate != TARGET_SAMPLE_RATE {
        super::audio::resample_audio(audio, source_sample_rate, TARGET_SAMPLE_RATE)?
    } else {
        audio.to_vec()
    };

    if resampled.is_empty() {
        return Ok(Array2::<f32>::zeros((0, N_MELS)));
    }

    let scaled: Vec<f32> = resampled.iter().map(|s| s * WAVEFORM_SCALE).collect();
    let mut fbank = compute_kaldi_fbank(&scaled)?;
    apply_per_utterance_mean(&mut fbank);
    Ok(fbank)
}

fn compute_kaldi_fbank(waveform: &[f32]) -> Result<Array2<f32>> {
    let mut frame_opts = FrameOptions::default();
    frame_opts.samp_freq = TARGET_SAMPLE_RATE as f32;
    frame_opts.dither = 0.0;
    frame_opts.window_type = "hamming".to_string();
    frame_opts.frame_shift_ms = 10.0;
    frame_opts.frame_length_ms = 25.0;
    frame_opts.snip_edges = true;

    let mut fbank_opts = FbankOptions::default();
    fbank_opts.frame_opts = frame_opts;
    fbank_opts.mel_opts.num_bins = N_MELS;
    fbank_opts.use_energy = false;
    fbank_opts.energy_floor = 0.0;
    fbank_opts.use_log_fbank = true;
    fbank_opts.use_power = true;

    let computer = FeatureComputer::Fbank(
        FbankComputer::new(fbank_opts.clone()).map_err(|e| anyhow::anyhow!(e))?,
    );
    let mut online = OnlineFeature::new(computer);
    online.accept_waveform(fbank_opts.frame_opts.samp_freq, waveform);
    online.input_finished();

    let num_frames = online.num_frames_ready();
    let dim = fbank_opts.mel_opts.num_bins;
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

fn apply_per_utterance_mean(features: &mut Array2<f32>) {
    if features.nrows() == 0 {
        return;
    }
    let mean = features.mean_axis(ndarray::Axis(0)).unwrap();
    for mut row in features.outer_iter_mut() {
        for (idx, value) in row.iter_mut().enumerate() {
            *value -= mean[idx];
        }
    }
}
