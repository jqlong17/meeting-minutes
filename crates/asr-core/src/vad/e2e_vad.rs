use serde::Deserialize;
use std::f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadStateMachine {
    StartPointNotDetected = 1,
    InSpeechSegment = 2,
    EndPointDetected = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameState {
    Invalid = -1,
    Speech = 1,
    Sil = 0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioChangeState {
    Speech2Speech = 0,
    Speech2Sil = 1,
    Sil2Sil = 2,
    Sil2Speech = 3,
    NoBegin = 4,
    Invalid = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadDetectMode {
    SingleUtterance = 0,
    MultipleUtterance = 1,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VadPostConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: i32,
    #[serde(default = "default_detect_mode")]
    pub detect_mode: i32,
    #[serde(default)]
    pub snr_mode: i32,
    #[serde(default = "default_max_end_silence_time")]
    pub max_end_silence_time: i32,
    #[serde(default = "default_max_start_silence_time")]
    pub max_start_silence_time: i32,
    #[serde(default = "default_true")]
    pub do_start_point_detection: bool,
    #[serde(default = "default_true")]
    pub do_end_point_detection: bool,
    #[serde(default = "default_window_size_ms")]
    pub window_size_ms: i32,
    #[serde(default = "default_sil_to_speech_time_thres")]
    pub sil_to_speech_time_thres: i32,
    #[serde(default = "default_speech_to_sil_time_thres")]
    pub speech_to_sil_time_thres: i32,
    #[serde(default = "default_speech_2_noise_ratio")]
    pub speech_2_noise_ratio: f64,
    #[serde(default = "default_do_extend")]
    pub do_extend: i32,
    #[serde(default = "default_lookback_time_start_point")]
    pub lookback_time_start_point: i32,
    #[serde(default = "default_lookahead_time_end_point")]
    pub lookahead_time_end_point: i32,
    #[serde(default = "default_max_single_segment_time")]
    pub max_single_segment_time: i32,
    #[serde(default = "default_snr_thres")]
    pub snr_thres: f64,
    #[serde(default = "default_noise_frame_num_used_for_snr")]
    pub noise_frame_num_used_for_snr: i32,
    #[serde(default = "default_decibel_thres")]
    pub decibel_thres: f64,
    #[serde(default = "default_speech_noise_thres")]
    pub speech_noise_thres: f64,
    #[serde(default = "default_fe_prior_thres")]
    pub fe_prior_thres: f64,
    #[serde(default = "default_silence_pdf_num")]
    pub silence_pdf_num: i32,
    #[serde(default = "default_sil_pdf_ids")]
    pub sil_pdf_ids: Vec<usize>,
    #[serde(default = "default_speech_noise_thresh_low")]
    pub speech_noise_thresh_low: f64,
    #[serde(default = "default_speech_noise_thresh_high")]
    pub speech_noise_thresh_high: f64,
    #[serde(default)]
    pub output_frame_probs: bool,
    #[serde(default = "default_frame_in_ms")]
    pub frame_in_ms: i32,
    #[serde(default = "default_frame_length_ms")]
    pub frame_length_ms: i32,
}

fn default_sample_rate() -> i32 {
    16000
}
fn default_detect_mode() -> i32 {
    VadDetectMode::MultipleUtterance as i32
}
fn default_max_end_silence_time() -> i32 {
    800
}
fn default_max_start_silence_time() -> i32 {
    3000
}
fn default_true() -> bool {
    true
}
fn default_window_size_ms() -> i32 {
    200
}
fn default_sil_to_speech_time_thres() -> i32 {
    150
}
fn default_speech_to_sil_time_thres() -> i32 {
    150
}
fn default_speech_2_noise_ratio() -> f64 {
    1.0
}
fn default_do_extend() -> i32 {
    1
}
fn default_lookback_time_start_point() -> i32 {
    200
}
fn default_lookahead_time_end_point() -> i32 {
    100
}
fn default_max_single_segment_time() -> i32 {
    60000
}
fn default_snr_thres() -> f64 {
    -100.0
}
fn default_noise_frame_num_used_for_snr() -> i32 {
    100
}
fn default_decibel_thres() -> f64 {
    -100.0
}
fn default_speech_noise_thres() -> f64 {
    0.6
}
fn default_fe_prior_thres() -> f64 {
    0.0001
}
fn default_silence_pdf_num() -> i32 {
    1
}
fn default_sil_pdf_ids() -> Vec<usize> {
    vec![0]
}
fn default_speech_noise_thresh_low() -> f64 {
    -0.1
}
fn default_speech_noise_thresh_high() -> f64 {
    0.3
}
fn default_frame_in_ms() -> i32 {
    10
}
fn default_frame_length_ms() -> i32 {
    25
}

struct E2EVadSpeechBufWithDoa {
    start_ms: i64,
    end_ms: i64,
    contain_seg_start_point: bool,
    contain_seg_end_point: bool,
}

impl E2EVadSpeechBufWithDoa {
    fn new() -> Self {
        Self {
            start_ms: 0,
            end_ms: 0,
            contain_seg_start_point: false,
            contain_seg_end_point: false,
        }
    }

    fn reset(&mut self) {
        self.start_ms = 0;
        self.end_ms = 0;
        self.contain_seg_start_point = false;
        self.contain_seg_end_point = false;
    }
}

struct WindowDetector {
    window_size_ms: i32,
    sil_to_speech_time: i32,
    speech_to_sil_time: i32,
    frame_size_ms: i32,
    win_size_frame: i32,
    win_sum: i32,
    win_state: Vec<i32>,
    cur_win_pos: usize,
    pre_frame_state: FrameState,
    sil_to_speech_frmcnt_thres: i32,
    speech_to_sil_frmcnt_thres: i32,
}

impl WindowDetector {
    fn new(
        window_size_ms: i32,
        sil_to_speech_time: i32,
        speech_to_sil_time: i32,
        frame_size_ms: i32,
    ) -> Self {
        let win_size_frame = window_size_ms / frame_size_ms;
        Self {
            window_size_ms,
            sil_to_speech_time,
            speech_to_sil_time,
            frame_size_ms,
            win_size_frame,
            win_sum: 0,
            win_state: vec![0; win_size_frame as usize],
            cur_win_pos: 0,
            pre_frame_state: FrameState::Sil,
            sil_to_speech_frmcnt_thres: sil_to_speech_time / frame_size_ms,
            speech_to_sil_frmcnt_thres: speech_to_sil_time / frame_size_ms,
        }
    }

    fn reset(&mut self) {
        self.cur_win_pos = 0;
        self.win_sum = 0;
        self.win_state = vec![0; self.win_size_frame as usize];
        self.pre_frame_state = FrameState::Sil;
    }

    fn win_size(&self) -> i32 {
        self.win_size_frame
    }

    fn detect_one_frame(&mut self, frame_state: FrameState, _frame_count: i32) -> AudioChangeState {
        let cur_frame_state = match frame_state {
            FrameState::Speech => 1,
            FrameState::Sil => 0,
            FrameState::Invalid => return AudioChangeState::Invalid,
        };

        self.win_sum -= self.win_state[self.cur_win_pos];
        self.win_sum += cur_frame_state;
        self.win_state[self.cur_win_pos] = cur_frame_state;
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame as usize;

        if self.pre_frame_state == FrameState::Sil
            && self.win_sum >= self.sil_to_speech_frmcnt_thres
        {
            self.pre_frame_state = FrameState::Speech;
            return AudioChangeState::Sil2Speech;
        }

        if self.pre_frame_state == FrameState::Speech
            && self.win_sum <= self.speech_to_sil_frmcnt_thres
        {
            self.pre_frame_state = FrameState::Sil;
            return AudioChangeState::Speech2Sil;
        }

        if self.pre_frame_state == FrameState::Sil {
            AudioChangeState::Sil2Sil
        } else if self.pre_frame_state == FrameState::Speech {
            AudioChangeState::Speech2Speech
        } else {
            AudioChangeState::Invalid
        }
    }
}

/// FunASR E2EVadModel — faithful port of `e2e_vad.py`.
pub struct E2EVadModel {
    opts: VadPostConfig,
    windows_detector: WindowDetector,
    data_buf_start_frame: i32,
    frm_cnt: i32,
    latest_confirmed_speech_frame: i32,
    lastest_confirmed_silence_frame: i32,
    continous_silence_frame_count: i32,
    vad_state_machine: VadStateMachine,
    confirmed_start_frame: i32,
    confirmed_end_frame: i32,
    number_end_time_detected: i32,
    sil_frame: i32,
    sil_pdf_ids: Vec<usize>,
    noise_average_decibel: f64,
    pre_end_silence_detected: bool,
    next_seg: bool,
    output_data_buf: Vec<E2EVadSpeechBufWithDoa>,
    output_data_buf_offset: usize,
    max_end_sil_frame_cnt_thresh: i32,
    speech_noise_thres: f64,
    scores: Vec<Vec<Vec<f64>>>,
    idx_pre_chunk: i32,
    max_time_out: bool,
    decibel: Vec<f64>,
    data_buf_size: i32,
    data_buf_all_size: i32,
    waveform: Vec<f32>,
}

impl E2EVadModel {
    pub fn new(vad_post_args: VadPostConfig) -> Self {
        let windows_detector = WindowDetector::new(
            vad_post_args.window_size_ms,
            vad_post_args.sil_to_speech_time_thres,
            vad_post_args.speech_to_sil_time_thres,
            vad_post_args.frame_in_ms,
        );
        let max_end_sil_frame_cnt_thresh =
            vad_post_args.max_end_silence_time - vad_post_args.speech_to_sil_time_thres;
        let speech_noise_thres = vad_post_args.speech_noise_thres;
        let sil_pdf_ids = vad_post_args.sil_pdf_ids.clone();

        let mut model = Self {
            opts: vad_post_args,
            windows_detector,
            data_buf_start_frame: 0,
            frm_cnt: 0,
            latest_confirmed_speech_frame: 0,
            lastest_confirmed_silence_frame: -1,
            continous_silence_frame_count: 0,
            vad_state_machine: VadStateMachine::StartPointNotDetected,
            confirmed_start_frame: -1,
            confirmed_end_frame: -1,
            number_end_time_detected: 0,
            sil_frame: 0,
            sil_pdf_ids,
            noise_average_decibel: -100.0,
            pre_end_silence_detected: false,
            next_seg: true,
            output_data_buf: Vec::new(),
            output_data_buf_offset: 0,
            max_end_sil_frame_cnt_thresh,
            speech_noise_thres,
            scores: Vec::new(),
            idx_pre_chunk: 0,
            max_time_out: false,
            decibel: Vec::new(),
            data_buf_size: 0,
            data_buf_all_size: 0,
            waveform: Vec::new(),
        };
        model.reset_detection();
        model
    }

    fn all_reset_detection(&mut self) {
        self.data_buf_start_frame = 0;
        self.frm_cnt = 0;
        self.latest_confirmed_speech_frame = 0;
        self.lastest_confirmed_silence_frame = -1;
        self.continous_silence_frame_count = 0;
        self.vad_state_machine = VadStateMachine::StartPointNotDetected;
        self.confirmed_start_frame = -1;
        self.confirmed_end_frame = -1;
        self.number_end_time_detected = 0;
        self.sil_frame = 0;
        self.sil_pdf_ids = self.opts.sil_pdf_ids.clone();
        self.noise_average_decibel = -100.0;
        self.pre_end_silence_detected = false;
        self.next_seg = true;
        self.output_data_buf.clear();
        self.output_data_buf_offset = 0;
        self.max_end_sil_frame_cnt_thresh =
            self.opts.max_end_silence_time - self.opts.speech_to_sil_time_thres;
        self.speech_noise_thres = self.opts.speech_noise_thres;
        self.scores.clear();
        self.idx_pre_chunk = 0;
        self.max_time_out = false;
        self.decibel.clear();
        self.data_buf_size = 0;
        self.data_buf_all_size = 0;
        self.waveform.clear();
        self.reset_detection();
    }

    fn reset_detection(&mut self) {
        self.continous_silence_frame_count = 0;
        self.latest_confirmed_speech_frame = 0;
        self.lastest_confirmed_silence_frame = -1;
        self.confirmed_start_frame = -1;
        self.confirmed_end_frame = -1;
        self.vad_state_machine = VadStateMachine::StartPointNotDetected;
        self.windows_detector.reset();
        self.sil_frame = 0;
    }

    fn compute_decibel(&mut self) {
        let frame_sample_length =
            (self.opts.frame_length_ms * self.opts.sample_rate / 1000) as usize;
        let frame_shift_length = (self.opts.frame_in_ms * self.opts.sample_rate / 1000) as usize;

        if self.data_buf_all_size == 0 {
            self.data_buf_all_size = self.waveform.len() as i32;
            self.data_buf_size = self.data_buf_all_size;
        } else {
            self.data_buf_all_size += self.waveform.len() as i32;
        }

        if self.waveform.len() < frame_sample_length {
            return;
        }

        let mut offset = 0usize;
        while offset + frame_sample_length <= self.waveform.len() {
            let sum_sq: f64 = self.waveform[offset..offset + frame_sample_length]
                .iter()
                .map(|s| f64::from(*s) * f64::from(*s))
                .sum();
            self.decibel.push(10.0 * (sum_sq + 0.000001).log10());
            offset += frame_shift_length;
        }
    }

    fn compute_scores(&mut self, scores: &[Vec<f64>]) {
        let block_size = scores.len() as i32;
        self.frm_cnt += block_size;
        self.scores = vec![scores.to_vec()];
    }

    fn nn_eval_block_size(&self) -> i32 {
        self.scores.first().map(|s| s.len() as i32).unwrap_or(0)
    }

    fn pop_data_buf_till_frame(&mut self, frame_idx: i32) {
        while self.data_buf_start_frame < frame_idx {
            let shift = self.opts.frame_in_ms * self.opts.sample_rate / 1000;
            if self.data_buf_size >= shift {
                self.data_buf_start_frame += 1;
                self.data_buf_size = self.data_buf_all_size - self.data_buf_start_frame * shift;
            } else {
                break;
            }
        }
    }

    fn pop_data_to_output_buf(
        &mut self,
        start_frm: i32,
        frm_cnt: i32,
        first_frm_is_start_point: bool,
        last_frm_is_end_point: bool,
        end_point_is_sent_end: bool,
    ) {
        self.pop_data_buf_till_frame(start_frm);
        let mut expected_sample_number =
            (frm_cnt * self.opts.sample_rate * self.opts.frame_in_ms / 1000) as i32;

        if last_frm_is_end_point {
            let extra_sample = ((self.opts.frame_length_ms * self.opts.sample_rate / 1000)
                - (self.opts.sample_rate * self.opts.frame_in_ms / 1000))
                .max(0);
            expected_sample_number += extra_sample;
        }
        if end_point_is_sent_end {
            expected_sample_number = expected_sample_number.max(self.data_buf_size);
        }

        if self.output_data_buf.is_empty() || first_frm_is_start_point {
            self.output_data_buf.push(E2EVadSpeechBufWithDoa::new());
            let cur_seg = self.output_data_buf.last_mut().unwrap();
            cur_seg.reset();
            cur_seg.start_ms = i64::from(start_frm * self.opts.frame_in_ms);
            cur_seg.end_ms = cur_seg.start_ms;
        }

        let cur_seg = self.output_data_buf.last_mut().unwrap();
        let data_to_pop = if end_point_is_sent_end {
            expected_sample_number
        } else {
            frm_cnt * self.opts.frame_in_ms * self.opts.sample_rate / 1000
        }
        .min(self.data_buf_size);

        self.data_buf_start_frame += frm_cnt;
        cur_seg.end_ms = i64::from((start_frm + frm_cnt) * self.opts.frame_in_ms);
        if first_frm_is_start_point {
            cur_seg.contain_seg_start_point = true;
        }
        if last_frm_is_end_point {
            cur_seg.contain_seg_end_point = true;
        }
        let _ = data_to_pop;
    }

    fn on_silence_detected(&mut self, valid_frame: i32) {
        self.lastest_confirmed_silence_frame = valid_frame;
        if self.vad_state_machine == VadStateMachine::StartPointNotDetected {
            self.pop_data_buf_till_frame(valid_frame);
        }
    }

    fn on_voice_detected(&mut self, valid_frame: i32) {
        self.latest_confirmed_speech_frame = valid_frame;
        self.pop_data_to_output_buf(valid_frame, 1, false, false, false);
    }

    fn on_voice_start(&mut self, start_frame: i32, fake_result: bool) {
        if self.opts.do_start_point_detection {
            // callback placeholder
        }
        if self.confirmed_start_frame != -1 {
            // not reset vad properly
        } else {
            self.confirmed_start_frame = start_frame;
        }

        if !fake_result && self.vad_state_machine == VadStateMachine::StartPointNotDetected {
            self.pop_data_to_output_buf(self.confirmed_start_frame, 1, true, false, false);
        }
    }

    fn on_voice_end(&mut self, end_frame: i32, fake_result: bool, is_last_frame: bool) {
        for t in (self.latest_confirmed_speech_frame + 1)..end_frame {
            self.on_voice_detected(t);
        }
        if self.opts.do_end_point_detection {
            // callback placeholder
        }
        if self.confirmed_end_frame != -1 {
            // not reset vad properly
        } else {
            self.confirmed_end_frame = end_frame;
        }
        if !fake_result {
            self.sil_frame = 0;
            self.pop_data_to_output_buf(self.confirmed_end_frame, 1, false, true, is_last_frame);
            self.number_end_time_detected += 1;
        }
    }

    fn maybe_on_voice_end_if_last_frame(&mut self, is_final_frame: bool, cur_frm_idx: i32) {
        if is_final_frame {
            self.on_voice_end(cur_frm_idx, false, true);
            self.vad_state_machine = VadStateMachine::EndPointDetected;
        }
    }

    fn latency_frm_num_at_start_point(&self) -> i32 {
        let mut vad_latency = self.windows_detector.win_size();
        if self.opts.do_extend != 0 {
            vad_latency += self.opts.lookback_time_start_point / self.opts.frame_in_ms;
        }
        vad_latency
    }

    fn get_frame_state(&mut self, t: i32) -> FrameState {
        let cur_decibel = self.decibel.get(t as usize).copied().unwrap_or(-100.0);
        let cur_snr = cur_decibel - self.noise_average_decibel;

        if cur_decibel < self.opts.decibel_thres {
            let frame_state = FrameState::Sil;
            self.detect_one_frame(frame_state, t, false);
            return frame_state;
        }

        let score_frame = (t - self.idx_pre_chunk) as usize;
        let batch_scores = match self.scores.first() {
            Some(s) if score_frame < s.len() => &s[score_frame],
            _ => return FrameState::Sil,
        };

        let sil_sum: f64 = self
            .sil_pdf_ids
            .iter()
            .filter_map(|&id| batch_scores.get(id).copied())
            .sum();

        let noise_prob = sil_sum.ln() * self.opts.speech_2_noise_ratio;
        let speech_prob = (1.0 - sil_sum).max(f64::MIN_POSITIVE).ln();

        let frame_state = if speech_prob.exp() >= noise_prob.exp() + self.speech_noise_thres {
            if cur_snr >= self.opts.snr_thres && cur_decibel >= self.opts.decibel_thres {
                FrameState::Speech
            } else {
                FrameState::Sil
            }
        } else {
            FrameState::Sil
        };

        if self.noise_average_decibel < -99.9 {
            self.noise_average_decibel = cur_decibel;
        } else {
            self.noise_average_decibel = (cur_decibel
                + self.noise_average_decibel
                    * f64::from(self.opts.noise_frame_num_used_for_snr - 1))
                / f64::from(self.opts.noise_frame_num_used_for_snr);
        }

        frame_state
    }

    pub fn process_chunk(
        &mut self,
        scores: &[Vec<f64>],
        waveform: &[f32],
        is_final: bool,
        max_end_sil: i32,
    ) -> Vec<(u64, u64)> {
        self.max_end_sil_frame_cnt_thresh = max_end_sil - self.opts.speech_to_sil_time_thres;
        self.waveform = waveform.to_vec();
        self.compute_decibel();
        self.compute_scores(scores);

        if is_final {
            self.detect_last_frames();
        } else {
            self.detect_common_frames();
        }

        let mut segments = Vec::new();
        for i in self.output_data_buf_offset..self.output_data_buf.len() {
            let buf = &self.output_data_buf[i];
            if !is_final && (!buf.contain_seg_start_point || !buf.contain_seg_end_point) {
                continue;
            }
            segments.push((buf.start_ms.max(0) as u64, buf.end_ms.max(0) as u64));
            self.output_data_buf_offset += 1;
        }

        if is_final {
            self.all_reset_detection();
        }

        segments
    }

    fn detect_common_frames(&mut self) {
        if self.vad_state_machine == VadStateMachine::EndPointDetected {
            return;
        }
        let block = self.nn_eval_block_size();
        for i in (0..block).rev() {
            let frame_idx = self.frm_cnt - 1 - i;
            let frame_state = self.get_frame_state(frame_idx);
            self.detect_one_frame(frame_state, frame_idx, false);
        }
        self.idx_pre_chunk += block;
    }

    fn detect_last_frames(&mut self) {
        if self.vad_state_machine == VadStateMachine::EndPointDetected {
            return;
        }
        let block = self.nn_eval_block_size();
        for i in (0..block).rev() {
            let frame_idx = self.frm_cnt - 1 - i;
            let frame_state = self.get_frame_state(frame_idx);
            if i != 0 {
                self.detect_one_frame(frame_state, frame_idx, false);
            } else {
                self.detect_one_frame(frame_state, self.frm_cnt - 1, true);
            }
        }
    }

    fn detect_one_frame(
        &mut self,
        cur_frm_state: FrameState,
        cur_frm_idx: i32,
        is_final_frame: bool,
    ) {
        let tmp_cur_frm_state = match cur_frm_state {
            FrameState::Speech => {
                if (1.0f64 - self.opts.fe_prior_thres).abs() > self.opts.fe_prior_thres {
                    FrameState::Speech
                } else {
                    FrameState::Sil
                }
            }
            FrameState::Sil => FrameState::Sil,
            FrameState::Invalid => FrameState::Invalid,
        };

        let state_change = self
            .windows_detector
            .detect_one_frame(tmp_cur_frm_state, cur_frm_idx);
        let frm_shift_in_ms = self.opts.frame_in_ms;

        match state_change {
            AudioChangeState::Sil2Speech => {
                self.continous_silence_frame_count = 0;
                self.pre_end_silence_detected = false;
                if self.vad_state_machine == VadStateMachine::StartPointNotDetected {
                    let start_frame = self
                        .data_buf_start_frame
                        .max(cur_frm_idx - self.latency_frm_num_at_start_point());
                    self.on_voice_start(start_frame, false);
                    self.vad_state_machine = VadStateMachine::InSpeechSegment;
                    for t in (start_frame + 1)..=cur_frm_idx {
                        self.on_voice_detected(t);
                    }
                } else if self.vad_state_machine == VadStateMachine::InSpeechSegment {
                    for t in (self.latest_confirmed_speech_frame + 1)..cur_frm_idx {
                        self.on_voice_detected(t);
                    }
                    if cur_frm_idx - self.confirmed_start_frame + 1
                        > self.opts.max_single_segment_time / frm_shift_in_ms
                    {
                        self.on_voice_end(cur_frm_idx, false, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if !is_final_frame {
                        self.on_voice_detected(cur_frm_idx);
                    } else {
                        self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx);
                    }
                }
            }
            AudioChangeState::Speech2Sil => {
                self.continous_silence_frame_count = 0;
                if self.vad_state_machine == VadStateMachine::InSpeechSegment {
                    if cur_frm_idx - self.confirmed_start_frame + 1
                        > self.opts.max_single_segment_time / frm_shift_in_ms
                    {
                        self.on_voice_end(cur_frm_idx, false, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if !is_final_frame {
                        self.on_voice_detected(cur_frm_idx);
                    } else {
                        self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx);
                    }
                }
            }
            AudioChangeState::Speech2Speech => {
                self.continous_silence_frame_count = 0;
                if self.vad_state_machine == VadStateMachine::InSpeechSegment {
                    if cur_frm_idx - self.confirmed_start_frame + 1
                        > self.opts.max_single_segment_time / frm_shift_in_ms
                    {
                        self.max_time_out = true;
                        self.on_voice_end(cur_frm_idx, false, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if !is_final_frame {
                        self.on_voice_detected(cur_frm_idx);
                    } else {
                        self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx);
                    }
                }
            }
            AudioChangeState::Sil2Sil => {
                self.continous_silence_frame_count += 1;
                if self.vad_state_machine == VadStateMachine::StartPointNotDetected {
                    let silence_timeout = self.continous_silence_frame_count * frm_shift_in_ms
                        > self.opts.max_start_silence_time;
                    let single_mode =
                        self.opts.detect_mode == VadDetectMode::SingleUtterance as i32;
                    if (single_mode && silence_timeout)
                        || (is_final_frame && self.number_end_time_detected == 0)
                    {
                        for t in (self.lastest_confirmed_silence_frame + 1)..cur_frm_idx {
                            self.on_silence_detected(t);
                        }
                        self.on_voice_start(0, true);
                        self.on_voice_end(0, true, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if cur_frm_idx >= self.latency_frm_num_at_start_point() {
                        self.on_silence_detected(
                            cur_frm_idx - self.latency_frm_num_at_start_point(),
                        );
                    }
                } else if self.vad_state_machine == VadStateMachine::InSpeechSegment {
                    if self.continous_silence_frame_count * frm_shift_in_ms
                        >= self.max_end_sil_frame_cnt_thresh
                    {
                        let mut lookback_frame =
                            self.max_end_sil_frame_cnt_thresh / frm_shift_in_ms;
                        if self.opts.do_extend != 0 {
                            lookback_frame -= self.opts.lookahead_time_end_point / frm_shift_in_ms;
                        }
                        lookback_frame -= 1;
                        lookback_frame = lookback_frame.max(0);
                        self.on_voice_end(cur_frm_idx - lookback_frame, false, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if cur_frm_idx - self.confirmed_start_frame + 1
                        > self.opts.max_single_segment_time / frm_shift_in_ms
                    {
                        self.on_voice_end(cur_frm_idx, false, false);
                        self.vad_state_machine = VadStateMachine::EndPointDetected;
                    } else if self.opts.do_extend != 0 && !is_final_frame {
                        if self.continous_silence_frame_count
                            <= self.opts.lookahead_time_end_point / frm_shift_in_ms
                        {
                            self.on_voice_detected(cur_frm_idx);
                        } else {
                            self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx);
                        }
                    }
                }
            }
            _ => {}
        }

        if self.vad_state_machine == VadStateMachine::EndPointDetected
            && self.opts.detect_mode == VadDetectMode::MultipleUtterance as i32
        {
            self.reset_detection();
        }
    }
}
